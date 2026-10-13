# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Nemotron4 vision-language model for processor-expanded image token sequences.

Nemotron4 uses the same media contract as Nemotron Omni: the processor emits one
image placeholder per projected image feature, and only those positions are
replaced in place. Insertion is one-for-one and length preserving, so the model
inserts media into the full token stream and then selects its own
context-parallel shard without rebuilding packed metadata.

The encoder is a Pixtral-style ViT (2D interleaved RoPE, gated SiLU MLP, RMSNorm,
no class token) whose 2x2 spatial patch merger runs inside the encoder, followed
by a two-layer non-gated MLP projector. The language stack is a ``HybridModel``
built from the provider's stack spec; the wide-residual, GDP and shortcut-MoE
layers it needs come from the Nemotron4 Megatron-LM fork.

Submodule names (``language_model``, ``vision_model``, ``vision_projection``)
match MCore ``LLaVAModel`` so the native non-MIMO training checkpoints load
unchanged and the HF weight mappings in :mod:`nemotron4_bridge` apply directly.

This forward is the production NM4 SFT path: packed THD context parallelism with
configurable linear/attention CP layouts, per-sequence zigzag outputs for
NeMo RL's loss reconstruction, optional vision sharding across language CP ranks
(``vision_dp_over_cp``), and GTP-safe vision process groups.
"""

import copy
import logging
from collections import namedtuple
from typing import Callable, Optional, Self

import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.context_parallel.layout import convert_cp_layout
from megatron.core.context_parallel.utils import get_batches_on_this_cp_rank
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.multimodal.context_parallel import (
    gather_from_context_parallel_ranks_dynamic_res,
    split_to_context_parallel_ranks_dynamic_res,
)
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.vit_model import ViTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_product import GatedDeltaProductMixer
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group

from megatron.bridge.models.logit_dtype import logit_dtype_kwarg


logger = logging.getLogger(__name__)


class Nemotron4GatedDeltaProductMixer(GatedDeltaProductMixer):
    """Materialize CP-sharded convolution inputs when their row stride is unaligned."""

    def _prepare_qkv(self, x, *args, **kwargs):
        # causal-conv1d's channel-last kernel requires the batch and sequence
        # strides after [B, L, D] -> [B, D, L] to be multiples of eight. CP8
        # can leave ``x`` as a slice of a wider projection even though its own
        # payload is aligned. A compact copy changes only the storage stride;
        # values, parameter shapes, and checkpoint keys stay unchanged.
        if x.stride(0) % 8 or x.stride(1) % 8:
            x = x.contiguous()
        return super()._prepare_qkv(x, *args, **kwargs)


class Nemotron4VisionMLP(MLP):
    """Keep TP1 GTP checkpoint shards contiguous across the GLU gate/up halves."""

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None
    ) -> ShardedStateDict:
        state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        if self.config.gated_linear_unit and self.tp_group.size() == 1:
            from megatron.core.tensor_parallel.gtp_api import is_gtp_param

            if is_gtp_param(self.linear_fc1.weight):
                # The generic GLU factory treats the GTP shard as a TP shard and
                # loads [gate_0, up_0] / [gate_1, up_1]. GTP gathers row shards
                # without interleaving, so TP1 must load [gate] / [up] instead.
                metadata = ensure_metadata_has_dp_cp_group(metadata)
                fc1_prefix = f"{prefix}linear_fc1."
                fc1_state = self.linear_fc1.sharded_state_dict(fc1_prefix, sharded_offsets, metadata)
                state_dict[f"{fc1_prefix}weight"] = fc1_state[f"{fc1_prefix}weight"]
        return state_dict


def _ignore_transformer_engine_extra_state(module: torch.nn.Module, incompatible_keys: namedtuple) -> None:
    """Allow checkpoints that store no TE extra state to load."""
    del module
    for mismatch_kind, keys in incompatible_keys._asdict().items():
        ignored = [key for key in keys if "extra_state" in key]
        if not ignored:
            continue
        for key in ignored:
            keys.remove(key)
        logger.warning(
            "Ignoring %d Transformer Engine %s _extra_state entries; first entries: %s%s",
            len(ignored),
            mismatch_kind,
            ", ".join(ignored[:5]),
            " ..." if len(ignored) > 5 else "",
        )


def _build_vision_packed_seq_params(imgs_sizes: torch.Tensor, patch_dim: int, device: torch.device) -> PackedSeqParams:
    """Build one THD vision-attention sequence per input image."""
    sequence_lengths = [
        (int(height) // patch_dim) * (int(width) // patch_dim) for height, width in imgs_sizes.tolist()
    ]
    cumulative_lengths = [0]
    for sequence_length in sequence_lengths:
        cumulative_lengths.append(cumulative_lengths[-1] + sequence_length)

    cu_seqlens = torch.tensor(cumulative_lengths, dtype=torch.int32, device=device)
    max_seqlen = max(sequence_lengths, default=0)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )


class _WithoutTEExtraState:
    """Exclude TE metadata absent from the native NM4 checkpoints, as :class:`Nemotron4VLModel` does."""

    def sharded_state_dict(self, *args, **kwargs) -> ShardedStateDict:
        from megatron.bridge.training.checkpointing import delete_extra_state

        state = super().sharded_state_dict(*args, **kwargs)
        delete_extra_state(state)
        return state


class Nemotron4MIMOVisionEncoder(_WithoutTEExtraState, ViTModel):
    """Pixtral-style ViT of the MIMO ``images`` module."""

    def forward(
        self,
        x: torch.Tensor,
        imgs_sizes: torch.Tensor,
        packed_seq_params: PackedSeqParams,
        num_frames: torch.Tensor | None = None,
        expected_output_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode the MIMO packed-image payload, one ``imgs_sizes`` row per image.

        Images have one frame, so ``num_frames`` is not used. The media balancer checks the
        output rows against ``expected_output_counts``.
        """
        del num_frames, expected_output_counts
        # The ViT reads the patch grid on the host; MIMO payloads carry it on the GPU.
        return super().forward(x, imgs_sizes=imgs_sizes.cpu(), packed_seq_params=packed_seq_params)


class Nemotron4MIMOProjector(_WithoutTEExtraState, MultimodalProjector):
    """Vision projector of the MIMO ``images`` module."""


class Nemotron4MIMOLanguageModel(_WithoutTEExtraState, HybridModel):
    """Hybrid decoder retaining Nemotron4's FP32 state during MIMO dtype casts.

    MIMO calls ``bfloat16()`` on the complete model. The residual controllers,
    GDP decay parameters, and router bookkeeping must keep their original FP32
    values, just as the HF model's strict FP32 module list requires.
    """

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> Self:
        retained = []
        for name, tensor in list(self.named_parameters()) + list(self.named_buffers()):
            if (
                getattr(tensor, "is_residual_map_parameter", False)
                or getattr(tensor, "is_wide_residual_retention_parameter", False)
                or name.endswith((".A_log", ".expert_bias", ".qb_bin_bounds"))
            ):
                grad = tensor.grad.detach() if tensor.grad is not None else None
                retained.append((name, isinstance(tensor, torch.nn.Parameter), tensor.detach(), grad))
        super()._apply(fn, recurse=recurse)
        # Keep parameter objects and their parallelism/optimizer attributes. A
        # detached reference retains the original storage across Module._apply.
        for name, is_parameter, original, original_grad in retained:
            tensor = self.get_parameter(name) if is_parameter else self.get_buffer(name)
            tensor.data = original.to(device=tensor.device, dtype=torch.float32)
            if original_grad is not None:
                tensor.grad.data = original_grad.to(device=tensor.device, dtype=torch.float32)
        return self


class Nemotron4VLModel(MegatronModule):
    """Nemotron4 VLM: Pixtral-style ViT + MLP projector + hybrid language model.

    The input sequence already carries one placeholder per projected image
    feature. Unlike MCore ``LLaVAModel`` this model does not collapse and
    re-expand media runs; insertion is one-for-one and length preserving, so
    the model can insert media into the full token stream and then select its
    own context-parallel shard, which is what
    ``model_slices_context_parallel_inputs`` promises NeMo RL.
    """

    model_owns_packing = False
    model_owns_mtp_loss_mask_packing = False
    model_slices_context_parallel_inputs = True

    def __init__(
        self,
        *,
        language_transformer_config: TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        vision_transformer_config: TransformerConfig,
        vision_transformer_layer_spec: ModuleSpec,
        vision_projection_config: TransformerConfig,
        vision_projection_layer_spec,
        image_token_index: int,
        img_h: int,
        img_w: int,
        patch_dim: int,
        spatial_merge_size: int,
        hybrid_layer_pattern: Optional[str],
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        language_position_embedding_type: str = "none",
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        pg_collection: ProcessGroupCollection | None = None,
        vision_dp_over_cp: bool = True,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=language_transformer_config)
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.vocab_size = language_vocab_size
        self.image_token_index = image_token_index
        self.patch_dim = patch_dim
        self.spatial_merge_size = spatial_merge_size
        self.sequence_parallel_lm = language_transformer_config.sequence_parallel
        self.context_parallel_lm = language_transformer_config.context_parallel_size
        self.vision_dp_over_cp = vision_dp_over_cp and self.context_parallel_lm > 1
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.encoder_hidden_state = None
        self.pg_collection = pg_collection

        self.language_model = None
        if add_decoder:
            self.language_model = HybridModel(
                config=language_transformer_config,
                hybrid_stack_spec=language_transformer_layer_spec,
                vocab_size=language_vocab_size,
                max_sequence_length=language_max_sequence_length,
                parallel_output=parallel_output,
                **logit_dtype_kwarg(HybridModel, language_transformer_config.logit_dtype),
                position_embedding_type=language_position_embedding_type,
                pre_process=pre_process,
                hybrid_layer_pattern=hybrid_layer_pattern,
                post_process=post_process,
                # The caller scatters for sequence parallelism after media merge.
                scatter_embedding_sequence_parallel=False,
                share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                pg_collection=pg_collection,
                vp_stage=vp_stage,
            )
            self.language_model.register_load_state_dict_post_hook(_ignore_transformer_engine_extra_state)

        self.vision_model = None
        self.vision_projection = None
        if add_encoder:
            # Vision encodes whole images, even when images are distributed over
            # language CP ranks. RoPE must therefore see a singleton vision CP
            # group rather than select positions as if each image were sharded.
            vision_pg = copy.copy(pg_collection)
            # GTP partners consume different media and can skip the encoder.
            # Keep vision weights replicated so their gathers cannot enter the
            # language weight stream. An explicit None prevents MPU fallback.
            vision_pg.gtp_remat = None
            vision_pg.expt_gtp_remat = None
            if pg_collection.cp.size() > 1:
                vision_pg.cp = torch.distributed.new_group(
                    ranks=[torch.distributed.get_rank()],
                    use_local_synchronization=True,
                )
                vision_pg.tp_cp = vision_pg.tp
                vision_pg.hcp = None
            self.vision_model = ViTModel(
                transformer_config=vision_transformer_config,
                transformer_layer_spec=vision_transformer_layer_spec,
                patch_dim=patch_dim,
                img_h=img_h,
                img_w=img_w,
                add_class_token=False,
                class_token_len=0,
                pos_emb_type="rope2d",
                ln_pre=True,
                use_merger=spatial_merge_size > 1,
                spatial_merge_size=spatial_merge_size,
                pg_collection=vision_pg,
                vp_stage=vp_stage,
            )
            # The merger keeps the ViT width, so the projector input is the ViT
            # hidden size rather than a pixel-shuffled multiple of it.
            self.vision_projection = MultimodalProjector(
                vision_projection_config,
                vision_projection_layer_spec,
                "mlp",
                getattr(self.vision_model, "out_hidden_size", vision_transformer_config.hidden_size),
                pg_collection=vision_pg,
            )
            self.vision_model.register_load_state_dict_post_hook(_ignore_transformer_engine_extra_state)
            self.vision_projection.register_load_state_dict_post_hook(_ignore_transformer_engine_extra_state)

    def sharded_state_dict(self, *args, **kwargs):
        """Exclude TE metadata absent from the native NM4 checkpoints."""
        from megatron.bridge.training.checkpointing import delete_extra_state

        state = super().sharded_state_dict(*args, **kwargs)
        delete_extra_state(state)
        return state

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ) -> None:
        """Freeze selected towers."""
        modules = (
            (freeze_language_model, self.language_model),
            (freeze_vision_model, self.vision_model),
            (freeze_vision_projection, self.vision_projection),
        )
        for should_freeze, module in modules:
            if should_freeze and module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = False

    def set_input_tensor(self, input_tensor) -> None:
        """Set the pipeline input on the language model."""
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor must contain exactly one tensor"
        if self.language_model is not None:
            self.language_model.set_input_tensor(input_tensor[0])

    def _encode_images(self, images, imgs_sizes, vision_packed_seq_params) -> torch.Tensor:
        """Return one projected row per image token."""
        if self.vision_model is None or self.vision_projection is None:
            raise RuntimeError("Image data was provided on a stage without the vision encoder")

        parameter = next(self.vision_model.parameters())
        if imgs_sizes is not None:
            imgs_sizes = imgs_sizes.cpu()
            if vision_packed_seq_params is None and imgs_sizes.numel() > 0:
                # Pixel patches from every image in a physical language pack
                # are concatenated into one THD tensor. Keep the ViT attention
                # independent per image; otherwise image i's projected tokens
                # depend on every other image that happened to share its pack.
                vision_packed_seq_params = _build_vision_packed_seq_params(imgs_sizes, self.patch_dim, images.device)
        shard_vision = self.vision_dp_over_cp and imgs_sizes is not None and imgs_sizes.numel() > 0
        num_padded_ranks = 0
        if shard_vision:
            if self.pg_collection.cp.size() != self.context_parallel_lm:
                raise ValueError("Nemotron4 vision_dp_over_cp does not match its process group")
            (
                images,
                imgs_sizes,
                vision_packed_seq_params,
                _has_fp8_padding,
                num_padded_ranks,
                _local_num_frames,
            ) = split_to_context_parallel_ranks_dynamic_res(
                images,
                imgs_sizes,
                vision_packed_seq_params,
                patch_dim=self.patch_dim,
                fp8_enabled=False,
            )
            # MCore gives empty ranks a 1x1 dummy patch grid. The merger
            # requires a grid divisible by spatial_merge_size, even for dummies.
            if num_padded_ranks and parallel_state.get_context_parallel_rank() >= (
                self.context_parallel_lm - num_padded_ranks
            ):
                merge = self.spatial_merge_size
                images = images.new_zeros((1, merge * merge, images.shape[-1]))
                imgs_sizes = imgs_sizes.new_full((1, 2), self.patch_dim * merge)
                vision_packed_seq_params = _build_vision_packed_seq_params(imgs_sizes, self.patch_dim, images.device)
        # The Pixtral ViT adds no class token and uses the patch merger rather
        # than pixel shuffle, so the encoder output already has exactly one row
        # per placeholder and needs no per-image regrouping.
        encoded = self.vision_model(
            images.to(dtype=parameter.dtype),
            imgs_sizes=imgs_sizes,
            packed_seq_params=vision_packed_seq_params,
        )
        features = encoded.squeeze(0).unsqueeze(1)
        projected = self.vision_projection(features).squeeze(1)
        if shard_vision:
            # Gather only projected features; encoder and projector stay local.
            projected = gather_from_context_parallel_ranks_dynamic_res(projected, num_padded_ranks)
        return projected.contiguous()

    def _merge_projected_media(self, embeddings, input_ids, features, validity_mask) -> torch.Tensor:
        """Replace each valid media placeholder with exactly one feature row."""
        media_mask = input_ids == self.image_token_index
        if validity_mask is not None:
            if validity_mask.shape != input_ids.shape:
                raise ValueError(
                    "The media token-validity mask must have the same shape as input_ids: "
                    f"got mask={tuple(validity_mask.shape)}, input_ids={tuple(input_ids.shape)}."
                )
            media_mask = media_mask & validity_mask.bool()

        expected_features = int(media_mask.sum().item())
        if expected_features != features.shape[0]:
            raise ValueError(
                "Expanded-sequence media alignment failed: "
                f"found {expected_features} valid placeholders for {features.shape[0]} projected "
                "features. Nemotron4VLModel requires the processor to emit one placeholder "
                "for every projected patch."
            )

        # With batch size one, transpose can remain contiguous and
        # ``contiguous()`` may return the original autograd view. Media
        # insertion is in-place, so force independent storage for the merge.
        merged = embeddings.transpose(0, 1).clone()
        if expected_features:
            merged[media_mask] = features.to(dtype=merged.dtype)
        return merged.transpose(0, 1).contiguous()

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        inference_context=None,
        runtime_gather_output: Optional[bool] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        images: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        imgs_sizes: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        vision_packed_seq_params: Optional[PackedSeqParams] = None,
        *,
        media_token_validity_mask: Optional[torch.Tensor] = None,
        inference_params=None,
        **kwargs,
    ):
        """Insert media into the expanded sequence, shard for CP, then run the hybrid LM.

        ``pixel_values``/``image_sizes`` are accepted as aliases of
        ``images``/``imgs_sizes`` because the HF Nemotron4 processor emits those
        names.

        Returns:
            Model output, or ``(output, local_loss_mask)`` when this model
            applied a context-parallel shard to the supervision tensors. Final
            logits, token losses, and the returned mask use per-sequence zigzag
            ordering; decoder inputs retain the configured linear CP layout.
        """
        del kwargs
        if images is None:
            images = pixel_values
        if image_sizes is not None:
            if imgs_sizes is not None:
                raise ValueError("Pass image sizes as either imgs_sizes or image_sizes, not both.")
            imgs_sizes = image_sizes

        lm_input_ids = input_ids
        combined_embeddings = None
        if self.pre_process:
            if input_ids is None:
                raise ValueError("The first Nemotron4 pipeline stage requires input_ids.")
            # An explicit mask from the caller wins: padding and attention masks
            # answer "is this a real token", which is a different question from
            # "is this a media anchor". They coincide only while every media
            # token in a valid position anchors an image. Otherwise a padding
            # mask is unambiguous for collator-owned packed inputs, and a 2-D
            # attention mask is the MBridge collators' token-validity mask;
            # MCore's 4-D causal mask cannot filter placeholders.
            if media_token_validity_mask is None:
                if padding_mask is not None:
                    media_token_validity_mask = ~padding_mask
                elif attention_mask is not None and attention_mask.dim() == input_ids.dim():
                    media_token_validity_mask = attention_mask
            if media_token_validity_mask is not None and media_token_validity_mask.shape != input_ids.shape:
                raise ValueError(
                    "The media token-validity mask must have the same shape as input_ids: "
                    f"got mask={tuple(media_token_validity_mask.shape)}, input_ids={tuple(input_ids.shape)}."
                )

            # Match LLaVAModel's execution order: embed text first, then merge.
            image_anchor_mask = input_ids == self.image_token_index
            if media_token_validity_mask is not None:
                image_anchor_mask &= media_token_validity_mask.bool()
            text_ids = input_ids.masked_fill(image_anchor_mask, 0)
            combined_embeddings = self.language_model.embedding(input_ids=text_ids, position_ids=position_ids)
            if images is not None and images.numel() > 0:
                features = self._encode_images(images, imgs_sizes, vision_packed_seq_params)
            else:
                features = combined_embeddings.new_empty((0, combined_embeddings.shape[-1]))
            combined_embeddings = self._merge_projected_media(
                combined_embeddings, input_ids, features, media_token_validity_mask
            )

        if packed_seq_params is not None:
            # THD boundaries are final collator output. Token-aligned tensors may
            # be CP-sharded here, but the packing metadata is never rebuilt.
            attention_mask = None

        sequence_tensors = (
            (lm_input_ids, 1),
            (combined_embeddings, 0),
            (position_ids, 1),
            (labels, 1),
            (loss_mask, 1),
            (padding_mask, 1),
        )
        full_lengths = {tensor.size(dim) for tensor, dim in sequence_tensors if tensor is not None}
        if len(full_lengths) > 1:
            raise ValueError(f"Nemotron4 token-aligned tensors have inconsistent lengths: {sorted(full_lengths)}.")
        sliced_loss_mask = False
        cp_batch = None
        output_cp_batch = None
        if full_lengths and self.context_parallel_lm > 1:
            if packed_seq_params is None:
                raise ValueError(
                    "Nemotron4 context parallelism requires caller-packed THD inputs; enable policy.sequence_packing."
                )
            cu_seqlens = packed_seq_params.cu_seqlens_q.unsqueeze(0)
            cu_seqlens_padded = packed_seq_params.cu_seqlens_q_padded
            if cu_seqlens_padded is not None:
                cu_seqlens_padded = cu_seqlens_padded.unsqueeze(0)
            batch = {
                "tokens": lm_input_ids,
                "decoder_input": (
                    combined_embeddings.transpose(0, 1).contiguous() if combined_embeddings is not None else None
                ),
                "position_ids": position_ids,
                "labels": labels,
                "loss_mask": loss_mask,
                # Reuse a sequence-valued CP batch slot for the valid-token mask. The packed
                # zigzag sharder fills synthetic positions with False, so inverting the local
                # attention-layout view below marks both source and synthetic padding correctly.
                "mtp_input_mask": None if padding_mask is None else ~padding_mask.bool(),
                "cu_seqlens": cu_seqlens,
                "cu_seqlens_padded": cu_seqlens_padded,
                "max_seqlen": torch.tensor(
                    [packed_seq_params.max_seqlen_q],
                    dtype=torch.int32,
                    device=cu_seqlens.device,
                ),
            }
            additional_layouts = {
                "zigzag",
                self.config.attention_cp_layout,
                self.config.linear_cp_layout,
            }
            cp_batch = get_batches_on_this_cp_rank(
                batch,
                boundary_layout=self.config.linear_cp_layout,
                is_hybrid_cp=False,
                cp_group=self.pg_collection.cp,
                additional_layouts=additional_layouts,
                sequence_parallel=self.sequence_parallel_lm,
                tp_group=self.pg_collection.tp,
                tp_cp_group=self.pg_collection.tp_cp,
            )
            output_cp_batch = cp_batch
            if self.post_process and self.sequence_parallel_lm and cp_batch.boundary_layout != "zigzag":
                # Output logits have gathered TP sequence shards. Build a CP-only
                # route rather than reuse the decoder's combined TP/SP route.
                output_cp_batch = get_batches_on_this_cp_rank(
                    {key: value for key, value in batch.items() if key != "decoder_input"},
                    boundary_layout=self.config.linear_cp_layout,
                    is_hybrid_cp=False,
                    cp_group=self.pg_collection.cp,
                    additional_layouts={"zigzag"},
                    sequence_parallel=False,
                    tp_group=self.pg_collection.tp,
                    tp_cp_group=self.pg_collection.tp_cp,
                )
            boundary_batch = cp_batch.get_batch()
            sliced_loss_mask = loss_mask is not None
            lm_input_ids = boundary_batch["tokens"]
            combined_embeddings = boundary_batch["decoder_input"]
            if combined_embeddings is not None:
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
            position_ids = boundary_batch["position_ids"]
            labels = boundary_batch["labels"]
            loss_mask = boundary_batch["loss_mask"]
            if padding_mask is not None:
                # HybridStack passes padding_mask only to Transformer/MoE consumers, whose
                # physical token order is attention_cp_layout. GDP/Mamba layers do not consume it.
                attention_valid_tokens = cp_batch.get_batch(self.config.attention_cp_layout)["mtp_input_mask"]
                padding_mask = ~attention_valid_tokens.bool()
            packed_seq_params = cp_batch.get_packed_seq_params()
            for layout_batch in cp_batch.batches_by_layout.values():
                layout_batch.pop("decoder_input", None)
                layout_batch.pop("mtp_input_mask", None)

        if combined_embeddings is not None and self.sequence_parallel_lm:
            combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
                combined_embeddings, group=self.pg_collection.tp
            ).contiguous()
        if padding_mask is not None and self.sequence_parallel_lm:
            padding_mask = (
                tensor_parallel.scatter_to_sequence_parallel_region(
                    padding_mask.transpose(0, 1).contiguous(),
                    group=self.pg_collection.tp,
                )
                .transpose(0, 1)
                .contiguous()
            )

        # Once media is merged into the decoder embeddings the language model
        # must not also receive the pre-merge token IDs. MTP is the exception:
        # its loss derives targets from input_ids.
        mtp_num_layers = getattr(self.config, "mtp_num_layers", None)
        if combined_embeddings is not None and not (mtp_num_layers and mtp_num_layers > 0):
            lm_input_ids = None

        output = self.language_model(
            input_ids=lm_input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            loss_mask=loss_mask,
            padding_mask=padding_mask,
            inference_context=inference_context,
            inference_params=inference_params,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
            cp_batch=cp_batch,
        )
        if output_cp_batch is not None and self.post_process:
            # NeMo RL reconstructs packed losses and logprobs in per-sequence
            # zigzag order. Keep decoder/MTP inputs in their configured layouts,
            # and adapt only the final token-aligned outputs. The LM head has
            # already gathered the TP sequence shards, so SP is disabled here.
            output = (
                convert_cp_layout(
                    output.transpose(0, 1).contiguous(),
                    source_layout=output_cp_batch.boundary_layout,
                    target_layout="zigzag",
                    cp_group=self.pg_collection.cp,
                    thd_plan=output_cp_batch.thd_plan,
                )
                .transpose(0, 1)
                .contiguous()
            )
            if sliced_loss_mask:
                loss_mask = output_cp_batch.get_batch("zigzag")["loss_mask"]
        if sliced_loss_mask:
            return output, loss_mask
        return output
