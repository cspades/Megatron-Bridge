# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Bridge provider for experimental NM4 tiny/Nano native LLaVA checkpoints."""

import argparse
import copy
import logging
import os
import sys
import tempfile
from collections import namedtuple
from dataclasses import dataclass, fields
from functools import partial
from pathlib import Path
from typing import Any, Literal, Optional

import megatron.core
import torch
from megatron.core import dist_checkpointing, parallel_state, tensor_parallel
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
from megatron.core.pipeline_parallel.utils import is_pp_first_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_product import GatedDeltaProductMixer
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.spec_utils import import_module
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group

from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.models.logit_dtype import logit_dtype_kwarg
from megatron.bridge.training.mlm_compat.arguments import _transformer_config_from_args


# These source-model configuration helpers live outside the installed core package.
sys.path.insert(0, str(Path(megatron.core.__file__).resolve().parents[2]))


class _NM4GatedDeltaProductMixer(GatedDeltaProductMixer):
    """Materialize CP-sharded convolution inputs when their row stride is unaligned."""

    def _prepare_qkv(self, x, *args, **kwargs):
        # causal-conv1d's channel-last kernel requires the batch and sequence
        # strides after [B, L, D] -> [B, D, L] to be multiples of eight.  CP8
        # can leave ``x`` as a slice of a 2316-wide projection even though its
        # own 2112-wide payload is aligned.  A compact copy changes only the
        # storage stride; values, parameter shapes, and checkpoint keys stay
        # unchanged.
        if x.stride(0) % 8 or x.stride(1) % 8:
            x = x.contiguous()
        return super()._prepare_qkv(x, *args, **kwargs)


class _NM4VisionMLP(MLP):
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
    """Allow the native checkpoint, which stores no TE extra state, to load."""

    del module
    for mismatch_kind, keys in incompatible_keys._asdict().items():
        ignored = [key for key in keys if "extra_state" in key]
        if not ignored:
            continue
        for key in ignored:
            keys.remove(key)
        logging.getLogger(__name__).warning(
            "Ignoring %d Transformer Engine %s _extra_state entries; first entries: %s%s",
            len(ignored),
            mismatch_kind,
            ", ".join(ignored[:5]),
            " ..." if len(ignored) > 5 else "",
        )


_TORCH_DTYPES = {str(dtype): dtype for dtype in (torch.float32, torch.float16, torch.bfloat16)}


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


def _dict_from_namespace(args: argparse.Namespace) -> dict[str, Any]:
    """YAML-safe copy of the native checkpoint args used by provide()."""
    from megatron.bridge.training.utils.config_utils import _ConfigContainerBase

    serialized: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key.startswith("_"):
            continue
        if isinstance(value, torch.dtype):
            serialized[key] = str(value)
            continue
        try:
            serialized[key] = _ConfigContainerBase._convert_value_to_dict(value)
        except Exception:
            continue
    return serialized


def _namespace_from_dict(saved_args: dict[str, Any]) -> argparse.Namespace:
    """Rebuild the native args Namespace from a previously serialized mapping."""
    from megatron.bridge.utils.instantiate_utils import instantiate

    restored: dict[str, Any] = {}
    for key, value in saved_args.items():
        if isinstance(value, str) and value in _TORCH_DTYPES:
            restored[key] = _TORCH_DTYPES[value]
        elif isinstance(value, dict) and "_target_" in value:
            restored[key] = instantiate(value)
        else:
            restored[key] = value
    return argparse.Namespace(**restored)


def _validate_native_llava_args(args: argparse.Namespace) -> None:
    if getattr(args, "vlm_checkpoint_layout", None) != "llava":
        raise ValueError("Expected the native LLaVA checkpoint produced by our NM4 converter")
    if (args.hidden_size, args.num_layers) not in ((512, 8), (3072, 72)):
        raise ValueError("Only the inspected NM4 tiny and Nano models are supported")
    if args.vision_projection_type != "mlp2":
        raise ValueError("Expected the NM4 mlp2 projector")


class NM4LlavaModel(MegatronModule):
    """NM4 LLaVA whose input sequence already carries one placeholder per feature.

    Submodule names match ``LLaVAModel`` so the native checkpoint loads
    unchanged. Unlike ``LLaVAModel`` this does not collapse and re-expand media
    runs: the Pixtral processor emits one placeholder per projected patch, so
    insertion is one-for-one and length preserving. That lets the model insert
    media into the full token stream and then select its own context-parallel
    shard, which is what ``model_slices_context_parallel_inputs`` promises
    NeMo-RL.
    """

    model_slices_context_parallel_inputs = True

    def __init__(
        self,
        *,
        language_transformer_config,
        language_transformer_layer_spec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        vision_transformer_config,
        vision_transformer_layer_spec,
        vision_projection_config,
        vision_projection_layer_spec,
        image_token_index: int,
        img_h: int,
        img_w: int,
        patch_dim: int,
        spatial_merge_size: int,
        hybrid_layer_pattern: Optional[str],
        parallel_output: bool,
        share_embeddings_and_output_weights: bool,
        language_position_embedding_type: str,
        pre_process: bool,
        post_process: bool,
        add_encoder: bool,
        add_decoder: bool,
        pg_collection: ProcessGroupCollection,
        vision_dp_over_cp: bool = True,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=language_transformer_config)

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
                pos_emb_type="rope2d",
                ln_pre=True,
                use_merger=spatial_merge_size > 1,
                spatial_merge_size=spatial_merge_size,
                pg_collection=vision_pg,
                vp_stage=vp_stage,
            )
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
        """Exclude TE metadata absent from the native NM4 checkpoint."""
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
        """Freeze selected NM4 towers."""
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
                raise ValueError("NM4 vision_dp_over_cp does not match its process group")
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
            # MCore gives empty ranks a 1x1 dummy patch grid. NM4's merger
            # requires a grid divisible by spatial_merge_size, even for dummies.
            if num_padded_ranks and parallel_state.get_context_parallel_rank() >= (
                self.context_parallel_lm - num_padded_ranks
            ):
                merge = self.spatial_merge_size
                images = images.new_zeros((1, merge * merge, images.shape[-1]))
                imgs_sizes = imgs_sizes.new_full((1, 2), self.patch_dim * merge)
                vision_packed_seq_params = _build_vision_packed_seq_params(imgs_sizes, self.patch_dim, images.device)
        # Mistral-ViT adds no class token and NM4 uses the patch merger rather
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
                "features. NM4LlavaModel requires the Pixtral processor to emit one placeholder "
                "for every projected patch."
            )

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
        vision_packed_seq_params: Optional[PackedSeqParams] = None,
        *,
        media_token_validity_mask: Optional[torch.Tensor] = None,
        inference_params=None,
        **kwargs,
    ):
        """Insert media into the expanded sequence, shard for CP, then call NemotronH.

        Returns:
            Model output, or ``(output, local_loss_mask)`` when this model
            applied a context-parallel shard to the supervision tensors. Final
            logits, token losses, and the returned mask use per-sequence zigzag
            ordering; decoder inputs retain the configured linear CP layout.
        """
        del kwargs
        if images is None:
            images = pixel_values

        lm_input_ids = input_ids
        combined_embeddings = None
        if self.pre_process:
            if input_ids is None:
                raise ValueError("The first NM4 LLaVA pipeline stage requires input_ids.")
            text_ids = input_ids.masked_fill(input_ids == self.image_token_index, 0)
            combined_embeddings = self.language_model.embedding(input_ids=text_ids, position_ids=position_ids)
            if images is not None and images.numel() > 0:
                features = self._encode_images(images, imgs_sizes, vision_packed_seq_params)
            else:
                features = combined_embeddings.new_empty((0, combined_embeddings.shape[-1]))

            # A padding mask answers "is this a real token", which coincides with
            # "is this a media anchor" only while every placeholder in a valid
            # position anchors a patch. An explicit mask from the caller wins.
            if media_token_validity_mask is None and padding_mask is not None:
                media_token_validity_mask = ~padding_mask
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
            raise ValueError(f"NM4 LLaVA token-aligned tensors have inconsistent lengths: {sorted(full_lengths)}.")
        sliced_loss_mask = False
        cp_batch = None
        output_cp_batch = None
        if full_lengths and self.context_parallel_lm > 1:
            if packed_seq_params is None:
                raise ValueError(
                    "NM4 LLaVA context parallelism requires caller-packed THD inputs; enable policy.sequence_packing."
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
            # NeMo-RL reconstructs packed losses and logprobs in per-sequence
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


PRECISION_RECIPE_FILENAME = "precision_recipe.yaml"

# Copied from the native checkpoint; runtime TP/GTP must come from this provider.
_CHECKPOINT_PARALLEL_FIELDS = frozenset(
    {
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
        "sequence_parallel",
        "tensor_parallel_num_weight_shards",
        "expert_tensor_parallel_num_weight_shards",
        "gtp_weight_remat_size",
        "expert_gtp_weight_remat_size",
    }
)


@dataclass
class ExperimentalNM4LlavaProvider(HybridModelProvider):
    """Reconstruct the native LLaVAModel without HF or AutoBridge.

    Architecture is read from the copied checkpoint's common args. Tensor,
    pipeline, context, expert parallel, and GTP sizes come from this provider
    after NeMo-RL copies ``policy.megatron_cfg`` onto it. Construction uses
    the Bridge ``provide()`` path so the training loop can wrap DDP and mixed
    precision.
    """

    checkpoint_path: str = ""
    precision_recipe_path: str = ""
    # Raw TE recipe YAML. Saved into run_config and copied next to it so resume
    # does not need the original 30K checkpoint directory.
    precision_recipe_yaml: str | None = None
    # Serialized native argparse payload. Present on NeMo-RL saves so resume
    # can reconstruct architecture without the original 30K checkpoint.
    saved_args: dict[str, Any] | None = None
    # Parent TransformerConfig marks these init=False (derived). Expose them so
    # YAML / model_overrides can set GTP. shards = TP * GTP.
    gtp_weight_remat_size: int = 1
    expert_gtp_weight_remat_size: int = 1
    freeze_language_model: bool | None = None
    freeze_vision_model: bool | None = None
    freeze_vision_projection: bool | None = None
    vision_dp_over_cp: bool = True
    """Encode whole images on separate language CP ranks, then gather features."""
    recompute_vision: bool = False
    vision_recompute_granularity: Literal["full", "selective"] | None = None
    vision_recompute_method: Literal["uniform", "block"] | None = None
    vision_recompute_num_layers: int | None = None

    def _apply_vision_recompute_settings(self, vision_config: Any) -> None:
        """Apply explicitly requested activation recompute to the ViT tower."""
        if not self.recompute_vision:
            vision_config.recompute_granularity = None
            vision_config.recompute_method = None
            vision_config.recompute_num_layers = None
            return

        recompute_settings = (
            self.vision_recompute_granularity,
            self.vision_recompute_method,
            self.vision_recompute_num_layers,
        )
        if any(value is None for value in recompute_settings):
            raise ValueError(
                "Vision recompute requires vision_recompute_granularity, "
                "vision_recompute_method, and vision_recompute_num_layers."
            )
        vision_config.recompute_granularity = self.vision_recompute_granularity
        vision_config.recompute_method = self.vision_recompute_method
        vision_config.recompute_num_layers = self.vision_recompute_num_layers

    def _apply_checkpoint_freeze_settings(self, args: argparse.Namespace) -> None:
        """Translate the native NM4 checkpoint's tower freeze settings."""
        if self.freeze_language_model is None:
            self.freeze_language_model = bool(getattr(args, "freeze_lm", False))
        if self.freeze_vision_model is None:
            self.freeze_vision_model = bool(getattr(args, "freeze_vit", False))
        if self.freeze_vision_projection is None:
            self.freeze_vision_projection = bool(getattr(args, "freeze_projection", False))

    def _apply_gtp_weight_shards(self) -> None:
        if self.gtp_weight_remat_size < 1 or self.expert_gtp_weight_remat_size < 1:
            raise ValueError("gtp_weight_remat_size and expert_gtp_weight_remat_size must be >= 1")
        if self.expert_tensor_parallel_size is None and self.expert_model_parallel_size > 1:
            self.expert_tensor_parallel_size = 1
        self.tensor_parallel_num_weight_shards = self.tensor_model_parallel_size * self.gtp_weight_remat_size
        self.expert_tensor_parallel_num_weight_shards = (
            self.expert_tensor_parallel_size or 1
        ) * self.expert_gtp_weight_remat_size

    def _load_precision_recipe_yaml(self) -> None:
        if self.precision_recipe_yaml:
            return
        if self.precision_recipe_path and os.path.isfile(self.precision_recipe_path):
            self.precision_recipe_yaml = Path(self.precision_recipe_path).read_text()

    def _materialize_precision_recipe(self) -> str:
        """Return a readable recipe file, writing one from saved YAML if needed."""
        self._load_precision_recipe_yaml()
        if self.precision_recipe_path and os.path.isfile(self.precision_recipe_path):
            if self.precision_recipe_yaml is None:
                self.precision_recipe_yaml = Path(self.precision_recipe_path).read_text()
            return self.precision_recipe_path
        if not self.precision_recipe_yaml:
            raise ValueError("precision_recipe_path is required")
        handle = tempfile.NamedTemporaryFile(
            prefix="nm4_precision_recipe_",
            suffix=".yaml",
            delete=False,
        )
        with handle:
            handle.write(self.precision_recipe_yaml.encode())
        self.precision_recipe_path = handle.name
        return self.precision_recipe_path

    def save_precision_recipe_assets(self, checkpoint_dir: str) -> None:
        """Copy the TE precision recipe next to run_config.yaml."""
        self._load_precision_recipe_yaml()
        if not self.precision_recipe_yaml:
            return
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(checkpoint_dir, PRECISION_RECIPE_FILENAME).write_text(self.precision_recipe_yaml)

    def _apply_precision_recipe(self, args: argparse.Namespace) -> None:
        args.te_precision_config_file = self._materialize_precision_recipe()

    def __post_init__(self) -> None:
        already_loaded = getattr(self, "_saved_args", None) is not None
        if already_loaded:
            # NeMo-RL calls this again after applying megatron_cfg TP/ETP.
            self._apply_gtp_weight_shards()
        super().__post_init__()
        if already_loaded:
            return
        if self.saved_args is not None:
            args = _namespace_from_dict(self.saved_args)
            _validate_native_llava_args(args)
            self._apply_precision_recipe(args)
            self._saved_args = args
            self._apply_checkpoint_freeze_settings(args)
            self._apply_gtp_weight_shards()
            return
        if not self.checkpoint_path or not (self.precision_recipe_path or self.precision_recipe_yaml):
            raise ValueError("checkpoint_path and precision_recipe_path are required")
        args = copy.deepcopy(dist_checkpointing.load_common_state_dict(self.checkpoint_path)["args"])
        _validate_native_llava_args(args)
        self._apply_precision_recipe(args)
        self._saved_args = args
        self.saved_args = _dict_from_namespace(args)
        config = _transformer_config_from_args(args)
        for item in fields(config):
            if item.name in _CHECKPOINT_PARALLEL_FIELDS or not hasattr(self, item.name):
                continue
            setattr(self, item.name, getattr(config, item.name))
        self.hybrid_layer_pattern = args.hybrid_layer_pattern
        self.hybrid_override_pattern = None
        self.mtp_hybrid_override_pattern = None
        self.mtp_use_repeated_layer = args.mtp_use_repeated_layer
        self.seq_length = args.seq_length
        self.vocab_size = args.padded_vocab_size
        self.position_embedding_type = "none"
        self.share_embeddings_and_output_weights = False
        self.logit_dtype = args.logit_dtype
        self._apply_checkpoint_freeze_settings(args)
        self._apply_gtp_weight_shards()

    def finalize(self) -> None:
        """Reconcile hybrid/MTP fields, then derive GTP from the provider TP/GTP config."""
        requested_gtp = self.gtp_weight_remat_size
        requested_expert_gtp = self.expert_gtp_weight_remat_size
        self._apply_gtp_weight_shards()
        super().finalize()
        if self.gtp_weight_remat_size != requested_gtp or self.expert_gtp_weight_remat_size != requested_expert_gtp:
            raise ValueError(
                "GTP shard reconciliation did not honor the provider config: "
                f"got gtp_weight_remat_size={self.gtp_weight_remat_size}, "
                f"expert_gtp_weight_remat_size={self.expert_gtp_weight_remat_size}, "
                f"requested {requested_gtp} and {requested_expert_gtp}"
            )

    def to_cfg_dict(self) -> dict:
        """Serialize hydrated architecture, runtime parallel sizes, and native args."""
        from megatron.bridge.training.utils.config_utils import _ConfigContainerBase

        if getattr(self, "_saved_args", None) is not None:
            self.saved_args = _dict_from_namespace(self._saved_args)
        self._load_precision_recipe_yaml()
        result = {"_target_": f"{type(self).__module__}.{type(self).__qualname__}"}
        for item in fields(self):
            if item.name.startswith("_"):
                continue
            result[item.name] = _ConfigContainerBase._convert_value_to_dict(getattr(self, item.name))
        if self.precision_recipe_yaml:
            result["precision_recipe_path"] = PRECISION_RECIPE_FILENAME
        return result

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """Build the NM4 LLaVA model for the current provider parallel sizes."""
        from examples.mimo.model_providers.nemotron_mistral_vit import (
            mistral_vit_mlp2_projection_config,
            mistral_vit_projection_layer_spec,
            mistral_vit_vision_config,
        )
        from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec

        # Newer MCore model-provider helpers accept optional vision-attention
        # overrides that are absent from older checkpoint argument namespaces.
        # Keep this compatibility shim in Bridge rather than modifying MCore.
        args = copy.copy(self._saved_args)
        args.mimo_vision_encoder_attention_backend = getattr(args, "mimo_vision_encoder_attention_backend", None)
        args.mimo_vision_encoder_flash_attention_version = getattr(
            args, "mimo_vision_encoder_flash_attention_version", None
        )
        language = self._copy_config_without_runtime_process_groups(deep=True)
        language.language_model_type = "nemotron6-moe"
        vision = mistral_vit_vision_config(args, self.tensor_model_parallel_size, 1)
        vision.vision_model_type = "mistral-vit"
        vision.context_parallel_size = 1
        vision.sequence_parallel = False
        self._apply_vision_recompute_settings(vision)
        projection = mistral_vit_mlp2_projection_config(args, self.tensor_model_parallel_size)
        for config in (vision, projection):
            config.fp8 = None
            config.fp8_param = False
            config.quant_recipe = None
        pg = getattr(self, "_pg_collection", None)
        if pg is None:
            pg = ProcessGroupCollection.use_mpu_process_groups()
        add_encoder = True if self.pipeline_model_parallel_size == 1 else is_pp_first_stage(pg.pp)
        language_layer_spec = copy.deepcopy(import_module(args.spec))
        gdp_mixer_spec = language_layer_spec.submodules.mamba_layer.submodules.mixer
        if gdp_mixer_spec.module is not GatedDeltaProductMixer:
            raise TypeError(
                f"NM4 expected its recurrent layer spec to use GatedDeltaProductMixer, got {gdp_mixer_spec.module!r}"
            )
        gdp_mixer_spec.module = _NM4GatedDeltaProductMixer
        vision_layer_spec = get_vit_layer_with_transformer_engine_spec()
        vision_mlp_spec = vision_layer_spec.submodules.mlp
        vision_layer_spec.submodules.mlp = partial(
            _NM4VisionMLP.as_mlp_submodule, *vision_mlp_spec.args, **vision_mlp_spec.keywords
        )
        model = NM4LlavaModel(
            language_transformer_config=language,
            language_transformer_layer_spec=language_layer_spec,
            language_vocab_size=self.vocab_size,
            language_max_sequence_length=self.seq_length,
            vision_transformer_config=vision,
            vision_transformer_layer_spec=vision_layer_spec,
            vision_projection_config=projection,
            vision_projection_layer_spec=mistral_vit_projection_layer_spec(),
            image_token_index=args.image_token_id,
            img_h=args.img_h,
            img_w=args.img_w,
            patch_dim=args.patch_dim,
            vision_dp_over_cp=self.vision_dp_over_cp,
            spatial_merge_size=args.spatial_merge_size,
            hybrid_layer_pattern=args.hybrid_layer_pattern,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            language_position_embedding_type=self.position_embedding_type,
            pre_process=True if pre_process is None else pre_process,
            post_process=True if post_process is None else post_process,
            add_encoder=add_encoder,
            add_decoder=True,
            pg_collection=pg,
            vp_stage=vp_stage,
        )
        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )
        return model
