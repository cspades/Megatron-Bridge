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
``<img>`` placeholder per projected image feature (rows of ``<img>`` separated by
``<image_break>`` and closed by ``</img>``), and only the ``<img>`` positions are
replaced in place. The media merge, context-parallel sharding and packed-sequence
handling are therefore inherited from :class:`NemotronOmniModel` unchanged.

What differs is the encoder side. Instead of RADIO with a pixel-shuffle projector,
Nemotron4 uses a Pixtral-style ViT (2D interleaved RoPE, gated SiLU MLP, RMSNorm,
no class token) whose 2x2 spatial patch merger runs inside the encoder, followed
by a two-layer non-gated MLP projector.

The language stack is a ``HybridModel`` built from the provider's stack spec. The
wide-residual, GDP and shortcut-MoE layers it needs are not in upstream MCore;
the provider is responsible for supplying an MCore that has them.
"""

import torch
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.vit_model import ViTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.nemotron_omni.modeling_nemotron_omni import (
    NemotronOmniModel,
    _build_vision_packed_seq_params,
    _ignore_transformer_engine_extra_state,
    _project_multimodal_embeddings,
)


class Nemotron4VLModel(NemotronOmniModel):
    """Nemotron4 VLM: Pixtral-style ViT + MLP projector + hybrid language model.

    Reuses :class:`NemotronOmniModel` for the language model, one-for-one media
    insertion, context-parallel sharding and pipeline plumbing. Only the vision
    encoder, projector and image encoding path are replaced. Sound inputs are not
    supported.
    """

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
        vision_projection_layer_spec: ModuleSpec,
        image_token_index: int,
        hybrid_layer_pattern: str,
        img_h: int,
        img_w: int,
        patch_dim: int,
        spatial_merge_size: int,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        language_position_embedding_type: str = "none",
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        pg_collection: ProcessGroupCollection | None = None,
        vp_stage: int | None = None,
    ) -> None:
        # Build the language model and all shared state through the Omni
        # constructor, but suppress its RADIO encoder; the Pixtral encoder is
        # attached below.
        super().__init__(
            language_transformer_config=language_transformer_config,
            language_transformer_layer_spec=language_transformer_layer_spec,
            language_vocab_size=language_vocab_size,
            language_max_sequence_length=language_max_sequence_length,
            vision_transformer_config=vision_transformer_config,
            vision_transformer_layer_spec=vision_transformer_layer_spec,
            vision_projection_config=vision_projection_config,
            vision_projection_layer_spec=vision_projection_layer_spec,
            image_token_index=image_token_index,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            language_position_embedding_type=language_position_embedding_type,
            pre_process=pre_process,
            post_process=post_process,
            add_encoder=False,
            add_decoder=add_decoder,
            hybrid_layer_pattern=hybrid_layer_pattern,
            patch_dim=patch_dim,
            dynamic_resolution=True,
            vision_dp_over_cp=False,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )

        self.add_encoder = add_encoder
        self.spatial_merge_size = spatial_merge_size
        if add_encoder:
            self.vision_model = ViTModel(
                vision_transformer_config,
                vision_transformer_layer_spec,
                patch_dim=patch_dim,
                img_h=img_h,
                img_w=img_w,
                add_class_token=False,
                class_token_len=0,
                ln_pre=True,
                pos_emb_type="rope2d",
                use_merger=spatial_merge_size > 1,
                spatial_merge_size=spatial_merge_size,
                pg_collection=self.pg_collection,
                vp_stage=vp_stage,
            )
            # The merger keeps the ViT width, so the projector input is the ViT
            # hidden size rather than Omni's pixel-shuffled ``hidden * 4``.
            self.vision_projection = MultimodalProjector(
                vision_projection_config,
                vision_projection_layer_spec,
                "mlp",
                vision_transformer_config.hidden_size,
                tp_group=self.pg_collection.tp,
            )
            self.vision_model.register_load_state_dict_post_hook(_ignore_transformer_engine_extra_state)
            self.vision_projection.register_load_state_dict_post_hook(_ignore_transformer_engine_extra_state)

    def _encode_images(
        self,
        images: torch.Tensor,
        imgs_sizes: torch.Tensor | None,
        vision_packed_seq_params: PackedSeqParams | None,
        num_frames: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode dynamic-resolution images and return one row per ``<img>`` token.

        Args:
            images: Padded pixels ``[N, C, H, W]`` or packed patches
                ``[1, total_patches, C * patch_dim**2]``.
            imgs_sizes: Per-image ``(height, width)`` in pixels. Required.
            vision_packed_seq_params: Per-image THD boundaries for the ViT; built
                from ``imgs_sizes`` when omitted.
            num_frames: Unsupported; must be ``None``.

        Returns:
            ``[num_merged_patches, language_hidden]`` projected features, in image
            order and row-major merged-patch order within each image.
        """

        if self.vision_model is None or self.vision_projection is None:
            raise RuntimeError("Image data was provided on a stage without the vision encoder")
        if num_frames is not None:
            raise ValueError("Nemotron4 does not support video inputs (num_frames must be None).")
        if imgs_sizes is None or imgs_sizes.numel() == 0:
            raise ValueError("Nemotron4 requires imgs_sizes: its ViT only runs the dynamic-resolution path.")

        parameter = next(self.vision_model.parameters())
        patches = self._patchify_dynamic_images(images.to(dtype=parameter.dtype), imgs_sizes)
        if vision_packed_seq_params is None:
            vision_packed_seq_params = _build_vision_packed_seq_params(imgs_sizes, self.patch_dim)

        # [1, merged_patches, vit_hidden]; the 2x2 merge happens inside the ViT.
        encoded = self.vision_model(
            patches,
            imgs_sizes=imgs_sizes,
            packed_seq_params=vision_packed_seq_params,
        )
        return _project_multimodal_embeddings(self.vision_projection, encoded.squeeze(0)).contiguous()
