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

"""Model providers for Nemotron4 (language-only and vision-language).

Defaults describe the Nemotron4 nano architecture, which the WSM vision-language
checkpoint shares exactly. Only architectural settings are defaults here; kernel
choices (MXFP8, fused weighted squared ReLU, CuTe-DSL GDP, flex dispatcher),
recompute and offload policies, and loss reduction belong to recipes. Pretraining
used ``calculate_per_token_loss=True`` with ``mtp_normalize_loss_by_main_tokens=True``;
MCore requires the second to be paired with the first.

Nemotron4 needs MCore features that upstream Megatron-LM does not have yet:
streamwise wide residuals, shortcut MoE blocks, global-batch quantile balancing,
tanh-clamped activations and the pre-up-projection MoE norm. The providers check
for them explicitly, because a dataclass subclass would otherwise accept these
settings as new, silently ignored fields on an MCore that lacks them.

Attention is dense by default. Checkpoints trained with simplified sparse attention (SSA)
(a learned single-head Q/K indexer choosing each query's top-k keys, run by MCore's CuTe
kernels) use :func:`nemotron4_ssa_stack_spec` and ``experimental_attention_variant="ssa"``;
``Nemotron4Bridge.provider_bridge`` sets both when the HF config has ``index_topk``.
"""

import argparse
import copy
import dataclasses
import importlib.util
import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, ClassVar, Literal

import torch
from megatron.core import dist_checkpointing
from megatron.core.activations import squared_relu
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.pipeline_parallel.utils import is_pp_first_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_product import GatedDeltaProductMixer
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec, import_module
from megatron.core.transformer.transformer_config import TransformerConfig as MCoreTransformerConfig

from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.models.nemotron4.modeling_nemotron4_vl import (
    Nemotron4GatedDeltaProductMixer,
    Nemotron4MIMOLanguageModel,
    Nemotron4MIMOProjector,
    Nemotron4MIMOVisionEncoder,
    Nemotron4VisionMLP,
    Nemotron4VLModel,
)
from megatron.bridge.training.mlm_compat.arguments import _transformer_config_from_args


# Nano/WSM: 32 GDP ("M"), 4 dense MLP ("-"), 8 attention ("*"), 28 MoE ("E"), then two MTP depths.
NEMOTRON4_NANO_HYBRID_LAYER_PATTERN = "M-M-M*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEM-M-/*E/*E"

# TransformerConfig fields Nemotron4 sets that exist only in MCore builds carrying the
# Nemotron4 architecture.
NEMOTRON4_REQUIRED_MCORE_FIELDS = (
    "wide_residual",
    "moe_shortcut_connection",
    "moe_shortcut_parallel",
    "moe_use_norm_before_up_proj",
    "moe_router_quantile_balancing_estimation_scope",
    "activation_func_tanh_clamp_scale",
    "moe_latent_size",
    "mtp_use_repeated_layer",
    "mtp_hsm",
)


def _missing_nemotron4_mcore_fields() -> list[str]:
    """Return the Nemotron4 config fields the installed MCore does not define."""
    defined = {config_field.name for config_field in dataclasses.fields(MCoreTransformerConfig)}
    return [name for name in NEMOTRON4_REQUIRED_MCORE_FIELDS if name not in defined]


def _has_apex_wgrad_fusion_kernel() -> bool:
    """Return whether MCore's non-TE ColumnParallelLinear can fuse weight-gradient accumulation."""
    return importlib.util.find_spec("fused_weight_gradient_mlp_cuda") is not None


def require_nemotron4_mcore() -> None:
    """Raise if the installed MCore cannot build Nemotron4.

    Raises:
        RuntimeError: If any required TransformerConfig field or the GDP stack spec is missing.
    """
    missing = _missing_nemotron4_mcore_fields()
    if missing:
        raise RuntimeError(
            "The installed megatron.core cannot build Nemotron4: TransformerConfig is missing "
            f"{missing}. Nemotron4 needs an MCore with streamwise wide residuals and shortcut MoE "
            "(the Nemotron4 Megatron-LM fork); upstream Megatron-LM does not provide them yet."
        )


def nemotron4_stack_spec() -> ModuleSpec:
    """Return the gated-delta-product hybrid stack spec Nemotron4 was trained with.

    A named function (not a lambda) so the provider stays serializable.
    """
    try:
        from megatron.core.models.hybrid.hybrid_layer_specs import gated_delta_product_stack_spec
    except ImportError as error:
        raise RuntimeError(
            "The installed megatron.core has no gated_delta_product_stack_spec, which Nemotron4 needs."
        ) from error
    return gated_delta_product_stack_spec


# TransformerConfig fields required by MCore's simplified sparse attention.
# The top-k and loss coefficient retain their historical dsa_ names in MCore.
NEMOTRON4_SSA_REQUIRED_MCORE_FIELDS = (
    "experimental_attention_variant",
    "ssa_train_indexer_only",
    "dsa_indexer_topk",
    "dsa_indexer_loss_coeff",
)

# Indexer-loss settings of the sparse-attention stage-2 recipe (KL loss on the selected top-k
# keys, coefficient 0.1). They only affect training; MCore's CuTe backend requires them to be set.
NEMOTRON4_SSA_INDEXER_LOSS_COEFF = 0.1


def require_nemotron4_ssa_mcore() -> None:
    """Raise if the installed MCore cannot build Nemotron4's simplified sparse attention.

    Raises:
        RuntimeError: If an SSA TransformerConfig field or the SSA layer spec is missing.
    """
    require_nemotron4_mcore()
    defined = {config_field.name for config_field in dataclasses.fields(MCoreTransformerConfig)}
    missing = [name for name in NEMOTRON4_SSA_REQUIRED_MCORE_FIELDS if name not in defined]
    if (
        missing
        or importlib.util.find_spec(
            "megatron.core.transformer.experimental_attention_variant.simplified_sparse_attention_layer_specs"
        )
        is None
    ):
        raise RuntimeError(
            "The installed megatron.core cannot build Nemotron4 sparse attention: it needs "
            "experimental_attention_variant.simplified_sparse_attention_layer_specs (CuTe backend); "
            f"missing TransformerConfig fields: {missing}."
        )


def nemotron4_ssa_stack_spec() -> ModuleSpec:
    """Return the gated-delta-product stack spec with simplified sparse attention.

    Identical to :func:`nemotron4_stack_spec` except that every attention layer (including the
    MTP block's) is MCore's ``SimplifiedSparseAttention``: ordinary GQA projections whose core
    attention attends to each query's top-k keys chosen by a learned single-head indexer.
    A named function (not a lambda) so the provider stays serializable.
    """
    require_nemotron4_ssa_mcore()
    from megatron.core.transformer.experimental_attention_variant.simplified_sparse_attention_layer_specs import (
        gdp_ssa_stack_spec,
    )

    return gdp_ssa_stack_spec


def nemotron4_nano_wide_residual() -> Any:
    """Return the nano wide-residual config: three streams with learned retention."""
    require_nemotron4_mcore()
    from megatron.core.transformer.transformer_config import WideResidualConfig

    return WideResidualConfig(
        num_streams=3,
        streamwise_sigmoid_init_scale=0.01,
        learned_retention=True,
        retention_init=0.999,
        retention_max_forget=0.10,
    )


def nemotron4_projection_submodules() -> MLPSubmodules:
    """Return the projector's linears: plain TE column/row parallel, with no fused pre-norm.

    The Nemotron Omni helper uses ``TELayerNormColumnParallelLinear`` for fc1, which would add
    an RMSNorm (``linear_fc1.layer_norm_weight``) before the first projection. The Nemotron4
    projector has none; its only norm is the patch merger's, inside the ViT.
    """
    return MLPSubmodules(linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear)


def nemotron4_projector_gelu(x: torch.Tensor) -> torch.Tensor:
    """Tanh-approximated GELU used by the vision projector, without MCore's torch.compile wrapper.

    Written out rather than ``megatron.core.activations.fast_gelu`` because the Nemotron4
    projector was trained with this exact uncompiled expression; the compiled variant can
    round differently.
    """
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


logger = logging.getLogger(__name__)

PRECISION_RECIPE_FILENAME = "precision_recipe.yaml"

# Runtime parallel sizes come from the provider (NeMo RL copies policy.megatron_cfg onto
# it), never from a native checkpoint's saved arguments.
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

_TORCH_DTYPES = {str(dtype): dtype for dtype in (torch.float32, torch.float16, torch.bfloat16)}

# Mistral-ViT geometries the Nemotron4 Megatron-LM fork selects by ``--mistral-vit-model-size``:
# (num_layers, hidden_size, ffn_hidden_size, kv_channels). Both use 16 heads.
_MISTRAL_VIT_GEOMETRY = {
    "410m": (24, 1024, 4096, 64),
    "1.6b": (30, 2048, 6144, 128),
}


def _dict_from_namespace(args: argparse.Namespace) -> dict[str, Any]:
    """YAML-safe copy of native checkpoint args, for run_config serialization."""
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


def _validate_native_args(args: argparse.Namespace) -> None:
    """Reject native checkpoints this non-MIMO model cannot represent."""
    if getattr(args, "vlm_checkpoint_layout", None) != "llava":
        raise ValueError(
            "Expected a native non-MIMO (LLaVA-layout) Nemotron4 checkpoint; "
            f"got vlm_checkpoint_layout={getattr(args, 'vlm_checkpoint_layout', None)!r}"
        )
    if getattr(args, "vision_projection_type", None) != "mlp2":
        raise ValueError("Expected the Nemotron4 mlp2 projector")
    model_size = getattr(args, "mistral_vit_model_size", "410m")
    if model_size not in _MISTRAL_VIT_GEOMETRY:
        raise ValueError(f"Unsupported Mistral-ViT model size in checkpoint args: {model_size!r}")


@dataclass
class Nemotron4ModelProvider(HybridModelProvider):
    """Provider for the Nemotron4 language model (nano architecture by default)."""

    hybrid_layer_pattern: str | None = NEMOTRON4_NANO_HYBRID_LAYER_PATTERN
    hybrid_stack_spec: ModuleSpec | Callable[[], ModuleSpec] | None = nemotron4_stack_spec
    is_hybrid_model: bool = True

    # Dense geometry.
    hidden_size: int = 3072
    ffn_hidden_size: int = 12288
    num_attention_heads: int = 16
    num_query_groups: int = 1
    kv_channels: int = 256
    normalization: str = "RMSNorm"
    layernorm_epsilon: float = 1e-5
    add_bias_linear: bool = False
    gated_linear_unit: bool = False
    activation_func: Callable = squared_relu
    activation_func_tanh_clamp_scale: float | None = 16.0
    attention_softmax_in_fp32: bool = False
    init_method_std: float = 0.0162
    # Runtime defaults the pretraining run used (Megatron-LM argument defaults). The native
    # checkpoint path hydrates these from saved_args; set them here so an HF-derived provider
    # builds the same config without a native checkpoint.
    cp_comm_type: str | list[str] | None = "p2p"
    masked_softmax_fusion: bool = True
    persist_layer_norm: bool = True

    # Gated delta product mixers reuse the Mamba head/group fields.
    mamba_num_heads: int | None = 24
    mamba_num_groups: int = 24

    # MoE.
    num_moe_experts: int | None = 512
    moe_ffn_hidden_size: int | None = 4096
    moe_latent_size: int | None = 768
    moe_shared_expert_intermediate_size: int | None = 4096
    moe_router_topk: int = 10
    moe_router_topk_scaling_factor: float | None = 3.16
    moe_router_score_function: str = "sigmoid"
    moe_router_dtype: str | None = "fp32"
    moe_router_load_balancing_type: str = "quantile_balancing"
    moe_router_quantile_balancing_estimation_scope: str = "global_batch"
    moe_use_norm_before_up_proj: bool = True
    moe_shortcut_connection: bool = True
    moe_shortcut_parallel: bool = True
    moe_grouped_gemm: bool = True

    # Streamwise wide residual.
    wide_residual: Any = field(default_factory=nemotron4_nano_wide_residual)

    # Multi-token prediction: one shared MTP layer applied at two depths.
    mtp_num_layers: int | None = 2
    mtp_use_repeated_layer: bool = True
    mtp_hsm: bool = True

    # Vocabulary, sequence and output.
    vocab_size: int | None = 262272
    seq_length: int = 8192
    position_embedding_type: str = "none"
    share_embeddings_and_output_weights: bool = False
    logit_dtype: torch.dtype | None = torch.float32

    def finalize(self) -> None:
        """Check MCore support, then finalize as a hybrid model."""
        require_nemotron4_mcore()
        if self.gradient_accumulation_fusion and not _has_apex_wgrad_fusion_kernel():
            # The hybrid output layer and MTP projections are MCore ColumnParallelLinear, which
            # implement this fusion only through the APEX kernel; TransformerEngine alone is not
            # enough, so the generic "TE is installed" probe would make construction fail.
            logger.warning(
                "Disabling gradient_accumulation_fusion for Nemotron4: the fused_weight_gradient_mlp_cuda "
                "APEX extension is not installed and MCore ColumnParallelLinear requires it."
            )
            self.gradient_accumulation_fusion = False
        super().finalize()

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """Build the Nemotron4 language model after checking MCore support."""
        require_nemotron4_mcore()
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)


@dataclass
class Nemotron4VLModelProvider(Nemotron4ModelProvider):
    """Provider for the Nemotron4 vision-language model (WSM architecture by default).

    The Pixtral-style ViT and projector are co-located with the first language pipeline
    stage. Unlike the training fork's heterogeneous MIMO layout, there is no separate
    encoder rank span.
    """

    image_token_index: int = 18
    mimo_supports_mtp: ClassVar[bool] = True

    # Pixtral-style ViT (Mistral-ViT "1.6b").
    vision_num_layers: int = 30
    vision_hidden_size: int = 2048
    vision_ffn_hidden_size: int = 6144
    vision_num_attention_heads: int = 16
    vision_kv_channels: int = 128
    vision_layernorm_epsilon: float = 1e-5
    patch_dim: int = 14
    spatial_merge_size: int = 2
    img_h: int = 1540
    img_w: int = 1540

    # Two-layer projector: vision_hidden -> projector_ffn -> language hidden.
    # ``None`` means "the language hidden size", which is what Nemotron4 uses.
    vision_projection_ffn_hidden_size: int | None = None

    # ``None`` means "not requested"; a native checkpoint's freeze_lm/freeze_vit/
    # freeze_projection fill it in when hydrating from saved args.
    freeze_language_model: bool | None = None
    freeze_vision_model: bool | None = None
    freeze_vision_projection: bool | None = None

    # --- training runtime controls (NeMo RL sets these through policy.megatron_cfg) ---
    # The Bridge TransformerConfig marks these init=False and derives them from the
    # weight-shard counts. Expose them so YAML / model_overrides can request GTP directly;
    # shards = TP * GTP is derived in _apply_gtp_weight_shards().
    gtp_weight_remat_size: int = 1
    expert_gtp_weight_remat_size: int = 1
    vision_dp_over_cp: bool = True
    """Encode whole images on separate language CP ranks, then gather features."""
    recompute_vision: bool = False
    vision_recompute_granularity: Literal["full", "selective"] | None = None
    vision_recompute_method: Literal["uniform", "block"] | None = None
    vision_recompute_num_layers: int | None = None
    vision_recompute_budget_gib: float | None = None
    """MIMO only: activation budget of one ViT micro-batch in GiB. Each micro-batch then sets its full-recompute
    layer count and its MLP-only recompute layers to fit the budget (see ``Nemotron4MIMOVisionEncoder``).
    Requires a positive finite budget, full / block vision recompute, and the fused ViT SwiGLU and RoPE (the per-patch
    sizes assume them). The encoder must use BF16: the estimates have not been validated for MXFP8 or other encoder
    precisions. The decoder may use FP8 independently. ``vision_recompute_num_layers`` then has no effect in training."""
    vision_bias_activation_fusion: bool = False
    """Run the ViT SwiGLU as one fused kernel (Core ``bias_swiglu_impl``) instead of unfused ops."""
    vision_apply_rope_fusion: bool = False
    """Apply the ViT 2D RoPE with Transformer Engine's fused kernel. Needs a Megatron Core whose fused RoPE
    applies per-token frequencies of packed inputs; with an older Core, packed images get wrong angles and no
    error is raised."""

    # --- legacy native-checkpoint hydration (non-MIMO checkpoints converted from the fork) ---
    checkpoint_path: str = ""
    """Native checkpoint whose common args describe the architecture. Empty for HF-derived providers."""
    precision_recipe_path: str = ""
    precision_recipe_yaml: str | None = None
    """Raw TE precision recipe YAML; saved into run_config and copied next to it so resume
    does not need the original checkpoint directory."""
    saved_args: dict[str, Any] | None = None
    """Serialized native argparse payload; present on resume so architecture can be
    reconstructed without the original checkpoint."""

    @property
    def special_token_ids(self) -> dict[str, int]:
        """Return the image placeholder used by MegatronMIMO."""
        return {"images": self.image_token_index}

    def build_language_model_spec(self, pp_rank: int = 0) -> ModuleSpec:
        """Build the same hybrid language stack as the standard Nemotron4 VLM."""
        if self.tensor_model_parallel_size != 1 or self.pipeline_model_parallel_size != 1 or pp_rank != 0:
            raise ValueError("Nemotron4 MIMO conversion requires language TP=1 and PP=1.")
        config = self._copy_config_without_runtime_process_groups(deep=True)
        config.finalize()
        return ModuleSpec(
            module=Nemotron4MIMOLanguageModel,
            params={
                "config": config,
                "hybrid_stack_spec": self._language_layer_spec(),
                "vocab_size": config.vocab_size,
                "max_sequence_length": config.seq_length,
                "hybrid_layer_pattern": config.hybrid_layer_pattern,
                "fp16_lm_cross_entropy": config.fp16_lm_cross_entropy,
                "logit_dtype": config.logit_dtype,
                "parallel_output": config.parallel_output,
                "share_embeddings_and_output_weights": config.share_embeddings_and_output_weights,
                "position_embedding_type": config.position_embedding_type,
                "pre_process": True,
                "post_process": True,
            },
        )

    def build_mimo_modality_submodules_spec(self) -> dict[str, ModuleSpec]:
        """Build a Pixtral encoder and separate projector on the image component."""
        vision_config, projection_config = self._vision_configs(
            self._copy_config_without_runtime_process_groups(deep=True)
        )
        return {
            "images": ModuleSpec(
                module=VisionModalitySubmodules,
                params={},
                submodules={
                    "encoders": {
                        "pixtral": ModuleSpec(
                            module=Nemotron4MIMOVisionEncoder,
                            params={
                                "transformer_config": vision_config,
                                "transformer_layer_spec": self._vision_layer_spec(),
                                "patch_dim": self.patch_dim,
                                "img_h": self.img_h,
                                "img_w": self.img_w,
                                "add_class_token": False,
                                "class_token_len": 0,
                                "ln_pre": True,
                                "pos_emb_type": "rope2d",
                                "use_merger": self.spatial_merge_size > 1,
                                "spatial_merge_size": self.spatial_merge_size,
                                "recompute_budget_gib": self.vision_recompute_budget_gib,
                            },
                        )
                    },
                    "input_projections": [
                        ModuleSpec(
                            module=Nemotron4MIMOProjector,
                            params={
                                "config": projection_config,
                                "submodules": nemotron4_projection_submodules(),
                                "projector_type": "mlp",
                                "input_size": self.vision_hidden_size,
                            },
                        )
                    ],
                },
            )
        }

    def _vision_configs(
        self, language_cfg: "Nemotron4VLModelProvider"
    ) -> tuple[MCoreTransformerConfig, MCoreTransformerConfig]:
        """Derive the BF16 ViT and projector configs, with the requested ViT recompute."""
        vision_cfg = _dense_config(self._build_vision_config(language_cfg))
        projection_cfg = _dense_config(self._build_vision_projection_config(language_cfg))
        self._apply_vision_recompute_settings(vision_cfg)
        for config in (vision_cfg, projection_cfg):
            # The encoder and projector stay in BF16 regardless of the decoder's FP8/MXFP8 recipe.
            config.fp8 = None
            config.fp8_param = False
            config.quant_recipe = None
        return vision_cfg, projection_cfg

    def _build_vision_config(self, language_cfg: "Nemotron4VLModelProvider") -> "Nemotron4VLModelProvider":
        """Derive the ViT config from a language config copy.

        Mirrors the training fork's ``mistral_vit_vision_config``: a dense, non-hybrid,
        replicated (no GTP) transformer with its own geometry.
        """
        vision_cfg = copy.deepcopy(language_cfg)
        vision_cfg.num_layers = self.vision_num_layers
        vision_cfg.hidden_size = self.vision_hidden_size
        vision_cfg.ffn_hidden_size = self.vision_ffn_hidden_size
        vision_cfg.num_attention_heads = self.vision_num_attention_heads
        vision_cfg.num_query_groups = self.vision_num_attention_heads
        vision_cfg.kv_channels = self.vision_kv_channels
        vision_cfg.normalization = "RMSNorm"
        vision_cfg.layernorm_epsilon = self.vision_layernorm_epsilon
        vision_cfg.activation_func = torch.nn.functional.silu
        vision_cfg.gated_linear_unit = True
        vision_cfg.add_bias_linear = False
        vision_cfg.add_qkv_bias = False
        vision_cfg.rotary_interleaved = True
        vision_cfg.qk_layernorm = False
        vision_cfg.hidden_dropout = 0.0
        vision_cfg.attention_dropout = 0.0
        vision_cfg.layernorm_zero_centered_gamma = False
        vision_cfg.bias_activation_fusion = self.vision_bias_activation_fusion
        vision_cfg.bias_dropout_fusion = False
        vision_cfg.attention_softmax_in_fp32 = True
        vision_cfg.apply_rope_fusion = self.vision_apply_rope_fusion
        _make_dense_non_hybrid(vision_cfg)
        _make_replicated_single_stage(vision_cfg)
        return vision_cfg

    def _build_vision_projection_config(self, language_cfg: "Nemotron4VLModelProvider") -> "Nemotron4VLModelProvider":
        """Derive the ``mlp2`` projector config: two bias-free linears with tanh-GELU."""
        projection_cfg = copy.deepcopy(language_cfg)
        projection_cfg.num_layers = 1
        projection_cfg.num_attention_heads = 1
        projection_cfg.ffn_hidden_size = (
            self.vision_projection_ffn_hidden_size
            if self.vision_projection_ffn_hidden_size is not None
            else language_cfg.hidden_size
        )
        projection_cfg.gated_linear_unit = False
        projection_cfg.activation_func = nemotron4_projector_gelu
        projection_cfg.add_bias_linear = False
        projection_cfg.bias_activation_fusion = False
        projection_cfg.bias_dropout_fusion = False
        _make_dense_non_hybrid(projection_cfg)
        _make_replicated_single_stage(projection_cfg)
        return projection_cfg

    # ------------------------------------------------------------------ runtime controls
    def _apply_vision_recompute_settings(self, vision_config: Any) -> None:
        """Apply explicitly requested activation recompute to the ViT tower."""
        if self.vision_recompute_budget_gib is not None and (
            not self.recompute_vision
            or (self.vision_recompute_granularity, self.vision_recompute_method) != ("full", "block")
            or not (self.vision_bias_activation_fusion and self.vision_apply_rope_fusion)
            or not 0 < self.vision_recompute_budget_gib < math.inf
        ):
            raise ValueError(
                "vision_recompute_budget_gib requires a positive finite budget, full / block vision recompute, and "
                "vision_bias_activation_fusion and vision_apply_rope_fusion (its per-patch sizes assume them)."
            )
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

    def _apply_gtp_weight_shards(self) -> None:
        """Reconcile the GTP sizes with the logical weight-shard counts.

        Two callers set these differently. NeMo RL sets ``gtp_weight_remat_size`` /
        ``expert_gtp_weight_remat_size`` through ``model_overrides`` and leaves the shard
        counts derived. Bridge's ``load_megatron_model`` resets the GTP sizes to 1 and passes
        the saved topology as shard counts through ``mp_overrides``. An explicit shard count
        larger than the parallel size therefore wins over a default GTP size of 1; otherwise
        the GTP size drives the shard count.
        """
        if self.gtp_weight_remat_size < 1 or self.expert_gtp_weight_remat_size < 1:
            raise ValueError("gtp_weight_remat_size and expert_gtp_weight_remat_size must be >= 1")
        if self.expert_tensor_parallel_size is None and self.expert_model_parallel_size > 1:
            self.expert_tensor_parallel_size = 1
        for shards_name, parallel_size, remat_name in (
            ("tensor_parallel_num_weight_shards", self.tensor_model_parallel_size, "gtp_weight_remat_size"),
            (
                "expert_tensor_parallel_num_weight_shards",
                self.expert_tensor_parallel_size or 1,
                "expert_gtp_weight_remat_size",
            ),
        ):
            shards = getattr(self, shards_name)
            remat = getattr(self, remat_name)
            if remat == 1 and isinstance(shards, int) and shards > parallel_size:
                if shards % parallel_size:
                    raise ValueError(f"{shards_name}={shards} must be a multiple of the parallel size {parallel_size}")
                remat = shards // parallel_size
                setattr(self, remat_name, remat)
            setattr(self, shards_name, parallel_size * remat)

    # ------------------------------------------------------------------ precision recipe
    def _load_precision_recipe_yaml(self) -> None:
        if self.precision_recipe_yaml:
            return
        if self.precision_recipe_path and os.path.isfile(self.precision_recipe_path):
            self.precision_recipe_yaml = Path(self.precision_recipe_path).read_text()

    def _materialize_precision_recipe(self) -> str:
        """Return a readable recipe file, writing one from saved YAML if needed."""
        self._load_precision_recipe_yaml()
        if self.precision_recipe_path and os.path.isfile(self.precision_recipe_path):
            return self.precision_recipe_path
        if not self.precision_recipe_yaml:
            raise ValueError("precision_recipe_path is required")
        handle = tempfile.NamedTemporaryFile(prefix="nemotron4_precision_recipe_", suffix=".yaml", delete=False)
        with handle:
            handle.write(self.precision_recipe_yaml.encode())
        self.precision_recipe_path = handle.name
        return self.precision_recipe_path

    def save_precision_recipe_assets(self, checkpoint_dir: str) -> None:
        """Copy the TE precision recipe next to run_config.yaml (called by save_checkpoint)."""
        self._load_precision_recipe_yaml()
        if not self.precision_recipe_yaml:
            return
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(checkpoint_dir, PRECISION_RECIPE_FILENAME).write_text(self.precision_recipe_yaml)

    # ------------------------------------------------------------------ native-checkpoint hydration
    def _apply_checkpoint_freeze_settings(self, args: argparse.Namespace) -> None:
        if self.freeze_language_model is None:
            self.freeze_language_model = bool(getattr(args, "freeze_lm", False))
        if self.freeze_vision_model is None:
            self.freeze_vision_model = bool(getattr(args, "freeze_vit", False))
        if self.freeze_vision_projection is None:
            self.freeze_vision_projection = bool(getattr(args, "freeze_projection", False))

    def _hydrate_from_native_args(self, args: argparse.Namespace) -> None:
        """Copy the architecture of a native Nemotron4 checkpoint onto this provider."""
        config = _transformer_config_from_args(args)
        for item in dataclasses.fields(config):
            if item.name in _CHECKPOINT_PARALLEL_FIELDS or not hasattr(self, item.name):
                continue
            setattr(self, item.name, getattr(config, item.name))
        self.hybrid_layer_pattern = args.hybrid_layer_pattern
        self.hybrid_override_pattern = None
        self.mtp_hybrid_override_pattern = None
        self.mtp_use_repeated_layer = args.mtp_use_repeated_layer
        self.seq_length = args.seq_length
        self.vocab_size = args.padded_vocab_size
        self.should_pad_vocab = False
        self.position_embedding_type = "none"
        self.share_embeddings_and_output_weights = False
        self.logit_dtype = args.logit_dtype
        # Vision tower and projector.
        num_layers, hidden, ffn, kv = _MISTRAL_VIT_GEOMETRY[getattr(args, "mistral_vit_model_size", "410m")]
        self.vision_num_layers = num_layers
        self.vision_hidden_size = hidden
        self.vision_ffn_hidden_size = ffn
        self.vision_kv_channels = kv
        self.vision_num_attention_heads = 16
        self.vision_projection_ffn_hidden_size = None
        self.image_token_index = args.image_token_id
        self.img_h = args.img_h
        self.img_w = args.img_w
        self.patch_dim = args.patch_dim
        self.spatial_merge_size = args.spatial_merge_size

    def __post_init__(self) -> None:
        already_loaded = getattr(self, "_saved_args", None) is not None
        if already_loaded:
            # NeMo RL calls this again after applying megatron_cfg TP/ETP.
            self._apply_gtp_weight_shards()
        super().__post_init__()
        if already_loaded:
            return
        self._saved_args = None
        if self.saved_args is not None:
            if "vlm_checkpoint_layout" not in self.saved_args:
                raise ValueError(
                    "saved_args is the argparse namespace of a hand-converted native Nemotron4 "
                    "checkpoint and is only set by that checkpoint's run_config.yaml; got a partial "
                    f"dict with keys {sorted(self.saved_args)}. Runtime knobs such as attention_backend "
                    "belong on the provider fields directly (for NeMo RL: policy.megatron_cfg.*), not under "
                    "model_overrides.saved_args."
                )
            # Resume: run_config already carries the hydrated architecture fields.
            args = _namespace_from_dict(self.saved_args)
            _validate_native_args(args)
            if self.precision_recipe_path or self.precision_recipe_yaml:
                args.te_precision_config_file = self._materialize_precision_recipe()
            self._saved_args = args
            self._apply_checkpoint_freeze_settings(args)
            self._apply_gtp_weight_shards()
            return
        if not self.checkpoint_path:
            # HF-derived or explicitly configured provider: nothing to hydrate.
            return
        if not (self.precision_recipe_path or self.precision_recipe_yaml):
            raise ValueError("checkpoint_path requires precision_recipe_path or precision_recipe_yaml")
        args = copy.deepcopy(dist_checkpointing.load_common_state_dict(self.checkpoint_path)["args"])
        _validate_native_args(args)
        args.te_precision_config_file = self._materialize_precision_recipe()
        self._saved_args = args
        self.saved_args = _dict_from_namespace(args)
        self._hydrate_from_native_args(args)
        self._apply_checkpoint_freeze_settings(args)
        self._apply_gtp_weight_shards()

    def finalize(self) -> None:
        """Check MCore support, reconcile hybrid/MTP fields, then derive GTP from TP/GTP."""
        if getattr(self, "_saved_args", None) is None:
            # No native checkpoint to fill these in: "not requested" means "not frozen".
            for name in ("freeze_language_model", "freeze_vision_model", "freeze_vision_projection"):
                if getattr(self, name) is None:
                    setattr(self, name, False)
        self._apply_gtp_weight_shards()
        _require_single_tensor_parallel(self)
        requested_gtp = self.gtp_weight_remat_size
        requested_expert_gtp = self.expert_gtp_weight_remat_size
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
        for item in dataclasses.fields(self):
            if item.name.startswith("_"):
                continue
            result[item.name] = _ConfigContainerBase._convert_value_to_dict(getattr(self, item.name))
        if self.precision_recipe_yaml:
            result["precision_recipe_path"] = PRECISION_RECIPE_FILENAME
        return result

    # ------------------------------------------------------------------ model construction
    def _language_layer_spec(self) -> ModuleSpec:
        """Hybrid stack spec with the stride-safe GDP mixer substituted."""
        saved_args = getattr(self, "_saved_args", None)
        if saved_args is not None and getattr(saved_args, "spec", None):
            # Native checkpoints name their own layer spec; honor it exactly.
            spec = copy.deepcopy(import_module(saved_args.spec))
        else:
            spec = copy.deepcopy(self._resolve_hybrid_stack_spec())
        gdp_mixer_spec = spec.submodules.mamba_layer.submodules.mixer
        if gdp_mixer_spec.module is not GatedDeltaProductMixer:
            raise TypeError(
                f"Nemotron4 expected its recurrent layer spec to use GatedDeltaProductMixer, got {gdp_mixer_spec.module!r}"
            )
        gdp_mixer_spec.module = Nemotron4GatedDeltaProductMixer
        return spec

    @staticmethod
    def _vision_layer_spec() -> ModuleSpec:
        """ViT layer spec whose MLP keeps GTP GLU shards contiguous."""
        vision_layer_spec = get_vit_layer_with_transformer_engine_spec()
        vision_mlp_spec = vision_layer_spec.submodules.mlp
        if isinstance(vision_mlp_spec, partial):
            vision_layer_spec.submodules.mlp = partial(
                Nemotron4VisionMLP.as_mlp_submodule, *vision_mlp_spec.args, **vision_mlp_spec.keywords
            )
        elif isinstance(vision_mlp_spec, ModuleSpec):
            vision_mlp_spec.module = Nemotron4VisionMLP
        else:
            raise TypeError(f"Unexpected ViT MLP spec type: {type(vision_mlp_spec)!r}")
        return vision_layer_spec

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> Nemotron4VLModel:
        """Build the Nemotron4 VLM for the current provider parallel sizes."""
        if self.vision_recompute_budget_gib is not None:
            raise ValueError("vision_recompute_budget_gib is supported only by the MIMO vision encoder.")
        require_nemotron4_mcore()
        language_cfg = self._copy_config_without_runtime_process_groups(deep=True)
        vision_cfg, projection_cfg = self._vision_configs(language_cfg)

        pg_collection = self._pg_collection
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        add_encoder = True if self.pipeline_model_parallel_size == 1 else is_pp_first_stage(pg_collection.pp)
        model = Nemotron4VLModel(
            language_transformer_config=language_cfg,
            language_transformer_layer_spec=self._language_layer_spec(),
            language_vocab_size=self.vocab_size,
            language_max_sequence_length=self.seq_length,
            vision_transformer_config=vision_cfg,
            vision_transformer_layer_spec=self._vision_layer_spec(),
            vision_projection_config=projection_cfg,
            vision_projection_layer_spec=nemotron4_projection_submodules(),
            image_token_index=self.image_token_index,
            hybrid_layer_pattern=self.hybrid_layer_pattern,
            img_h=self.img_h,
            img_w=self.img_w,
            patch_dim=self.patch_dim,
            spatial_merge_size=self.spatial_merge_size,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            language_position_embedding_type=self.position_embedding_type,
            pre_process=True if pre_process is None else pre_process,
            post_process=True if post_process is None else post_process,
            add_encoder=add_encoder,
            add_decoder=True,
            pg_collection=pg_collection,
            vision_dp_over_cp=self.vision_dp_over_cp,
            vp_stage=vp_stage,
        )
        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=bool(self.freeze_language_model),
                freeze_vision_model=bool(self.freeze_vision_model),
                freeze_vision_projection=bool(self.freeze_vision_projection),
            )
        return model


def _require_single_tensor_parallel(provider: "Nemotron4VLModelProvider") -> None:
    """Reject tensor parallelism before any model is built.

    Nemotron4 runs at TP=1 (weights may still be GTP-sharded): its attention has a single KV
    group, the sparse-attention CuTe kernels and indexer require TP=1, the GDP weight mappings
    refuse TP>1, and the projector config (one attention head) cannot be split. Without this
    check TP>1 fails later, inside model construction, with a misleading head-count error.

    Raises:
        ValueError: If ``tensor_model_parallel_size`` is not 1.
    """
    if provider.tensor_model_parallel_size != 1:
        sparse = getattr(provider, "experimental_attention_variant", None) == "ssa"
        raise ValueError(
            f"Nemotron4 requires tensor_model_parallel_size=1 (got {provider.tensor_model_parallel_size}): one KV "
            "group, TP=1-only GDP conversion"
            + (" and the CuTe sparse-attention kernels" if sparse else "")
            + ". Use expert parallelism, context parallelism or GTP weight sharding "
            "(gtp_weight_remat_size) instead."
        )


def _make_dense_non_hybrid(config: MCoreTransformerConfig) -> None:
    """Strip language-only architecture from an encoder/projector config copy.

    Mirrors the training fork's ``_make_dense_non_hybrid`` so the ViT and projector
    inherit no MoE, MTP, hybrid, shortcut or wide-residual settings.
    """
    config.activation_func_tanh_clamp_scale = None
    config.fp32_residual_connection = False
    config.pipeline_dtype = config.params_dtype
    config.first_last_layers_bf16 = False
    config.num_layers_at_start_in_bf16 = 0
    config.num_layers_at_end_in_bf16 = 0
    config.mtp_num_layers = None
    config.mtp_use_repeated_layer = False
    config.num_moe_experts = None
    config.moe_ffn_hidden_size = None
    config.moe_shared_expert_intermediate_size = None
    config.moe_grouped_gemm = False
    config.moe_use_grouped_tensor = False
    config.moe_router_fusion = False
    config.moe_permute_fusion = False
    config.moe_shared_expert_overlap = False
    config.moe_shortcut_connection = False
    config.moe_shortcut_parallel = False
    config.moe_shortcut_post_norm = False
    config.is_hybrid_model = False
    config.use_fused_weighted_squared_relu = False
    config.wide_residual = None
    config.residual_stream_recompute_num_layers = None
    # The ViT and projector use ordinary dense attention; the language model's sparse-attention
    # settings would otherwise fail the CuTe SSA validation for their head geometry.
    if getattr(config, "experimental_attention_variant", None) is not None:
        config.experimental_attention_variant = None
    if getattr(config, "ssa_train_indexer_only", False):
        config.ssa_train_indexer_only = False
    if config.recompute_modules is not None:
        decoder_only = {"residual_stream", "shortcut_pre_mlp_layernorm"}
        config.recompute_modules = [module for module in config.recompute_modules if module not in decoder_only]


def _make_replicated_single_stage(config: MCoreTransformerConfig) -> None:
    """Keep an encoder/projector config on one pipeline stage, unsharded by CP, SP or GTP."""
    config.sequence_parallel = False
    config.context_parallel_size = 1
    config.tp_comm_overlap = False
    config.pipeline_model_parallel_size = 1
    config.virtual_pipeline_model_parallel_size = None
    config.num_layers_in_first_pipeline_stage = None
    config.num_layers_in_last_pipeline_stage = None
    config.pipeline_model_parallel_layout = None
    config.account_for_embedding_in_pipeline_split = False
    config.account_for_loss_in_pipeline_split = False
    config.tensor_parallel_num_weight_shards = config.tensor_model_parallel_size
    config.gtp_weight_remat_size = 1
    expert_tp = config.expert_tensor_parallel_size or config.tensor_model_parallel_size
    config.expert_tensor_parallel_num_weight_shards = expert_tp
    config.expert_gtp_weight_remat_size = 1
    config.recompute_granularity = None
    config.recompute_method = None
    config.recompute_num_layers = None


def _dense_config(config: Nemotron4VLModelProvider) -> MCoreTransformerConfig:
    # Encoder/projector configs must not run HybridModelProvider.finalize(), which
    # would overwrite their layer counts with the language pattern's layer count.
    config.mtp_hsm = False
    config.moe_router_load_balancing_type = "aux_loss"
    config.expert_model_parallel_size = 1
    config.expert_tensor_parallel_size = 1
    return MCoreTransformerConfig(
        **{f.name: getattr(config, f.name) for f in dataclasses.fields(MCoreTransformerConfig) if f.init}
    )
