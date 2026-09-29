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
"""

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from megatron.core import parallel_state
from megatron.core.activations import squared_relu
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig as MCoreTransformerConfig

from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.models.nemotron4.modeling_nemotron4_vl import Nemotron4VLModel


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

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

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
        vision_cfg.bias_activation_fusion = False
        vision_cfg.bias_dropout_fusion = False
        vision_cfg.attention_softmax_in_fp32 = True
        vision_cfg.apply_rope_fusion = False
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

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> Nemotron4VLModel:
        """Build the Nemotron4 VLM with the encoder on the first pipeline stage."""
        require_nemotron4_mcore()
        language_cfg = self._copy_config_without_runtime_process_groups(deep=True)
        vision_cfg = self._build_vision_config(language_cfg)
        projection_cfg = self._build_vision_projection_config(language_cfg)

        add_encoder = parallel_state.is_pipeline_first_stage() if self.pipeline_model_parallel_size > 1 else True
        model = Nemotron4VLModel(
            language_transformer_config=language_cfg,
            language_transformer_layer_spec=self._resolve_hybrid_stack_spec(),
            language_vocab_size=self.vocab_size,
            language_max_sequence_length=self.seq_length,
            vision_transformer_config=vision_cfg,
            vision_transformer_layer_spec=get_vit_layer_with_transformer_engine_spec(),
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
            pg_collection=self._pg_collection,
            vp_stage=vp_stage,
        )

        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )
        return model


def _make_dense_non_hybrid(config: MCoreTransformerConfig) -> None:
    """Strip language-only architecture from an encoder/projector config copy.

    Mirrors the training fork's ``_make_dense_non_hybrid`` so the ViT and projector
    inherit no MoE, MTP, hybrid, shortcut or wide-residual settings.
    """
    config.activation_func_tanh_clamp_scale = None
    config.fp32_residual_connection = False
    config.pipeline_dtype = config.params_dtype
    config.mtp_num_layers = None
    config.mtp_use_repeated_layer = False
    config.num_moe_experts = None
    config.moe_ffn_hidden_size = None
    config.moe_shared_expert_intermediate_size = None
    config.moe_grouped_gemm = False
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
