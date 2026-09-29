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

"""HF <-> Megatron bridge for Nemotron4 (``Nemotron4ForConditionalGeneration``).

Index convention: HF ``layers.N`` is a *block*, and so is the live Megatron
``decoder.layers.N`` -- MCore regroups each ``*E`` / ``ME`` pair into one
``ShortcutMoEBlock`` -- so block indices map one-to-one. (Only ``torch_dist``
checkpoints use physical layer numbers, and MCore handles those itself.) This
holds at pipeline-parallel size 1, which wide residuals already require.

Transforms that are not plain copies:

* Wide-residual controllers are zero-padded to MCore's minimum vector length in
  Megatron and stored at ``num_streams`` entries in HF.
* The vision tower's Q/K rows are permuted per head between Megatron's
  interleaved RoPE pairs ``(2i, 2i+1)`` and HF Pixtral's ``rotate_half`` pairs
  ``(i, i + d/2)``.
* GDP ``in_proj`` / ``conv1d`` are single fused tensors on both sides with the
  same row order at tensor-parallel size 1. Nemotron4's attention has one KV
  group, so tensor parallelism is 1; the GDP mappings assert it rather than
  implement the per-piece sharding they would need beyond that.

Megatron state with no HF counterpart:

* ``router.qb_bin_bounds`` -- quantile-balancing bookkeeping. Left at its
  initial value on import, not exported.
* Multi-token prediction, only when the HF config has none
  (``num_nextn_predict_layers == 0``): the provider then builds no MTP, and a
  Megatron model that does carry MTP exports without it, with a warning.

When the HF config has MTP, ``mtp.layers.i`` maps one-to-one: ``enorm``,
``hnorm``, ``eh_proj`` and ``final_layernorm`` directly, and the inner
``mtp_model_layer.layers.0`` to HF's ``block`` -- a shortcut attention + MoE
block on an ordinary residual, i.e. the backbone block's tensors without the
wide-residual controllers.
"""

import logging
from typing import Mapping

import torch
from torch import nn

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    FusedExpertMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    QKVMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.nemotron4.modeling_nemotron4_vl import Nemotron4VLModel
from megatron.bridge.models.nemotron4.nemotron4_provider import Nemotron4VLModelProvider, require_nemotron4_mcore


logger = logging.getLogger(__name__)

# HF ``layer_types`` block -> Megatron hybrid-pattern symbols (one symbol per physical layer).
_LAYER_TYPE_TO_PATTERN = {
    "linear_attention": "M",
    "mlp": "-",
    "shortcut_moe_attn": "*E",
    "shortcut_moe_gdp": "ME",
}

# Synthesized HF name prefix for Megatron-only state; never present in an HF checkpoint.
_MEGATRON_ONLY_PREFIX = "megatron_only::"


def layer_types_to_hybrid_pattern(layer_types: list[str]) -> str:
    """Expand HF ``layer_types`` blocks into the Megatron ``hybrid_layer_pattern`` (no MTP suffix).

    Raises:
        ValueError: On a layer type Nemotron4 does not define.
    """
    unknown = sorted(set(layer_types) - set(_LAYER_TYPE_TO_PATTERN))
    if unknown:
        raise ValueError(f"Unsupported Nemotron4 layer_types {unknown}; expected {sorted(_LAYER_TYPE_TO_PATTERN)}.")
    return "".join(_LAYER_TYPE_TO_PATTERN[layer_type] for layer_type in layer_types)


class _ControllerMapping(ReplicatedMapping):
    """Wide-residual controller vector: ``num_streams`` entries in HF, zero-padded in Megatron."""

    def __init__(self, megatron_param: str, hf_param: str, num_streams: int):
        super().__init__(megatron_param, hf_param)
        self.num_streams = num_streams

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        """Zero-pad the HF vector to the live parameter's length, then replicate."""
        target = getattr(megatron_module, self.megatron_param.rsplit(".", 1)[-1])
        if hf_weights.shape != (self.num_streams,):
            raise ValueError(f"{self.hf_param}: expected shape ({self.num_streams},), got {tuple(hf_weights.shape)}.")
        padded = torch.zeros(target.shape, dtype=hf_weights.dtype, device=hf_weights.device)
        padded[: self.num_streams] = hf_weights
        return super().hf_to_megatron(padded, megatron_module)

    def megatron_to_hf(self, megatron_weights: torch.Tensor | None, megatron_module: nn.Module | None):
        """Replicate, then keep only the active ``num_streams`` entries."""
        converted = super().megatron_to_hf(megatron_weights, megatron_module)
        return {name: tensor[: self.num_streams] for name, tensor in converted.items()}

    def resolve(self, captures: tuple[str, ...]) -> MegatronParamMapping:
        """Resolve wildcards, keeping ``num_streams``."""
        megatron_param, hf_param = self._resolve_names(captures)
        return type(self)(megatron_param, hf_param, self.num_streams)


def _rotate_half_rows_to_interleaved(weight: torch.Tensor, head_dim: int) -> torch.Tensor:
    """HF Pixtral Q/K rows -> Megatron interleaved-RoPE rows, per head."""
    heads = weight.shape[0] // head_dim
    return weight.view(heads, 2, head_dim // 2, -1).transpose(1, 2).reshape(weight.shape)


def _interleaved_rows_to_rotate_half(weight: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Megatron interleaved-RoPE Q/K rows -> HF Pixtral rows, per head (inverse of the above)."""
    heads = weight.shape[0] // head_dim
    return weight.view(heads, head_dim // 2, 2, -1).transpose(1, 2).reshape(weight.shape)


class _PixtralQKVMapping(QKVMapping):
    """Vision QKV: interleave Q/K/V and convert the Q/K rotary row convention."""

    def __init__(self, megatron_param: str, q: str, k: str, v: str, head_dim: int):
        super().__init__(megatron_param, q, k, v)
        self.head_dim = head_dim

    def hf_to_megatron(self, hf_weights: dict[str, torch.Tensor], megatron_module: nn.Module) -> torch.Tensor:
        """Permute HF Q/K rows to Megatron's interleaved pairs, then interleave Q/K/V."""
        if self.tp_rank == 0:
            hf_weights = {
                **hf_weights,
                "q": _rotate_half_rows_to_interleaved(hf_weights["q"], self.head_dim),
                "k": _rotate_half_rows_to_interleaved(hf_weights["k"], self.head_dim),
            }
        return super().hf_to_megatron(hf_weights, megatron_module)

    def megatron_to_hf(self, megatron_weights: torch.Tensor | None, megatron_module: nn.Module | None):
        """Split Q/K/V, then permute Q/K rows back to HF's ``rotate_half`` pairs."""
        converted = super().megatron_to_hf(megatron_weights, megatron_module)
        if not converted:
            return converted
        for role in ("q", "k"):
            name = self.hf_param[role]
            converted[name] = _interleaved_rows_to_rotate_half(converted[name], self.head_dim)
        return converted

    def resolve(self, captures: tuple[str, ...]) -> MegatronParamMapping:
        """Resolve wildcards, keeping ``head_dim``."""
        megatron_param, hf_param = self._resolve_names(captures)
        return type(self)(megatron_param, hf_param["q"], hf_param["k"], hf_param["v"], self.head_dim)


class _SingleTPColumnMapping(ColumnParallelMapping):
    """Column mapping for GDP tensors whose fused row layout is only valid at TP=1.

    Under tensor parallelism MCore shards each GDP piece (``z``, ``V*``, ``K*``, ``Q``,
    ``b*``, ``a``) separately, so a plain dim-0 chunk of the fused tensor would be wrong.
    Nemotron4 never runs at TP>1 (its attention has a single KV group), so this refuses
    rather than guessing.
    """

    def _require_single_tp(self) -> None:
        if self.tp_size != 1:
            raise NotImplementedError(
                f"{self.megatron_param}: Nemotron4 GDP conversion supports tensor_model_parallel_size=1 "
                f"only (got {self.tp_size})."
            )

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        """Copy the fused tensor at TP=1."""
        self._require_single_tp()
        return super().hf_to_megatron(hf_weights, megatron_module)

    def megatron_to_hf(self, megatron_weights: torch.Tensor | None, megatron_module: nn.Module | None):
        """Export the fused tensor at TP=1."""
        self._require_single_tp()
        return super().megatron_to_hf(megatron_weights, megatron_module)


_warned_dropped_megatron_state: set[str] = set()


class _MegatronOnlyStateMapping(MegatronParamMapping[torch.Tensor]):
    """Megatron state with no HF counterpart: keep it on import, omit it on export."""

    def __init__(self, megatron_param: str, reason: str):
        super().__init__(megatron_param, _MEGATRON_ONLY_PREFIX + megatron_param)
        self.allow_hf_name_mismatch = True
        self.reason = reason

    def hf_to_megatron(self, hf_weights: torch.Tensor | None, megatron_module: nn.Module) -> None:
        """Leave the Megatron tensor at its current value."""
        return None

    def megatron_to_hf(self, megatron_weights: torch.Tensor | None, megatron_module: nn.Module | None):
        """Export nothing; warn once per reason."""
        if self.reason not in _warned_dropped_megatron_state:
            _warned_dropped_megatron_state.add(self.reason)
            logger.warning("Nemotron4 export omits Megatron-only state: %s", self.reason)
        return {}

    def resolve(self, captures: tuple[str, ...]) -> MegatronParamMapping:
        """Resolve wildcards, keeping the reason."""
        megatron_param, _ = self._resolve_names(captures)
        return type(self)(megatron_param, self.reason)


@MegatronModelBridge.register_bridge(
    source="Nemotron4ForConditionalGeneration",
    target=Nemotron4VLModel,
    provider=Nemotron4VLModelProvider,
    model_type="nemotron4",
)
class Nemotron4Bridge(MegatronModelBridge):
    """Bridge between HF ``Nemotron4ForConditionalGeneration`` and :class:`Nemotron4VLModel`."""

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Nemotron4VLModelProvider:
        """Build a :class:`Nemotron4VLModelProvider` from the HF config.

        MTP layers are built exactly when the HF config has them
        (``num_nextn_predict_layers``), one ``*E`` pattern depth each.

        Raises:
            ValueError: If the HF config uses a setting this bridge cannot represent.
        """
        require_nemotron4_mcore()
        from megatron.core.transformer.transformer_config import WideResidualConfig

        hf_config = hf_pretrained.config
        text = hf_config.text_config
        vision = hf_config.vision_config
        _validate_hf_config(hf_config)

        mtp_depths = getattr(text, "num_nextn_predict_layers", 0)
        pattern = layer_types_to_hybrid_pattern(list(text.layer_types))
        if mtp_depths:
            # Every Nemotron4 MTP depth is one shortcut attention + MoE block.
            pattern += "/*E" * mtp_depths
        return Nemotron4VLModelProvider(
            hybrid_layer_pattern=pattern,
            mtp_num_layers=mtp_depths or None,
            mtp_use_repeated_layer=bool(mtp_depths) and text.mtp_use_repeated_layer,
            # Training-only hidden-state mixing, as pretraining used it; MCore requires >= 2 depths.
            mtp_hsm=mtp_depths >= 2,
            hidden_size=text.hidden_size,
            ffn_hidden_size=text.intermediate_size,
            num_attention_heads=text.num_attention_heads,
            num_query_groups=text.num_key_value_heads,
            kv_channels=text.head_dim,
            layernorm_epsilon=text.rms_norm_eps,
            init_method_std=text.initializer_range,
            activation_func_tanh_clamp_scale=text.activation_clamp_value,
            vocab_size=text.vocab_size,
            seq_length=text.max_position_embeddings,
            share_embeddings_and_output_weights=hf_config.tie_word_embeddings,
            mamba_num_heads=text.gdp_num_heads,
            mamba_num_groups=text.gdp_num_groups,
            mamba_head_dim=text.gdp_head_dim,
            mamba_state_dim=text.gdp_key_head_dim,
            gdp_num_householder=text.gdp_num_householder,
            num_moe_experts=text.n_routed_experts,
            moe_router_topk=text.num_experts_per_tok,
            moe_ffn_hidden_size=text.moe_intermediate_size,
            moe_shared_expert_intermediate_size=text.moe_shared_expert_intermediate_size,
            moe_latent_size=text.moe_latent_size,
            moe_use_norm_before_up_proj=text.moe_use_norm_before_up_proj,
            moe_router_topk_scaling_factor=text.routed_scaling_factor,
            moe_shortcut_post_norm=text.moe_shortcut_post_norm,
            wide_residual=WideResidualConfig(
                num_streams=text.wide_residual_num_streams,
                streamwise_sigmoid_init_scale=text.wide_residual_streamwise_sigmoid_init_scale,
                learned_retention=text.wide_residual_learned_retention,
                retention_init=text.wide_residual_retention_init,
                retention_max_forget=text.wide_residual_retention_max_forget,
            ),
            image_token_index=hf_config.image_token_id,
            spatial_merge_size=hf_config.spatial_merge_size,
            vision_num_layers=vision.num_hidden_layers,
            vision_hidden_size=vision.hidden_size,
            vision_ffn_hidden_size=vision.intermediate_size,
            vision_num_attention_heads=vision.num_attention_heads,
            vision_kv_channels=vision.head_dim,
            vision_layernorm_epsilon=1e-5,
            patch_dim=vision.patch_size,
            img_h=vision.image_size,
            img_w=vision.image_size,
            vision_projection_ffn_hidden_size=None,
        )

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor | dict[str, torch.Tensor] | None:
        """Skip the lookup for Megatron-only state, which has no HF tensor."""
        if isinstance(hf_param, str) and hf_param.startswith(_MEGATRON_ONLY_PREFIX):
            return None
        return super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return the Nemotron4 VLM parameter mappings."""
        text = self.hf_config.text_config
        vision = self.hf_config.vision_config
        num_streams = text.wide_residual_num_streams
        learned_retention = text.wide_residual_learned_retention

        L = "language_model.decoder.layers.*"
        HL = "model.language_model.layers.*"
        mappings: list[MegatronParamMapping] = [
            ColumnParallelMapping(
                "language_model.embedding.word_embeddings.weight", "model.language_model.embed_tokens.weight"
            ),
            ColumnParallelMapping("language_model.output_layer.weight", "lm_head.weight"),
            ReplicatedMapping("language_model.decoder.final_norm.weight", "model.language_model.norm.weight"),
            _ControllerMapping(
                "language_model.decoder.residual_stream_readout.exit_map.logit",
                "model.language_model.residual_readout.logit",
                num_streams,
            ),
        ]

        def controllers(megatron_prefix: str, hf_prefix: str) -> list[MegatronParamMapping]:
            entries = [
                _ControllerMapping(f"{megatron_prefix}.read_map.logit", f"{hf_prefix}.read_logit", num_streams),
                _ControllerMapping(f"{megatron_prefix}.write_map.logit", f"{hf_prefix}.write_logit", num_streams),
            ]
            if learned_retention:
                entries.append(
                    _ControllerMapping(
                        f"{megatron_prefix}.retention.retention_logit", f"{hf_prefix}.retention_logit", num_streams
                    )
                )
            return entries

        def gdp(mixer: str, hf: str, residual: str, hf_residual: str) -> list[MegatronParamMapping]:
            return [
                AutoMapping(f"{mixer}.in_proj.layer_norm_weight", f"{hf}.input_layernorm.weight"),
                _SingleTPColumnMapping(f"{mixer}.in_proj.weight", f"{hf}.mixer.in_proj.weight"),
                _SingleTPColumnMapping(f"{mixer}.conv1d.weight", f"{hf}.mixer.conv1d.weight"),
                _SingleTPColumnMapping(f"{mixer}.dt_bias", f"{hf}.mixer.dt_bias"),
                _SingleTPColumnMapping(f"{mixer}.A_log", f"{hf}.mixer.A_log"),
                _SingleTPColumnMapping(f"{mixer}.norm.weight", f"{hf}.mixer.norm.weight"),
                RowParallelMapping(f"{mixer}.out_proj.weight", f"{hf}.mixer.out_proj.weight"),
                *controllers(residual, hf_residual),
            ]

        def attention(compute: str, hf: str) -> list[MegatronParamMapping]:
            return [
                AutoMapping(f"{compute}.self_attention.linear_qkv.layer_norm_weight", f"{hf}.input_layernorm.weight"),
                QKVMapping(
                    f"{compute}.self_attention.linear_qkv.weight",
                    q=f"{hf}.self_attn.q_proj.weight",
                    k=f"{hf}.self_attn.k_proj.weight",
                    v=f"{hf}.self_attn.v_proj.weight",
                ),
                RowParallelMapping(f"{compute}.self_attention.linear_proj.weight", f"{hf}.self_attn.o_proj.weight"),
            ]

        def shortcut_moe(block: str, hf: str) -> list[MegatronParamMapping]:
            """The shortcut block's MoE half, without wide-residual controllers."""
            mlp = f"{block}.moe_layer.mlp"
            moe = f"{hf}.moe"
            entries = [
                ReplicatedMapping(
                    f"{block}.shortcut_pre_mlp_layernorm.weight", f"{hf}.shortcut_input_layernorm.weight"
                ),
                ReplicatedMapping(f"{block}.moe_layer.pre_mlp_layernorm.weight", f"{hf}.pre_moe_layernorm.weight"),
                ReplicatedMapping(f"{mlp}.router.weight", f"{moe}.gate.weight"),
                ReplicatedMapping(f"{mlp}.router.expert_bias", f"{moe}.gate.e_score_correction_bias"),
                _MegatronOnlyStateMapping(
                    f"{mlp}.router.qb_bin_bounds", "router.qb_bin_bounds (quantile-balancing bookkeeping)"
                ),
                ReplicatedMapping(f"{mlp}.fc1_latent_proj.weight", f"{moe}.fc1_latent_proj.weight"),
                ReplicatedMapping(f"{mlp}.fc2_latent_proj.weight", f"{moe}.fc2_latent_proj.weight"),
                FusedExpertMapping(f"{mlp}.experts.linear_fc1.weight*", f"{moe}.experts.up_proj"),
                FusedExpertMapping(f"{mlp}.experts.linear_fc2.weight*", f"{moe}.experts.down_proj"),
                ColumnParallelMapping(
                    f"{mlp}.shared_experts.linear_fc1.weight", f"{moe}.shared_experts.up_proj.weight"
                ),
                RowParallelMapping(
                    f"{mlp}.shared_experts.linear_fc2.weight", f"{moe}.shared_experts.down_proj.weight"
                ),
            ]
            if text.moe_use_norm_before_up_proj:
                entries.append(ReplicatedMapping(f"{mlp}.fc2_norm.weight", f"{moe}.fc2_norm.weight"))
            if text.moe_shortcut_post_norm:
                entries.append(
                    ReplicatedMapping(f"{block}.shortcut_post_norm.weight", f"{hf}.shortcut_post_norm.weight")
                )
            return entries

        # Shortcut blocks first: their names nest one level deeper than standalone layers.
        C = f"{L}.compute_layer"
        mappings += [
            # Attention compute layer (shortcut_moe_attn).
            *attention(C, HL),
            *controllers(f"{C}.residual_connection_self_attn", f"{HL}.mixer_residual_connection"),
            # GDP compute layer (shortcut_moe_gdp).
            *gdp(f"{C}.mixer", HL, f"{C}.residual_connection", f"{HL}.mixer_residual_connection"),
            # Shortcut routing read and MoE.
            _ControllerMapping(f"{L}.shortcut_residual_read.read_map.logit", f"{HL}.shortcut_read.logit", num_streams),
            *shortcut_moe(L, HL),
            *controllers(f"{L}.moe_layer.residual_connection_mlp", f"{HL}.moe_residual_connection"),
        ]

        # Standalone GDP (linear_attention) and dense MLP layers.
        mappings += [
            *gdp(f"{L}.mixer", HL, f"{L}.residual_connection", f"{HL}.residual_connection"),
            AutoMapping(f"{L}.mlp.linear_fc1.layer_norm_weight", f"{HL}.input_layernorm.weight"),
            AutoMapping(f"{L}.mlp.linear_fc1.weight", f"{HL}.mlp.up_proj.weight"),
            AutoMapping(f"{L}.mlp.linear_fc2.weight", f"{HL}.mlp.down_proj.weight"),
            *controllers(f"{L}.residual_connection_mlp", f"{HL}.residual_connection"),
        ]

        # Multi-token prediction.
        if getattr(text, "num_nextn_predict_layers", 0):
            M = "language_model.mtp.layers.*"
            HMTP = "model.language_model.mtp.layers.*"
            inner = f"{M}.mtp_model_layer.layers.0"
            mappings += [
                ReplicatedMapping(f"{M}.enorm.weight", f"{HMTP}.enorm.weight"),
                ReplicatedMapping(f"{M}.hnorm.weight", f"{HMTP}.hnorm.weight"),
                AutoMapping(f"{M}.eh_proj.weight", f"{HMTP}.eh_proj.weight"),
                ReplicatedMapping(f"{M}.final_layernorm.weight", f"{HMTP}.final_layernorm.weight"),
                *attention(f"{inner}.compute_layer", f"{HMTP}.block"),
                *shortcut_moe(inner, f"{HMTP}.block"),
            ]
        else:
            mappings.append(
                _MegatronOnlyStateMapping("language_model.mtp.**", "multi-token prediction (HF config has none)")
            )

        # Vision tower (Pixtral-style ViT), its patch merger, and the projector.
        V = "vision_model"
        VL = f"{V}.decoder.layers.*"
        HV = "model.vision_tower"
        HVL = f"{HV}.transformer.layers.*"
        HP = "model.multi_modal_projector"
        mappings += [
            ReplicatedMapping(f"{V}.patch_embed.proj.weight", f"{HV}.patch_conv.weight"),
            ReplicatedMapping(f"{V}.ln_pre.weight", f"{HV}.ln_pre.weight"),
            AutoMapping(f"{VL}.self_attention.linear_qkv.layer_norm_weight", f"{HVL}.attention_norm.weight"),
            _PixtralQKVMapping(
                f"{VL}.self_attention.linear_qkv.weight",
                q=f"{HVL}.attention.q_proj.weight",
                k=f"{HVL}.attention.k_proj.weight",
                v=f"{HVL}.attention.v_proj.weight",
                head_dim=vision.head_dim,
            ),
            RowParallelMapping(f"{VL}.self_attention.linear_proj.weight", f"{HVL}.attention.o_proj.weight"),
            AutoMapping(f"{VL}.mlp.linear_fc1.layer_norm_weight", f"{HVL}.ffn_norm.weight"),
            GatedMLPMapping(
                f"{VL}.mlp.linear_fc1.weight",
                gate=f"{HVL}.feed_forward.gate_proj.weight",
                up=f"{HVL}.feed_forward.up_proj.weight",
            ),
            RowParallelMapping(f"{VL}.mlp.linear_fc2.weight", f"{HVL}.feed_forward.down_proj.weight"),
            # Megatron keeps the 2x2 merger inside the ViT; HF places it in the projector.
            ReplicatedMapping(f"{V}.merger.pre_norm.weight", f"{HP}.norm.weight"),
            ReplicatedMapping(f"{V}.merger.linear_fc1.weight", f"{HP}.patch_merger.merging_layer.weight"),
            ColumnParallelMapping("vision_projection.encoder.linear_fc1.weight", f"{HP}.linear_1.weight"),
            RowParallelMapping("vision_projection.encoder.linear_fc2.weight", f"{HP}.linear_2.weight"),
        ]
        return MegatronMappingRegistry(*mappings)


def _validate_hf_config(hf_config) -> None:
    """Reject HF settings the Megatron model cannot represent, instead of converting them wrongly."""
    text = hf_config.text_config
    vision = hf_config.vision_config
    expected = {
        "text_config.mlp_hidden_act": (text.mlp_hidden_act, "relu2"),
        "text_config.gdp_conv_kernel": (text.gdp_conv_kernel, 4),
        "text_config.gdp_norm_eps": (text.gdp_norm_eps, 1e-5),
        "text_config.residual_in_fp32": (text.residual_in_fp32, False),
        "vision_config.hidden_act": (vision.hidden_act, "silu"),
        "vision_config.rope_parameters.rope_theta": (vision.rope_parameters["rope_theta"], 10000.0),
        "projector_hidden_act": (hf_config.projector_hidden_act, "gelu_fast"),
        "multimodal_projector_bias": (hf_config.multimodal_projector_bias, False),
        # HF Pixtral hard-codes eps=1e-5 in every tower norm and uses projector_norm_eps only for
        # the merger norm; Megatron's ViT uses one epsilon for both, so they must agree.
        "projector_norm_eps": (hf_config.projector_norm_eps, 1e-5),
        "vision_feature_layer": (hf_config.vision_feature_layer, -1),
    }
    mismatched = {name: actual for name, (actual, required) in expected.items() if actual != required}
    if mismatched:
        raise ValueError(
            f"Unsupported Nemotron4 HF config values (bridge supports only the trained values): {mismatched}"
        )
    if vision.num_attention_heads * vision.head_dim != vision.hidden_size:
        raise ValueError("Nemotron4 vision_config requires num_attention_heads * head_dim == hidden_size.")
