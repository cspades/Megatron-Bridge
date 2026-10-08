"""Nemotron4 MIMO construction, routing, and supported-layout contracts."""

import importlib.util
from types import SimpleNamespace

import pytest
import torch


if importlib.util.find_spec("megatron.core.models.hybrid.shortcut_block") is None:
    pytest.skip("Nemotron4 requires its MCore fork", allow_module_level=True)

from megatron.bridge.models.megatron_mimo.conversion import (
    build_route_local_registry,
    get_mimo_conversion_spec,
    validate_route_table,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)
from megatron.bridge.models.nemotron4 import Nemotron4Bridge, Nemotron4VLModelProvider
from megatron.bridge.models.nemotron4.modeling_nemotron4_vl import Nemotron4MIMOLanguageModel
from megatron.bridge.models.nemotron4.nemotron4_bridge import nemotron4_mimo_conversion_spec


pytestmark = pytest.mark.unit


def test_family_import_registers_mimo_conversion():
    assert get_mimo_conversion_spec(Nemotron4Bridge) is nemotron4_mimo_conversion_spec


def test_bfloat16_cast_preserves_fp32_control_state():
    model = Nemotron4MIMOLanguageModel.__new__(Nemotron4MIMOLanguageModel)
    torch.nn.Module.__init__(model)
    model.block = torch.nn.Module()
    original = torch.tensor([0.1234567, -0.7654321], dtype=torch.float32)
    model.block.weight = torch.nn.Parameter(original.clone())
    model.block.A_log = torch.nn.Parameter(original.clone())
    model.block.logit = torch.nn.Parameter(original.clone())
    model.block.logit.is_residual_map_parameter = True
    model.block.logit.grad = original.clone()
    model.block.retention_logit = torch.nn.Parameter(original.clone())
    model.block.retention_logit.is_wide_residual_retention_parameter = True
    model.block.register_buffer("expert_bias", original.clone())
    model.block.register_buffer("qb_bin_bounds", original.clone())
    model.bfloat16()
    assert model.block.weight.dtype == torch.bfloat16
    for name in ("A_log", "logit", "retention_logit", "expert_bias", "qb_bin_bounds"):
        torch.testing.assert_close(getattr(model.block, name), original, rtol=0, atol=0)
    assert model.block.logit.is_residual_map_parameter
    torch.testing.assert_close(model.block.logit.grad, original, rtol=0, atol=0)


def _parallelism(**image_kwargs):
    return MegatronMIMOParallelismConfig(
        module_parallelisms={
            "language": ModuleParallelismConfig(tensor_model_parallel_size=1),
            "images": ModuleParallelismConfig(tensor_model_parallel_size=1, **image_kwargs),
        }
    )


def _source(monkeypatch, mtp_depths):
    provider = Nemotron4VLModelProvider(
        hybrid_layer_pattern="M-*EME" + "/*E" * mtp_depths,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        kv_channels=32,
        vocab_size=128,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_ffn_hidden_size=128,
        moe_latent_size=64,
        moe_shared_expert_intermediate_size=128,
        mtp_num_layers=mtp_depths or None,
        mtp_hsm=mtp_depths >= 2,
        mtp_use_repeated_layer=bool(mtp_depths),
        vision_num_layers=1,
        vision_hidden_size=64,
        vision_ffn_hidden_size=128,
        vision_num_attention_heads=4,
        vision_kv_channels=16,
    )
    config = SimpleNamespace(
        torch_dtype=torch.bfloat16,
        text_config=SimpleNamespace(
            wide_residual_num_streams=3,
            wide_residual_learned_retention=True,
            moe_use_norm_before_up_proj=True,
            moe_shortcut_post_norm=False,
            num_nextn_predict_layers=mtp_depths,
        ),
        vision_config=SimpleNamespace(head_dim=16),
    )
    bridge = Nemotron4Bridge()
    bridge.hf_config = config
    monkeypatch.setattr(bridge, "provider_bridge", lambda _: provider)
    return bridge, SimpleNamespace(config=config)


@pytest.mark.parametrize("mtp_depths", [0, 2])
@pytest.mark.parametrize("ep_size", [1, 2])
def test_specs_preserve_mtp_and_cover_all_weights(monkeypatch, mtp_depths, ep_size):
    bridge, hf = _source(monkeypatch, mtp_depths)
    parallelism = _parallelism()
    parallelism.module_parallelisms["language"].expert_model_parallel_size = ep_size
    parallelism.module_parallelisms["language"].data_parallel_size = ep_size
    provider, routes = nemotron4_mimo_conversion_spec(bridge, hf, parallelism)
    validate_route_table(
        routes, parallelism_config=parallelism, modality_submodules_spec=provider.modality_submodules_spec
    )
    assert provider.standard_provider.mtp_num_layers == (mtp_depths or None)
    assert provider.language_model_spec.params["config"].mtp_num_layers == (mtp_depths or None)
    assert provider.language_model_spec.params["config"].params_dtype == torch.bfloat16
    assert provider.special_token_ids == {"images": 18}
    image_spec = provider.modality_submodules_spec["images"]
    vision = image_spec.submodules["encoders"]["pixtral"].params["transformer_config"]
    projector = image_spec.submodules["input_projections"][0].params["config"]
    assert vision.num_layers == 1
    for config in (vision, projector):
        assert config.wide_residual is None
        assert config.num_moe_experts is None
        assert not config.mtp_hsm
        assert not config.is_hybrid_model
        assert config.expert_model_parallel_size == 1
    registry = bridge.mapping_registry()
    assert sum(len(build_route_local_registry(registry, route).mappings) for route in routes) == len(registry.mappings)
    assert routes[-1].parallelism_name == "images"
    assert routes[-1].target_module_path.endswith("input_projections.0")
    # Rebuild from durable provider state, as checkpoint loading does.
    provider.language_model_spec = None
    provider.modality_submodules_spec = {}
    provider._rebuild_derived_specs_from_standard_provider()
    assert provider.language_model_spec.params["config"].mtp_num_layers == (mtp_depths or None)


@pytest.mark.parametrize(
    "field", ["pipeline_model_parallel_size", "context_parallel_size", "expert_model_parallel_size"]
)
def test_rejects_unsupported_image_parallelism(monkeypatch, field):
    bridge, hf = _source(monkeypatch, 0)
    with pytest.raises(ValueError, match=field):
        nemotron4_mimo_conversion_spec(bridge, hf, _parallelism(**{field: 2}))


def test_rejects_language_tp(monkeypatch):
    bridge, hf = _source(monkeypatch, 0)
    parallelism = _parallelism()
    parallelism.module_parallelisms["language"].tensor_model_parallel_size = 2
    with pytest.raises(ValueError, match="language.tensor_model_parallel_size"):
        nemotron4_mimo_conversion_spec(bridge, hf, parallelism)


def test_export_restores_fp32_for_hf_strict_fp32_state(monkeypatch):
    """HF keeps controllers, A_log and the router correction bias in FP32; export must too."""
    bridge, _ = _source(monkeypatch, 2)
    converted = {
        "model.language_model.layers.0.mixer.A_log": torch.ones(4, dtype=torch.bfloat16),
        "model.language_model.layers.0.residual_connection.read_logit": torch.ones(3, dtype=torch.bfloat16),
        "model.language_model.layers.0.shortcut_read.logit": torch.ones(3, dtype=torch.bfloat16),
        "model.language_model.residual_readout.logit": torch.ones(3, dtype=torch.bfloat16),
        "model.language_model.layers.0.moe.gate.e_score_correction_bias": torch.ones(4, dtype=torch.bfloat16),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        "model.vision_tower.patch_conv.weight": torch.ones(2, 2, dtype=torch.bfloat16),
    }
    out = bridge.maybe_modify_converted_hf_weight(SimpleNamespace(), dict(converted), {})
    fp32 = {name for name, tensor in out.items() if tensor.dtype == torch.float32}
    assert fp32 == {
        "model.language_model.layers.0.mixer.A_log",
        "model.language_model.layers.0.residual_connection.read_logit",
        "model.language_model.layers.0.shortcut_read.logit",
        "model.language_model.residual_readout.logit",
        "model.language_model.layers.0.moe.gate.e_score_correction_bias",
    }
    assert out["model.language_model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16
    assert out["model.vision_tower.patch_conv.weight"].dtype == torch.bfloat16
