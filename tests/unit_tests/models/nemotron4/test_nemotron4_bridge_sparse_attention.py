"""Nemotron4 bridge: SSA provider settings and indexer mappings."""

import dataclasses
import importlib.util
from types import SimpleNamespace

import pytest


if importlib.util.find_spec("megatron.core.models.hybrid.shortcut_block") is None:
    pytest.skip("Nemotron4 requires its MCore fork", allow_module_level=True)

from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.nemotron4 import Nemotron4Bridge
from megatron.bridge.models.nemotron4.nemotron4_bridge import ssa_provider_kwargs
from megatron.bridge.models.nemotron4.nemotron4_provider import (
    NEMOTRON4_SSA_REQUIRED_MCORE_FIELDS,
    nemotron4_ssa_stack_spec,
    nemotron4_stack_spec,
)


pytestmark = pytest.mark.unit

_HAS_SSA = importlib.util.find_spec(
    "megatron.core.transformer.experimental_attention_variant.simplified_sparse_attention_layer_specs"
) is not None and all(
    name in {f.name for f in dataclasses.fields(TransformerConfig)} for name in NEMOTRON4_SSA_REQUIRED_MCORE_FIELDS
)
requires_ssa = pytest.mark.skipif(not _HAS_SSA, reason="installed MCore has no SSA")

# Every SSA-related provider field the bridge sets; nothing else may change between dense and sparse.
_SSA_FIELDS = {
    "hybrid_stack_spec",
    "experimental_attention_variant",
    "dsa_indexer_topk",
    "dsa_indexer_loss_coeff",
    "cp_comm_type",
    "apply_rope_fusion",
}


def _hf_config(index_topk=None, mtp_depths=2, **text_overrides):
    text = dict(
        layer_types=["linear_attention", "mlp", "shortcut_moe_attn", "shortcut_moe_gdp", "shortcut_moe_attn"],
        num_nextn_predict_layers=mtp_depths,
        mtp_use_repeated_layer=True,
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=16,
        num_key_value_heads=1,
        head_dim=256,
        rms_norm_eps=1e-5,
        initializer_range=0.0162,
        activation_clamp_value=16.0,
        vocab_size=512,
        max_position_embeddings=4096,
        gdp_num_heads=4,
        gdp_num_groups=4,
        gdp_head_dim=32,
        gdp_key_head_dim=32,
        gdp_num_householder=3,
        gdp_conv_kernel=4,
        gdp_norm_eps=1e-5,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        moe_shared_expert_intermediate_size=128,
        moe_latent_size=64,
        moe_use_norm_before_up_proj=True,
        routed_scaling_factor=3.16,
        moe_shortcut_post_norm=False,
        wide_residual_num_streams=3,
        wide_residual_streamwise_sigmoid_init_scale=0.01,
        wide_residual_learned_retention=True,
        wide_residual_retention_init=0.999,
        wide_residual_retention_max_forget=0.10,
        mlp_hidden_act="relu2",
        residual_in_fp32=False,
        index_topk=index_topk,
        index_head_dim=128,
    )
    text.update(text_overrides)
    vision = SimpleNamespace(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        head_dim=32,
        patch_size=14,
        image_size=56,
        hidden_act="silu",
        rope_parameters={"rope_theta": 10000.0},
    )
    return SimpleNamespace(
        text_config=SimpleNamespace(**text),
        vision_config=vision,
        tie_word_embeddings=False,
        image_token_id=18,
        spatial_merge_size=2,
        projector_hidden_act="gelu_fast",
        multimodal_projector_bias=False,
        projector_norm_eps=1e-5,
        vision_feature_layer=-1,
    )


def _provider(hf_config):
    return Nemotron4Bridge().provider_bridge(SimpleNamespace(config=hf_config))


def _hf_names(hf_config, megatron_names):
    bridge = Nemotron4Bridge()
    bridge.hf_config = hf_config
    registry = bridge.mapping_registry()
    resolved = {}
    for name in megatron_names:
        mapping = registry.megatron_to_hf_lookup(name)
        resolved[name] = None if mapping is None else mapping.hf_param
    return resolved


_INDEXER_NAMES = {
    "language_model.decoder.layers.2.compute_layer.self_attention.core_attention.indexer.linear_q.weight": (
        "model.language_model.layers.2.self_attn.indexer.q_proj.weight"
    ),
    "language_model.decoder.layers.4.compute_layer.self_attention.core_attention.indexer.linear_k.weight": (
        "model.language_model.layers.4.self_attn.indexer.k_proj.weight"
    ),
    "language_model.mtp.layers.0.mtp_model_layer.layers.0.compute_layer.self_attention.core_attention.indexer.linear_q.weight": (
        "model.language_model.mtp.layers.0.block.self_attn.indexer.q_proj.weight"
    ),
    "language_model.mtp.layers.0.mtp_model_layer.layers.0.compute_layer.self_attention.core_attention.indexer.linear_k.weight": (
        "model.language_model.mtp.layers.0.block.self_attn.indexer.k_proj.weight"
    ),
}


def test_dense_config_has_no_sparse_attention():
    assert ssa_provider_kwargs(_hf_config().text_config) == {}
    # Configs written before sparse attention existed have no index_topk at all.
    legacy_text = SimpleNamespace(**{k: v for k, v in vars(_hf_config().text_config).items() if k != "index_topk"})
    assert ssa_provider_kwargs(legacy_text) == {}

    provider = _provider(_hf_config())
    assert provider.hybrid_stack_spec is nemotron4_stack_spec
    assert getattr(provider, "experimental_attention_variant", None) is None
    assert set(_hf_names(_hf_config(), _INDEXER_NAMES).values()) == {None}


@requires_ssa
def test_sparse_config_sets_only_ssa_fields():
    dense = _provider(_hf_config())
    sparse = _provider(_hf_config(index_topk=512))
    differing = {
        field.name
        for field in dataclasses.fields(dense)
        if getattr(dense, field.name) != getattr(sparse, field.name)
        and field.name not in {"wide_residual"}  # fresh WideResidualConfig per provider, equal by value
    }
    # An SSA field may coincide with the dense default (e.g. apply_rope_fusion); nothing else may differ.
    assert differing <= _SSA_FIELDS
    assert {"hybrid_stack_spec", "experimental_attention_variant", "dsa_indexer_topk"} <= differing
    assert vars(dense.wide_residual) == vars(sparse.wide_residual)
    assert sparse.hybrid_stack_spec is nemotron4_ssa_stack_spec
    expected = {
        "experimental_attention_variant": "ssa",
        "dsa_indexer_topk": 512,
        "dsa_indexer_loss_coeff": 0.1,
        "cp_comm_type": "all_gather",
        "apply_rope_fusion": False,
    }
    assert {name: getattr(sparse, name) for name in expected} == expected


@requires_ssa
def test_ssa_stack_spec_swaps_only_attention():
    dense_spec = nemotron4_stack_spec()
    sparse_spec = nemotron4_ssa_stack_spec()
    from megatron.core.transformer.experimental_attention_variant.simplified_sparse_attention import (
        SimplifiedSparseAttention,
    )

    assert sparse_spec.submodules.attention_layer.submodules.self_attention.module is SimplifiedSparseAttention
    assert sparse_spec.submodules.mamba_layer.submodules.mixer.module is (
        dense_spec.submodules.mamba_layer.submodules.mixer.module
    )


def test_missing_ssa_spec_is_rejected(monkeypatch):
    from megatron.bridge.models.nemotron4 import nemotron4_provider

    monkeypatch.setattr(nemotron4_provider.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="simplified_sparse_attention_layer_specs"):
        nemotron4_provider.require_nemotron4_ssa_mcore()


def test_missing_ssa_config_field_is_rejected(monkeypatch):
    from megatron.bridge.models.nemotron4 import nemotron4_provider

    fields = [f for f in dataclasses.fields(TransformerConfig) if f.name != "dsa_indexer_loss_coeff"]
    monkeypatch.setattr(nemotron4_provider.dataclasses, "fields", lambda cls: fields)
    with pytest.raises(RuntimeError, match="missing TransformerConfig fields: .*dsa_indexer_loss_coeff"):
        nemotron4_provider.require_nemotron4_ssa_mcore()


@requires_ssa
def test_vision_and_projector_configs_stay_dense():
    from megatron.bridge.models.nemotron4.nemotron4_provider import _make_dense_non_hybrid

    config = SimpleNamespace(
        recompute_modules=None,
        params_dtype=None,
        experimental_attention_variant="ssa",
        ssa_train_indexer_only=True,
    )
    _make_dense_non_hybrid(config)
    assert config.experimental_attention_variant is None
    assert config.ssa_train_indexer_only is False


@requires_ssa
def test_indexer_mappings_backbone_and_mtp():
    assert _hf_names(_hf_config(index_topk=512), _INDEXER_NAMES) == _INDEXER_NAMES
    # Without MTP the MTP indexer falls under the Megatron-only MTP mapping, not an HF tensor.
    no_mtp = _hf_names(_hf_config(index_topk=512, mtp_depths=0), _INDEXER_NAMES)
    for megatron_name, hf_name in no_mtp.items():
        if ".mtp." in megatron_name:
            assert hf_name.startswith("megatron_only::")
        else:
            assert hf_name == _INDEXER_NAMES[megatron_name]


@requires_ssa
@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"index_topk": 256}, "index_topk=256"),
        ({"index_topk": 512, "index_head_dim": 64}, "index_head_dim=64"),
        ({"index_topk": 512, "num_attention_heads": 8}, "num_attention_heads=8"),
        ({"index_topk": 512, "num_key_value_heads": 2, "num_attention_heads": 32}, "num_key_value_heads=2"),
        ({"index_topk": 512, "head_dim": 128}, "head_dim=128"),
    ],
)
def test_unsupported_sparse_attention_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        _provider(_hf_config(**overrides))


def test_unsupported_geometry_is_fine_when_dense():
    provider = _provider(_hf_config(index_topk=None, num_attention_heads=8, head_dim=128, index_head_dim=64))
    assert provider.num_attention_heads == 8


@requires_ssa
@pytest.mark.parametrize("index_topk", [512, 1024, 2048])
@pytest.mark.parametrize("num_attention_heads", [16, 32, 96])
def test_sparse_provider_passes_mcore_ssa_validation(index_topk, num_attention_heads):
    """MCore validates the CuTe SSA contract in finalize(); the bridge's settings must satisfy it."""
    import torch

    provider = _provider(_hf_config(index_topk=index_topk, num_attention_heads=num_attention_heads))
    provider.params_dtype = torch.bfloat16
    provider.bf16 = True
    provider.finalize()
    assert provider.experimental_attention_variant == "ssa"
    assert provider.dsa_indexer_topk == index_topk


@pytest.mark.parametrize("index_topk", [None, 512])
def test_tensor_parallel_is_rejected_at_finalize(index_topk):
    """TP>1 must fail in finalize() with a Nemotron4 message, not later inside model construction."""
    import torch

    if index_topk is not None and not _HAS_SSA:
        pytest.skip("installed MCore has no SSA")
    provider = _provider(_hf_config(index_topk=index_topk))
    provider.params_dtype = torch.bfloat16
    provider.bf16 = True
    provider.tensor_model_parallel_size = 2
    with pytest.raises(ValueError, match="Nemotron4 requires tensor_model_parallel_size=1"):
        provider.finalize()
