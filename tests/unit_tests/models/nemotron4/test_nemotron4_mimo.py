"""Nemotron4 MIMO construction, routing, and supported-layout contracts."""

import importlib.util
from types import SimpleNamespace

import pytest
import torch
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.vit_model import ViTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import Float16Module, mark_keep_in_fp32
from megatron.core.transformer.transformer_config import TransformerConfig


if importlib.util.find_spec("megatron.core.models.hybrid.shortcut_block") is None:
    pytest.skip("Nemotron4 requires its MCore fork", allow_module_level=True)

from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules

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
from megatron.bridge.models.nemotron4.modeling_nemotron4_vl import (
    Nemotron4GatedDeltaProductMixer,
    Nemotron4MIMOLanguageModel,
    Nemotron4MIMOProjector,
    Nemotron4MIMOVisionEncoder,
    _build_vision_packed_seq_params,
)
from megatron.bridge.models.nemotron4.nemotron4_bridge import nemotron4_mimo_conversion_spec


pytestmark = pytest.mark.unit


def test_family_import_registers_mimo_conversion():
    assert get_mimo_conversion_spec(Nemotron4Bridge) is nemotron4_mimo_conversion_spec


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_dtype_cast_preserves_fp32_control_state(dtype):
    model = Nemotron4MIMOLanguageModel.__new__(Nemotron4MIMOLanguageModel)
    torch.nn.Module.__init__(model)
    model.block = torch.nn.Module()
    model.block.gdp = torch.nn.Module()
    original = torch.tensor([0.1234567, -0.7654321], dtype=torch.float32)
    model.block.weight = torch.nn.Parameter(original.clone())
    model.block.A_log = torch.nn.Parameter(original.clone())
    model.block.logit = torch.nn.Parameter(original.clone())
    model.block.logit.is_residual_map_parameter = True
    model.block.retention_logit = torch.nn.Parameter(original.clone())
    model.block.retention_logit.is_wide_residual_retention_parameter = True
    model.block.register_buffer("expert_bias", original.clone())
    model.block.register_buffer("qb_bin_bounds", original.clone())
    model.block.register_buffer("ordinary_buffer", original.clone())
    model.block.gdp.dt_bias = mark_keep_in_fp32(torch.nn.Parameter(original.clone()))
    model.block.gdp.other_parameter = mark_keep_in_fp32(torch.nn.Parameter(original.clone()))
    model.block.gdp.register_buffer("other_buffer", mark_keep_in_fp32(original.clone()))
    for parameter in model.parameters():
        parameter.grad = original.clone()
    model.block.gdp.dt_bias.tensor_model_parallel = True
    parameter_ids = {name: id(parameter) for name, parameter in model.named_parameters()}
    config = TransformerConfig(
        num_layers=1,
        hidden_size=4,
        num_attention_heads=1,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        params_dtype=dtype,
    )
    model.to(dtype=dtype)
    # The real MIMO provider casts before wrapping. A buffer's marker must
    # survive the first cast so this second conversion still protects it.
    for wrap in (False, True):
        if wrap:
            model = Float16Module(config, model).module
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
            expected = original.to(dtype) if name in ("block.weight", "block.ordinary_buffer") else original
            torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
        for name, parameter in model.named_parameters():
            assert id(parameter) == parameter_ids[name]
            expected = original.to(dtype) if name == "block.weight" else original
            torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)
        assert model.block.gdp.dt_bias.keep_in_fp32
        assert model.block.gdp.other_parameter.keep_in_fp32
        assert model.block.gdp.other_buffer.keep_in_fp32
        assert model.block.gdp.dt_bias.tensor_model_parallel
        assert model.block.logit.is_residual_map_parameter
        assert model.block.retention_logit.is_wide_residual_retention_parameter


def _parallelism(**image_kwargs):
    return MegatronMIMOParallelismConfig(
        module_parallelisms={
            "language": ModuleParallelismConfig(tensor_model_parallel_size=1),
            "images": ModuleParallelismConfig(tensor_model_parallel_size=1, **image_kwargs),
        }
    )


def _provider(mtp_depths=0, **overrides):
    return Nemotron4VLModelProvider(
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
        **overrides,
    )


def _source(monkeypatch, mtp_depths):
    provider = _provider(mtp_depths)
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


def test_export_copies_processor_config_from_source():
    """OpenCV-free environments cannot instantiate the Nemotron4 processor; copy its config."""
    assert "processor_config.json" in Nemotron4Bridge.ADDITIONAL_FILE_PATTERNS


def test_training_specs_use_the_mimo_components_bf16_encoder_and_vision_recompute():
    provider = _provider(
        fp8="e4m3",
        fp8_recipe="mxfp8",
        fp8_param=True,
        recompute_vision=True,
        vision_recompute_granularity="full",
        vision_recompute_method="block",
        vision_recompute_num_layers=1,
    )
    images = provider.build_mimo_modality_submodules_spec()["images"]
    encoder = images.submodules["encoders"]["pixtral"]
    (projector,) = images.submodules["input_projections"]
    assert (images.module, images.params) == (VisionModalitySubmodules, {})
    assert encoder.module is Nemotron4MIMOVisionEncoder
    assert projector.module is Nemotron4MIMOProjector
    vision = encoder.params["transformer_config"]
    for config in (vision, projector.params["config"]):
        assert (config.fp8, config.fp8_param, config.quant_recipe) == (None, False, None)
    assert (vision.recompute_granularity, vision.recompute_method, vision.recompute_num_layers) == ("full", "block", 1)


def test_language_spec_uses_the_saved_spec_with_the_stride_safe_gdp_mixer():
    """NM4 run configs store hybrid_stack_spec: null; the checkpoint's saved spec names the GDP stack."""
    provider = _provider(hybrid_stack_spec=None)
    provider._saved_args = SimpleNamespace(
        spec=("megatron.core.models.hybrid.hybrid_layer_specs", "gated_delta_product_stack_spec")
    )
    language = provider.build_language_model_spec()
    mixer = language.params["hybrid_stack_spec"].submodules.mamba_layer.submodules.mixer
    assert mixer.module is Nemotron4GatedDeltaProductMixer


def test_vision_encoder_takes_the_mimo_payload(monkeypatch):
    received = {}

    def vit_forward(self, pixel_values, attention_mask=None, imgs_sizes=None, packed_seq_params=None):
        received.update(pixel_values=pixel_values, imgs_sizes=imgs_sizes, packed_seq_params=packed_seq_params)
        return pixel_values

    monkeypatch.setattr(ViTModel, "forward", vit_forward)
    encoder = object.__new__(Nemotron4MIMOVisionEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.recompute_budget_gib = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.zeros(1, 8, 588, device=device)
    imgs_sizes = torch.tensor([[28, 28], [28, 28]], dtype=torch.int32, device=device)
    packed_seq_params = PackedSeqParams(qkv_format="thd")
    ones = torch.ones(2, dtype=torch.int32, device=device)

    encoder(
        x=x,
        imgs_sizes=imgs_sizes,
        num_frames=ones,
        expected_output_counts=ones,
        packed_seq_params=packed_seq_params,
    )

    assert received["pixel_values"] is x
    assert received["imgs_sizes"].device.type == "cpu"
    assert torch.equal(received["imgs_sizes"], imgs_sizes.cpu())
    assert received["packed_seq_params"] is packed_seq_params


@pytest.mark.parametrize(
    "component,parent",
    [
        (Nemotron4MIMOLanguageModel, HybridModel),
        (Nemotron4MIMOProjector, MultimodalProjector),
        (Nemotron4MIMOVisionEncoder, ViTModel),
    ],
)
def test_components_omit_te_extra_state(monkeypatch, component, parent):
    state = {"linear.weight": object(), "linear._extra_state": object()}
    monkeypatch.setattr(parent, "sharded_state_dict", lambda self, *args, **kwargs: dict(state))

    assert object.__new__(component).sharded_state_dict(prefix="") == {"linear.weight": state["linear.weight"]}


_BUDGETED_RECOMPUTE = dict(
    recompute_vision=True,
    vision_recompute_granularity="full",
    vision_recompute_method="block",
    vision_recompute_num_layers=1,
    vision_bias_activation_fusion=True,
    vision_apply_rope_fusion=True,
)


def test_vision_recompute_budget_reaches_only_the_mimo_encoder():
    def encoder_params(**overrides):
        images = _provider(**overrides).build_mimo_modality_submodules_spec()["images"]
        return images.submodules["encoders"]["pixtral"].params

    assert encoder_params(**_BUDGETED_RECOMPUTE, vision_recompute_budget_gib=105.0)["recompute_budget_gib"] == 105.0
    with pytest.raises(ValueError, match="only by the MIMO vision encoder"):
        _provider(**_BUDGETED_RECOMPUTE, vision_recompute_budget_gib=105.0).provide()


@pytest.mark.parametrize(
    "overrides",
    [
        dict(recompute_vision=False),
        dict(vision_recompute_granularity="selective"),
        dict(vision_recompute_method="uniform"),
        dict(vision_bias_activation_fusion=False),
        dict(vision_apply_rope_fusion=False),
        dict(vision_recompute_budget_gib=0.0),
        dict(vision_recompute_budget_gib=float("inf")),
    ],
)
def test_vision_recompute_budget_rejects_unsupported_settings(overrides):
    provider = _provider(**{**_BUDGETED_RECOMPUTE, "vision_recompute_budget_gib": 105.0, **overrides})
    with pytest.raises(ValueError, match="vision_recompute_budget_gib requires"):
        provider.build_mimo_modality_submodules_spec()


def _budgeted_encoder(budget_gib, hidden=2048, ffn=6144, layers=30):
    encoder = object.__new__(Nemotron4MIMOVisionEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.config = SimpleNamespace(
        hidden_size=hidden,
        ffn_hidden_size=ffn,
        bf16=True,
        fp8=None,
        fp4=None,
        quant_recipe=None,
    )
    encoder.decoder = SimpleNamespace(layers=[SimpleNamespace(recompute_mlp=False) for _ in range(layers)])
    encoder.recompute_budget_gib = budget_gib
    return encoder


@pytest.mark.parametrize(
    "precision",
    [
        dict(bf16=False),
        dict(fp8="e4m3", fp8_recipe="mxfp8"),
        dict(fp8="hybrid", fp8_recipe="tensorwise"),
        dict(fp4="e2m1"),
        dict(quant_recipe={}),
    ],
    ids=["non-bf16", "mxfp8", "tensorwise-fp8", "fp4", "quant-recipe"],
)
def test_recompute_budget_rejects_unvalidated_encoder_precision(precision):
    encoder = _budgeted_encoder(105.0)
    encoder.config = SimpleNamespace(**(vars(encoder.config) | precision))

    with pytest.raises(ValueError, match="requires a BF16 encoder"):
        encoder.recompute_plan(83_984)


def test_recompute_budget_allows_mxfp8_decoder_with_bf16_encoder():
    provider = _provider(
        fp8="e4m3",
        fp8_recipe="mxfp8",
        fp8_param=True,
        **_BUDGETED_RECOMPUTE,
        vision_recompute_budget_gib=105.0,
    )
    spec = provider.build_mimo_modality_submodules_spec()["images"].submodules["encoders"]["pixtral"]
    encoder = _budgeted_encoder(spec.params["recompute_budget_gib"], layers=provider.vision_num_layers)
    encoder.config = spec.params["transformer_config"]

    assert encoder.recompute_plan(1) == (0, 0)
    assert provider.fp8 == "e4m3"
    assert provider.fp8_recipe == "mxfp8"


@pytest.mark.parametrize(
    "patches,plan",
    # 84k-99k: plans logged by the 18n run at 105 GiB (job 1293415).
    [(1, (0, 0)), (83_984, (0, 20)), (93_184, (0, 23)), (98_956, (0, 25)), (199_000, (14, 16)), (10**7, (30, 0))],
)
def test_recompute_plan(patches, plan):
    assert _budgeted_encoder(105.0).recompute_plan(patches) == plan


def test_fit_sets_the_layer_count_and_mlp_switches():
    encoder = _budgeted_encoder(105.0)
    n_full, n_mlp = encoder.recompute_plan(199_000)
    encoder._fit_recompute_to_budget(199_000)
    assert encoder.config.recompute_num_layers == n_full
    flags = [layer.recompute_mlp for layer in encoder.decoder.layers]
    assert flags == [False] * n_full + [True] * n_mlp + [False] * (30 - n_full - n_mlp)


@pytest.mark.gpu
def test_budgeted_recompute_keeps_outputs_and_gradients_bitwise(tmp_path, monkeypatch):
    """Training order F0 F1 B0 F2 B1 F3 B2 B3 with a different plan per micro-batch. Each forward must run its own
    plan, and mb0's fully recomputed layers 0-1 replay in B0 with mb1's MLP-only switch. Outputs and gradients must
    equal those without recompute."""
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.enums import AttnBackend

    # Deterministic cuDNN attention, so that two runs compare bitwise (production allows a non-deterministic
    # backward). setenv restores these process-wide values after the test.
    for name, value in (
        ("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0"),
        ("NVTE_FLASH_ATTN", "0"),
        ("NVTE_FUSED_ATTN", "1"),
        ("NVTE_UNFUSED_ATTN", "0"),
    ):
        monkeypatch.setenv(name, value)
    torch.distributed.init_process_group("nccl", init_method=(tmp_path / "dist-init").as_uri(), rank=0, world_size=1)
    try:
        parallel_state.initialize_model_parallel()
        tensor_parallel.model_parallel_cuda_manual_seed(1234)
        provider = _provider(
            bf16=True,
            params_dtype=torch.bfloat16,
            **_BUDGETED_RECOMPUTE,
            vision_recompute_budget_gib=1.0,
        )
        provider.attention_backend = AttnBackend.fused
        provider.gradient_accumulation_fusion = False  # no DDP main_grad buffers here
        provider.vision_num_layers = 4
        provider.vision_hidden_size = 128
        provider.vision_ffn_hidden_size = 256
        provider.vision_kv_channels = 32
        spec = provider.build_mimo_modality_submodules_spec()["images"].submodules["encoders"]["pixtral"]
        encoder = spec.module(**spec.params, pg_collection=ProcessGroupCollection.use_mpu_process_groups())
        encoder = encoder.cuda().bfloat16().train()
        layers = encoder.decoder.layers
        assert encoder.decoder.config is encoder.config
        seen, mlp_seen, plan_calls = [], [], []
        for index, layer in enumerate(layers):
            layer.register_forward_pre_hook(
                lambda module, args, index=index: seen.append(
                    (index, torch.is_grad_enabled(), encoder.config.recompute_num_layers, module.recompute_mlp)
                )
            )
            # The MLP runs without grad inside a checkpoint forward and with grad when it is replayed.
            layer.mlp.register_forward_pre_hook(
                lambda module, args, index=index: mlp_seen.append((index, torch.is_grad_enabled()))
            )

        generator = torch.Generator(device="cuda").manual_seed(0)
        batches = []
        for sizes in ([[56, 84], [28, 112]], [[112, 112], [28, 28]], [[84, 56]], [[56, 56], [28, 56], [84, 28]]):
            imgs_sizes = torch.tensor(sizes, dtype=torch.int32, device="cuda")
            patches = int((imgs_sizes // 14).prod(dim=1).sum())
            x = torch.randn(1, patches, 588, generator=generator, device="cuda").bfloat16()
            batches.append((x, imgs_sizes, _build_vision_packed_seq_params(imgs_sizes, 14, x.device)))
        # (n_full, n_mlp) per micro-batch: all full, MLP-only + none, full + MLP-only, none.
        wanted = [(4, 0), (0, 2), (2, 2), (0, 0)]

        def run(budgeted):
            encoder.config.recompute_granularity = "full" if budgeted else None
            for layer in layers:
                layer.recompute_mlp = False
            encoder.zero_grad(set_to_none=True)
            outputs, steps, grad_rng = {}, {}, torch.Generator(device="cuda").manual_seed(1)
            mlp_steps = {}
            for kind, mb in (("F", 0), ("F", 1), ("B", 0), ("F", 2), ("B", 1), ("F", 3), ("B", 2), ("B", 3)):
                seen.clear()
                mlp_seen.clear()
                if kind == "F":
                    encoder.recompute_budget_gib = 1.0 if budgeted else None

                    def plan_of(patches, plan=wanted[mb]):
                        plan_calls.append(patches)
                        return plan

                    monkeypatch.setattr(encoder, "recompute_plan", plan_of)
                    x, imgs_sizes, packed = batches[mb]
                    outputs[mb] = encoder(x=x, imgs_sizes=imgs_sizes, packed_seq_params=packed)
                else:
                    out = outputs[mb]
                    out.backward(torch.randn(out.shape, generator=grad_rng, device="cuda").to(out.dtype))
                steps[f"{kind}{mb}"] = sorted(seen)
                mlp_steps[f"{kind}{mb}"] = sorted(mlp_seen)
            grads = {name: param.grad.clone() for name, param in encoder.named_parameters()}
            return [outputs[mb].detach() for mb in range(4)], grads, steps, mlp_steps

        reference_outputs, reference_grads, _, _ = run(budgeted=False)
        outputs, grads, steps, mlp_steps = run(budgeted=True)
        # The planner sees the patch count of each micro-batch (only the budgeted run plans).
        assert plan_calls == [batch[0].shape[1] for batch in batches]
        # (layer, grad enabled, config.recompute_num_layers, layer.recompute_mlp) when each layer runs.
        expected = {
            f"F{mb}": [(i, i >= n_full, n_full, n_full <= i < n_full + n_mlp) for i in range(4)]
            for mb, (n_full, n_mlp) in enumerate(wanted)
        }
        expected |= {
            "B0": [(i, True, 0, i < 2) for i in range(4)],
            "B1": [],
            "B2": [(i, True, 0, False) for i in range(2)],
            "B3": [],
        }
        assert steps == expected
        # An MLP runs without grad when its layer or the MLP itself is checkpointed, and again with grad in the
        # backward. B0: layers 0-1 replay with mb1's MLP-only switch, so their MLP checkpoints again.
        mlp_expected = {
            f"F{mb}": [(i, i >= n_full + n_mlp) for i in range(4)] for mb, (n_full, n_mlp) in enumerate(wanted)
        }
        mlp_expected |= {
            "B0": [(0, False), (0, True), (1, False), (1, True), (2, True), (3, True)],
            "B1": [(0, True), (1, True)],
            "B2": [(i, True) for i in range(4)],
            "B3": [],
        }
        assert mlp_steps == mlp_expected
        for reference, output in zip(reference_outputs, outputs):
            assert torch.equal(reference, output)
        assert reference_grads.keys() == grads.keys()
        for name in grads:
            assert torch.equal(reference_grads[name], grads[name]), name
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()
