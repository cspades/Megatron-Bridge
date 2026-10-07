# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Opt-in two-GPU Nemotron4 HF/MIMO checkpoint round trip.

Set NEMOTRON4_HF_MODEL to a local snapshot containing the preview's remote code
and config. Tests create tiny random HF checkpoints; they do not download weights.
Requires the Nemotron4 MCore fork and Transformers >= 5.17.
"""

import gc
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoConfig, AutoModelForImageTextToText, PreTrainedTokenizerFast

from megatron.bridge.models.megatron_mimo import (
    MegatronMIMOBridge,
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)
from tests.functional_tests.utils import broadcast_path, initialize_distributed


def _create_tiny_checkpoint(source: str, output: Path, mtp_depths: int) -> None:
    config = AutoConfig.from_pretrained(source, trust_remote_code=True)
    text_overrides = {
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "vocab_size": 128,
        "max_position_embeddings": 128,
        "gdp_num_heads": 4,
        "gdp_num_groups": 4,
        "gdp_head_dim": 32,
        "gdp_key_head_dim": 32,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 128,
        "moe_latent_size": 64,
        "moe_shared_expert_intermediate_size": 128,
        "num_nextn_predict_layers": mtp_depths,
        "mtp_use_repeated_layer": bool(mtp_depths),
        "layer_types": ["linear_attention", "mlp", "shortcut_moe_attn", "shortcut_moe_gdp"],
    }
    vision_overrides = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "head_dim": 16,
        "num_hidden_layers": 1,
        "image_size": 28,
        "patch_size": 14,
    }
    for target, overrides in ((config.text_config, text_overrides), (config.vision_config, vision_overrides)):
        for key, value in overrides.items():
            if not hasattr(target, key):
                raise ValueError(f"HF config has no field {key}")
            setattr(target, key, value)
    torch.manual_seed(42)
    model = AutoModelForImageTextToText.from_config(config, trust_remote_code=True, dtype=torch.float32)
    # Match the real preview's mixed BF16/FP32 state, including nonzero routing
    # biases so a lossy BF16 cast cannot pass unnoticed.
    fp32_names = set(model._keep_in_fp32_modules_strict)
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if name.rsplit(".", 1)[-1] in fp32_names:
            tensor.data = tensor.data.float()
            if name.endswith("e_score_correction_bias"):
                tensor.data.uniform_(-0.01, 0.01)
        elif tensor.is_floating_point():
            tensor.data = tensor.data.to(torch.bfloat16)
    model.config.dtype = torch.bfloat16
    model.save_pretrained(output)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({f"token{i}": i for i in range(128)}, unk_token="token0")),
        unk_token="token0",
    )
    tokenizer.save_pretrained(output)


def _read_weights(path: Path) -> dict[str, torch.Tensor]:
    weights = {}
    for shard in sorted(path.glob("*.safetensors")):
        weights.update(load_file(shard))
    return weights


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize("mtp_depths", [0, 2])
def test_nemotron4_hf_mimo_roundtrip(tmp_path, mtp_depths):
    source = os.environ.get("NEMOTRON4_HF_MODEL")
    if not source:
        pytest.skip("Set NEMOTRON4_HF_MODEL to the local Nemotron4 preview snapshot")
    initialize_distributed()
    if dist.get_world_size() != 2:
        pytest.skip("Requires exactly two ranks")
    root = Path(broadcast_path(tmp_path))
    hf_path, checkpoint, exported = root / "hf", root / "mimo", root / "exported"
    if dist.get_rank() == 0:
        _create_tiny_checkpoint(source, hf_path, mtp_depths)
    dist.barrier()
    parallelism = MegatronMIMOParallelismConfig(
        module_parallelisms={
            "language": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=1, rank_offset=0),
            "images": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=1, rank_offset=1),
        }
    )
    bridge = MegatronMIMOBridge.from_hf_pretrained(hf_path, parallelism_config=parallelism, trust_remote_code=True)
    model = bridge.to_megatron_model(load_weights=True, wrap_with_ddp=False)[0]
    expected = _read_weights(hf_path) if dist.get_rank() == 0 else {}

    # Check actual import/export, including every HF key, before checkpoint I/O.
    seen = set()
    for name, weight in bridge.export_hf_weights(model, show_progress=False):
        if dist.get_rank() == 0:
            assert name not in seen
            seen.add(name)
            torch.testing.assert_close(weight, expected[name], rtol=0, atol=0, msg=lambda detail: f"{name}: {detail}")
    if dist.get_rank() == 0:
        assert seen == set(expected)
    bridge.save_megatron_model(model, checkpoint)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    restored = bridge.load_megatron_model(checkpoint)
    bridge.save_hf_pretrained(restored, exported, strict=True, show_progress=False)
    if dist.get_rank() == 0:
        actual = _read_weights(exported)
        assert actual.keys() == expected.keys()
        for name in expected:
            torch.testing.assert_close(
                actual[name], expected[name], rtol=0, atol=0, msg=lambda detail: f"{name}: {detail}"
            )
        restored_config = AutoConfig.from_pretrained(exported, trust_remote_code=True)
        assert restored_config.text_config.num_nextn_predict_layers == mtp_depths
    dist.barrier()
