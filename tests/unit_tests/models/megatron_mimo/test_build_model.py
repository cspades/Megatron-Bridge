# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from megatron.core import parallel_state as mpu

from megatron.bridge.models.megatron_mimo import build_model
from megatron.bridge.models.megatron_mimo.build_model import _bridge_parallel_state_globals


def test_bridge_parallel_state_uses_mcore_expert_group_fields(monkeypatch):
    groups = SimpleNamespace(
        tp=object(),
        dp=object(),
        dp_cp=object(),
        pp=object(),
        ep=object(),
        expt_tp=object(),
        tp_ep=object(),
        tp_ep_pp=object(),
        cp=object(),
        mp=object(),
    )
    for global_name in (
        "_GLOBAL_MEMORY_BUFFER",
        "_TENSOR_MODEL_PARALLEL_GROUP",
        "_DATA_PARALLEL_GROUP",
        "_DATA_PARALLEL_GROUP_WITH_CP",
        "_PIPELINE_MODEL_PARALLEL_GROUP",
        "_EXPERT_MODEL_PARALLEL_GROUP",
        "_EXPERT_TENSOR_PARALLEL_GROUP",
        "_EXPERT_DATA_PARALLEL_GROUP",
        "_INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP",
        "_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP",
        "_EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP",
        "_CONTEXT_PARALLEL_GROUP",
        "_MODEL_PARALLEL_GROUP",
    ):
        monkeypatch.setattr(mpu, global_name, None)

    _bridge_parallel_state_globals(groups)

    assert mpu._EXPERT_MODEL_PARALLEL_GROUP is groups.ep
    assert mpu._EXPERT_TENSOR_PARALLEL_GROUP is groups.expt_tp
    assert mpu._EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP is groups.tp_ep
    assert mpu._EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP is groups.tp_ep_pp


def test_build_model_forwards_layer_wise_optimizer_flag(monkeypatch):
    provider = MagicMock()
    provider.provide_distributed_model.return_value = [MagicMock()]
    monkeypatch.setattr(build_model, "dist", MagicMock())
    monkeypatch.setattr(build_model, "_set_per_module_random_seeds", MagicMock())
    monkeypatch.setattr(build_model, "_bridge_parallel_state_globals", MagicMock())

    with (
        patch("megatron.bridge.training.megatron_mimo_parallel_utils.validate_no_stub_ranks"),
        patch(
            "megatron.bridge.training.megatron_mimo_parallel_utils.get_active_module_pg",
            return_value=("language", MagicMock()),
        ),
    ):
        build_model.build_megatron_mimo_model(
            provider, ddp_config=MagicMock(), use_layer_wise_distributed_optimizer=True
        )

    assert provider.provide_distributed_model.call_args.kwargs["use_layer_wise_distributed_optimizer"] is True
