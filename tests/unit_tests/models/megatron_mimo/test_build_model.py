# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.megatron_mimo.build_model import _bridge_parallel_state_globals, build_megatron_mimo_model
from megatron.bridge.models.megatron_mimo.megatron_mimo_builder import build_hypercomm_grids
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOProvider


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


def test_build_megatron_mimo_model_creates_process_groups_once(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: [0])
    monkeypatch.setattr(dist, "new_subgroups_by_enumeration", lambda ranks, **kwargs: (object(), []))
    monkeypatch.setattr(dist, "new_group", lambda ranks=None, **kwargs: object())
    build_module = "megatron.bridge.models.megatron_mimo.build_model"
    monkeypatch.setattr(f"{build_module}._set_per_module_random_seeds", lambda infra, *, seed: None)
    monkeypatch.setattr(f"{build_module}._bridge_parallel_state_globals", lambda pg_collection: None)

    provider = MegatronMIMOProvider(
        language_model_spec=ModuleSpec(module=MagicMock, params={"config": MagicMock()}),
        megatron_mimo_parallelism_config=MegatronMIMOParallelismConfig(
            module_parallelisms={"language": ModuleParallelismConfig(data_parallel_size=1)}
        ),
        use_cpu_initialization=True,
    )
    provider_module = "megatron.bridge.models.megatron_mimo.megatron_mimo_provider"
    with (
        patch(f"{provider_module}.build_hypercomm_grids", wraps=build_hypercomm_grids) as build_grids_spy,
        patch(f"{provider_module}.MimoModel") as mock_mimo_model,
        patch(f"{provider_module}.get_model_config"),
        patch(f"{provider_module}.wrap_megatron_mimo_model_distributed") as mock_wrap_ddp,
    ):
        mock_wrap_ddp.side_effect = lambda megatron_mimo_model, **kwargs: megatron_mimo_model
        _, infra = build_megatron_mimo_model(provider, ddp_config=MagicMock(), bf16=False)

    build_grids_spy.assert_called_once()
    mimo_config = mock_mimo_model.call_args.args[0]
    assert mimo_config.module_to_grid_map is infra.module_to_grid_map
    assert mimo_config.language_model_spec.params["pg_collection"] is infra.pg_collections["language"]
    assert mock_wrap_ddp.call_args.kwargs["pg_collections"] is infra.pg_collections
