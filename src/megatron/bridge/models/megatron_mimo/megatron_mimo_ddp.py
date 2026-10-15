# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Mixed-precision and DDP wrapping utilities for MegatronMIMO models.

Called from the training layer after MegatronMIMOProvider.provide().

Note: This module only supports DDP wrapping. FSDP is not yet implemented.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import torch
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.model import MimoEncoderFloat16Module
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn


if TYPE_CHECKING:
    from megatron.core.distributed import (
        DistributedDataParallel,
        DistributedDataParallelConfig,
    )
    from megatron.core.hyper_comm_grid import HyperCommGrid
    from megatron.core.models.mimo import MimoModel
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.transformer_config import TransformerConfig

    from megatron.bridge.models.megatron_mimo.megatron_mimo_config import MegatronMIMOParallelismConfig


def _wrap_mixed_precision(
    module: torch.nn.Module,
    config: TransformerConfig,
    *,
    encoder: bool,
) -> torch.nn.Module:
    """Apply the MCore mixed-precision wrapper when the component requests it."""
    if not (config.fp16 or config.bf16):
        return module
    wrapper = MimoEncoderFloat16Module if encoder else Float16Module
    return wrapper(config, module)


def wrap_megatron_mimo_model_distributed(
    megatron_mimo_model: "MimoModel",
    ddp_config: "DistributedDataParallelConfig",
    megatron_mimo_parallelism_config: "MegatronMIMOParallelismConfig",
    grids: Dict[str, "HyperCommGrid"],
    pg_collections: Dict[str, Optional["ProcessGroupCollection"]],
    *,
    use_layer_wise_distributed_optimizer: bool = False,
    use_layer_wise_param_layout: bool = True,
) -> "MimoModel":
    """Wrap MegatronMIMO model's submodules with mixed precision and DDP.

    Modifies megatron_mimo_model in-place and returns it.

    Args:
        megatron_mimo_model: The MimoModel to wrap.
        ddp_config: DDP configuration from Bridge.
        megatron_mimo_parallelism_config: MegatronMIMO parallelism configuration.
        grids: Module name to HyperCommGrid mapping.
        pg_collections: Module name to ProcessGroupCollection mapping.
        use_layer_wise_distributed_optimizer: Build shard-aligned layouts for layer-wise optimizers.
        use_layer_wise_param_layout: With the layer-wise optimizer, use shard-aligned DDP buffers.
            False keeps the default DDP buffers and ``ddp_config.use_distributed_optimizer``, so
            the layer-wise optimizer takes its legacy variable-size parameter sync (the NM4
            compact-Muon recipes). Mirrors Megatron-Core's ``_ddp_wrap``.

    Returns:
        The same megatron_mimo_model with wrapped submodules.
    """
    from megatron.core.distributed import DistributedDataParallel
    from megatron.core.optimizer.layer_wise_optimizer import (
        LayerWiseDistributedOptimizer,
        tag_params_for_buffer_routing,
    )

    # Lazy import to avoid circular dependency (models layer loads before training layer)
    from megatron.bridge.training.megatron_mimo_parallel_utils import is_current_rank_in_grid

    def wrap_module(
        module: nn.Module, config: "TransformerConfig", pg: "ProcessGroupCollection"
    ) -> "DistributedDataParallel":
        layout = None
        if use_layer_wise_distributed_optimizer and use_layer_wise_param_layout:
            ddp_config.use_distributed_optimizer = True
            tag_params_for_buffer_routing([module])
            groups = pg.setup_process_groups_for_ddp(pg, config, ddp_config)
            bucket_size = ddp_config.bucket_size
            if ddp_config.num_buckets is not None:
                bucket_size = sum(p.numel() for p in module.parameters()) // ddp_config.num_buckets
            elif bucket_size is None:
                bucket_size = max(40000000, 1000000 * groups["dp_group"].size())
            if not ddp_config.overlap_grad_reduce or groups["pp_group"].rank() > 0:
                bucket_size = None
            layout = LayerWiseDistributedOptimizer.compute_full_param_layout(
                [p for p in module.parameters() if p.requires_grad],
                bucket_size,
                groups["intra_dp_cp_group"].size(),
                ddp_config,
                expert_data_parallel_world_size=groups["intra_expt_dp_group"].size(),
            )
        return DistributedDataParallel(
            config=config,
            ddp_config=ddp_config,
            module=module,
            pg_collection=pg,
            full_param_layout=layout,
        )

    # Wrap language model if present and rank participates
    if megatron_mimo_model.language_model is not None:
        llm_grid = grids.get(MIMO_LANGUAGE_MODULE_KEY)
        if llm_grid is not None and is_current_rank_in_grid(llm_grid):
            llm_pg = pg_collections.get(MIMO_LANGUAGE_MODULE_KEY)
            if llm_pg is not None:
                language_config = megatron_mimo_model.language_model.config
                language_model = _wrap_mixed_precision(
                    megatron_mimo_model.language_model,
                    language_config,
                    encoder=False,
                )
                wrapped_lm = wrap_module(
                    language_model,
                    language_config,
                    llm_pg,
                )
                # MCore's DDP wrapper does not proxy arbitrary module methods.
                # MimoModel._forward_language_module() checks for and calls
                # language_model.set_input_tensor(...) on non-first PP stages.
                # Preserve that method on the wrapper so decoder input tensors
                # are wired correctly when language_model is DDP-wrapped.
                if hasattr(wrapped_lm.module, "set_input_tensor"):
                    wrapped_lm.set_input_tensor = wrapped_lm.module.set_input_tensor
                megatron_mimo_model.language_model = wrapped_lm

    # Wrap modality submodules
    if hasattr(megatron_mimo_model, "modality_submodules"):
        for module_name, submodule in megatron_mimo_model.modality_submodules.items():
            if submodule is None:
                continue
            module_grid = grids.get(module_name)
            if module_grid is None:
                continue
            if not is_current_rank_in_grid(module_grid):
                continue

            module_pg = pg_collections.get(module_name)
            if module_pg is None:
                continue

            # Get config from first encoder in the submodule.
            # Note: We use the first encoder's config for DDP bucket sizing.
            # This assumes all encoders in a modality submodule share similar
            # parallelism settings, which is typical for MegatronMIMO models.
            if hasattr(submodule, "encoders") and submodule.encoders:
                encoder_key = next(iter(submodule.encoders.keys()))
                first_encoder = submodule.encoders[encoder_key]

                if not hasattr(first_encoder, "config"):
                    raise AttributeError(
                        f"Encoder '{encoder_key}' in modality '{module_name}' does not have "
                        f"a 'config' attribute. Encoders must be MegatronModule subclasses."
                    )

                component_config = first_encoder.config
                precision_wrapped = _wrap_mixed_precision(
                    submodule,
                    component_config,
                    encoder=True,
                )
                wrapped = wrap_module(
                    precision_wrapped,
                    component_config,
                    module_pg,
                )
                # MCore optimizers unwrap DDP and read process groups from the wrapped module.
                wrapped.module.pg_collection = module_pg
                megatron_mimo_model.modality_submodules[module_name] = wrapped

    return megatron_mimo_model
