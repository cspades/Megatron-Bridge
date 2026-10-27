"""Regression for HF conversion routes through MIMO component wrappers."""

import torch.nn as nn
from megatron.core.distributed import DistributedDataParallel
from megatron.core.transformer.module import Float16Module

from megatron.bridge.models.megatron_mimo.conversion.orchestrator import _get_route_submodule


def wrap(cls, module):
    result = cls.__new__(cls)
    nn.Module.__init__(result)
    result.module = module
    return result


leaf = nn.Linear(2, 2)
component = nn.Module()
component.encoders = nn.ModuleDict({"vision": leaf})
root = nn.Module()
root.modality_submodules = nn.ModuleDict({"images": wrap(DistributedDataParallel, wrap(Float16Module, component))})
root.language_model = wrap(DistributedDataParallel, nn.Linear(2, 2))
assert _get_route_submodule(root, "modality_submodules.images.encoders.vision") is leaf
assert _get_route_submodule(wrap(Float16Module, root), "modality_submodules.images.encoders.vision") is leaf
assert _get_route_submodule(root, "language_model") is root.language_model.module
try:
    _get_route_submodule(root, "modality_submodules.images.missing")
except AttributeError:
    pass
else:
    raise AssertionError("Missing routes must fail")
print("PASS: nested DDP/Float16 route traversal and missing-route rejection")
