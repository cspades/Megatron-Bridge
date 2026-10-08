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

"""Deprecated aliases for the NM4 SFT training provider.

The training model and provider now live in :mod:`megatron.bridge.models.nemotron4`.
These names remain so that ``run_config.yaml`` files written by earlier NM4 SFT
runs (``_target_: ...experimental_nm4_llava_provider.ExperimentalNM4LlavaProvider``)
keep resolving. New code should use :class:`Nemotron4VLModelProvider` and
:class:`Nemotron4VLModel` directly.
"""

from dataclasses import dataclass

from megatron.bridge.models.nemotron4.modeling_nemotron4_vl import (
    Nemotron4GatedDeltaProductMixer as _NM4GatedDeltaProductMixer,
)
from megatron.bridge.models.nemotron4.modeling_nemotron4_vl import (
    Nemotron4VisionMLP as _NM4VisionMLP,
)
from megatron.bridge.models.nemotron4.modeling_nemotron4_vl import (
    Nemotron4VLModel as NM4LlavaModel,
)
from megatron.bridge.models.nemotron4.nemotron4_provider import (
    PRECISION_RECIPE_FILENAME,
    Nemotron4VLModelProvider,
)


@dataclass
class ExperimentalNM4LlavaProvider(Nemotron4VLModelProvider):
    """Deprecated alias of :class:`Nemotron4VLModelProvider` kept for existing run configs."""


__all__ = [
    "PRECISION_RECIPE_FILENAME",
    "ExperimentalNM4LlavaProvider",
    "NM4LlavaModel",
    "_NM4GatedDeltaProductMixer",
    "_NM4VisionMLP",
]
