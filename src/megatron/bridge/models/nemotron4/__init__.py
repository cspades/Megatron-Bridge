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


"""Nemotron4 model family (hybrid wide-residual LM + Pixtral-style ViT) for Megatron Bridge."""

from megatron.bridge.models.nemotron4.modeling_nemotron4_vl import Nemotron4VLModel
from megatron.bridge.models.nemotron4.nemotron4_bridge import Nemotron4Bridge
from megatron.bridge.models.nemotron4.nemotron4_provider import (
    Nemotron4ModelProvider,
    Nemotron4VLModelProvider,
)


__all__ = [
    "Nemotron4Bridge",
    "Nemotron4ModelProvider",
    "Nemotron4VLModel",
    "Nemotron4VLModelProvider",
]
