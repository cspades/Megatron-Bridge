# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from megatron.bridge.models.deepseek.deepseek_v2_bridge import DeepSeekV2Bridge  # noqa: F401
from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge  # noqa: F401


try:
    from megatron.bridge.models.deepseek.deepseek_v4_bridge import DeepSeekV4Bridge  # noqa: F401
except AttributeError:
    # DeepSeek V4 needs the WINDOW/CSA/HCA hybrid-layer symbols from current Megatron-Core.
    # The Nemotron4 Megatron-LM fork predates them; keep the rest of the DeepSeek family
    # importable there and register V4 only when its MCore support is present.
    DeepSeekV4Bridge = None


__all__ = ["DeepSeekV2Bridge", "DeepSeekV3Bridge"]
if DeepSeekV4Bridge is not None:
    __all__.append("DeepSeekV4Bridge")
