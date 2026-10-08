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

import pytest

from megatron.bridge.models.megatron_mimo.conversion import orchestrator


@pytest.fixture(autouse=True)
def _preserve_mimo_conversion_registry():
    """Restore import-time MIMO conversion registrations after each test.

    Several tests here call ``_reset_registry_for_tests`` to work on an empty registry.
    Bridges register their MIMO specs at import time (e.g. Nemotron4), so without this
    restore those registrations stay lost for every later test in the session.
    """
    saved = dict(orchestrator._CONVERSION_SPECS)
    yield
    orchestrator._CONVERSION_SPECS.clear()
    orchestrator._CONVERSION_SPECS.update(saved)
