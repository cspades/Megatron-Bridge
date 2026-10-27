"""One-GPU regression: Bridge HF import refreshes TE's original-weight initializer."""

from types import SimpleNamespace as NS

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import MXFP8BlockScaling

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge


torch.manual_seed(42)
with te.quantized_model_init(enabled=True, recipe=MXFP8BlockScaling(), preserve_high_precision_init_val=True):
    model = te.Linear(64, 64, bias=False, params_dtype=torch.bfloat16, device="cuda")
p = model.weight
assert p.get_high_precision_init_val() is not None
source = torch.randn_like(torch.empty(64, 64, device="cuda", dtype=torch.bfloat16))
original = p.get_high_precision_init_val().clone()
task = NS(
    megatron_module=model,
    param_weight=p,
    mapping=NS(hf_param="weight", megatron_param="weight"),
    param_name="weight",
    vp_stage=None,
)
bridge = NS(
    build_conversion_tasks=lambda hf, models: [task],
    _with_progress_tracking=lambda tasks, desc: tasks,
    maybe_modify_loaded_hf_weight=lambda key, state: state[key],
    _convert_loaded_hf_weight=lambda task, weight: weight,
    finalize_hf_import=lambda models: None,
)
MegatronModelBridge.load_weights_hf_to_megatron(
    bridge, NS(state={"weight": source}, model_name_or_path="synthetic-hf"), model
)
retained = p.get_high_precision_init_val()
assert not torch.equal(original, retained)
assert torch.equal(retained, source.cpu())
assert torch.equal(retained.float(), source.cpu().float())
assert not torch.equal(p.detach().float().cpu(), source.cpu().float()), (
    "Test must distinguish quantized weights from original values"
)
print("PASS: original BF16 HF values retained exactly for FP32 master initialization; model weights are MXFP8")
