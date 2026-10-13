#!/usr/bin/env python3
"""test_reload_fix.py — CPU unit tests for reload_fix_patch.py (run on the box: CUDA_VISIBLE_DEVICES= ~/venv/bin/python ~/test_reload_fix.py).
Covers: overlap-tolerant copy, replace_parameter(prefer_copy=True) pointer stability, in-place fused-MoE quant-config refresh,
the gated broadcast helper, and (if a GPU is visible) the derived-tensor walker of the checksum hook."""
import os
import sys

import torch

os.environ.setdefault("IMPALA_RELOAD_FIX", "1")
fails = 0


def check(name, cond, detail=""):
    global fails
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        fails += 1


# 1. overlap-tolerant copy ------------------------------------------------------------------------------------------
from vllm.model_executor.model_loader.reload.layerwise import _impala_overlap_copy  # noqa: E402

base = torch.tensor([1.0])
dst = base.expand(4)  # stride-0 view, 1 storage element
src = torch.full((4,), 7.0)
try:
    dst.copy_(src)
    check("stock copy_ refuses an expanded destination", False, "no error raised")
except RuntimeError:
    check("stock copy_ refuses an expanded destination", True)
_impala_overlap_copy(dst, src, "x")
check("overlap copy writes the unique element", float(base[0]) == 7.0 and torch.equal(dst, src))
src2 = torch.tensor([3.0]).expand(4)
_impala_overlap_copy(dst, src2, "x")
check("overlap copy from an expanded source", float(base[0]) == 3.0)
try:
    _impala_overlap_copy(dst, torch.tensor([1.0, 2.0, 3.0, 4.0]), "x")
    check("overlap copy rejects a non-broadcast source", False, "no error raised")
except RuntimeError:
    check("overlap copy rejects a non-broadcast source", True)
plain = torch.zeros(4)
_impala_overlap_copy(plain, src, "y")
check("overlap copy is a plain copy_ for contiguous destinations", torch.equal(plain, src))

# 2. replace_parameter(prefer_copy=True) ---------------------------------------------------------------------------
from vllm.model_executor.utils import replace_parameter  # noqa: E402

m = torch.nn.Module()
m.register_parameter("w", torch.nn.Parameter(torch.zeros(3, 2), requires_grad=False))
ptr0 = m.w.data_ptr()
replace_parameter(m, "w", torch.ones(3, 2), prefer_copy=True)
check("prefer_copy keeps the storage address", m.w.data_ptr() == ptr0 and isinstance(m.w, torch.nn.Parameter))
check("prefer_copy copies the values", torch.equal(m.w, torch.ones(3, 2)))
replace_parameter(m, "w", torch.ones(3, 2) * 2)
check("default replace_parameter re-registers", m.w.data_ptr() != ptr0 and float(m.w[0, 0]) == 2.0)
replace_parameter(m, "w", torch.ones(5, 2), prefer_copy=True)
check("prefer_copy falls back on shape change", tuple(m.w.shape) == (5, 2))
check("prefer_copy on a missing attribute registers a parameter",
      (replace_parameter(m, "v", torch.ones(2), prefer_copy=True) or True) and isinstance(m.v, torch.nn.Parameter))

# 3. in-place fused-MoE quant-config refresh ------------------------------------------------------------------------
from vllm.model_executor.layers.fused_moe.config import nvfp4_moe_quant_config  # noqa: E402
from vllm.model_executor.layers.quantization.modelopt import _impala_refresh_quant_config_inplace  # noqa: E402

E = 4


def make_qc(v):
    return nvfp4_moe_quant_config(
        g1_alphas=torch.full((E,), v), g2_alphas=torch.full((E,), v + 1), a1_gscale=torch.full((E,), v + 2),
        a2_gscale=torch.full((E,), v + 3), w1_scale=torch.full((E, 8), v + 4), w2_scale=torch.full((E, 8), v + 5),
    )


old_qc, new_qc = make_qc(1.0), make_qc(10.0)
ids = {n: id(getattr(old_qc, n)) for n in ("g1_alphas", "g2_alphas", "a1_gscale", "a2_gscale", "w1_scale", "w2_scale")}
_impala_refresh_quant_config_inplace(old_qc, new_qc, torch.nn.Module())
check("refresh keeps the live tensor objects", all(id(getattr(old_qc, n)) == i for n, i in ids.items()))
check("refresh copies a1_gscale/a2_gscale in place",
      float(old_qc.a1_gscale[0]) == 12.0 and float(old_qc.a2_gscale[0]) == 13.0)
check("refresh copies alphas and block scales in place",
      float(old_qc.g1_alphas[0]) == 10.0 and float(old_qc.g2_alphas[0]) == 11.0 and float(old_qc.w1_scale[0, 0]) == 14.0
      and float(old_qc.w2_scale[0, 0]) == 15.0)
exp_old = make_qc(1.0)
exp_old._a1.alpha_or_gscale = torch.tensor([5.0]).expand(E)  # stride-0 live tensor (stock expand layout)
_impala_refresh_quant_config_inplace(exp_old, make_qc(20.0), torch.nn.Module())
check("refresh tolerates a stride-0 live tensor", float(exp_old.a1_gscale[0]) == 22.0)

# 4. gated broadcast helper ----------------------------------------------------------------------------------------
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import _impala_broadcast_scale  # noqa: E402

t = _impala_broadcast_scale(torch.tensor(2.0), E)
check("broadcast helper (fix on) returns a writable contiguous tensor", t.is_contiguous() and t.stride() == (1,) and t.numel() == E)
os.environ["IMPALA_RELOAD_FIX"] = "0"
t2 = _impala_broadcast_scale(torch.tensor(2.0), E)
check("broadcast helper (fix off) keeps the stock expand", t2.stride() == (0,))
os.environ["IMPALA_RELOAD_FIX"] = "1"

# 5. derived-tensor walker (GPU only) ------------------------------------------------------------------------------
if torch.cuda.is_available():
    from vllm.model_executor.model_loader.base_loader import _impala_derived_tensors

    class QM:
        pass

    mod = torch.nn.Module()
    mod.register_parameter("p", torch.nn.Parameter(torch.ones(2, device="cuda"), requires_grad=False))
    mod.register_buffer("b", torch.ones(2, device="cuda"))
    mod.plain = torch.ones(3, device="cuda")
    mod.quant_method = QM()
    mod.quant_method.moe_quant_config = nvfp4_moe_quant_config(
        g1_alphas=torch.ones(E, device="cuda"), g2_alphas=torch.ones(E, device="cuda"), a1_gscale=torch.ones(E, device="cuda"),
        a2_gscale=torch.ones(E, device="cuda"), w1_scale=torch.ones(E, 8, device="cuda"), w2_scale=torch.ones(E, 8, device="cuda"))
    names = [n for n, _ in _impala_derived_tensors(mod)]
    check("derived walker finds plain attributes and quant-config tensors",
          any(n.endswith("<attr>plain") for n in names) and any("<qc>_a1" in n for n in names)
          and not any(n.endswith("<attr>p") or n.endswith("<attr>b") for n in names), str(names))
else:
    print("skip derived-tensor walker (no GPU visible)")

print("ALL_OK" if fails == 0 else f"FAILS={fails}")
sys.exit(1 if fails else 0)
