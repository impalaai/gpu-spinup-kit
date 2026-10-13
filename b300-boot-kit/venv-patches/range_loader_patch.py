#!/usr/bin/env python3
"""range_loader_patch.py [vllm-package-dir] [module-src] — Experiment 4b: install the per-rank byte-range reader
(impala_range_loader.py) into vLLM 0.20.2's model_loader package and hook RunaiModelStreamerLoader.load_weights so that
IMPALA_RANGE_LOADER=1 routes TP>1 loads through it (stock path otherwise). Idempotent; requires pinned_loader_patch.py
(it reuses _impala_pin_streamer_buffer). Default target ~/venv/lib/python3.12/site-packages/vllm, module next to this script."""
import os
import shutil
import sys

root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/venv/lib/python3.12/site-packages/vllm")
src = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "impala_range_loader.py")
dst = os.path.join(root, "model_executor/model_loader/impala_range_loader.py")
rl = os.path.join(root, "model_executor/model_loader/runai_streamer_loader.py")

cur = open(dst).read() if os.path.exists(dst) else None
new = open(src).read()
if cur == new:
    print("module: already installed (identical)")
else:
    shutil.copyfile(src, dst)
    print("module: installed" if cur is None else "module: updated")

s = open(rl).read()
if "load_weights_ranged" in s:
    print("runai_streamer_loader: already patched")
else:
    old = '''        model_weights = model_config.model
        if model_weights_override := model_config.model_weights:
            model_weights = model_weights_override
        model.load_weights(
            self._get_weights_iterator(model_weights, model_config.revision)
        )
'''
    new_body = '''        model_weights = model_config.model
        if model_weights_override := model_config.model_weights:
            model_weights = model_weights_override
        if os.environ.get("IMPALA_RANGE_LOADER", "0") == "1":
            # Experiment 4b (Impala): per-rank byte-range reads for TP>1; returns False -> stock path
            from vllm.model_executor.model_loader.impala_range_loader import load_weights_ranged

            if load_weights_ranged(self, model, model_weights, model_config.revision):
                return
        model.load_weights(
            self._get_weights_iterator(model_weights, model_config.revision)
        )
'''
    assert s.count(old) == 1, s.count(old)
    s = s.replace(old, new_body, 1)
    assert "\nimport os\n" in s
    open(rl, "w").write(s)
    print("runai_streamer_loader: patched (env-gated IMPALA_RANGE_LOADER=1)")
