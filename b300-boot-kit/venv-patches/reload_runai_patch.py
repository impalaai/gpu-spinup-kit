#!/usr/bin/env python3
"""reload_runai_patch.py [vllm-package-dir] — Experiment 5 step 2 (2026-10-11): let `reload_weights` run through the Run:ai
streamer loader (and therefore through the pinned-staging path of pinned_loader_patch.py / patches/0001) instead of the default
safetensors mmap iterator. vLLM 0.20.2's `GPUModelRunner.reload_weights` raises NotImplementedError unless the loader has
`get_all_weights`; the Run:ai loader lacks it although its iterator is exactly what the first load used. Additive, idempotent,
inert unless the server was started with --load-format runai_streamer (the LoadConfig is captured in the sleep image, so the
bake must use it). With IMPALA_PINNED_LOADER=1 in the process env, the reload streams every tensor from a page-locked 16 GB
buffer straight to the worker's GPU (RESEARCH-4: 14 GB/s per worker) and the four TP=2 pairs reload from ONE page-cached copy.
"""
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/venv/lib/python3.12/site-packages/vllm")
p = os.path.join(root, "model_executor/model_loader/runai_streamer_loader.py")
s = open(p).read()
if "def get_all_weights" in s:
    print("runai_streamer_loader: already patched")
    raise SystemExit
old = '''    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into a model."""
'''
new = '''    def get_all_weights(
        self, model_config: ModelConfig, model: nn.Module
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Experiment 5 (Impala): the iterator `reload_weights` needs, same source and path as the first load."""
        model_weights = model_config.model
        if model_weights_override := model_config.model_weights:
            model_weights = model_weights_override
        yield from self._get_weights_iterator(model_weights, model_config.revision)

''' + old
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new, 1)
assert "Generator" in s and "import torch" in s, "runai_streamer_loader: expected imports missing"
open(p, "w").write(s)
print("runai_streamer_loader: patched (get_all_weights for reload_weights)")
