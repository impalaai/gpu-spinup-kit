# precapture_patch.py — env-gated pause right before CUDA-graph capture in vLLM's GPU worker (idempotent).
# With VLLM_PRECAPTURE_GATE=<path>, each worker writes <path>.waiting.<pid>, synchronizes, and spins until <path> exists.
# Lets us checkpoint the fully-loaded, compiled, autotuned TP worker tree BEFORE graphs exist (graphs with multi-GPU
# collectives are not restorable by cuda-checkpoint on driver 615); capture resumes after restore.
import os
p = os.path.expanduser("~/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py")
s = open(p).read()
if "VLLM_PRECAPTURE_GATE" in s:
    print("already patched"); raise SystemExit
anchor = "        if not self.model_config.enforce_eager:\n            cuda_graph_memory_bytes = self.model_runner.capture_model()"
assert s.count(anchor) == 1, f"anchor count={s.count(anchor)}"
gate = '''        _pcg = os.environ.get("VLLM_PRECAPTURE_GATE")
        if _pcg:
            import time as _pcg_time
            torch.cuda.synchronize()
            with open(f"{_pcg}.waiting.{os.getpid()}", "w") as _pcg_f:
                _pcg_f.write("1")
            while not os.path.exists(_pcg):
                _pcg_time.sleep(0.5)
            torch.cuda.synchronize()
'''
s = s.replace(anchor, gate + anchor)
if "\nimport os\n" not in s:
    s = s.replace("\nimport ", "\nimport os\nimport ", 1)
open(p, "w").write(s)
print("patched")
