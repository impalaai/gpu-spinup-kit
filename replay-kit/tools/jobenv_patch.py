# jobenv_patch.py — libcuda appears to consume/unset CUDA_CHECKPOINT_JOB_FILE at init, so spawned vLLM children never join
# the checkpoint job. Re-export it in every vLLM process from IMPALA_CKPT_JOB_FILE (which nothing touches). Idempotent.
import os
p = os.path.expanduser("~/venv/lib/python3.12/site-packages/vllm/__init__.py")
s = open(p).read()
if "IMPALA_CKPT_JOB_FILE" in s:
    print("already patched"); raise SystemExit
snippet = '''import os as _impala_os
if _impala_os.environ.get("IMPALA_CKPT_JOB_FILE") and not _impala_os.environ.get("CUDA_CHECKPOINT_JOB_FILE"):
    _impala_os.environ["CUDA_CHECKPOINT_JOB_FILE"] = _impala_os.environ["IMPALA_CKPT_JOB_FILE"]
'''
open(p, "w").write(snippet + s)
print("patched")
