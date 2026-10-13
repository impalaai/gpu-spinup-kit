# Patch vllm kernel_warmup.py: env-gated flashinfer autotune cache. Idempotent.
import os
p = os.path.expanduser("~/venv/lib/python3.12/site-packages/vllm/model_executor/warmup/kernel_warmup.py")
s = open(p).read()
if "VLLM_FI_AUTOTUNE_CACHE" in s:
    print("already patched"); raise SystemExit
old = "    with torch.inference_mode(), fi_utils.autotune():"
new = """    _fi_cache = os.environ.get("VLLM_FI_AUTOTUNE_CACHE")
    with torch.inference_mode(), (
        fi_utils.autotune(True, cache=_fi_cache) if _fi_cache else fi_utils.autotune()
    ):"""
assert s.count(old) == 1, f"anchor count={s.count(old)}"
s = s.replace(old, new)
if "\nimport os\n" not in s and not s.startswith("import os\n"):
    lines = s.splitlines(True)
    for i, l in enumerate(lines):
        if l.startswith("import ") or l.startswith("from "):
            lines.insert(i, "import os\n"); break
    s = "".join(lines)
open(p, "w").write(s)
print("patched")
