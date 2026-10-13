# fi_numsms_patch.py — flashinfer 0.6.8 gdn_decode_bf16_state.py queries torch.cuda.get_device_properties(0) at IMPORT time,
# which initialises CUDA in vLLM's API server and forces the spawn start method. Make NUM_SMS lazy. Idempotent.
import os, re
p = os.path.expanduser("~/venv/lib/python3.12/site-packages/flashinfer/gdn_kernels/gdn_decode_bf16_state.py")
s = open(p).read()
if "_impala_num_sms" in s:
    print("already patched"); raise SystemExit
old = "NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count\n"
assert s.count(old) == 1, s.count(old)
new = '''_IMPALA_NUM_SMS = None
def _impala_num_sms():
    global _IMPALA_NUM_SMS
    if _IMPALA_NUM_SMS is None:
        _IMPALA_NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
    return _IMPALA_NUM_SMS
'''
s = s.replace(old, new)
n = len(re.findall(r"\bNUM_SMS\b", s))
s = re.sub(r"(?<![_A-Za-z])NUM_SMS(?![_A-Za-z])", "_impala_num_sms()", s)
open(p, "w").write(s)
print(f"patched; {n} NUM_SMS uses rewritten")
