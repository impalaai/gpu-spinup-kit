#!/usr/bin/env python3
"""exp6_patch_new.py <venv-root> — Experiment 6 (2026-10-11): the small env-gated venv patches of the 0.20.2 golden venv, ported to a
vLLM 0.31.0 venv (default ~/venv-new). Idempotent (marker strings). Does NOT touch ~/venv.

 1. upstream #57635 (vLLM 3403e0f176, 2026-10-06, in 0.31.1rc0, not in 0.31.0): per-rank FlashInfer autotune cache files — without it a
    TP>1 MoE serve with a warm cache can deadlock (rank 0 hits the cache and skips the per-tactic reduce). Applied from the diff file
    patches/0005-upstream-57635-autotune-cache-per-rank-v0.31.0.diff (copied to ~/exp6-0005.diff on the box) with `patch -p1`.
 2. flashinfer 0.7.0.post1 gdn_decode_bf16_state.py: module-level NUM_SMS = torch.cuda.get_device_properties(0) and (2b)
    _GPU_MAJOR/_USE_PACKED_FMA = torch.cuda.get_device_capability(0) → lazy (found by exp6_cudatrace.py: imported by
    vllm/compilation/passes/fusion/rms_quant_fusion.py:98 `from flashinfer import add_rmsnorm_fp4quant` in the API server; so the vLLM
    API server never initialises CUDA before forking the workers). Same edit as fi_numsms_patch.py.
 3. vllm/__init__.py: re-export CUDA_CHECKPOINT_JOB_FILE from IMPALA_CKPT_JOB_FILE (jobenv_patch.py).
 4. vllm/v1/executor/multiproc_executor.py: env-gated per-rank NUMA bind inside forked workers (numa_rankbind_patch.py; the upstream
    --numa-bind still forces spawn in 0.31.0, utils/system_utils.py:138).
 The pinned loader + checksum hook are applied by pinned_loader_patch_031.py <vllm-package-dir> (separate script).
"""
import os
import re
import subprocess
import sys

venv = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/venv-new")
sp = os.path.join(venv, "lib/python3.12/site-packages")
vllm_root = os.path.join(sp, "vllm")
diff_file = os.path.abspath(os.path.expanduser(os.environ.get("EXP6_0005_DIFF", "~/exp6-0005.diff")))

# ---------------------------------------------------------------- 1. #57635
kw = os.path.join(vllm_root, "model_executor/warmup/kernel_warmup.py")
s = open(kw).read()
if "_all_ranks_have_matching_cache" in s:
    print("kernel_warmup: #57635 already applied")
else:
    r = subprocess.run(["patch", "-p1", "--forward", "-d", sp, "-i", diff_file], capture_output=True, text=True)
    print("kernel_warmup: patch rc=%d %s" % (r.returncode, (r.stdout + r.stderr).strip().replace("\n", " | ")[:300]))
    assert r.returncode == 0, "#57635 did not apply"
    assert "_all_ranks_have_matching_cache" in open(kw).read()

# ---------------------------------------------------------------- 2. flashinfer NUM_SMS
p = os.path.join(sp, "flashinfer/gdn_kernels/gdn_decode_bf16_state.py")
s = open(p).read()
if "_impala_num_sms" in s:
    print("flashinfer gdn: already patched")
else:
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
    n = len(re.findall(r"(?<![_A-Za-z])NUM_SMS(?![_A-Za-z])", s))
    s = re.sub(r"(?<![_A-Za-z])NUM_SMS(?![_A-Za-z])", "_impala_num_sms()", s)
    open(p, "w").write(s)
    print(f"flashinfer gdn: patched; {n} NUM_SMS uses rewritten")
# 2b. the second import-time query in the same module (flashinfer 0.7.0.post1:2709): _GPU_MAJOR/_USE_PACKED_FMA → lazy
s = open(p).read()
if "_impala_use_packed_fma" in s:
    print("flashinfer gdn: _USE_PACKED_FMA already lazy")
else:
    old2 = "_GPU_MAJOR, _ = torch.cuda.get_device_capability(0)\n_USE_PACKED_FMA = _GPU_MAJOR >= 10\n"
    assert s.count(old2) == 1, s.count(old2)
    new2 = '''_IMPALA_USE_PACKED_FMA = None
def _impala_use_packed_fma():
    global _IMPALA_USE_PACKED_FMA
    if _IMPALA_USE_PACKED_FMA is None:
        _IMPALA_USE_PACKED_FMA = torch.cuda.get_device_capability(0)[0] >= 10
    return _IMPALA_USE_PACKED_FMA
'''
    s = s.replace(old2, new2)
    n2 = len(re.findall(r"(?<![_A-Za-z])_USE_PACKED_FMA(?![_A-Za-z])", s))
    s = re.sub(r"(?<![_A-Za-z])_USE_PACKED_FMA(?![_A-Za-z])", "_impala_use_packed_fma()", s)
    assert "_GPU_MAJOR" not in s, "a _GPU_MAJOR use remains"
    open(p, "w").write(s)
    print(f"flashinfer gdn: _USE_PACKED_FMA made lazy; {n2} uses rewritten")

# 2c. flashinfer/kda_kernels/__init__.py:59–62 queries the device capability at IMPORT time (guarded by cuda.is_available()) to
#     decide whether the CuTe-DSL KDA path is usable; in the API server that initialises CUDA. Guard it by cuda.is_initialized()
#     instead: processes that have not touched CUDA skip the arch check (the DSL path is simply enabled; B300/sm103 is supported).
p = os.path.join(sp, "flashinfer/kda_kernels/__init__.py")
s = open(p).read()
if "_torch.cuda.is_initialized():" in s:
    print("flashinfer kda: already patched")
else:
    old3 = "    if _torch.cuda.is_available():\n"
    assert s.count(old3) == 1, s.count(old3)
    s = s.replace(old3, "    if _torch.cuda.is_initialized():  # Impala: no CUDA init at import time (was is_available())\n")
    open(p, "w").write(s)
    print("flashinfer kda: import-time arch check gated on cuda.is_initialized()")

# 2d. kernel_warmup.py (after #57635): env-gated IMPALA_FI_TRUST_CACHE=1 — when every rank loaded a matching per-rank autotune
#     file, run the autotune dummy runs with tune_mode=False so FlashInfer serves the persisted tactics. Why: flashinfer 0.7's fused-MoE
#     TuningConfig sets use_cold_l2_cache=True, and AutoTuner.search_cache skips persisted v1 entries while tuning whenever the
#     requested policy (cold L2) differs from the default (hot) → every boot re-profiles all 22 trtllm_fp4_block_scale_moe shapes
#     (~250 s) even though "Loaded 22 configs" (N1). Measurement aid; the gap is an upstream vLLM/FlashInfer issue.
p = os.path.join(vllm_root, "model_executor/warmup/kernel_warmup.py")
s = open(p).read()
if "IMPALA_FI_TRUST_CACHE" in s:
    print("kernel_warmup: trust-cache already patched")
else:
    old4 = """    if _all_ranks_have_matching_cache(cache_path, tune_group):
        loaded = tuner.load_configs(str(cache_path))
"""
    new4 = """    _impala_loaded = False
    if _all_ranks_have_matching_cache(cache_path, tune_group):
        loaded = tuner.load_configs(str(cache_path))
"""
    assert s.count(old4) == 1, s.count(old4)
    s = s.replace(old4, new4)
    old5 = """        if not loaded:
            tuner.clear_cache()
"""
    new5 = """        if not loaded:
            tuner.clear_cache()
        _impala_loaded = bool(loaded)
    _impala_tune = not (
        _impala_loaded and os.environ.get("IMPALA_FI_TRUST_CACHE", "0") == "1"
    )
    if not _impala_tune:
        logger.info_once("[impala-fi] every rank loaded the autotune cache; running the warmup with tune_mode=False")
"""
    assert s.count(old5) == 1, s.count(old5)
    s = s.replace(old5, new5)
    old6 = "            fi_utils.autotune(tune_mode=True, **autotune_kwargs),\n"
    assert s.count(old6) == 1, s.count(old6)
    s = s.replace(old6, "            fi_utils.autotune(tune_mode=_impala_tune, **autotune_kwargs),\n")
    if "\nimport os\n" not in s:
        s = s.replace("\nimport json\n", "\nimport json\nimport os\n", 1)
    open(p, "w").write(s)
    print("kernel_warmup: trust-cache patched (IMPALA_FI_TRUST_CACHE=1)")

# 2e. the dummy runs re-enter `fi_utils.autotune(tuning_buckets=…)` = tune_mode=True (vllm/utils/flashinfer.py wrapper default), which
#     re-enables tuning inside the trust-cache context (N2 still profiled 22 shapes). Propagate the trust decision to that inner context.
p = os.path.join(vllm_root, "model_executor/warmup/kernel_warmup.py")
s = open(p).read()
if "_IMPALA_TUNE_MODE" in s:
    print("kernel_warmup: inner tune mode already patched")
else:
    old7 = "        with fi_utils.autotune(tuning_buckets=tuning_buckets):\n"
    assert s.count(old7) == 1, s.count(old7)
    s = s.replace(old7, "        with fi_utils.autotune(tune_mode=_IMPALA_TUNE_MODE, tuning_buckets=tuning_buckets):\n")
    old8 = """    if not _impala_tune:
        logger.info_once("[impala-fi] every rank loaded the autotune cache; running the warmup with tune_mode=False")
"""
    assert s.count(old8) == 1, s.count(old8)
    s = s.replace(old8, old8 + """    global _IMPALA_TUNE_MODE
    _IMPALA_TUNE_MODE = _impala_tune
""")
    old9 = "\n\ndef _run_flashinfer_autotune_dummy_runs("
    assert s.count(old9) == 1, s.count(old9)
    s = s.replace(old9, "\n\n_IMPALA_TUNE_MODE = True  # Impala: set by flashinfer_autotune() when the persisted cache is trusted\n" + old9)
    open(p, "w").write(s)
    print("kernel_warmup: inner dummy-run tune mode follows the trust decision")

# ---------------------------------------------------------------- 3. jobenv
p = os.path.join(vllm_root, "__init__.py")
s = open(p).read()
if "IMPALA_CKPT_JOB_FILE" in s:
    print("vllm/__init__: already patched")
else:
    snippet = '''import os as _impala_os
if _impala_os.environ.get("IMPALA_CKPT_JOB_FILE") and not _impala_os.environ.get("CUDA_CHECKPOINT_JOB_FILE"):
    _impala_os.environ["CUDA_CHECKPOINT_JOB_FILE"] = _impala_os.environ["IMPALA_CKPT_JOB_FILE"]
'''
    open(p, "w").write(snippet + s)
    print("vllm/__init__: patched (job-file re-export)")

# ---------------------------------------------------------------- 4. numa rank bind
p = os.path.join(vllm_root, "v1/executor/multiproc_executor.py")
s = open(p).read()
if "IMPALA_NUMA_RANK_BIND" in s:
    print("multiproc_executor: already patched")
else:
    anchor = '''        worker = None
        ready_writer = kwargs.pop("ready_pipe")
'''
    assert s.count(anchor) == 1, s.count(anchor)
    snippet = '''        if os.environ.get("IMPALA_NUMA_RANK_BIND") == "1":
            try:
                import ctypes
                _lr = kwargs.get("local_rank", kwargs.get("rank", 0))
                _nodes = [int(x) for x in os.environ.get("IMPALA_NUMA_NODES", "").split(",") if x != ""]
                if _lr < len(_nodes):
                    _node = _nodes[_lr]
                    _cpus = set()
                    for _part in open(f"/sys/devices/system/node/node{_node}/cpulist").read().strip().split(","):
                        _a, _, _b = _part.partition("-")
                        _cpus.update(range(int(_a), int(_b or _a) + 1))
                    os.sched_setaffinity(0, _cpus)
                    _mask = ctypes.c_ulong(1 << _node)
                    _rc = ctypes.CDLL(None, use_errno=True).syscall(238, 1, ctypes.byref(_mask), 64)  # set_mempolicy(MPOL_PREFERRED)
                    logger.info("IMPALA numa rank bind: local_rank=%s node=%s cpus=%d mempolicy_rc=%s", _lr, _node, len(_cpus), _rc)
            except Exception as _e:  # placement only — never fail the worker
                logger.warning("IMPALA numa rank bind failed: %s", _e)
'''
    s = s.replace(anchor, snippet + anchor)
    assert "\nimport os\n" in s
    open(p, "w").write(s)
    print("multiproc_executor: patched (numa rank bind)")
print("EXP6_PATCH_NEW_DONE")
