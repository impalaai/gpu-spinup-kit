# numa_rankbind_patch.py — env-gated per-rank NUMA binding inside vLLM's forked TP workers (Experiment 3, TP=8 cold boot).
# vLLM 0.20.2's --numa-bind wraps each worker in numactl but forces the spawn start method (loses the fork lever). With
# IMPALA_NUMA_RANK_BIND=1 and IMPALA_NUMA_NODES="0,0,0,0,1,1,1,1" (node per local rank, i.e. per CUDA_VISIBLE_DEVICES slot),
# every worker pins its CPU affinity to its GPU's node and sets MPOL_PREFERRED for that node before loading anything.
# Behaviour-neutral (placement only). Idempotent.
import os
p = os.path.expanduser("~/venv/lib/python3.12/site-packages/vllm/v1/executor/multiproc_executor.py")
s = open(p).read()
if "IMPALA_NUMA_RANK_BIND" in s:
    print("already patched"); raise SystemExit
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
if "\nimport os\n" not in s:
    s = s.replace("\nimport ", "\nimport os\nimport ", 1)
open(p, "w").write(s)
print("patched")
