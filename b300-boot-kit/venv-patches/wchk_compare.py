#!/usr/bin/env python3
"""wchk_compare.py <report-A.json> <report-B.json> [--same-process]
Compare two checksum reports (reload_fix_patch.py hook v2; v1 reports lack `derived`). Prints the parameter/buffer digest and
the derived-tensor digest of both, lists every tensor whose hash differs, and — with --same-process (A and B taken in the same
process, e.g. fresh load vs after wake_up+reload_weights) — every tensor whose data_ptr moved (a captured CUDA graph would read
the old address). Exit 0 iff both digests match and (with --same-process) no pointer moved."""
import json
import sys


def load(p):
    with open(p) as f:
        return json.load(f)


a, b = load(sys.argv[1]), load(sys.argv[2])
same_proc = "--same-process" in sys.argv
import re
ign = [re.compile(sys.argv[i + 1]) for i, a in enumerate(sys.argv) if a == "--ignore" and i + 1 < len(sys.argv)]
rc = 0
print(f"A: tag={a.get('tag', 'load')} rank={a['rank']} pid={a['pid']} params/buffers {a['n_hashed']} ({a['gpu_bytes'] / 2**30:.1f} GiB) "
      f"all={a['all_tensors_sha256'][:16]} derived={a.get('n_derived', '?')} {str(a.get('derived_sha256', '-'))[:16]}")
print(f"B: tag={b.get('tag', 'load')} rank={b['rank']} pid={b['pid']} params/buffers {b['n_hashed']} ({b['gpu_bytes'] / 2**30:.1f} GiB) "
      f"all={b['all_tensors_sha256'][:16]} derived={b.get('n_derived', '?')} {str(b.get('derived_sha256', '-'))[:16]}")
for section in ("tensors", "derived"):
    ta, tb = a.get(section, {}), b.get(section, {})
    skip = lambda n: n.endswith("<attr>kv_cache") or any(r.search(n) for r in ign)  # KV tensors are state; --ignore <regex>
    ta = {n: e for n, e in ta.items() if not skip(n)}
    tb = {n: e for n, e in tb.items() if not skip(n)}
    if not ta and not tb:
        continue
    only_a = sorted(set(ta) - set(tb))
    only_b = sorted(set(tb) - set(ta))
    diff = [n for n in sorted(set(ta) & set(tb)) if ta[n].get("h64") != tb[n].get("h64") or ta[n].get("shape") != tb[n].get("shape")]
    moved = [n for n in sorted(set(ta) & set(tb)) if same_proc and "ptr" in ta[n] and "ptr" in tb[n] and ta[n]["ptr"] != tb[n]["ptr"]]
    print(f"[{section}] common {len(set(ta) & set(tb))}, only-A {len(only_a)}, only-B {len(only_b)}, hash-diff {len(diff)}, ptr-moved {len(moved)}")
    for n in only_a[:10]:
        print(f"   only-A  {n}")
    for n in only_b[:10]:
        print(f"   only-B  {n}")
    for n in diff[:40]:
        print(f"   DIFF    {n}  {ta[n].get('shape')} {ta[n].get('h64')} -> {tb[n].get('h64')}")
    for n in moved[:40]:
        print(f"   MOVED   {n}  {hex(ta[n]['ptr'])} -> {hex(tb[n]['ptr'])}")
    if diff or moved or only_a or only_b:
        rc = 1
print("DIGESTS_MATCH" if a["all_tensors_sha256"] == b["all_tensors_sha256"] else "DIGESTS_DIFFER",
      "DERIVED_MATCH" if a.get("derived_sha256") == b.get("derived_sha256") else "DERIVED_DIFFER", "RC", rc)
sys.exit(rc)
