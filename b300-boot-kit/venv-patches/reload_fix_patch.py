#!/usr/bin/env python3
"""reload_fix_patch.py [vllm-package-dir] — Experiment 5 (2026-10-11): make vLLM 0.20.2's in-place `reload_weights` correct after a
level-2 sleep for ModelOpt NVFP4 MoE + MLA models (GLM-5.2-NVFP4), and extend the checksum oracle. Idempotent (marker strings),
env-gated: the behaviour changes only when IMPALA_RELOAD_FIX=1 is set in the serving process (the sleep-L2 recipe sets it).

Root cause (RESEARCH-5.md, step 0): sleep level 2 discards EVERY allocation of the "weights" memory pool. `reload_weights` refills
parameters/buffers in place (kernel-tensor copy), but tensors DERIVED at load time and held as plain attributes — the NVFP4 MoE
activation global scales `a1_gscale`/`a2_gscale` inside the fused-MoE quant config (read by the captured input-quant kernel) and
the MLA absorbed weights `W_UV`/`W_UK_T` (read by the captured bmm) — are re-derived into NEW tensors while the CUDA graphs keep
the OLD (now garbage) pointers. Eager steps use the new tensors, graph replays the stale ones -> HTTP 200 with garbage tokens.
Also: `expand(num_experts)` made the kernel-format MoE input scales stride-0 views, which Tensor.copy_ refuses as a destination.

Edits (all inert unless IMPALA_RELOAD_FIX=1, except the additive ones):
 1. flashinfer_fp4_moe.py: shared NVFP4 MoE input scales via `.repeat` instead of `.expand` (port of upstream #49489, b07ec92fa).
 2. model_executor/utils.py: `replace_parameter(..., prefer_copy=True)` copies into the existing parameter in place (port of
    upstream #40390, 2e777d21a8) — additive, default False.
 3. mla_attention.py: `W_UV`/`W_UK_T` become parameters refreshed in place on reload (port of upstream #48251, 17fdd4210).
 4. modelopt.py ModelOptNvFp4FusedMoE.process_weights_after_loading: on a RELOAD keep the MoE kernel the graphs were captured with
    and refresh its quant-config tensors in place from the freshly processed layer (no upstream equivalent as of 2026-10-11 main:
    upstream still rebuilds the kernel, so its `a1_gscale` goes stale after a level-2 sleep).
 5. reload/layerwise.py: overlap-tolerant kernel-tensor copy (defensive; the Exp-2 venv patch `reload_overlap_patch.py` did the
    same — this edit is skipped if that marker is present).
 6. base_loader.py checksum hook v2 (IMPALA_WEIGHT_CHECKSUM=<dir>): hashes parameters + buffers (digest `all_tensors_sha256`, same
    definition as Exp 4) AND derived tensors (plain CUDA tensor attributes of every module + fused-MoE quant-config descriptors +
    experts attributes; digest `derived_sha256`), records data_ptr per tensor; `Worker.impala_weight_checksum(tag)` exposes it to
    /collective_rpc so a report can be taken after wake_up + reload_weights (file wchk-rank<r>-<pid>-<tag>.json).
"""
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/venv/lib/python3.12/site-packages/vllm")
VERSION = 2


def rd(p):
    return open(p).read()


def wr(p, s):
    open(p, "w").write(s)


def sub1(s, old, new, what):
    n = s.count(old)
    assert n == 1, f"{what}: expected 1 anchor, found {n}"
    return s.replace(old, new, 1)


GATE_HELPER = '''

def _impala_reload_fix_enabled() -> bool:
    """Experiment 5 (Impala): IMPALA_RELOAD_FIX=1 enables the in-place reload fixes."""
    import os as _os

    return _os.environ.get("IMPALA_RELOAD_FIX", "0") == "1"
'''

# ---------------------------------------------------------------- 1. flashinfer_fp4_moe.py: expand -> repeat (gated)
p = os.path.join(root, "model_executor/layers/quantization/utils/flashinfer_fp4_moe.py")
s = rd(p)
if "_impala_broadcast_scale" in s:
    print("flashinfer_fp4_moe: already patched")
else:
    o13 = "a13_scale = a13_scale.max().to(torch.float32).expand(num_experts)"
    o2 = "a2_scale = a2_scale.max().to(torch.float32).expand(num_experts)"
    n = s.count(o13) + s.count(o2)
    assert s.count(o13) >= 1 and s.count(o2) >= 1, f"flashinfer_fp4_moe: expand anchors {s.count(o13)}/{s.count(o2)}"
    s = s.replace(o13, "a13_scale = _impala_broadcast_scale(a13_scale.max().to(torch.float32), num_experts)")
    s = s.replace(o2, "a2_scale = _impala_broadcast_scale(a2_scale.max().to(torch.float32), num_experts)")
    assert ".expand(num_experts)" not in s, "flashinfer_fp4_moe: an expand site of another shape remains"
    helper = GATE_HELPER + '''

def _impala_broadcast_scale(scalar: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Shared per-expert scale: a writable `.repeat` under the reload fix (upstream #49489), stock stride-0 `.expand` otherwise."""
    if _impala_reload_fix_enabled():
        return scalar.repeat(num_experts)
    return scalar.expand(num_experts)
'''
    anchor = "\n\ndef is_flashinfer_fp4_cutlass_moe_available("
    s = sub1(s, anchor, helper + anchor, "flashinfer_fp4_moe helper anchor")
    wr(p, s)
    print(f"flashinfer_fp4_moe: patched ({n} expand sites -> gated repeat)")

# ---------------------------------------------------------------- 2. utils.py: replace_parameter(prefer_copy=)
p = os.path.join(root, "model_executor/utils.py")
s = rd(p)
if "prefer_copy" in s:
    print("model_executor/utils: already patched (prefer_copy)")
else:
    s = sub1(
        s,
        "    layer: torch.nn.Module, param_name: str, new_data: torch.Tensor | None\n):\n",
        "    layer: torch.nn.Module,\n    param_name: str,\n    new_data: torch.Tensor | None,\n    prefer_copy: bool = False,\n):\n",
        "replace_parameter signature",
    )
    old = '''    if isinstance(new_data, torch.nn.Parameter):
        new_data = new_data.data
    new_param = torch.nn.Parameter(new_data, requires_grad=False)

    old_param: torch.nn.Parameter | None = getattr(layer, param_name, None)
'''
    new = '''    if isinstance(new_data, torch.nn.Parameter):
        new_data = new_data.data

    old_param: torch.nn.Parameter | None = getattr(layer, param_name, None)
    # Experiment 5 (Impala) / upstream #40390: copy in place so captured CUDA graphs keep a valid pointer across reloads
    if (
        prefer_copy
        and old_param is not None
        and old_param.shape == new_data.shape
        and old_param.dtype == new_data.dtype
        and old_param.device == new_data.device
    ):
        old_param.copy_(new_data)
        return

    new_param = torch.nn.Parameter(new_data, requires_grad=False)
'''
    s = sub1(s, old, new, "replace_parameter body")
    wr(p, s)
    print("model_executor/utils: patched (prefer_copy)")

# ---------------------------------------------------------------- 3. mla_attention.py: W_UV / W_UK_T in place (gated)
p = os.path.join(root, "model_executor/layers/attention/mla_attention.py")
s = rd(p)
if "_impala_set_absorbed" in s:
    print("mla_attention: already patched")
else:
    old = '''            # Convert from (L, N, V) to (N, L, V)
            self.W_UV = W_UV.transpose(0, 1)
            # Convert from (L, N, P) to (N, P, L)
            self.W_UK_T = W_UK.permute(1, 2, 0)
'''
    new = '''            # Convert from (L, N, V) to (N, L, V)
            _impala_set_absorbed(self, "W_UV", W_UV.transpose(0, 1))
            # Convert from (L, N, P) to (N, P, L)
            _impala_set_absorbed(self, "W_UK_T", W_UK.permute(1, 2, 0))
'''
    s = sub1(s, old, new, "mla W_UV/W_UK_T")
    helper = GATE_HELPER + '''

def _impala_set_absorbed(layer: nn.Module, name: str, value: torch.Tensor) -> None:
    """Experiment 5 (Impala) / upstream #48251: under the reload fix the absorbed MLA weights are parameters that a reload
    refreshes IN PLACE (the captured bmm keeps its pointer); stock behaviour rebinds a fresh tensor."""
    if not _impala_reload_fix_enabled():
        setattr(layer, name, value)
        return
    from vllm.model_executor.utils import replace_parameter

    old = getattr(layer, name, None)
    if old is not None and not isinstance(old, nn.Parameter):
        delattr(layer, name)
    replace_parameter(layer, name, value, prefer_copy=True)
'''
    anchor = "\n\nclass MLAAttention("
    s = sub1(s, anchor, helper + anchor, "mla class anchor")
    assert "import torch.nn as nn" in s or "from torch import nn" in s, "mla_attention: nn import not found"
    wr(p, s)
    print("mla_attention: patched (absorbed weights in place)")

# ---------------------------------------------------------------- 4. modelopt.py: keep the MoE kernel on reload (gated)
p = os.path.join(root, "model_executor/layers/quantization/modelopt.py")
s = rd(p)
if "_impala_refresh_quant_config_inplace" in s:
    print("modelopt: already patched")
else:
    old = '''        # Setup modular kernel.
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.experts_cls is not None
        self.moe_kernel = make_nvfp4_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            shared_experts=layer.shared_experts,
            routing_tables=layer._maybe_init_expert_routing_tables(),
        )
        self.moe_kernel.fused_experts.process_weights_after_loading(layer)
'''
    new = '''        # Setup modular kernel.
        _impala_old_kernel = getattr(self, "moe_kernel", None)
        _impala_old_qc = getattr(self, "moe_quant_config", None)
        new_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.experts_cls is not None
        new_kernel = make_nvfp4_moe_kernel(
            moe_quant_config=new_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            shared_experts=layer.shared_experts,
            routing_tables=layer._maybe_init_expert_routing_tables(),
        )
        new_kernel.fused_experts.process_weights_after_loading(layer)
        if (
            _impala_reload_fix_enabled()
            and _impala_old_kernel is not None
            and _impala_old_qc is not None
        ):
            # Experiment 5 (Impala): a RELOAD. Keep the kernel the CUDA graphs were captured with and refresh its
            # quant-config tensors in place (a1/a2_gscale are not layer parameters; the generic layerwise copy
            # only covers parameters/buffers). The layer's parameters are copied back into their original
            # storage by _copy_and_restore_kernel_tensors right after this returns.
            _impala_refresh_quant_config_inplace(_impala_old_qc, new_quant_config, layer)
            del new_kernel
        else:
            self.moe_quant_config = new_quant_config
            self.moe_kernel = new_kernel
'''
    s = sub1(s, old, new, "modelopt NVFP4 MoE kernel setup")
    helper = GATE_HELPER + '''

def _impala_overlap_copy_(dst: torch.Tensor, src: torch.Tensor) -> None:
    """copy_ that tolerates an expanded (stride-0) destination: write its unique storage elements once."""
    if torch._debug_has_internal_overlap(dst) == 0:
        dst.copy_(src)
        return
    n = dst.untyped_storage().nbytes() // dst.element_size()
    dst_flat = torch.as_strided(dst, (n,), (1,), 0)
    if src.shape == dst.shape and src.stride() == dst.stride() and src.untyped_storage().nbytes() == dst.untyped_storage().nbytes():
        src_flat = torch.as_strided(src, (n,), (1,), 0)
    else:
        idx = tuple(0 if st == 0 else slice(None) for st in dst.stride())
        src_flat = src[idx].contiguous().reshape(-1)
        if src_flat.numel() != n:
            raise RuntimeError(f"overlap copy: dst storage {n} elements, src sub-view {src_flat.numel()}")
        if not torch.equal(src, src[idx].expand_as(src)):
            raise RuntimeError("overlap copy: source is not uniform along the destination's stride-0 dims")
    dst_flat.copy_(src_flat)


def _impala_refresh_quant_config_inplace(old_qc, new_qc, layer: torch.nn.Module) -> None:
    """Experiment 5 (Impala): copy every tensor of the freshly built fused-MoE quant config into the tensor the LIVE quant
    config references (same object the captured kernels read). Parameters of the layer are refreshed again by the
    layerwise machinery afterwards (harmless). A shape/dtype mismatch rebinds the field and warns (graphs would be stale)."""
    n_copied = n_same = n_rebound = 0
    for desc_name in ("_a1", "_a2", "_w1", "_w2"):
        od = getattr(old_qc, desc_name, None)
        nd = getattr(new_qc, desc_name, None)
        if od is None or nd is None:
            continue
        for field in list(vars(od).keys()):
            ov = getattr(od, field, None)
            nv = getattr(nd, field, None)
            if not (isinstance(ov, torch.Tensor) and isinstance(nv, torch.Tensor)):
                continue
            if ov is nv:
                n_same += 1
                continue
            if ov.shape == nv.shape and ov.dtype == nv.dtype and ov.device == nv.device:
                _impala_overlap_copy_(ov, nv)
                n_copied += 1
            else:
                logger.warning(
                    "[impala-reload] %s.%s: shape/dtype changed on reload (%s/%s -> %s/%s); rebinding (captured graphs stale)",
                    desc_name, field, tuple(ov.shape), ov.dtype, tuple(nv.shape), nv.dtype,
                )
                object.__setattr__(od, field, nv)
                n_rebound += 1
    logger.info_once(
        "[impala-reload] fused-MoE quant config refreshed in place: %d tensors copied, %d shared, %d rebound",
        n_copied, n_same, n_rebound,
    )
'''
    anchor = "\n\nclass ModelOptNvFp4FusedMoE("
    s = sub1(s, anchor, helper + anchor, "modelopt class anchor")
    wr(p, s)
    print("modelopt: patched (MoE kernel kept on reload, quant config refreshed in place)")

# ---------------------------------------------------------------- 5. reload/layerwise.py: overlap-tolerant copy (defensive)
p = os.path.join(root, "model_executor/model_loader/reload/layerwise.py")
s = rd(p)
if "_impala_overlap_copy" in s:
    print("reload/layerwise: already patched (overlap copy)")
else:
    old = '''    parameters, buffers = info.kernel_tensors
    for name, param in parameters.items():
        param.data.copy_(getattr(layer, name))
    for name, buffer in buffers.items():
        buffer.data.copy_(getattr(layer, name))
'''
    new = '''    parameters, buffers = info.kernel_tensors
    for name, param in parameters.items():
        _impala_overlap_copy(param.data, getattr(layer, name), name)
    for name, buffer in buffers.items():
        _impala_overlap_copy(buffer.data, getattr(layer, name), name)
'''
    s = sub1(s, old, new, "layerwise copy loop")
    helper = '''

def _impala_overlap_copy(dst: torch.Tensor, src: torch.Tensor, name: str = "") -> None:
    """copy_ that tolerates an expanded (stride-0) destination: write its unique storage elements once (Experiment 2/5)."""
    if torch._debug_has_internal_overlap(dst) == 0:
        dst.copy_(src)
        return
    n = dst.untyped_storage().nbytes() // dst.element_size()
    dst_flat = torch.as_strided(dst, (n,), (1,), 0)
    if src.shape == dst.shape and src.stride() == dst.stride() and src.untyped_storage().nbytes() == dst.untyped_storage().nbytes():
        src_flat = torch.as_strided(src, (n,), (1,), 0)
    else:
        idx = tuple(0 if st == 0 else slice(None) for st in dst.stride())
        src_flat = src[idx].contiguous().reshape(-1)
        if src_flat.numel() != n:
            raise RuntimeError(f"overlap copy for {name}: dst storage {n} elements, src sub-view {src_flat.numel()}")
        if not torch.equal(src, src[idx].expand_as(src)):
            raise RuntimeError(f"overlap copy for {name}: source is not uniform along the destination's stride-0 dims")
    dst_flat.copy_(src_flat)
'''
    s = sub1(s, "\n\ndef _place_kernel_tensors(", helper + "\n\ndef _place_kernel_tensors(", "layerwise helper anchor")
    wr(p, s)
    print("reload/layerwise: patched (overlap-tolerant copy)")

# ---------------------------------------------------------------- 6. base_loader.py: checksum hook v2 + Worker RPC
p = os.path.join(root, "model_executor/model_loader/base_loader.py")
s = rd(p)
CHK_V2 = '''

_IMPALA_CHECKSUM_VERSION = 2


def _impala_h64(t: torch.Tensor) -> tuple[str, int]:
    """64-bit position-mixed hash of a CUDA tensor's bytes (device side), plus its byte count."""
    mask = (1 << 64) - 1
    chunk = 1 << 26
    u8 = t.detach().contiguous().view(-1).view(torch.uint8)
    nb = u8.numel()
    pad = (-nb) % 8
    if pad:
        u8 = torch.cat([u8, torch.zeros(pad, dtype=torch.uint8, device=u8.device)])
    x = u8.view(torch.int64)
    h = 0
    for off in range(0, x.numel(), chunk):
        c = x[off : off + chunk]
        idx = torch.arange(off, off + c.numel(), device=c.device, dtype=torch.int64)
        h = (h + int(((c * 0x5BD1E995) ^ (idx * 0x1B873593)).sum().item())) & mask
    return f"{h:016x}", nb


def _impala_derived_tensors(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    """Derived (non-parameter, non-buffer) CUDA tensors reachable from the modules: plain tensor attributes, fused-MoE quant
    config descriptors, and the experts objects' tensor attributes. These are what captured CUDA graphs read besides params."""
    out: list[tuple[str, torch.Tensor]] = []
    seen: set[int] = set()

    def add(name: str, t) -> None:
        if isinstance(t, torch.Tensor) and t.device.type == "cuda" and t.numel() > 0 and id(t) not in seen:
            seen.add(id(t))
            out.append((name, t))

    for mod_name, mod in model.named_modules():
        pb = set(mod._parameters.keys()) | set(mod._buffers.keys())
        for k, v in list(vars(mod).items()):
            if k in pb or k == "kv_cache":  # kv_cache: the attention layers' KV tensors (bound after load), not weights
                continue
            add(f"{mod_name}.<attr>{k}", v)
        qm = getattr(mod, "quant_method", None)
        qc = getattr(qm, "moe_quant_config", None)
        if qc is not None:
            for desc_name in ("_a1", "_a2", "_w1", "_w2"):
                d = getattr(qc, desc_name, None)
                if d is None:
                    continue
                for f, v in list(vars(d).items()):
                    add(f"{mod_name}.<qc>{desc_name}.{f}", v)
        kern = getattr(qm, "moe_kernel", None)
        ex = getattr(kern, "fused_experts", None)
        if ex is not None:
            for f, v in list(vars(ex).items()):
                add(f"{mod_name}.<experts>{f}", v)
    out.sort(key=lambda kv: kv[0])
    return out


def _impala_weight_checksum(model: nn.Module, tag: str = "load") -> str | None:
    """Experiment 4/5 (Impala): IMPALA_WEIGHT_CHECKSUM=<dir> -> write a per-rank report hashing every GPU parameter/buffer
    (digest `all_tensors_sha256`, Exp-4 definition) and every derived CUDA tensor (digest `derived_sha256`), with data_ptr per
    tensor, plus sha256 of a fixed sample. Returns the report path (None when the hook is inert)."""
    import hashlib
    import json
    import os
    import time

    out_dir = os.environ.get("IMPALA_WEIGHT_CHECKSUM")
    if not out_dir:
        return None
    from vllm.distributed import get_tensor_model_parallel_rank

    t0 = time.time()
    rank = get_tensor_model_parallel_rank()
    items = [(n, p.data) for n, p in model.named_parameters()]
    items += [(n, b) for n, b in model.named_buffers()]
    items.sort(key=lambda kv: kv[0])
    res: dict[str, dict] = {}
    total = 0
    for name, t in items:
        entry = {"shape": list(t.shape), "dtype": str(t.dtype), "dev": str(t.device)}
        if t.device.type == "cuda" and t.numel() > 0:
            h, nb = _impala_h64(t)
            entry["nbytes"] = nb
            entry["h64"] = h
            entry["ptr"] = t.data_ptr()
            total += nb
        res[name] = entry
    names = [n for n, t in items if t.device.type == "cuda" and t.numel() > 0]
    k = max(1, len(names) // 20)
    chosen = list(dict.fromkeys(names[::k][:20] + [n for n in names if "scale" in n][:6]))
    by_name = dict(items)
    sample = {}
    for n in chosen:
        t = by_name[n]
        sample[n] = hashlib.sha256(
            t.detach().contiguous().view(-1).view(torch.uint8).cpu().numpy().tobytes()
        ).hexdigest()
    # digest over h64/shape/dtype only (ptr excluded) so it is comparable across processes
    digest_src = {n: {k2: v for k2, v in e.items() if k2 != "ptr"} for n, e in res.items()}
    digest = hashlib.sha256(json.dumps(digest_src, sort_keys=True).encode()).hexdigest()
    derived: dict[str, dict] = {}
    dtotal = 0
    for name, t in _impala_derived_tensors(model):
        h, nb = _impala_h64(t)
        derived[name] = {"shape": list(t.shape), "dtype": str(t.dtype), "nbytes": nb, "h64": h, "ptr": t.data_ptr()}
        dtotal += nb
    dsrc = {n: {k2: v for k2, v in e.items() if k2 != "ptr"} for n, e in derived.items()}
    ddigest = hashlib.sha256(json.dumps(dsrc, sort_keys=True).encode()).hexdigest()
    report = {
        "version": _IMPALA_CHECKSUM_VERSION,
        "tag": tag,
        "rank": rank,
        "pid": os.getpid(),
        "n_tensors": len(res),
        "n_hashed": len(names),
        "gpu_bytes": total,
        "all_tensors_sha256": digest,
        "n_derived": len(derived),
        "derived_bytes": dtotal,
        "derived_sha256": ddigest,
        "sample_sha256": sample,
        "tensors": res,
        "derived": derived,
        "elapsed_s": round(time.time() - t0, 1),
        "pinned_loader": os.environ.get("IMPALA_PINNED_LOADER", "0"),
        "reload_fix": os.environ.get("IMPALA_RELOAD_FIX", "0"),
    }
    os.makedirs(out_dir, exist_ok=True)
    suffix = "" if tag == "load" else f"-{tag}"
    path = os.path.join(out_dir, f"wchk-rank{rank}-{os.getpid()}{suffix}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(report, f, indent=0)
    os.replace(path + ".tmp", path)
    logger.info(
        "[impala-checksum] v2 tag=%s rank %d: %d tensors / %.1f GiB hashed, all-tensors %s, derived %d / %.1f MiB %s (%.1f s) -> %s",
        tag, rank, len(names), total / 1024**3, digest[:16], len(derived), dtotal / 1024**2, ddigest[:16],
        time.time() - t0, path,
    )
    return path
'''
if "_IMPALA_CHECKSUM_VERSION = 2" in s:
    print("base_loader: already patched (checksum v2)")
else:
    i = s.find("\n\ndef _impala_weight_checksum(")
    if i < 0:
        # fresh tree without the Exp-4 hook: add the call site too
        old = '''            process_weights_after_loading(model, model_config, target_device)

        return model.eval()
'''
        new = '''            process_weights_after_loading(model, model_config, target_device)

            _impala_weight_checksum(model)

        return model.eval()
'''
        s = sub1(s, old, new, "base_loader call site")
        s = s.rstrip("\n") + "\n" + CHK_V2
    else:
        s = s[:i].rstrip("\n") + "\n" + CHK_V2
    wr(p, s)
    print("base_loader: patched (checksum hook v2)")

p = os.path.join(root, "v1/worker/gpu_worker.py")
s = rd(p)
if "def impala_weight_checksum" in s:
    print("gpu_worker: already patched")
else:
    old = '''    def reload_weights(self, *args, **kwargs) -> None:
        self.model_runner.reload_weights(*args, **kwargs)
'''
    new = '''    def reload_weights(self, *args, **kwargs) -> None:
        self.model_runner.reload_weights(*args, **kwargs)

    def impala_weight_checksum(self, tag: str = "rpc") -> str:
        """Experiment 5 (Impala): write the weight/derived-tensor checksum report (IMPALA_WEIGHT_CHECKSUM=<dir>) via RPC."""
        from vllm.model_executor.model_loader.base_loader import _impala_weight_checksum

        torch.cuda.synchronize()
        return _impala_weight_checksum(self.model_runner.model, tag=str(tag)) or ""
'''
    s = sub1(s, old, new, "gpu_worker reload_weights")
    wr(p, s)
    print("gpu_worker: patched (impala_weight_checksum RPC)")

# ---------------------------------------------------------------- 7. KV-init runtime state survives sleep (port of upstream #59158)
# In 0.20.2 the whole initialize_kv_cache (attention backends, metadata builders with their persistent buffers, the InputBatch)
# runs inside the "kv_cache" pool, so a sleep of any level DISCARDS that runtime state and wake_up remaps it blank — only the
# per-step-rewritten parts recover; padding rows / once-initialised buffers come back as garbage (nondeterministic output after
# wake + reload on the FlashInfer sparse-MLA path). Upstream main allocates it under a "runtime" tag that is backed up to CPU at
# every sleep level, with only the KV tensors in "kv_cache". Gated by IMPALA_RELOAD_FIX=1; the backup is pageable host memory
# (IMPALA_SLEEP_PAGEABLE_BACKUP, default 1 under the fix) so a CRIU/cuda-checkpoint image never carries cudaHostAlloc'd pages.
p = os.path.join(root, "v1/worker/gpu_worker.py")
s = rd(p)
if "_impala_runtime_tag" in s:
    print("gpu_worker: already patched (runtime pool)")
else:
    old = '''            allocator = CuMemAllocator.get_instance()
            with allocator.use_memory_pool(tag="kv_cache"):
                self.model_runner.initialize_kv_cache(kv_cache_config)
'''
    new = '''            allocator = CuMemAllocator.get_instance()
            with allocator.use_memory_pool(tag=_impala_runtime_tag()):
                self.model_runner.initialize_kv_cache(kv_cache_config)
'''
    s = sub1(s, old, new, "gpu_worker kv_cache pool")
    old = '''        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
'''
    new = '''        allocator = CuMemAllocator.get_instance()
        if _impala_runtime_tag() == "runtime":
            # Experiment 5 (Impala) / upstream #59158: KV-init runtime state is kept at every level
            allocator.sleep(offload_tags=("weights", "runtime") if level == 1 else ("runtime",))
        else:
            allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
'''
    s = sub1(s, old, new, "gpu_worker sleep tags")
    helper = '''

def _impala_runtime_tag() -> str:
    """Experiment 5 (Impala): "runtime" (offloaded on sleep) under IMPALA_RELOAD_FIX=1, stock "kv_cache" otherwise."""
    import os as _os

    return "runtime" if _os.environ.get("IMPALA_RELOAD_FIX", "0") == "1" else "kv_cache"
'''
    anchor = "\n\nclass Worker("
    s = sub1(s, anchor, helper + anchor, "gpu_worker class anchor")
    wr(p, s)
    print("gpu_worker: patched (runtime pool + sleep tags)")

p = os.path.join(root, "v1/worker/gpu_model_runner.py")
s = rd(p)
if "_impala_kv_tensor_pool" in s:
    print("gpu_model_runner: already patched (nested kv_cache pool)")
else:
    old = '''        self.may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)
        kv_caches = self.initialize_kv_cache_tensors(
            kv_cache_config, kernel_block_sizes
        )
'''
    new = '''        self.may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)
        with _impala_kv_tensor_pool(self.vllm_config):
            kv_caches = self.initialize_kv_cache_tensors(
                kv_cache_config, kernel_block_sizes
            )
'''
    s = sub1(s, old, new, "gpu_model_runner kv tensors call")
    helper = '''

def _impala_kv_tensor_pool(vllm_config):
    """Experiment 5 (Impala) / upstream #59158: with the reload fix on, the worker runs initialize_kv_cache inside a "runtime"
    pool (backed up on sleep); only the KV tensors themselves go into the discardable "kv_cache" pool (nested)."""
    import os as _os
    from contextlib import nullcontext

    if _os.environ.get("IMPALA_RELOAD_FIX", "0") != "1" or not vllm_config.model_config.enable_sleep_mode:
        return nullcontext()
    from vllm.device_allocator.cumem import CuMemAllocator

    return CuMemAllocator.get_instance().use_memory_pool(tag="kv_cache")
'''
    anchor = "\n\nclass GPUModelRunner("
    s = sub1(s, anchor, helper + anchor, "gpu_model_runner class anchor")
    wr(p, s)
    print("gpu_model_runner: patched (nested kv_cache pool)")

p = os.path.join(root, "device_allocator/cumem.py")
s = rd(p)
if "_impala_pin_backup" in s:
    print("cumem: already patched (pageable backup)")
else:
    old = '''                cpu_backup_tensor = torch.empty(
                    size_in_bytes,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=is_pin_memory_available(),
                )
'''
    new = '''                cpu_backup_tensor = torch.empty(
                    size_in_bytes,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=_impala_pin_backup(),
                )
'''
    s = sub1(s, old, new, "cumem backup alloc")
    helper = '''

def _impala_pin_backup() -> bool:
    """Experiment 5 (Impala): pageable CPU backups under the reload fix (IMPALA_SLEEP_PAGEABLE_BACKUP=1 default) so a
    checkpointed sleeping process carries no cudaHostAlloc'd pages; stock pinned backups otherwise."""
    import os as _os

    fix = _os.environ.get("IMPALA_RELOAD_FIX", "0") == "1"
    if _os.environ.get("IMPALA_SLEEP_PAGEABLE_BACKUP", "1" if fix else "0") == "1":
        return False
    return is_pin_memory_available()
'''
    anchor = "\n\nclass CuMemAllocator:"
    s = sub1(s, anchor, helper + anchor, "cumem class anchor")
    wr(p, s)
    print("cumem: patched (pageable backup option)")

# ---------------------------------------------------------------- 8. layerwise reload: process a layer on LAYER SWITCH, not on count
# The layerwise trigger fires as soon as the counted elements reach the layer's total. For the ModelOpt NVFP4 FusedMoE the count
# overshoots by 2 elements per expert (a per-tensor scale loader writes both shard columns per call), so the trigger can fire
# while the layer's last per-tensor scale loads are still in flight; those arrive after the layer was processed and reset and
# are dropped ("Excessive loading") → uninitialised w13/w2 weight_scale_2 / input_scale for the tail experts of some layers
# (measured: 16–384 dropped loads on 2–4 of 75 MoE layers per reload, different layers each time → garbage output). Under the
# fix a ready layer is only processed when a load for a DIFFERENT layer arrives (or at finalize), so late loads of the same
# layer are still buffered and applied. At most one extra materialised layer is alive at any time.
p = os.path.join(root, "model_executor/model_loader/reload/layerwise.py")
s = rd(p)
if "_IMPALA_PENDING" in s:
    print("reload/layerwise: already patched (layer-switch trigger)")
else:
    old = '''        # Process and copy when all weights are loaded
        if info.load_numel >= info.load_numel_total and not isinstance(  # type: ignore[operator]
            layer, (Attention, MLAAttention)
        ):
            _layerwise_process(layer, info)

        return ret
'''
    new = '''        # Process and copy when all weights are loaded
        if _impala_reload_fix_enabled():
            # Experiment 5 (Impala): defer processing of a ready layer until a load for another layer arrives, so a
            # count overshoot cannot drop this layer's late loads (see _IMPALA_PENDING).
            _impala_flush_pending(except_info=info)
            if info.load_numel >= info.load_numel_total and not isinstance(  # type: ignore[operator]
                layer, (Attention, MLAAttention)
            ):
                _IMPALA_PENDING[id(info)] = (layer, info)
            return ret
        if info.load_numel >= info.load_numel_total and not isinstance(  # type: ignore[operator]
            layer, (Attention, MLAAttention)
        ):
            _layerwise_process(layer, info)

        return ret
'''
    s = sub1(s, old, new, "layerwise trigger")
    old = '''    deferred_attn: list[tuple[torch.nn.Module, LayerReloadingInfo]] = []
'''
    new = '''    _impala_flush_pending(except_info=None)

    deferred_attn: list[tuple[torch.nn.Module, LayerReloadingInfo]] = []
'''
    s = sub1(s, old, new, "layerwise finalize flush")
    helper = '''

_IMPALA_PENDING: dict[int, tuple[torch.nn.Module, LayerReloadingInfo]] = {}


def _impala_reload_fix_enabled() -> bool:
    import os as _os

    return _os.environ.get("IMPALA_RELOAD_FIX", "0") == "1"


def _impala_flush_pending(except_info) -> None:
    """Experiment 5 (Impala): process every ready layer except the one currently receiving loads."""
    for key in list(_IMPALA_PENDING.keys()):
        layer, info = _IMPALA_PENDING[key]
        if info is except_info:
            continue
        del _IMPALA_PENDING[key]
        if info.can_load():
            _layerwise_process(layer, info)
'''
    s = sub1(s, "\n\ndef get_layerwise_info(", helper + "\n\ndef get_layerwise_info(", "layerwise helper anchor")
    wr(p, s)
    print("reload/layerwise: patched (layer-switch trigger, pending flush at finalize)")

# ---------------------------------------------------------------- 9. exact layerwise totals measured at the FIRST load
# The reload trigger compares the counted load elements with the layer's parameter numel; the ModelOpt NVFP4 MoE loaders write
# some per-tensor scale rows twice (+512 per layer), and the Run:ai streamer yields tensors in chunk-completion order, so a
# layer can hit its total while a few of its scalar loads are still in flight (dropped as "Excessive loading"). Under the fix,
# the first load (BaseModelLoader.load_model) counts every layer's loaded elements with the same CopyCounter and stores the
# measured total on the layer; the reload uses it as the trigger total, so processing fires exactly on the last load of the
# layer regardless of arrival order. Layers with no measured loads keep the stock total.
p = os.path.join(root, "model_executor/model_loader/base_loader.py")
s = rd(p)
if "_impala_count_first_load" in s:
    print("base_loader: already patched (first-load counting)")
else:
    old = '''            logger.debug("Loading weights on %s ...", load_device)
            self.load_weights(model, model_config)
'''
    new = '''            logger.debug("Loading weights on %s ...", load_device)
            with _impala_count_first_load(model):
                self.load_weights(model, model_config)
'''
    s = sub1(s, old, new, "base_loader load_weights call")
    helper = '''

@contextlib.contextmanager
def _impala_count_first_load(model: nn.Module):
    """Experiment 5 (Impala): count every layer's loaded elements during the first load (IMPALA_RELOAD_FIX=1) and store the
    measured total as layer._impala_first_load_numel, the exact trigger total for a later layerwise reload."""
    import os as _os

    if _os.environ.get("IMPALA_RELOAD_FIX", "0") != "1":
        yield
        return
    import inspect

    from vllm.model_executor.model_loader.reload.meta import SKIP_TENSORS, get_numel_loaded
    from vllm.model_executor.model_loader.reload.utils import get_layer_tensors
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    counts: dict[int, int] = {}
    wrapped: list[tuple[torch.Tensor, object]] = []

    def make(layer, tensor):
        original = getattr(tensor, "weight_loader", default_weight_loader)
        sig = inspect.signature(original)

        def counting_loader(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            n, ret = get_numel_loaded(original, bound)
            counts[id(layer)] = counts.get(id(layer), 0) + n
            return ret

        return original, counting_loader

    layers = list(model.modules())
    for layer in layers:
        for name, tensor in get_layer_tensors(layer).items():
            if name in SKIP_TENSORS:
                continue
            original, cl = make(layer, tensor)
            tensor.weight_loader = cl
            wrapped.append((tensor, original))
    try:
        yield
    finally:
        for tensor, original in wrapped:
            try:
                if original is default_weight_loader and "weight_loader" not in tensor.__dict__:
                    continue
                tensor.weight_loader = original
            except Exception:  # noqa: BLE001
                pass
        n_layers = 0
        for layer in layers:
            c = counts.get(id(layer), 0)
            if c > 0:
                layer._impala_first_load_numel = c
                n_layers += 1
        logger.info("[impala-reload] first-load totals recorded for %d layers", n_layers)
'''
    s = s.rstrip("\n") + "\n" + helper
    if "import contextlib" not in s:
        s = s.replace("import torch\n", "import contextlib\nimport torch\n", 1)
    wr(p, s)
    print("base_loader: patched (first-load counting)")

p = os.path.join(root, "model_executor/model_loader/reload/layerwise.py")
s = rd(p)
if "_impala_first_load_numel" in s:
    print("reload/layerwise: already patched (measured totals)")
else:
    old = '''    info.load_numel = 0
    info.load_numel_total = get_layer_size(layer)
'''
    new = '''    info.load_numel = 0
    info.load_numel_total = get_layer_size(layer)
    _measured = getattr(layer, "_impala_first_load_numel", None)
    if _impala_reload_fix_enabled() and _measured:
        # Experiment 5 (Impala): exact trigger total measured at the first load (see base_loader._impala_count_first_load)
        info.load_numel_total = _measured
'''
    s = sub1(s, old, new, "layerwise total override")
    wr(p, s)
    print("reload/layerwise: patched (measured totals)")

# ---------------------------------------------------------------- 10. diagnostic RPC: zero the KV cache tensors
p = os.path.join(root, "v1/worker/gpu_worker.py")
s = rd(p)
if "def impala_memory_stats" in s:
    print("gpu_worker: already patched (zero kv rpc)")
else:
    old = '''    def impala_weight_checksum(self, tag: str = "rpc") -> str:
'''
    new = '''    def impala_zero_kv_cache(self) -> str:
        """Experiment 5 (Impala, diagnostic): zero every KV cache tensor (after wake_up the remapped pages hold leftovers)."""
        n = 0
        for t in getattr(self.model_runner, "kv_caches", []) or []:
            if isinstance(t, torch.Tensor) and t.numel() > 0:
                t.zero_()
                n += t.numel() * t.element_size()
        torch.cuda.synchronize()
        return f"zeroed {n / 2**30:.1f} GiB"

    def impala_memory_stats(self) -> str:
        """Experiment 5 (Impala, diagnostic): allocator state + KV cache bytes bound (json)."""
        import json as _json

        st = torch.cuda.memory_stats()
        free, total = torch.cuda.mem_get_info()
        kv = sum(t.numel() * t.element_size() for t in (getattr(self.model_runner, "kv_caches", []) or []) if isinstance(t, torch.Tensor))
        keys = ("allocated_bytes.all.current", "reserved_bytes.all.current", "inactive_split_bytes.all.current",
                "segment.all.current", "num_alloc_retries", "num_ooms", "active_bytes.all.current")
        out = {k: int(st.get(k, -1)) for k in keys}
        out.update({"kv_cache_bytes": int(kv), "device_used_bytes": int(total - free)})
        return _json.dumps(out)

    def impala_empty_cache(self) -> str:
        """Experiment 5 (Impala, diagnostic): torch.cuda.empty_cache(); returns reserved bytes before/after."""
        before = int(torch.cuda.memory_stats().get("reserved_bytes.all.current", -1))
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        after = int(torch.cuda.memory_stats().get("reserved_bytes.all.current", -1))
        return f"reserved {before / 2**30:.1f} -> {after / 2**30:.1f} GiB"

    def impala_weight_checksum(self, tag: str = "rpc") -> str:
'''
    s = sub1(s, old, new, "gpu_worker zero kv anchor")
    wr(p, s)
    print("gpu_worker: patched (zero kv rpc)")

print(f"RELOAD_FIX_PATCH v{VERSION} DONE")
