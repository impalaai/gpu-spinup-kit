#!/usr/bin/env python3
"""reload_fix_patch_031.py [vllm-package-dir] — Experiment 6 (2026-10-11): the Experiment-5 in-place `reload_weights` fixes
(patches/0003 + 0004 on vLLM 0.20.2) ported to vLLM 0.31.0. Idempotent (marker strings); every behaviour change is gated by
IMPALA_RELOAD_FIX=1 in the serving process (the sleep-L2 recipe sets it), the checksum hook by IMPALA_WEIGHT_CHECKSUM=<dir>.

Already upstream in 0.31.0 (not ported): §1 expand→repeat (#49489), §2 replace_parameter(prefer_copy=) (#40390), §3 MLA W_UV/W_UK_T in
place (#48251). Not needed for correctness (RESEARCH-5): §7 KV-init runtime pool (#59158, only in 0.31.1rc0 anyway).
Ported:
 4. modelopt.py ModelOptNvFp4FusedMoE._build_moe_kernel: on a RELOAD keep the MoE kernel the CUDA graphs were captured with and
    refresh its quant-config tensors (a1/a2 gscales etc.) in place — upstream 0.31.0 still rebuilds the kernel on every
    process_weights_after_loading (modelopt.py:1030–1041), so a1_gscale goes stale after a level-2 sleep.
 5. reload/layerwise.py: overlap-tolerant kernel-tensor copy (defensive).
 6. base_loader.py checksum hook v2 (params/buffers + derived CUDA tensors, data_ptr) + Worker RPC impala_weight_checksum(tag).
 8. reload/layerwise.py: process a ready layer on LAYER SWITCH (pending set flushed at finalize) instead of on the count.
 9. base_loader.py: measure each layer's exact loaded-element count at the FIRST load (IMPALA_RELOAD_FIX=1) and use it as the reload
    trigger total (layerwise.py sets load_numel_total at TWO sites in 0.31.0: initialize_online_processing and the per-load
    refresh inside online_process_loader; both routed through _impala_layer_total).
10. gpu_worker.py diagnostic RPCs: impala_zero_kv_cache, impala_memory_stats, impala_empty_cache.
0004. runai_streamer_loader.py: get_all_weights so reload_weights can use the Run:ai (+ pinned) path.
"""
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/venv-new/lib/python3.12/site-packages/vllm")
VERSION = 1


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
    """Experiment 5/6 (Impala): IMPALA_RELOAD_FIX=1 enables the in-place reload fixes."""
    import os as _os

    return _os.environ.get("IMPALA_RELOAD_FIX", "0") == "1"
'''

# ---------------------------------------------------------------- 4. modelopt.py: keep the MoE kernel on reload (gated)
p = os.path.join(root, "model_executor/layers/quantization/modelopt.py")
s = rd(p)
if "_impala_refresh_quant_config_inplace" in s:
    print("modelopt: already patched")
else:
    old = '''    def _build_moe_kernel(self, layer: RoutedExperts) -> None:
        """Build the modular MoE kernel from the (already in-format) weights."""
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.experts_cls is not None
        self.moe_kernel = make_nvfp4_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            backend=self.nvfp4_backend,
            routing_tables=layer._expert_routing_tables(),
        )
        self.moe_kernel.fused_experts.process_weights_after_loading(layer)
'''
    new = '''    def _build_moe_kernel(self, layer: RoutedExperts) -> None:
        """Build the modular MoE kernel from the (already in-format) weights."""
        _impala_old_kernel = getattr(self, "moe_kernel", None)
        _impala_old_qc = getattr(self, "moe_quant_config", None)
        new_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.experts_cls is not None
        new_kernel = make_nvfp4_moe_kernel(
            moe_quant_config=new_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            backend=self.nvfp4_backend,
            routing_tables=layer._expert_routing_tables(),
        )
        new_kernel.fused_experts.process_weights_after_loading(layer)
        if (
            _impala_reload_fix_enabled()
            and _impala_old_kernel is not None
            and _impala_old_qc is not None
        ):
            # Experiment 5/6 (Impala): a RELOAD. Keep the kernel the CUDA graphs were captured with and refresh its
            # quant-config tensors in place (a1/a2_gscale are not layer parameters; the generic layerwise copy
            # only covers parameters/buffers). The layer's parameters are copied back into their original
            # storage by _copy_and_restore_kernel_tensors right after this returns.
            _impala_refresh_quant_config_inplace(_impala_old_qc, new_quant_config, layer)
            del new_kernel
        else:
            self.moe_quant_config = new_quant_config
            self.moe_kernel = new_kernel
'''
    s = sub1(s, old, new, "modelopt _build_moe_kernel")
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
    """Experiment 5/6 (Impala): copy every tensor of the freshly built fused-MoE quant config into the tensor the LIVE quant
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
    old = '''    for name, param in parameters.items():
        param.data.copy_(getattr(layer, name))
    for name, buffer in buffers.items():
        if name not in layer._buffers:
            continue
        if name in non_persistent and name not in loaded_tensor_names:
            continue
        buffer.data.copy_(getattr(layer, name))
'''
    new = '''    for name, param in parameters.items():
        _impala_overlap_copy(param.data, getattr(layer, name), name)
    for name, buffer in buffers.items():
        if name not in layer._buffers:
            continue
        if name in non_persistent and name not in loaded_tensor_names:
            continue
        _impala_overlap_copy(buffer.data, getattr(layer, name), name)
'''
    s = sub1(s, old, new, "layerwise copy loop")
    helper = '''

def _impala_overlap_copy(dst: torch.Tensor, src: torch.Tensor, name: str = "") -> None:
    """copy_ that tolerates an expanded (stride-0) destination: write its unique storage elements once (Experiment 2/5/6)."""
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
    """Experiment 4/5/6 (Impala): IMPALA_WEIGHT_CHECKSUM=<dir> -> write a per-rank report hashing every GPU parameter/buffer
    (digest `all_tensors_sha256`) and every derived CUDA tensor (digest `derived_sha256`), with data_ptr per tensor, plus
    sha256 of a fixed sample. Returns the report path (None when the hook is inert)."""
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
        # replace the Exp-4 v1 hook (pinned_loader_patch_031.py) with v2
        s = s[:i].rstrip("\n") + "\n" + CHK_V2
    wr(p, s)
    print("base_loader: patched (checksum hook v2)")

p = os.path.join(root, "v1/worker/gpu_worker.py")
s = rd(p)
if "def impala_weight_checksum" in s:
    print("gpu_worker: already patched (rpcs)")
else:
    old = '''    def reload_weights(self, *args, **kwargs) -> None:
        with set_current_vllm_config(self.vllm_config):
            self.model_runner.reload_weights(*args, **kwargs)
'''
    new = '''    def reload_weights(self, *args, **kwargs) -> None:
        with set_current_vllm_config(self.vllm_config):
            self.model_runner.reload_weights(*args, **kwargs)

    def impala_weight_checksum(self, tag: str = "rpc") -> str:
        """Experiment 5/6 (Impala): write the weight/derived-tensor checksum report (IMPALA_WEIGHT_CHECKSUM=<dir>) via RPC."""
        from vllm.model_executor.model_loader.base_loader import _impala_weight_checksum

        torch.cuda.synchronize()
        return _impala_weight_checksum(self.model_runner.model, tag=str(tag)) or ""

    def impala_zero_kv_cache(self) -> str:
        """Experiment 5 (Impala, diagnostic): zero every KV cache tensor."""
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
'''
    s = sub1(s, old, new, "gpu_worker reload_weights")
    wr(p, s)
    print("gpu_worker: patched (checksum + diagnostic RPCs)")

# ---------------------------------------------------------------- 8. layerwise reload: process a layer on LAYER SWITCH
p = os.path.join(root, "model_executor/model_loader/reload/layerwise.py")
s = rd(p)
if "_IMPALA_PENDING" in s:
    print("reload/layerwise: already patched (layer-switch trigger)")
else:
    old = '''        # Process and copy when all weights are loaded
        if info.load_numel >= info.load_numel_total:  # type: ignore[operator]
            _layerwise_process(layer, info)
            LOADING_LAYERS.discard(layer)

        return ret
'''
    new = '''        # Process and copy when all weights are loaded
        if _impala_reload_fix_enabled():
            # Experiment 5/6 (Impala): defer processing of a ready layer until a load for another layer arrives, so a
            # count overshoot cannot drop this layer's late loads (see _IMPALA_PENDING).
            _impala_flush_pending(except_info=info)
            if info.load_numel >= info.load_numel_total:  # type: ignore[operator]
                _IMPALA_PENDING[id(info)] = (layer, info)
            return ret
        if info.load_numel >= info.load_numel_total:  # type: ignore[operator]
            _layerwise_process(layer, info)
            LOADING_LAYERS.discard(layer)

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
    """Experiment 5/6 (Impala): process every ready layer except the one currently receiving loads."""
    for key in list(_IMPALA_PENDING.keys()):
        layer, info = _IMPALA_PENDING[key]
        if info is except_info:
            continue
        del _IMPALA_PENDING[key]
        if info.can_load():
            _layerwise_process(layer, info)
            LOADING_LAYERS.discard(layer)


def _impala_layer_total(layer: torch.nn.Module) -> int:
    """Experiment 5/6 (Impala): the exact trigger total measured at the first load (base_loader._impala_count_first_load)
    under IMPALA_RELOAD_FIX=1; the stock parameter numel otherwise."""
    _measured = getattr(layer, "_impala_first_load_numel", None)
    if _impala_reload_fix_enabled() and _measured:
        return _measured
    return get_layer_size(layer)
'''
    s = sub1(s, "\n\ndef get_layerwise_info(", helper + "\n\ndef get_layerwise_info(", "layerwise helper anchor")
    wr(p, s)
    print("reload/layerwise: patched (layer-switch trigger, pending flush at finalize)")

# ---------------------------------------------------------------- 9. exact layerwise totals measured at the FIRST load
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
    """Experiment 5/6 (Impala): count every layer's loaded elements during the first load (IMPALA_RELOAD_FIX=1) and store the
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
if "_impala_layer_total(layer)" in s:
    print("reload/layerwise: already patched (measured totals)")
else:
    import re as _re

    pat = _re.compile(r"^([ \t]*)info\.load_numel_total = get_layer_size\(layer\)\n", _re.M)
    n = len(pat.findall(s))
    assert n == 2, f"layerwise total sites: expected 2 (init + per-load refresh), found {n}"
    s = pat.sub(r"\1info.load_numel_total = _impala_layer_total(layer)\n", s)
    wr(p, s)
    print("reload/layerwise: patched (measured totals at both sites)")

# ---------------------------------------------------------------- 0004. Run:ai loader get_all_weights
p = os.path.join(root, "model_executor/model_loader/runai_streamer_loader.py")
s = rd(p)
if "def get_all_weights" in s:
    print("runai_streamer_loader: already patched")
else:
    old = '''    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into a model."""
'''
    new = '''    def get_all_weights(
        self, model_config: ModelConfig, model: nn.Module
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Experiment 5/6 (Impala): the iterator `reload_weights` needs, same source and path as the first load."""
        model_weights = model_config.model
        if model_weights_override := model_config.model_weights:
            model_weights = model_weights_override
        yield from self._get_weights_iterator(model_weights, model_config.revision)

''' + old
    s = sub1(s, old, new, "runai load_weights anchor")
    assert "Generator" in s and "import torch" in s, "runai_streamer_loader: expected imports missing"
    wr(p, s)
    print("runai_streamer_loader: patched (get_all_weights for reload_weights)")

print(f"RELOAD_FIX_PATCH_031 v{VERSION} DONE")
