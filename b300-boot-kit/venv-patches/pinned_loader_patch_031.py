#!/usr/bin/env python3
"""pinned_loader_patch_031.py [vllm-package-dir] — Experiment 6 (2026-10-11): the Experiment-4 pinned-staging loader ported to vLLM 0.31.0
(anchors: the Run:ai iterator now ends in a per-tensor `.clone()` loop, #43464; under GPU-yield the clone is skipped).
Idempotent (marker strings), like fi_numsms_patch.py. Default target: ~/venv/lib/python3.12/site-packages/vllm.

Two edits:
 1. weight_utils.runai_safetensors_weights_iterator: when IMPALA_PINNED_LOADER=1, page-lock the Run:ai streamer's CPU buffer
    (parallel MADV_POPULATE_WRITE + cudaHostRegister, ~2-3 s for 40 GB) so every host->device copy out of it is a pinned DMA
    copy instead of a pageable staged memcpy; with IMPALA_PINNED_GPU_YIELD=1 (default) the streamer moves each tensor to the
    worker's GPU itself (`device=cuda:N`, a DMA from the pinned buffer) and the model's weight_loaders consume GPU tensors, so
    every TP narrow/copy is device-to-device. Bounded buffer: RUNAI_STREAMER_MEMORY_LIMIT defaults to IMPALA_PINNED_BUFFER_BYTES
    (40e9) when unset, also in distributed mode (which otherwise sizes the buffer to the whole rank partition).
    Unregisters the buffer when the iterator closes. Falls back to the stock path on any failure (logged).
 2. base_loader.BaseModelLoader.load_model: when IMPALA_WEIGHT_CHECKSUM=<dir>, after process_weights_after_loading write
    <dir>/wchk-rank<r>-<pid>.json with a 64-bit position-mixed hash of EVERY GPU parameter/buffer plus sha256 of a fixed sample
    (20 evenly spaced tensors + up to 6 *scale* tensors) — the bit-identity proof between the stock and the pinned path.
"""
import os, sys

root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/venv/lib/python3.12/site-packages/vllm")
wu = os.path.join(root, "model_executor/model_loader/weight_utils.py")
bl = os.path.join(root, "model_executor/model_loader/base_loader.py")

# ---------------------------------------------------------------- 1. weight_utils.py
PIN_VERSION = 3   # v3: vLLM 0.31.0 anchors (clone loop); v2: MADV_HUGEPAGE before the parallel populate
s = open(wu).read()
helper = '''
_IMPALA_PINNED_VERSION = %d


def _impala_pin_streamer_buffer(streamer) -> int | None:
    """Experiment 4 (Impala): page-lock the Run:ai streamer's CPU staging buffer so that host->device copies out of it are
    DMA (pinned) copies. Returns the registered address (for cudaHostUnregister) or None if the stock path stays."""
    import ctypes
    import threading
    import time

    try:
        buf = streamer.file_streamer.file_streamer.requests_iterator.buffer  # np.ndarray[uint8]
    except AttributeError as e:
        logger.warning("[impala-pinned] streamer buffer not found (%%s); staying on the pageable path", e)
        return None
    nbytes = int(buf.nbytes)
    if nbytes == 0:
        return None
    addr = int(buf.ctypes.data)
    t0 = time.time()
    try:
        # populate the pages in parallel without touching their content (the streamer threads may already be writing)
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        page = 4096
        madv_populate_write = 23
        madv_hugepage = 14
        start = addr & ~(page - 1)
        end = (addr + nbytes + page - 1) & ~(page - 1)
        if os.environ.get("IMPALA_PINNED_THP", "1") == "1":
            libc.madvise(ctypes.c_void_p(start), ctypes.c_size_t(end - start), madv_hugepage)
        nthreads = max(1, int(os.environ.get("IMPALA_PINNED_TOUCH_THREADS", "16")))
        step = (((end - start) // nthreads) + page - 1) & ~(page - 1)

        def _touch(a: int, b: int) -> None:
            libc.madvise(ctypes.c_void_p(a), ctypes.c_size_t(b - a), madv_populate_write)

        ths = [
            threading.Thread(target=_touch, args=(a, min(a + step, end)))
            for a in range(start, end, step)
        ]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        t1 = time.time()
        rc = torch.cuda.cudart().cudaHostRegister(addr, nbytes, 0)
        t2 = time.time()
        if int(rc) != 0:
            logger.warning("[impala-pinned] cudaHostRegister rc=%%s; staying on the pageable path", rc)
            return None
        logger.info(
            "[impala-pinned] v%%d streamer buffer %%.1f GB page-locked (thp=%%s): populate %%.2f s, register %%.2f s",
            _IMPALA_PINNED_VERSION, nbytes / 1e9, os.environ.get("IMPALA_PINNED_THP", "1"), t1 - t0, t2 - t1,
        )
        return addr
    except Exception as e:  # noqa: BLE001
        logger.warning("[impala-pinned] pinning failed (%%s); staying on the pageable path", e)
        return None


def runai_safetensors_weights_iterator(''' % PIN_VERSION
anchor = "\ndef runai_safetensors_weights_iterator("
if f"_IMPALA_PINNED_VERSION = {PIN_VERSION}\n" in s:
    print("weight_utils: already patched (v%d)" % PIN_VERSION)
elif "_impala_pin_streamer_buffer" in s:
    # upgrade: replace the helper block (from the version marker or the old helper up to the iterator) with the current one
    i = s.find("\n_IMPALA_PINNED_VERSION = ")
    if i < 0:
        i = s.find("\ndef _impala_pin_streamer_buffer(")
    j = s.find(anchor)
    assert 0 <= i < j, (i, j)
    s = s[:i] + helper + s[j + len(anchor):]
    open(wu, "w").write(s)
    print("weight_utils: helper upgraded to v%d" % PIN_VERSION)
else:
    assert s.count(anchor) == 1, s.count(anchor)
    s = s.replace(anchor, helper, 1)

    old_body = '''        streamer.stream_files(
            hf_weights_files,
            device=device,
            is_distributed=is_distributed,
        )
'''
    new_body = '''        _impala_pinned = os.environ.get("IMPALA_PINNED_LOADER", "0") == "1"
        if _impala_pinned:
            os.environ.setdefault(
                "RUNAI_STREAMER_MEMORY_LIMIT",
                os.environ.get("IMPALA_PINNED_BUFFER_BYTES", "40000000000"),
            )
            if is_cuda_alike and os.environ.get("IMPALA_PINNED_GPU_YIELD", "1") == "1":
                device = f"cuda:{current_platform.current_device()}"
        streamer.stream_files(
            hf_weights_files,
            device=device,
            is_distributed=is_distributed,
        )
        _impala_reg = _impala_pin_streamer_buffer(streamer) if _impala_pinned else None
'''
    assert s.count(old_body) == 1, s.count(old_body)
    s = s.replace(old_body, new_body, 1)

    old_tail = '''        for name, tensor in tensor_iter:
            yield name, tensor.clone()


def fastsafetensors_weights_iterator('''
    new_tail = '''        try:
            if _impala_reg is not None and device != "cpu":
                # GPU-yield: the streamer already copied each tensor out of its reusable
                # CPU buffer with `.to(device)`; the upstream per-tensor clone (#43464) is
                # only needed for CPU tensors that alias that buffer.
                for name, tensor in tensor_iter:
                    yield name, tensor
            else:
                for name, tensor in tensor_iter:
                    yield name, tensor.clone()
        finally:
            if _impala_reg is not None:
                torch.cuda.cudart().cudaHostUnregister(_impala_reg)


def fastsafetensors_weights_iterator('''
    assert s.count(old_tail) == 1, s.count(old_tail)
    s = s.replace(old_tail, new_tail, 1)
    assert "\nimport os\n" in s, "weight_utils.py must import os"
    open(wu, "w").write(s)
    print("weight_utils: patched (pinned runai staging, env-gated IMPALA_PINNED_LOADER=1)")

# ---------------------------------------------------------------- 2. base_loader.py
s = open(bl).read()
if "_impala_weight_checksum" in s:
    print("base_loader: already patched")
else:
    old = '''            process_weights_after_loading(model, model_config, target_device)

        return model.eval()
'''
    new = '''            process_weights_after_loading(model, model_config, target_device)

            _impala_weight_checksum(model)

        return model.eval()
'''
    assert s.count(old) == 1, s.count(old)
    s = s.replace(old, new, 1)
    func = '''

def _impala_weight_checksum(model: nn.Module) -> None:
    """Experiment 4 (Impala): IMPALA_WEIGHT_CHECKSUM=<dir> -> write a per-rank report hashing every GPU parameter/buffer
    (64-bit position-mixed sum on the device) plus sha256 of a fixed sample, after process_weights_after_loading."""
    import hashlib
    import json
    import os
    import time

    out_dir = os.environ.get("IMPALA_WEIGHT_CHECKSUM")
    if not out_dir:
        return
    from vllm.distributed import get_tensor_model_parallel_rank

    t0 = time.time()
    rank = get_tensor_model_parallel_rank()
    items = [(n, p.data) for n, p in model.named_parameters()]
    items += [(n, b) for n, b in model.named_buffers()]
    items.sort(key=lambda kv: kv[0])
    mask = (1 << 64) - 1
    chunk = 1 << 26
    res: dict[str, dict] = {}
    total = 0
    for name, t in items:
        entry = {"shape": list(t.shape), "dtype": str(t.dtype), "dev": str(t.device)}
        if t.device.type == "cuda" and t.numel() > 0:
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
            entry["nbytes"] = nb
            entry["h64"] = f"{h:016x}"
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
    digest = hashlib.sha256(json.dumps(res, sort_keys=True).encode()).hexdigest()
    report = {
        "rank": rank,
        "pid": os.getpid(),
        "n_tensors": len(res),
        "n_hashed": len(names),
        "gpu_bytes": total,
        "all_tensors_sha256": digest,
        "sample_sha256": sample,
        "tensors": res,
        "elapsed_s": round(time.time() - t0, 1),
        "pinned_loader": os.environ.get("IMPALA_PINNED_LOADER", "0"),
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"wchk-rank{rank}-{os.getpid()}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(report, f, indent=0)
    os.replace(path + ".tmp", path)
    logger.info(
        "[impala-checksum] rank %d: %d tensors, %.1f GiB hashed, all-tensors sha256 %s (%.1f s) -> %s",
        rank, len(names), total / 1024**3, digest[:16], time.time() - t0, path,
    )
'''
    s = s.rstrip("\n") + "\n" + func
    open(bl, "w").write(s)
    print("base_loader: patched (checksum hook, env-gated IMPALA_WEIGHT_CHECKSUM=<dir>)")
