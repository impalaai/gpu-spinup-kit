# reload_overlap_patch.py — vLLM 0.20.2 layerwise reload copies the re-processed kernel tensors INTO the original storage (so the
# CUDA graphs keep their pointers). For modelopt NVFP4 MoE the original kernel tensor is an expanded (stride-0) view of a
# per-tensor scale, and Tensor.copy_ refuses a destination with internal overlap. Make the copy overlap-aware: write the unique
# storage elements of the destination from the matching sub-view of the source. Idempotent.
import os
p = os.path.expanduser("~/venv/lib/python3.12/site-packages/vllm/model_executor/model_loader/reload/layerwise.py")
s = open(p).read()
if "_impala_overlap_copy" in s:
    print("already patched"); raise SystemExit
old = '''    parameters, buffers = info.kernel_tensors
    for name, param in parameters.items():
        param.data.copy_(getattr(layer, name))
    for name, buffer in buffers.items():
        buffer.data.copy_(getattr(layer, name))
'''
assert s.count(old) == 1, s.count(old)
new = '''    parameters, buffers = info.kernel_tensors
    for name, param in parameters.items():
        _impala_overlap_copy(param.data, getattr(layer, name), name)
    for name, buffer in buffers.items():
        _impala_overlap_copy(buffer.data, getattr(layer, name), name)
'''
s = s.replace(old, new)
helper = '''

def _impala_overlap_copy(dst: torch.Tensor, src: torch.Tensor, name: str = "") -> None:
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
            raise RuntimeError(f"overlap copy for {name}: dst storage {n} elements, src sub-view {src_flat.numel()}")
    dst_flat.copy_(src_flat)
'''
s = s.replace("\n\ndef _place_kernel_tensors(", helper + "\n\ndef _place_kernel_tensors(", 1)
open(p, "w").write(s)
print("patched")
