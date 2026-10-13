# b300-boot-kit: tuned cold boot of GLM-5.2-NVFP4 on an 8×B300 node

The cold-boot side of the same research: every lever that shortens a vLLM start without
changing serving behaviour, applied to `nvidia/GLM-5.2-NVFP4` (433 GiB) on a Nebius
8×B300 preemptible node with the model on a shared filesystem.

| script | shape | from script start to all replicas serving |
|---|---|---|
| `boot_4x_tp2_pinned.sh` | four TP=2 replicas, vLLM 0.20.2 | 133 s (115–118 s with `OVERLAP=1`, the paced prewarm) |
| `boot_tp8_paced_prewarm.sh` | one TP=8 replica, vLLM 0.20.2 | 120 s |
| `boot_4x_tp2_vllm031.sh` | four TP=2 replicas, vLLM 0.31.0 | 131 s, +29–31 % tokens/s; not checkpointable on driver 615 |

The naive recurring boot of the same four replicas was 780 s (13 minutes); the first-ever boot 1,464 s.

## Levers, in the order they paid

1. **Prewarm the model into the page cache** 16-way from the shared filesystem; four replicas then read one copy. 13.0 → 4.5 min.
2. **FlashInfer autotune cache** persisted across boots (`venv-patches/uks_fipatch.py`, a one-line backport of vLLM #42537; newer vLLM has it natively). → 3.3 min.
3. **Run:ai streamer as the local loader**, **forked workers** (needs `FLASHINFER_CUDA_ARCH_LIST` and `venv-patches/fi_numsms_patch.py`, otherwise FlashInfer initialises CUDA in the API server and vLLM silently falls back to spawn), **venv prewarm** alongside the model prewarm, **per-pair NUMA binding**, **explicit `--kv-cache-memory-bytes`** (also part of the compile-cache key). 262 → 170 s.
4. **Pinned-staging loader** (`venv-patches/pinned_loader_patch.py`, same change as `../patches/vllm/0001`): a 16 GB page-locked staging buffer for the streamer; model-loading stage −48 % at TP=2, every GPU parameter bit-identical. 170 → 133 s.
5. **Paced, overlapped prewarm** (`OVERLAP=1`): launch the servers first and warm shards in the loader's order behind them. Works at TP=8 (ranks consume the same files in lock-step) and, paced, at 4×TP=2. 133 → 115–118 s.

Things that did not pay: a per-rank byte-range reader at TP=8 (`range_loader_patch.py`, bytes per rank halved, −5 %: the stage is a per-tensor floor of 232k tensors × ~130 µs), and the pinned loader at TP=8 (every rank reads the full checkpoint through the page cache; the ceiling is a memcpy).

## venv-patches/

Idempotent, env-gated edits applied to an installed vLLM (`~/venv/lib/python3.12/site-packages`), each with a marker string so it runs once. They are the working form of the `git format-patch` series in `../patches/vllm`.

| file | gate | effect |
|---|---|---|
| `fi_numsms_patch.py` | always | FlashInfer: lazy `NUM_SMS`, so importing it does not initialise CUDA |
| `uks_fipatch.py` | `VLLM_FI_AUTOTUNE_CACHE` | persist the FlashInfer autotune cache |
| `pinned_loader_patch.py` / `_031.py` | `IMPALA_PINNED_LOADER=1` | pinned-staging Run:ai loader (0.20.2 / 0.31.0) |
| `numa_rankbind_patch.py` | `IMPALA_NUMA_RANK_BIND=1` | per-rank NUMA binding inside forked workers |
| `reload_fix_patch.py` / `_031.py` | `IMPALA_RELOAD_FIX=1` | correct in-place `reload_weights` after level-2 sleep for NVFP4 MoE + MLA |
| `reload_runai_patch.py` | always (inert unless `--load-format runai_streamer`) | `get_all_weights` on the streamer loader so reloads use it |
| `reload_overlap_patch.py` | always | overlap-aware kernel-tensor copy for stride-0 scale views |
| `range_loader_patch.py` + `impala_range_loader.py` | `IMPALA_RANGE_LOADER=1` | per-rank byte-range reader (kept opt-in) |
| `exp6_patch_new.py` | n/a | the 0.31.0 port of the small patches, incl. vLLM #57635 |
| `test_reload_fix.py`, `wchk_compare.py` | n/a | the checksum oracle: hash every parameter, buffer and derived tensor and compare |

The scripts assume caches under `~/vllm-cache`, `~/inductor-cache`, `~/triton-cache`, `~/.cache/flashinfer` on the golden disk and the model at `/mnt/fs/glm52-nvfp4`. Read the header comments: each script documents which lever it adds and what it measured.
