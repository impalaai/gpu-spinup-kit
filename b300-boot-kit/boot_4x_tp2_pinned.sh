#!/usr/bin/env bash
# uks_boot9.sh — optimized cold boot: 4× vLLM GLM-5.2-NVFP4 TP=2 from the shared FS = uks_boot7.sh + the pinned-staging weight
# loader (Experiment 4, 2026-10-11; ledger b300-criu/RESEARCH-4.md): box/pinned_loader_patch.py (idempotent venv patch, env-gated)
# page-locks the Run:ai streamer's CPU buffer (16 GB, IMPALA_PINNED_BUFFER_BYTES) and streams tensors to the GPU from pinned memory
# (IMPALA_PINNED_LOADER=1) — weight stage 68 s → 30 s per worker, READY per pair −32…−36 s, checksums identical, throughput unchanged.
# Levers inherited from uks_boot7 (= uks_boot5 + the five behaviour-neutral levers that won their Exp-1 A/B):
#   1. --load-format runai_streamer (local FS path; the default mmap path is bound by pageable H2D copies)   −19…−24 s
#   2. --kv-cache-memory-bytes = the value profiling allocates (31.18 GiB → 606,144 tokens; profile_run still runs)  −7 s
#   3. VLLM_WORKER_MULTIPROC_METHOD=fork — needs FLASHINFER_CUDA_ARCH_LIST (flashinfer jit/env.py) and the lazy
#      NUM_SMS patch (fi_numsms_patch.py) so the API server never initialises CUDA before spawning          −5…−13 s
#   4. per-pair numactl --cpunodebind/--preferred to the pair's GPU node                                        −1…−5 s
#   8a. venv + cache prewarm (8.1 G site-packages, flashinfer JIT, torch_compile/triton/inductor caches) from the NRD
#      disk, 8-way, in parallel with the 16-way model prewarm (both ~25–30 s; different devices)              −18 s
# Caches on this golden disk live under ~/vllm-cache (VLLM_CACHE_ROOT), ~/inductor-cache, ~/triton-cache, ~/.cache/flashinfer.
# OVERLAP=1 launches the servers first and prewarms concurrently (lever 5); default is the serial order unless the ledger says
# otherwise. Reports READY both from script start and from T0 (after the prewarm) like uks_boot5.
set -uo pipefail
cd "$HOME"
export PATH="$HOME/venv/bin:/usr/local/bin:$PATH"
export VLLM_CACHE_ROOT=$HOME/vllm-cache TORCHINDUCTOR_CACHE_DIR=$HOME/inductor-cache TRITON_CACHE_DIR=$HOME/triton-cache HF_HOME=$HOME/hf
MODEL=${MODEL:-/mnt/fs/glm52-nvfp4}; OVERLAP=${OVERLAP:-0}; PREWARM=${PREWARM:-1}
KVB=${KVB:-33478800000}   # = 31.18 GiB as profiled by vLLM 0.20.2 for this recipe on B300 (log: "Available KV cache memory: 31.18 GiB")
now(){ python3 -c 'import time; print(time.time())'; }
el(){ python3 -c "import time; print(round(time.time()-$1,1))"; }
gpu_node(){ local B; B=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$1" 2>/dev/null | sed 's/^0000//' | tr 'A-F' 'a-f'); cat /sys/bus/pci/devices/$B/numa_node 2>/dev/null; }
# kill every vLLM shape (API server, forked EngineCore/workers, anything holding VRAM) and WAIT for the drain
pkill -9 -f "vllm.[e]ntrypoints" 2>/dev/null; pkill -9 -f "EngineCor[e]" 2>/dev/null; pkill -9 -f "VLLM[:]" 2>/dev/null
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do sudo kill -9 "$p" 2>/dev/null; done
for i in $(seq 1 90); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$USED" -lt 1024 ] && break; sleep 2
done
echo "vram drained: max ${USED} MiB"
sudo rm -f /dev/shm/psm_* /dev/shm/sem.* /dev/shm/link_remap.* 2>/dev/null
[ "${NODROP:-0}" = 1 ] || { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; }
free -g | sed -n 2p
mountpoint -q /mnt/fs || sudo mount -t virtiofs sharedfs /mnt/fs
~/venv/bin/python ~/uks_fipatch.py >/dev/null 2>&1; ~/venv/bin/python ~/fi_numsms_patch.py >/dev/null 2>&1; ~/venv/bin/python ~/pinned_loader_patch.py >/dev/null 2>&1   # idempotent venv patches
venvwarm(){ local T; T=$(now); find "$HOME/venv/lib/python3.12/site-packages" "$HOME/.cache/flashinfer" "$HOME/vllm-cache" "$HOME/triton-cache" "$HOME/inductor-cache" /usr/lib/python3.12 -type f -print0 2>/dev/null | xargs -0 -P 8 -n 64 cat > /dev/null; echo "venvwarm: $(el $T)s"; }
prewarm(){ venvwarm & [ "$PREWARM" = 1 ] && ls "$MODEL"/* | xargs -P 16 -n 1 cat > /dev/null; wait; }
launch(){ local i=$1 PORT G0 G1 N NB=""
  PORT=$((8201+i)); G0=$((i*2)); G1=$((i*2+1))
  N=$(gpu_node $G0); [ -n "$N" ] && [ "$N" -ge 0 ] 2>/dev/null && NB="numactl --cpunodebind=$N --preferred=$N"
  CUDA_VISIBLE_DEVICES=$G0,$G1 VLLM_FI_AUTOTUNE_CACHE=$HOME/fi-autotune-$PORT.json \
  VLLM_WORKER_MULTIPROC_METHOD=fork FLASHINFER_CUDA_ARCH_LIST=10.3a \
  IMPALA_PINNED_LOADER=${PINNED:-1} IMPALA_PINNED_BUFFER_BYTES=${PINNED_BYTES:-16000000000} \
  VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 NO_COLOR=1 \
  setsid $NB ~/venv/bin/python -m vllm.entrypoints.cli.main serve "$MODEL" \
    --tensor-parallel-size 2 --served-model-name glm-5.2-nvfp4 --port $PORT \
    --max-model-len 32768 --max-num-seqs 64 \
    --load-format runai_streamer --kv-cache-memory-bytes "$KVB" \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64]}' < /dev/null > "$HOME/glm-$PORT.log" 2>&1 &
}
TS=$(now)
if [ "$OVERLAP" = 1 ]; then
  T0=$TS; for i in 0 1 2 3; do launch $i; done; TP=$(now); prewarm; echo "prewarm (overlapped): $(el $TP)s"
else
  TP=$(now); prewarm; echo "prewarm: $(el $TP)s"; T0=$(now); echo "T0=$T0"; for i in 0 1 2 3; do launch $i; done
fi
UPLIST=""
for s in $(seq 1 3000); do
  for PORT in 8201 8202 8203 8204; do
    case " $UPLIST " in *" $PORT "*) continue;; esac
    curl -sf -o /dev/null --max-time 1 "http://127.0.0.1:$PORT/health" && { UPLIST="$UPLIST $PORT"; echo "READY $PORT at $(el $T0)s from T0 ($(el $TS)s from start)"; }
  done
  N=$(echo $UPLIST | wc -w); [ "$N" -eq 4 ] && break
  if [ "$s" -gt 90 ]; then dead=1; for p in 8201 8202 8203 8204; do grep -aq "Traceback\|Engine core initialization failed" "$HOME/glm-$p.log" || dead=0; done
    [ "$dead" = 1 ] && { echo ALL_VLLM_DEAD; for p in 8201 8202 8203 8204; do echo "== log $p:"; tail -4 "$HOME/glm-$p.log"; done; exit 1; }; fi
  sleep 1
done
N=$(echo $UPLIST | wc -w)
[ "$N" -eq 4 ] || { echo "BOOT_TIMEOUT ($N/4 up)"; for p in 8201 8202 8203 8204; do echo "== log $p:"; tail -4 "$HOME/glm-$p.log"; done; exit 1; }
echo "ALL 4 READY at $(el $T0)s from T0, $(el $TS)s from script start"
# content-verified sanity on every port (never health alone)
for PORT in 8201 8202 8203 8204; do
  R=$(curl -s --max-time 240 http://127.0.0.1:$PORT/v1/completions -H 'content-type: application/json' \
    -d '{"model":"glm-5.2-nvfp4","prompt":"Q: What is 9+9? Answer with just the number.\nA:","max_tokens":12,"temperature":0}' | grep -ao '"text":"[^"]*"' | head -1 | cut -c1-60)
  echo "sanity $PORT: $R"; echo "$R" | grep -q 18 || echo "SANITY_BAD $PORT"
done
grep -ah "GPU KV cache size" "$HOME"/glm-820?.log | sed 's/.*GPU KV/KV/' | sort | uniq -c
grep -ah "impala-pinned\|Model loading took" "$HOME"/glm-820?.log | sed 's/^.*INFO //' | cut -c1-140
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' '; echo
echo UKS_BOOT9_DONE
