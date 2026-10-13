#!/usr/bin/env bash
# uks_boot8.sh — optimized cold boot of ONE vLLM GLM-5.2-NVFP4 TP=8 server (port 8201) from the shared FS
# (Experiment 3, 2026-10-11; ledger b300-criu/RESEARCH-3.md) = uks_boot7.sh's levers adapted to one 8-rank process:
#   runai_streamer loader · fork workers (FLASHINFER_CUDA_ARCH_LIST=10.3a + lazy-NUM_SMS patch) · venv prewarm ∥ 16-way model
#   prewarm · explicit --kv-cache-memory-bytes (KVB; empty = let vLLM profile) · own autotune cache ~/fi-autotune-tp8.json.
# Experiment 4 (2026-10-11, RESEARCH-4.md): + pinned-staging loader (pinned_loader_patch.py, PINNED=1 default: 16 GB page-locked
# streamer buffer, GPU-yield) — at TP=8 only −2…−4 s on the 47 s weight stage (read-bound shape), checksums identical, no tput change.
# Secondary B (RESEARCH-4.md): OVERLAP=1 (default) launches the server FIRST and runs the model prewarm concurrently, 4-way, in the
# streamer's sorted file order (PREWARM_P=4): READY 119.8–120.0 s vs 136.6–138.8 s from script start (−17 s, 3/3). OVERLAP=0 = the
# Exp-3 serial 16-way prewarm. NOTE: the golden-disk copy installed on 2026-10-11 (md5 04d2e7b8) predates this knob (serial prewarm).
# Experiment 4b: RANGE=1 (opt-in, default 0) adds the per-rank byte-range reader (range_loader_patch.py + impala_range_loader.py;
# first boot per model runs a 12 s probe pass, cached): bytes per rank 433 → 213 GB, checksums identical, READY only −4…−6 s (−5 %).
# NUMA: one tree spans both nodes (GPUs 0-3 node 0, 4-7 node 1) — NUMA_MODE=rank (default: fork + env-gated per-rank bind patch, −0…−4 s) | none (fork, unbound) | vllm (--numa-bind:
# vLLM wraps every worker in numactl --cpunodebind/--membind, forces spawn) | rank (fork + env-gated per-rank bind patch).
# Reports READY from script start and from T0 (after the prewarm). Env: KVB, NUMA_MODE, LABEL, NODROP=1, PREWARM=0, KEEP=1.
set -uo pipefail
cd "$HOME"
export PATH="$HOME/venv/bin:/usr/local/bin:$PATH"
export VLLM_CACHE_ROOT=$HOME/vllm-cache TORCHINDUCTOR_CACHE_DIR=$HOME/inductor-cache TRITON_CACHE_DIR=$HOME/triton-cache HF_HOME=$HOME/hf
MODEL=${MODEL:-/mnt/fs/glm52-nvfp4}; PREWARM=${PREWARM:-1}; NUMA_MODE=${NUMA_MODE:-rank}; LABEL=${LABEL:-boot8}; PORT=8201
KVB=${KVB:-}   # TP=8 profiled value goes here once known (part of the torch.compile cache key: freeze it after the first compile)
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
~/venv/bin/python ~/uks_fipatch.py >/dev/null 2>&1; ~/venv/bin/python ~/fi_numsms_patch.py >/dev/null 2>&1; ~/venv/bin/python ~/numa_rankbind_patch.py >/dev/null 2>&1; ~/venv/bin/python ~/pinned_loader_patch.py >/dev/null 2>&1; [ "${RANGE:-0}" = 1 ] && ~/venv/bin/python ~/range_loader_patch.py >/dev/null 2>&1   # idempotent venv patches
venvwarm(){ local T; T=$(now); find "$HOME/venv/lib/python3.12/site-packages" "$HOME/.cache/flashinfer" "$HOME/vllm-cache" "$HOME/triton-cache" "$HOME/inductor-cache" /usr/lib/python3.12 -type f -print0 2>/dev/null | xargs -0 -P 8 -n 64 cat > /dev/null; echo "venvwarm: $(el $T)s"; }
prewarm(){ venvwarm & [ "$PREWARM" = 1 ] && ls "$MODEL"/*.safetensors | sort | xargs -P "${PREWARM_P:-$([ "${OVERLAP:-1}" = 1 ] && echo 4 || echo 16)}" -n 1 cat > /dev/null; wait; }
launch(){ local MP=fork EXTRA=()
  [ -n "$KVB" ] && EXTRA+=(--kv-cache-memory-bytes "$KVB")
  case "$NUMA_MODE" in
    vllm) MP=spawn; EXTRA+=(--numa-bind);;
    rank) NODES=""; for g in 0 1 2 3 4 5 6 7; do NODES="$NODES,$(gpu_node $g)"; done; export IMPALA_NUMA_RANK_BIND=1 IMPALA_NUMA_NODES="${NODES#,}"; echo "rank bind nodes=${NODES#,}";;
  esac
  echo "launch $PORT tp=8 numa=$NUMA_MODE mp=$MP kvb=[$KVB] extra=[${EXTRA[*]:-}]"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_FI_AUTOTUNE_CACHE=$HOME/fi-autotune-tp8.json \
  VLLM_WORKER_MULTIPROC_METHOD=$MP FLASHINFER_CUDA_ARCH_LIST=10.3a \
  IMPALA_PINNED_LOADER=${PINNED:-1} IMPALA_PINNED_BUFFER_BYTES=${PINNED_BYTES:-16000000000} \
  IMPALA_RANGE_LOADER=${RANGE:-0} IMPALA_RANGE_READER=pool \
  VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 NO_COLOR=1 ${EXTRA_ENV:-} \
  setsid ~/venv/bin/python -m vllm.entrypoints.cli.main serve "$MODEL" \
    --tensor-parallel-size 8 --served-model-name glm-5.2-nvfp4 --port $PORT \
    --max-model-len 32768 --max-num-seqs 64 \
    --load-format runai_streamer "${EXTRA[@]}" \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64]}' < /dev/null > "$HOME/glm-$PORT.log" 2>&1 &
}
TS=$(now)
PWFILE=$HOME/uks_boot8-prewarm.txt; rm -f "$PWFILE"
if [ "${OVERLAP:-1}" = 1 ]; then T0=$TS; echo "T0=$T0 TS=$TS (overlap, prewarm P=${PREWARM_P:-4})"; launch; ( TP=$(now); prewarm; echo "$(el $TP)" > "$PWFILE" ) > /dev/null 2>&1 &
else TP=$(now); prewarm; PW=$(el $TP); echo "prewarm: ${PW}s"; T0=$(now); echo "T0=$T0 TS=$TS"; launch; fi
READY=""
for s in $(seq 1 3000); do
  curl -sf -o /dev/null --max-time 1 "http://127.0.0.1:$PORT/health" && { READY=$(el $T0); echo "READY $PORT at ${READY}s from T0 ($(el $TS)s from start)"; break; }
  if [ "$s" -gt 90 ] && grep -aq "Traceback\|Engine core initialization failed" "$HOME/glm-$PORT.log"; then echo VLLM_DEAD; tail -6 "$HOME/glm-$PORT.log"; exit 1; fi
  sleep 1
done
[ -n "$READY" ] || { echo BOOT_TIMEOUT; tail -4 "$HOME/glm-$PORT.log"; exit 1; }
READYS=$(el $TS)
echo "READY at ${READY}s from T0, ${READYS}s from script start"
if [ "${OVERLAP:-1}" = 1 ]; then for i in $(seq 1 60); do [ -f "$PWFILE" ] && break; sleep 5; done; PW=$(cat "$PWFILE" 2>/dev/null || echo -1); echo "prewarm (overlapped): ${PW}s from launch"; fi
# content-verified sanity (never health alone)
R=$(curl -s --max-time 240 http://127.0.0.1:$PORT/v1/completions -H 'content-type: application/json' \
  -d '{"model":"glm-5.2-nvfp4","prompt":"Q: What is 9+9? Answer with just the number.\nA:","max_tokens":12,"temperature":0}' | grep -ao '"text":"[^"]*"' | head -1 | cut -c1-60)
echo "sanity $PORT: $R"; echo "$R" | grep -q 18 && SAN=OK || { SAN=BAD; echo "SANITY_BAD $PORT"; }
grep -ah "GPU KV cache size\|Available KV cache memory\|torch.compile took\|Loading weights took\|Model loading took\|impala-pinned\|Graph capturing finished\|spawn\|numactl\|Binding\|numa rank bind\|init engine" "$HOME/glm-$PORT.log" | cut -c1-200
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' '; echo
mkdir -p ~/exp3-logs; cp "$HOME/glm-$PORT.log" ~/exp3-logs/$LABEL.log; python3 ~/stages.py "$HOME/glm-$PORT.log" "$T0" > ~/exp3-logs/$LABEL.stages.json 2>/dev/null
echo "{\"label\":\"$LABEL\",\"numa\":\"$NUMA_MODE\",\"kvb\":\"$KVB\",\"prewarm_s\":$PW,\"ready_t0_s\":$READY,\"ready_start_s\":$READYS,\"sanity\":\"$SAN\",\"t0\":$T0,\"ts\":$TS}" >> ~/exp3.jsonl
[ "${KEEP:-0}" = 1 ] || { pkill -9 -f "vllm.[e]ntrypoints" 2>/dev/null; pkill -9 -f "EngineCor[e]" 2>/dev/null; pkill -9 -f "VLLM[:]" 2>/dev/null; }
echo UKS_BOOT8_DONE
