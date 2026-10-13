#!/usr/bin/env bash
# uks_boot10.sh — Experiment 6 (2026-10-11, ledger b300-criu/RESEARCH-6.md): the uks_boot9 cold boot (4× GLM-5.2-NVFP4 TP=2, Experiment-1/4
# levers) on vLLM 0.31.0 in ~/venv-new (torch 2.13 cu13, flashinfer 0.7.0.post1, runai 0.16.1, transformers 5.11), with the ported
# env-gated patches (box/exp6_patch_new.py: upstream #57635 + the trust-cache warmup (IMPALA_FI_TRUST_CACHE=1), 3 flashinfer
# import-time CUDA-init edits so fork engages, job-file re-export, per-rank NUMA bind; box/pinned_loader_patch_031.py: pinned
# Run:ai staging v3 + checksum hook; box/reload_fix_patch_031.py: in-place reload after sleep level 2). Measured: ALL 4 READY
# 130.9 / 131.8 s from script start (92.0 / 93.2 from T0) vs 139.7 / 139.7 for uks_boot9 on the same day; throughput @32×256 +26 %.
# Requirements baked in: explicit --kv-cache-memory-bytes (KVB_NEW=34225657406; the 0.31.0 startup plan is per-rank gated and
# deadlocks TP pairs), per-venv CUDA JIT cache (CUDA_CACHE_PATH=~/.nv/cc-new; sharing ~/.nv/ComputeCache between two CUDA stacks
# re-JITs ~1 GB per switch, +50–60 s), caches under ~/vllm-cache-new / ~/inductor-cache-new / ~/triton-cache-new / ~/fi-new.
# Same knobs as exp6_boot9.sh (VENV, OVERLAP/PREWARM_P, PAIRS, KEEP, NODROP, LABEL). ~/venv and uks_boot7/8/9 are untouched.
set -uo pipefail
cd "$HOME"
MODEL=${MODEL:-/mnt/fs/glm52-nvfp4}; PREWARM=${PREWARM:-1}; LABEL=${LABEL:-boot10}; PAIRS=${PAIRS:-"0 1 2 3"}; KEEP=${KEEP:-0}
VENV=${VENV:-$HOME/venv-new}; OVERLAP=${OVERLAP:-0}; PREWARM_P=${PREWARM_P:-4}
KVB=${KVB:-33478800000}; KVB_NEW=${KVB_NEW:-34225657406}
mkdir -p ~/exp6-logs
now(){ python3 -c 'import time; print(time.time())'; }
el(){ python3 -c "import time; print(round(time.time()-$1,1))"; }
gpu_node(){ local B; B=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$1" 2>/dev/null | sed 's/^0000//' | tr 'A-F' 'a-f'); cat /sys/bus/pci/devices/$B/numa_node 2>/dev/null; }
is_new(){ case "$1" in *venv-new*) return 0;; *) return 1;; esac; }
venv_of(){ local v="VENV_$1"; echo "${!v:-$VENV}"; }
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
# idempotent venv patches (env-gated; applied to every venv in use)
VENVS=""; for i in $PAIRS; do v=$(venv_of $i); case " $VENVS " in *" $v "*) ;; *) VENVS="$VENVS $v";; esac; done
for v in $VENVS; do
  if is_new "$v"; then python3 ~/exp6_patch_new.py "$v" >/dev/null 2>&1; "$v/bin/python" ~/pinned_loader_patch_031.py "$v/lib/python3.12/site-packages/vllm" >/dev/null 2>&1
  else "$v/bin/python" ~/uks_fipatch.py >/dev/null 2>&1; "$v/bin/python" ~/fi_numsms_patch.py >/dev/null 2>&1; "$v/bin/python" ~/pinned_loader_patch.py >/dev/null 2>&1; fi
done
echo "venvs in use:$VENVS"
venvwarm(){ local T v SET=""; T=$(now)
  for v in $VENVS; do SET="$SET $v/lib/python3.12/site-packages"; if is_new "$v"; then SET="$SET $HOME/fi-new/.cache/flashinfer $HOME/vllm-cache-new $HOME/triton-cache-new $HOME/inductor-cache-new ${CUDA_CC_NEW:-$HOME/.nv/cc-new}"; else SET="$SET $HOME/.cache/flashinfer $HOME/vllm-cache $HOME/triton-cache $HOME/inductor-cache ${CUDA_CC_OLD:-$HOME/.nv/cc-old}"; fi; done
  find $SET /usr/lib/python3.12 -type f -print0 2>/dev/null | xargs -0 -P 8 -n 64 cat > /dev/null; echo "venvwarm: $(el $T)s"; }
prewarm(){ venvwarm & if [ "$PREWARM" = 1 ]; then if [ "$OVERLAP" = 1 ]; then ls "$MODEL"/*.safetensors | sort | xargs -P "$PREWARM_P" -n 1 cat > /dev/null; else ls "$MODEL"/* | xargs -P 16 -n 1 cat > /dev/null; fi; fi; wait; }
launch(){ local i=$1 PORT G0 G1 N NB="" EV AR v VE CE KV
  PORT=$((8201+i)); G0=$((i*2)); G1=$((i*2+1))
  v="ENV_$i"; EV=${!v:-}; v="ARGS_$i"; AR=${!v:-}; VE=$(venv_of $i)
  N=$(gpu_node $G0); [ -n "$N" ] && [ "$N" -ge 0 ] 2>/dev/null && NB="numactl --cpunodebind=$N --preferred=$N"
  if is_new "$VE"; then
    CE="VLLM_CACHE_ROOT=$HOME/vllm-cache-new TORCHINDUCTOR_CACHE_DIR=$HOME/inductor-cache-new TRITON_CACHE_DIR=$HOME/triton-cache-new FLASHINFER_WORKSPACE_BASE=$HOME/fi-new HF_HOME=$HOME/hf IMPALA_FI_TRUST_CACHE=${FI_TRUST:-1} CUDA_CACHE_PATH=${CUDA_CC_NEW:-$HOME/.nv/cc-new} CUDA_CACHE_MAXSIZE=4294967296"
    if [ -n "$KVB_NEW" ]; then KV="--kv-cache-memory-bytes $KVB_NEW"; else KV=""; CE="$CE VLLM_ENABLE_STARTUP_PLAN=1"; fi
  else
    CE="VLLM_CACHE_ROOT=$HOME/vllm-cache TORCHINDUCTOR_CACHE_DIR=$HOME/inductor-cache TRITON_CACHE_DIR=$HOME/triton-cache HF_HOME=$HOME/hf VLLM_FI_AUTOTUNE_CACHE=$HOME/fi-autotune-$PORT.json CUDA_CACHE_PATH=${CUDA_CC_OLD:-$HOME/.nv/cc-old} CUDA_CACHE_MAXSIZE=4294967296"
    KV="--kv-cache-memory-bytes $KVB"
  fi
  echo "launch $PORT gpus=$G0,$G1 venv=$VE env=[$EV] args=[$AR] kv=[$KV]"
  env $CE CUDA_VISIBLE_DEVICES=$G0,$G1 \
  VLLM_WORKER_MULTIPROC_METHOD=fork FLASHINFER_CUDA_ARCH_LIST=10.3a \
  IMPALA_PINNED_LOADER=${PINNED:-1} IMPALA_PINNED_BUFFER_BYTES=${PINNED_BYTES:-16000000000} \
  VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 NO_COLOR=1 PATH="$VE/bin:/usr/local/bin:$PATH" $EV \
  setsid $NB "$VE/bin/python" -m vllm.entrypoints.cli.main serve "$MODEL" \
    --tensor-parallel-size 2 --served-model-name glm-5.2-nvfp4 --port $PORT \
    --max-model-len 32768 --max-num-seqs 64 \
    --load-format runai_streamer $KV \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64]}' $AR < /dev/null > "$HOME/glm-$PORT.log" 2>&1 &
}
TS=$(now); PWFILE=~/exp6-prewarm-$LABEL.txt; rm -f "$PWFILE"; PW=-1
if [ "$OVERLAP" = 1 ]; then
  T0=$TS; echo "T0=$T0 TS=$TS (overlap, sorted prewarm P=$PREWARM_P)"; for i in $PAIRS; do launch $i; done
  ( TP=$(now); prewarm; echo "$(el $TP)" > "$PWFILE" ) > /dev/null 2>&1 &
  PWPID=$!
else
  TP=$(now); prewarm; PW=$(el $TP); echo "prewarm: ${PW}s"; T0=$(now); echo "T0=$T0 TS=$TS"; for i in $PAIRS; do launch $i; done
fi
NP=$(echo $PAIRS | wc -w); UPLIST=""; declare -A RT RTS
for s in $(seq 1 3600); do
  for i in $PAIRS; do PORT=$((8201+i))
    case " $UPLIST " in *" $PORT "*) continue;; esac
    curl -sf -o /dev/null --max-time 1 "http://127.0.0.1:$PORT/health" && { UPLIST="$UPLIST $PORT"; RT[$PORT]=$(el $T0); RTS[$PORT]=$(el $TS); echo "READY $PORT at ${RT[$PORT]}s from T0 (${RTS[$PORT]}s from start)"; }
  done
  N=$(echo $UPLIST | wc -w); [ "$N" -eq "$NP" ] && break
  if [ "$s" -gt 90 ]; then dead=1; for i in $PAIRS; do PORT=$((8201+i)); case " $UPLIST " in *" $PORT "*) continue;; esac; grep -aq "Traceback\|Engine core initialization failed" "$HOME/glm-$PORT.log" || dead=0; done
    [ "$dead" = 1 ] && { echo ALL_PENDING_DEAD; break; }; fi
  sleep 1
done
ALL=$(el $T0); ALLS=$(el $TS); N=$(echo $UPLIST | wc -w)
if [ "$OVERLAP" = 1 ]; then for i in $(seq 1 120); do [ -f "$PWFILE" ] && break; sleep 5; done; [ -f "$PWFILE" ] && PW=$(cat "$PWFILE") || { PW=-2; kill ${PWPID:-0} 2>/dev/null; }; echo "prewarm (overlapped, P=$PREWARM_P): ${PW}s from launch"; fi
echo "ALL $N/$NP READY at ${ALL}s from T0, ${ALLS}s from script start (prewarm ${PW}s)"
SAN=""
for i in $PAIRS; do PORT=$((8201+i))
  case " $UPLIST " in *" $PORT "*) ;; *) SAN="$SAN $PORT:DOWN"; echo "== log $PORT:"; grep -a "Error\|error\|Traceback" "$HOME/glm-$PORT.log" | tail -3 | cut -c1-200; continue;; esac
  R=$(curl -s --max-time 240 http://127.0.0.1:$PORT/v1/completions -H 'content-type: application/json' \
    -d '{"model":"glm-5.2-nvfp4","prompt":"Q: What is 9+9? Answer with just the number.\nA:","max_tokens":12,"temperature":0}' | grep -ao '"text":"[^"]*"' | head -1 | cut -c1-80)
  echo "sanity $PORT: $R"; echo "$R" | grep -q 18 && SAN="$SAN $PORT:OK" || SAN="$SAN $PORT:BAD"
done
for i in $PAIRS; do PORT=$((8201+i)); echo "== $PORT:"; grep -ah "Loading weights took\|Model loading took\|impala-pinned\|impala-checksum\|GPU KV cache size\|Available KV cache memory\|spawn\|torch.compile took\|torch.compile takes\|startup plan\|Startup plan\|autotune cache\|Graph capturing finished\|Model Runner\|model runner\|Reloading\|init engine" "$HOME/glm-$PORT.log" | sed 's/^.*INFO //' | cut -c1-160 | sort | uniq -c | sed 's/^ *//' | head -16; done
J="{\"label\":\"$LABEL\",\"venv\":\"$VENV\",\"overlap\":$OVERLAP,\"prewarm_p\":$PREWARM_P,\"prewarm_s\":$PW,\"all_ready_t0_s\":$ALL,\"all_ready_start_s\":$ALLS,\"n_ready\":$N,\"sanity\":\"$SAN\",\"ports\":{"
for i in $PAIRS; do PORT=$((8201+i)); v="ENV_$i"; EV=${!v:-}; v="ARGS_$i"; AR=${!v:-}; AR=${AR//\"/\\\"}; VE=$(venv_of $i)
  WL=$(grep -ah "Loading weights took" "$HOME/glm-$PORT.log" | grep -ao "[0-9.]* seconds" | cut -d' ' -f1 | tr '\n' ',' | sed 's/,$//')
  ML=$(grep -ah "Model loading took" "$HOME/glm-$PORT.log" | grep -ao "and [0-9.]* seconds" | cut -d' ' -f2 | tr '\n' ',' | sed 's/,$//')
  KVT=$(grep -ah "GPU KV cache size" "$HOME/glm-$PORT.log" | grep -ao "[0-9,]* tokens" | head -1 | tr -d ', tokens')
  J="$J\"$PORT\":{\"ready_t0_s\":${RT[$PORT]:-null},\"ready_start_s\":${RTS[$PORT]:-null},\"venv\":\"$VE\",\"env\":\"$EV\",\"args\":\"$AR\",\"weights_s\":\"$WL\",\"model_load_s\":\"$ML\",\"kv_tokens\":\"$KVT\"},"; done
J="${J%,}},\"t0\":$T0,\"ts\":$TS}"
echo "$J" >> ~/exp6.jsonl
for i in $PAIRS; do PORT=$((8201+i)); cp "$HOME/glm-$PORT.log" ~/exp6-logs/$LABEL-$PORT.log; python3 ~/stages.py "$HOME/glm-$PORT.log" "$T0" > ~/exp6-logs/$LABEL-$PORT.stages.json 2>/dev/null; done
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' '; echo
[ "$KEEP" = 1 ] || { pkill -9 -f "vllm.[e]ntrypoints" 2>/dev/null; pkill -9 -f "EngineCor[e]" 2>/dev/null; pkill -9 -f "VLLM[:]" 2>/dev/null; }
echo UKS_BOOT10_DONE
