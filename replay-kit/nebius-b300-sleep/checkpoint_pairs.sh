#!/usr/bin/env bash
# b300_criu.sh <serve|serve-default|gate|freeze|restore> <idx 0..3> [restore-args]
# TP=2 GPU-state CRIU per vLLM pair on the 8xB300 box. Pair idx -> port 8201+idx, GPUs 2*idx,2*idx+1.
# serve         : checkpointable recipe (symm-mem off, NCCL IPC paths off, custom all-reduce off, spawn, host IP lo).
# serve-default : stock recipe (NVLink on, default NCCL) — for the throughput baseline only.
# gate          : toggle round-trip sanity.  freeze : toggles + fd/sock hygiene + criu dump -> /mnt/fs/snap-<port>.
# restore       : same-boot restore from /mnt/fs/snap-<port> (+ toggle-up) with optional extra criu args (e.g. --premap-threads 12).
# All nebius/lyceum/verda lessons baked in: per-tree toggle sets, NVML fd close, non-loopback socket close,
# --tcp-established --link-remap, shm ghosts backup, logsize, hostip, chmod 600 on reseed.
set -uo pipefail
cd "$HOME"
PHASE=${1:?phase}; IDX=${2:?idx}; shift 2; EXTRA_RESTORE=("$@")
PORT=$((8201+IDX)); G0=$((IDX*2)); G1=$((IDX*2+1)); TP=${TP:-2}; [ "$TP" = 1 ] && G1=$G0
export PATH="$HOME/venv/bin:/usr/local/bin:$PATH"
export VLLM_CACHE_ROOT="$HOME/vllm-cache" VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_CONFIGURE_LOGGING=1 VLLM_LOGGING_LEVEL=INFO NO_COLOR=1 \
  HF_HOME="$HOME/hf" UV_USE_IO_URING=0 USE_LIBUV=0 TORCHINDUCTOR_COMPILE_THREADS=1 \
  VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 \
  TORCHINDUCTOR_CACHE_DIR=$HOME/inductor-cache TRITON_CACHE_DIR=$HOME/triton-cache \
  VLLM_FI_AUTOTUNE_CACHE=$HOME/fi-autotune-$PORT.json \
  CUDA_VISIBLE_DEVICES=$( [ "$TP" = 1 ] && echo $G0 || echo $G0,$G1 )
PY=$HOME/venv/bin/python
CKPT=/usr/local/bin/cuda-checkpoint
MODEL=${MODEL:-/mnt/fs/glm52-nvfp4}
SRV=glm-5.2-nvfp4
SNAP=/mnt/fs/snap-$PORT
LOGF=$HOME/glm-$PORT.log
KVB=${KVB:-8589934592}   # 8 GiB KV per rank keeps the image small
mkdir -p "$VLLM_CACHE_ROOT"

now(){ python3 -c 'import time; print(time.time())'; }
el(){ python3 -c "import time; print(round(time.time()-$1,1))"; }
vram(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader -i $G0,$G1 2>/dev/null | tr '\n' ' '; }
wait_ready(){ local t0=$1 lim=${2:-3000} pid=${3:-}
  for i in $(seq 1 "$lim"); do
    curl -sf -o /dev/null --max-time 2 http://127.0.0.1:$PORT/health && { el "$t0"; return 0; }
    [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null && { echo DEAD; return 1; }
    sleep 1
  done; echo TIMEOUT; return 1; }
strict(){ local A=$((RANDOM % 8 + 2)) B=$((RANDOM % 8 + 3)); local W=$((A+B))
  curl -s --max-time 240 http://127.0.0.1:$PORT/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"'"$SRV"'","messages":[{"role":"user","content":"What is '"$A"'+'"$B"'? Reply with just the number."}],"max_tokens":512,"temperature":0}' \
  | W=$W python3 -c 'import json,sys,os
w=os.environ["W"]
try:
  m=json.load(sys.stdin)["choices"][0]["message"]
  c=(m.get("content") or "").strip()
  if w in c: print("STRICT_OK")
  else: print("STRICT_BAD want=" + w + " content=" + repr(c[-80:]))
except Exception as e: print("STRICT_FAIL", e)'; }
root_pid(){ pgrep -f "vllm.entrypoints.cli.main serve .*--port $PORT( |$)" | sort -n | head -1; }
ckpt_down(){ # lock pass then checkpoint pass, sequentially over the job (order recorded); TOGGLE=1 = legacy --toggle per pid
  ORDER="$HEAVY $REST"
  if [ "${TOGGLE:-0}" = 1 ]; then for p in $ORDER; do sudo "$CKPT" --toggle --pid "$p" 2>&1 | grep -v '^$' || true; done
  else
  for p in $ORDER; do t=$(now); O=$(sudo "$CKPT" --action lock --pid "$p" --timeout 120000 2>&1); echo "  lock $p rc=$? $(el $t)s $(echo $O | cut -c1-120)"; done
  for p in $ORDER; do t=$(now); O=$(sudo "$CKPT" --action checkpoint --pid "$p" 2>&1); echo "  checkpoint $p rc=$? $(el $t)s $(echo $O | cut -c1-120)"; done
  fi
  echo "$ORDER" > "$HOME/jobs/order-$PORT"
}
ckpt_up(){ # restore+unlock in the same order as checkpointed
  local ORDER; ORDER=$(cat "$HOME/jobs/order-$PORT" 2>/dev/null || echo "$HEAVY $REST")
  for p in $ORDER; do
    sudo "$CKPT" --get-state --pid "$p" 2>/dev/null | grep -qi checkpointed || continue
    if [ "${TOGGLE:-0}" = 1 ]; then sudo "$CKPT" --toggle --pid "$p" 2>&1 | grep -v '^$' || true; else
    t=$(now); O=$(sudo "$CKPT" --action restore --pid "$p" 2>&1); echo "  restore $p rc=$? $(el $t)s $(echo $O | cut -c1-120)"
    t=$(now); O=$(sudo "$CKPT" --action unlock --pid "$p" 2>&1); echo "  unlock $p rc=$? $(el $t)s $(echo $O | cut -c1-120)"
    fi
    echo "resumed $p"
  done
}
pcg_release(){ local G="$HOME/jobs/pcg-$PORT"; if ls "$G".waiting.* >/dev/null 2>&1 && [ ! -e "$G" ]; then touch "$G"; echo "pcg released"; fi; }
toggle_sets(){
  ROOT=$(root_pid)
  [ -n "$ROOT" ] || { echo "no server on port $PORT"; return 1; }
  TREE=$(pstree -p -T "$ROOT" | grep -oE "[0-9]+" | sort -un | tr '\n' ' ')
  ALLHEAVY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -un | tr '\n' ' ')
  HEAVY=""; for p in $ALLHEAVY; do case " $TREE " in *" $p "*) HEAVY="$HEAVY $p";; esac; done
  SET="$HEAVY"
  for p in $TREE; do
    case " $SET " in *" $p "*) continue;; esac
    grep -qi "/dev/nvidia" /proc/$p/maps 2>/dev/null && SET="$SET $p"
  done
  case " $SET " in *" $ROOT "*) : ;; *) SET="$SET $ROOT";; esac
  SET=$(echo $SET | tr ' ' '\n' | sort -un | tr '\n' ' ')
  HEAVY=$(echo $HEAVY | tr ' ' '\n' | sort -un | tr '\n' ' ')
  REST=$(echo $SET | tr ' ' '\n' | grep -vxF -f <(echo $HEAVY | tr ' ' '\n') | tr '\n' ' ' 2>/dev/null || echo "")
}

case "$PHASE" in
serve|serve-default)
  if [ "$PHASE" = serve ]; then
    # NVLink stays ON: P2P enabled. Legacy cudaIpc (NCCL_CUMEM_ENABLE=0) is checkpointable on driver >=610;
    # cuMemExportToShareableHandle paths (NVLS multicast, vLLM symm-mem) are not -> off. Whole tree = one checkpoint job.
    export VLLM_ALLREDUCE_USE_SYMM_MEM=0 VLLM_HOST_IP=127.0.0.1 \
      NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo ${NCCL_EXTRA:-}
    if [ "${CAR:-off}" = on ]; then EXTRA=(); else EXTRA=(--disable-custom-all-reduce); fi
    [ "${EAGER:-0}" = 1 ] && EXTRA+=(--enforce-eager)
    mkdir -p "$HOME/jobs"
    if [ "${NOJOB:-0}" = 1 ]; then unset CUDA_CHECKPOINT_JOB_FILE; echo "no job file (NOJOB=1)"; : > "$HOME/jobs/current-$PORT"; else
      JOBF="$HOME/jobs/job-$PORT-$(date +%s)"
      "$CKPT" --launch-job bash -c "cp \"\$CUDA_CHECKPOINT_JOB_FILE\" \"$JOBF\"" && export CUDA_CHECKPOINT_JOB_FILE="$JOBF" IMPALA_CKPT_JOB_FILE="$JOBF"
      echo "job file: $JOBF ($(stat -c%s "$JOBF" 2>/dev/null) bytes)"; echo "$JOBF" > "$HOME/jobs/current-$PORT"
    fi
  else
    EXTRA=()
  fi
  if [ "${PCG:-0}" = 1 ]; then export VLLM_PRECAPTURE_GATE="$HOME/jobs/pcg-$PORT"; rm -f "$VLLM_PRECAPTURE_GATE" "$VLLM_PRECAPTURE_GATE".waiting.* 2>/dev/null; fi
  CC='{"cudagraph_capture_sizes":[1,2,4,8,16,32,64]}'
  [ "${FIAR:-on}" = off ] && CC='{"cudagraph_capture_sizes":[1,2,4,8,16,32,64],"pass_config":{"fuse_allreduce_rms":false}}'
  : > "$LOGF"
  T0=$(now)
  setsid "$PY" -m vllm.entrypoints.cli.main serve "$MODEL" \
    --tensor-parallel-size "$TP" --served-model-name "$SRV" --port "$PORT" \
    --max-model-len 32768 --max-num-seqs 64 --kv-cache-memory-bytes "$KVB" \
    --compilation-config "$CC" "${EXTRA[@]}" \
    < /dev/null > "$LOGF" 2>&1 &
  SPID=$!
  if [ "${PCG:-0}" = 1 ]; then
    READY=TIMEOUT
    for i in $(seq 1 3000); do n=$(ls "$VLLM_PRECAPTURE_GATE".waiting.* 2>/dev/null | wc -l); [ "$n" -ge "$TP" ] && { READY=$(el "$T0"); break; }; kill -0 "$SPID" 2>/dev/null || { READY=DEAD; break; }; sleep 1; done
    echo "{\"phase\":\"serve-pcg\",\"port\":$PORT,\"parked_s\":${READY/TIMEOUT/null},\"waiting\":$(ls "$VLLM_PRECAPTURE_GATE".waiting.* 2>/dev/null | wc -l),\"vram\":\"$(vram)\"}"
    [ "$READY" != DEAD ] && [ "$READY" != TIMEOUT ]; exit $?
  fi
  READY=$(wait_ready "$T0" 3000 "$SPID")
  case "$READY" in DEAD|TIMEOUT) echo "server $READY:"; grep -aE "Error|error|Traceback" "$LOGF" | tail -4 | cut -c1-220; READY=null;; esac
  echo "{\"phase\":\"$PHASE\",\"port\":$PORT,\"ready_s\":$READY,\"strict\":\"$(strict)\",\"vram\":\"$(vram)\"}"
  true ;;
gate)
  toggle_sets || exit 1; echo "toggle set: $SET (heavy: $HEAVY, rest: $REST)"
  for p in $SET; do sudo prlimit --pid "$p" --memlock=unlimited:unlimited 2>/dev/null; done
  TD=$(now); ckpt_down
  echo "down: $(el $TD)s vram=$(vram)"; for p in $SET; do echo "  $p $(sudo "$CKPT" --get-state --pid $p 2>&1)"; done
  sleep 2
  TU=$(now); ckpt_up
  echo "up: $(el $TU)s vram=$(vram)"
  TG=$(now); pcg_release; READY=$(wait_ready "$TG" 900 "$ROOT"); echo "ready after release: ${READY}s"
  sleep 2; echo "{\"phase\":\"gate\",\"port\":$PORT,\"down_s\":$(el $TD),\"ready_after_release_s\":\"$READY\",\"strict\":\"$(strict)\"}" ;;
freeze)
  toggle_sets || exit 1; echo "toggle set: $SET"
  for p in $SET; do sudo prlimit --pid "$p" --memlock=unlimited:unlimited 2>/dev/null; done
  T0=$(now)
  ckpt_down
  TT=$(el $T0); echo "checkpoint passes done: ${TT}s vram=$(vram)"; for p in $SET; do echo "  $p $(sudo "$CKPT" --get-state --pid $p 2>&1)"; done
  cp "$HOME/jobs/order-$PORT" "$SNAP.order" 2>/dev/null; cp "$(cat $HOME/jobs/current-$PORT)" "$SNAP.jobfile" 2>/dev/null
  # close leftover /dev/nvidia* fds (pynvml etc.) in EVERY process of the tree — CRIU cannot dump chr 195:255 fds
  for p in $TREE; do
    for fd in $(sudo ls /proc/$p/fd 2>/dev/null); do
      L=$(sudo readlink /proc/$p/fd/$fd 2>/dev/null)
      case "$L" in /dev/nvidia*) echo "close pid=$p fd=$fd $L"; sudo gdb -p "$p" --batch -ex "call (int)close($fd)" >/dev/null 2>&1;; esac
    done
  done
  for p in $SET; do
    sudo python3 "$HOME/nonloop_socks.py" "$p" 2>/dev/null | while read -r fd ip; do
      echo "close sock pid=$p fd=$fd ip=$ip"
      sudo gdb -p "$p" --batch -ex "call (int)close($fd)" >/dev/null 2>&1
    done
  done
  # dump to tmpfs first (CRIU's 4K page writes crawl at ~170 MB/s on virtiofs), then copy to the FS with parallel cp
  DUMPDIR=/dev/shm/snap-$PORT
  sudo rm -rf "$SNAP" "$DUMPDIR"; mkdir -p "$SNAP" "$DUMPDIR"
  TD=$(now)
  sudo env "PATH=$PATH" criu dump -t "$ROOT" --images-dir "$DUMPDIR" --ghost-limit 100M --tcp-established --skip-in-flight --link-remap 2> "$HOME/criu-dump-$PORT.log"; RC=$?
  DT=$(el $TD); SZ=$(du -sb "$DUMPDIR" | cut -f1)
  echo "dump rc=$RC ${DT}s image=$(du -sh $DUMPDIR | cut -f1) (tmpfs)"
  TC=$(now)
  if [ $RC -eq 0 ]; then
    sudo chown -R spinup:spinup "$DUMPDIR"
    # big files via dd with 64 MiB direct writes (virtiofs small-write pathology; cp ran ~0.2 GB/s per file), small ones via cp
    ( cd "$DUMPDIR" && find . -maxdepth 1 -type f -size +256M -printf '%P\n' | xargs -P 8 -I{} sh -c 'dd if="{}" of="'"$SNAP"'/{}" bs=64M iflag=fullblock oflag=direct status=none' ) \
    && ( cd "$DUMPDIR" && find . -mindepth 1 -maxdepth 1 \( -type d -o -type f -size -256M \) -printf '%P\n' | xargs -P 8 -I{} cp -a "{}" "$SNAP/{}" ) && sync
    echo "copy to FS: $(el $TC)s $(du -sh $SNAP | cut -f1)"
    sudo rm -rf "$DUMPDIR"
  fi
  CT=$(el $TC)
  cp "$HOME/jobs/order-$PORT" "$SNAP/.order" 2>/dev/null; cp "$(cat $HOME/jobs/current-$PORT)" "$SNAP/.jobfile" 2>/dev/null
  [ "${TOGGLE:-0}" = 1 ] && touch "$SNAP/.toggle"
  stat -c%s "$LOGF" > "$SNAP/.logsize"
  if [ -n "${VLLM_PRECAPTURE_GATE:-}" ] || ls "$HOME/jobs/pcg-$PORT".waiting.* >/dev/null 2>&1; then echo "$HOME/jobs/pcg-$PORT" > "$SNAP/.pcg"; for w in "$HOME/jobs/pcg-$PORT".waiting.*; do cp "$w" "$SNAP/.pcg.waiting.${w##*.}"; done; fi
  hostname -I | awk '{print $1}' > "$SNAP/.hostip"
  # CRIU's --link-remap re-links this tree's deleted semaphores at their ORIGINAL /dev/shm names (deterministic across
  # trees -> the next pair's dump collides "File exists"). Back up ALL ghosts incl. sem.* then clear the remap artifacts.
  mkdir -p "$SNAP/shmfiles"; cp -a /dev/shm/psm_* /dev/shm/link_remap.* /dev/shm/sem.* "$SNAP/shmfiles/" 2>/dev/null || true
  rm -f /dev/shm/sem.* /dev/shm/link_remap.* 2>/dev/null
  sudo chown -R spinup:spinup "$SNAP" 2>/dev/null || true
  [ $RC -ne 0 ] && tail -5 "$HOME/criu-dump-$PORT.log"
  echo "{\"phase\":\"freeze\",\"port\":$PORT,\"toggle_down_s\":$TT,\"dump_s\":$DT,\"copy_s\":$CT,\"dump_rc\":$RC,\"image_bytes\":$SZ}"
  [ $RC -eq 0 ] ;;
restore)
  DIR="$SNAP"
  for f in "$DIR"/shmfiles/psm_* "$DIR"/shmfiles/link_remap.* "$DIR"/shmfiles/sem.*; do
    [ -e "$f" ] || continue; B=$(basename "$f"); [ -e "/dev/shm/$B" ] || cp -a "$f" "/dev/shm/$B"
  done
  chmod 600 /dev/shm/psm_* /dev/shm/link_remap.* /dev/shm/sem.* 2>/dev/null || true
  [ -f "$DIR/.logsize" ] && truncate -s "$(cat "$DIR/.logsize")" "$LOGF"
  if [ -f "$DIR/.hostip" ]; then
    OLDIP=$(cat "$DIR/.hostip"); CUR=$(hostname -I | awk '{print $1}')
    [ -n "$OLDIP" ] && [ "$OLDIP" != "$CUR" ] && sudo ip addr add "$OLDIP/32" dev lo 2>/dev/null && echo "aliased $OLDIP"
  fi
  [ "${DROP_CACHES:-1}" = 1 ] && { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; }
  T0=$(now)
  sudo env "PATH=$PATH" criu restore -d --tcp-established --link-remap "${EXTRA_RESTORE[@]}" --images-dir "$DIR" 2> "$HOME/criu-restore-$PORT.log"; RC=$?
  TR=$(el $T0)
  [ $RC -ne 0 ] && { echo "RESTORE FAILED rc=$RC"; grep -a -E 'Error|error' "$HOME/criu-restore-$PORT.log" | head -6; exit 1; }
  toggle_sets || exit 1
  for p in $SET; do sudo prlimit --pid "$p" --memlock=unlimited:unlimited 2>/dev/null; done
  TW=$(now)
  [ -f "$SNAP/.order" ] && cp "$SNAP/.order" "$HOME/jobs/order-$PORT"
  [ -f "$SNAP/.toggle" ] && export TOGGLE=1
  ckpt_up
  TU=$(el $TW); echo "restore+unlock passes: ${TU}s"
  [ -f "$SNAP/.pcg" ] && { rm -f "$HOME/jobs/pcg-$PORT"; mkdir -p "$HOME/jobs"; cp "$SNAP"/.pcg.waiting.* "$HOME/jobs/" 2>/dev/null; pcg_release; }
  READY=$(wait_ready "$T0" 900 "$ROOT"); case "$READY" in DEAD|TIMEOUT) READY=null;; esac
  echo "{\"phase\":\"restore\",\"port\":$PORT,\"args\":\"${EXTRA_RESTORE[*]:-}\",\"criu_restore_s\":$TR,\"toggle_up_s\":$TU,\"ready_s\":$READY,\"strict\":\"$(strict)\",\"vram\":\"$(vram)\"}" ;;
strictonly)
  strict ;;
kill)
  ROOT=$(root_pid); [ -n "$ROOT" ] && { TREE=$(pstree -p -T "$ROOT" | grep -oE "[0-9]+" | sort -un); sudo kill -9 $TREE 2>/dev/null; }
  for i in $(seq 1 60); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G0,$G1 | sort -n | tail -1); [ "$U" -lt 1024 ] && break; sleep 2; done
  echo "killed port $PORT; vram max ${U:-?} MiB" ;;
*) echo unknown; exit 1;;
esac
