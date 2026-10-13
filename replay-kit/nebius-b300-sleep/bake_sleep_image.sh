#!/usr/bin/env bash
# exp2_glm_sleep.sh — H2b-2 bake: kill everything, prewarm the model, serve 4 GLM TP=2 pairs with --enable-sleep-mode (job mode,
# CAR on, FIAR off, KV 8 GiB = the warm compile key), strict each, sleep level 2 each, freeze each sequentially into
# /mnt/fs/snapL2-<port> (never touches snap-<port>), report image sizes. Marker ~/exp2-glm-bake.done.
set -uo pipefail
exec > ~/exp2-glm-bake.log 2>&1
export CAR=on FIAR=off SLEEP=1
for i in 0 1 2 3; do PORT=$((8201+i)) ~/exp2_criu.sh kill $i; done; PORT=8205 ~/exp2_criu.sh kill 3
sudo rm -f /dev/shm/psm_* /dev/shm/sem.* /dev/shm/link_remap.* /dev/shm/*.cr_link 2>/dev/null
T=$(date +%s); ls /mnt/fs/glm52-nvfp4/* | xargs -P 16 -n 1 cat > /dev/null; echo "$(date -u +%T) prewarm $(( $(date +%s) - T ))s"
TS=$(date +%s)
for i in 0 1 2 3; do ( ~/exp2_criu.sh serve $i > ~/exp2-serve-$((8201+i)).out 2>&1 ) & done; wait
echo "$(date -u +%T) all serves returned +$(( $(date +%s) - TS ))s"
for i in 0 1 2 3; do echo "serve $((8201+i)): $(tail -1 ~/exp2-serve-$((8201+i)).out)"; done
for i in 0 1 2 3; do echo "sleep $((8201+i)): $(~/exp2_criu.sh sleep2 $i)"; done
# verify the sleep -> wake -> reload_weights path on ONE pair before freezing anything (reload bug check), then re-sleep it
echo "$(date -u +%T) wake/reload check on 8201: $(~/exp2_criu.sh wake 0)"
grep -q STRICT_OK <<< "$(tail -1 ~/exp2-serve-8201.out)" 2>/dev/null; W=$(~/exp2_criu.sh strictonly 0); echo "post-reload strict 8201: $W"
case "$W" in STRICT_OK) echo "sleep 8201 again: $(~/exp2_criu.sh sleep2 0)";; *) echo "RELOAD CHECK FAILED — not freezing"; touch ~/exp2-glm-bake.done; echo BAKE_ABORT; exit 1;; esac
for i in 0 1 2 3; do P=$((8201+i)); echo "$(date -u +%T) freeze $P"; SNAP=/mnt/fs/snapL2-$P ~/exp2_criu.sh freeze $i 2>&1 | grep -vE "^close " | tail -5; done
du -sh /mnt/fs/snapL2-*; ls -la /mnt/fs/snapL2-8201 | head -20
touch ~/exp2-glm-bake.done; echo BAKE_DONE
