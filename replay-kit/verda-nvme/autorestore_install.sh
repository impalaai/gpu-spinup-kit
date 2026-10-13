#!/usr/bin/env bash
# Installs the boot-time autonomous CRIU replay onto this Verda golden volume.
# Verda flavor: image lives ON the OS volume (/root/snap-tp1) — no pull, no FS mount.
# RESTORE_ARGS is parameterized; default = --premap-threads 12 (flip after matrix results).
set -uo pipefail
RESTORE_ARGS=${RESTORE_ARGS:---premap-threads 12}

# cuinit-probe: true CUDA readiness gate (dlopen libcuda, cuInit(0)) — nvidia-smi -L is NOT
# sufficient: toggling before CUDA is truly up fails with "OS call failed", and a failed
# toggle POISONS the rank (retry exits 0 but engine dies on first request).
if [ ! -x /usr/local/bin/cuinit-probe ]; then
  cat > /tmp/cuinit-probe.c <<'EOF'
#include <dlfcn.h>
#include <stdio.h>
int main(void){
  void *h = dlopen("libcuda.so.1", RTLD_NOW);
  if(!h){ fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
  int (*cuInit)(unsigned) = (int(*)(unsigned))dlsym(h, "cuInit");
  if(!cuInit){ fprintf(stderr, "no cuInit\n"); return 3; }
  int rc = cuInit(0);
  if(rc){ fprintf(stderr, "cuInit=%d\n", rc); return 1; }
  return 0;
}
EOF
  gcc -O2 -o /usr/local/bin/cuinit-probe /tmp/cuinit-probe.c -ldl || exit 1
fi
/usr/local/bin/cuinit-probe && echo "cuinit-probe ok now" || echo "cuinit-probe rc=$? now (gate will wait at boot)"
tee /usr/local/bin/criu-autorestore.sh >/dev/null <<SCRIPT
#!/usr/bin/env bash
# Boot-time: jump pid counter past image pids, restore from on-volume image, toggle-up.
set -uo pipefail
exec > /var/log/criu-autorestore.log 2>&1
echo "start \$(date -u +%T)"
echo 10000 | tee /proc/sys/kernel/ns_last_pid   # image pids stay free forever
export PATH=/root/venv/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
T0=\$(date +%s)
DIR=/root/snap-tp1
# GATE: CRIU revives the API listener BEFORE cuda-checkpoint toggles; any request reaching
# the still-checkpointed engine wedges a thread mid-CUDA-call and the first toggle then
# fails with "OS call failed" — permanently poisoning the rank. Block the port until safe.
iptables -I INPUT -p tcp --dport 8200 ! -i lo -j DROP && echo "port gated"
echo "on-volume image, no pull +\$(( \$(date +%s) - T0 ))s"
for f in "\$DIR"/shmfiles/psm_* "\$DIR"/shmfiles/link_remap.*; do
  [ -e "\$f" ] || continue; B=\$(basename "\$f"); [ -e "/dev/shm/\$B" ] || cp -a "\$f" "/dev/shm/\$B"
done
rm -f /dev/shm/sem.* /dev/shm/*.cr_link 2>/dev/null
chmod 600 /dev/shm/psm_* /dev/shm/link_remap.* 2>/dev/null || true
[ -f "\$DIR/.logsize" ] && truncate -s "\$(cat "\$DIR/.logsize")" /root/tp1-serve.log
if [ -f "\$DIR/.hostip" ]; then
  OLDIP=\$(cat "\$DIR/.hostip"); CUR=\$(hostname -I | awk '{print \$1}')
  [ -n "\$OLDIP" ] && [ "\$OLDIP" != "\$CUR" ] && ip addr add "\$OLDIP/32" dev lo 2>/dev/null && echo "aliased \$OLDIP"
fi
ionice -c1 -n0 nice -n -10 criu restore -d --tcp-established --link-remap $RESTORE_ARGS --images-dir "\$DIR" 2> /var/log/criu-restore-boot.log; RC=\$?
echo "criu restore rc=\$RC +\$(( \$(date +%s) - T0 ))s"
[ \$RC -ne 0 ] && { grep -a Error /var/log/criu-restore-boot.log | head -4; exit 1; }
# dark-pool starts reach the unit before CUDA is truly up. nvidia-smi is NOT a sufficient
# gate; require an actual cuInit(0) success. A toggle attempted too early fails AND poisons
# the rank permanently (retry exits 0, engine dies on first request) — prevention only.
nvidia-modprobe -u -c=0 2>/dev/null || modprobe nvidia-uvm 2>/dev/null || true
for i in \$(seq 1 90); do /usr/local/bin/cuinit-probe 2>/dev/null && { echo "cuda up +\$(( \$(date +%s) - T0 ))s"; break; }; sleep 1; done
CKPT=/usr/local/bin/cuda-checkpoint
HEAVY=\$(ps -eo pid,comm | awk '\$2 ~ /EngineCor|Worker_TP/ {print \$1}' | sort -un | tr '\n' ' ')
RT=\$(pgrep -f "vllm.entrypoints.cli.main serve" | sort -n | head -1)
for p in \$HEAVY \$RT; do prlimit --pid "\$p" --memlock=unlimited:unlimited 2>/dev/null; done
toggle_up(){ local p=\$1 a
  for a in \$(seq 1 12); do
    "\$CKPT" --get-state --pid "\$p" 2>/dev/null | grep -qi checkpointed || { echo "resumed \$p (attempt \$a)"; return 0; }
    "\$CKPT" --toggle --pid "\$p" 2>&1 && { echo "resumed \$p (attempt \$a)"; return 0; }
    sleep 2
  done
  echo "TOGGLE FAILED pid=\$p after 12 attempts"; return 1
}
for p in \$HEAVY; do toggle_up "\$p"; done
[ -n "\$RT" ] && "\$CKPT" --get-state --pid "\$RT" 2>/dev/null | grep -qi checkpointed && toggle_up "\$RT"
for i in \$(seq 1 120); do curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8200/health && break; sleep 1; done
iptables -D INPUT -p tcp --dport 8200 ! -i lo -j DROP && echo "port opened +\$(( \$(date +%s) - T0 ))s"
echo "READY +\$(( \$(date +%s) - T0 ))s"
SCRIPT
chmod +x /usr/local/bin/criu-autorestore.sh
tee /etc/systemd/system/criu-autorestore.service >/dev/null <<'UNIT'
[Unit]
Description=Autonomous CRIU vLLM replay at boot
After=network-online.target
Wants=network-online.target
[Service]
Type=oneshot
ExecStart=/usr/local/bin/criu-autorestore.sh
RemainAfterExit=yes
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload && systemctl enable criu-autorestore.service
echo "--- installed unit restores with: $RESTORE_ARGS"
grep -a "criu restore" /usr/local/bin/criu-autorestore.sh
echo INSTALL_DONE
