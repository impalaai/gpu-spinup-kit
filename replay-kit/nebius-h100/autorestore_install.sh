#!/usr/bin/env bash
# Installs the boot-time autonomous CRIU replay onto this (golden) disk.
set -uo pipefail
: "${SNAP_S3_BASE:?set SNAP_S3_BASE=s3://<bucket>/<image-prefix>}"
sudo tee /usr/local/bin/criu-autorestore.sh >/dev/null <<SCRIPT
#!/usr/bin/env bash
# Boot-time: jump pid counter past image pids, pull image from S3, restore, toggle-up.
set -uo pipefail
exec > /var/log/criu-autorestore.log 2>&1
echo "start \$(date -u +%T)"
echo 10000 | tee /proc/sys/kernel/ns_last_pid   # image pids (2300-2600) stay free forever
export PATH=/home/spinup/venv/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
set -a; . /home/spinup/nebius-s3.env; set +a
BASE=$SNAP_S3_BASE
T0=\$(date +%s)
mkdir -p /dev/shm/snap-pull
BIG=pages-12.img
( s5cmd --endpoint-url "\$AWS_ENDPOINT_URL" --numworkers 16 cp --concurrency 16 --part-size 128 "\$BASE/\$BIG" /dev/shm/snap-pull/ ) &
( s5cmd --endpoint-url "\$AWS_ENDPOINT_URL" --numworkers 64 cp --exclude "\$BIG" "\$BASE/*" /dev/shm/snap-pull/ ) &
wait
echo "pull done +\$(( \$(date +%s) - T0 ))s"
DIR=/dev/shm/snap-pull
for f in "\$DIR"/shmfiles/psm_* "\$DIR"/shmfiles/link_remap.*; do
  [ -e "\$f" ] || continue; B=\$(basename "\$f"); [ -e "/dev/shm/\$B" ] || cp -a "\$f" "/dev/shm/\$B"
done
rm -f /dev/shm/sem.* /dev/shm/*.cr_link 2>/dev/null
chmod 600 /dev/shm/psm_* /dev/shm/link_remap.* 2>/dev/null || true
[ -f "\$DIR/.logsize" ] && truncate -s "\$(cat "\$DIR/.logsize")" /home/spinup/tp1-serve.log && chown spinup:spinup /home/spinup/tp1-serve.log
if [ -f "\$DIR/.hostip" ]; then
  OLDIP=\$(cat "\$DIR/.hostip"); CUR=\$(hostname -I | awk '{print \$1}')
  [ -n "\$OLDIP" ] && [ "\$OLDIP" != "\$CUR" ] && ip addr add "\$OLDIP/32" dev lo 2>/dev/null && echo "aliased \$OLDIP"
fi
criu restore -d --tcp-established --link-remap --images-dir "\$DIR" 2> /var/log/criu-restore-boot.log; RC=\$?
echo "criu restore rc=\$RC +\$(( \$(date +%s) - T0 ))s"
[ \$RC -ne 0 ] && { grep -a Error /var/log/criu-restore-boot.log | head -4; exit 1; }
CKPT=/usr/local/bin/cuda-checkpoint
HEAVY=\$(ps -eo pid,comm | awk '\$2 ~ /EngineCor|Worker_TP/ {print \$1}' | sort -un | tr '\n' ' ')
RT=\$(pgrep -f "vllm.entrypoints.cli.main serve" | sort -n | head -1)
for p in \$HEAVY \$RT; do prlimit --pid "\$p" --memlock=unlimited:unlimited 2>/dev/null; done
for p in \$HEAVY; do "\$CKPT" --toggle --pid "\$p" && echo "resumed \$p"; done
[ -n "\$RT" ] && "\$CKPT" --get-state --pid "\$RT" 2>/dev/null | grep -qi checkpointed && "\$CKPT" --toggle --pid "\$RT"
for i in \$(seq 1 120); do curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8200/health && break; sleep 1; done
echo "READY +\$(( \$(date +%s) - T0 ))s"
SCRIPT
sudo chmod +x /usr/local/bin/criu-autorestore.sh
sudo tee /etc/systemd/system/criu-autorestore.service >/dev/null <<'UNIT'
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
sudo systemctl daemon-reload && sudo systemctl enable criu-autorestore.service
echo INSTALL_DONE
