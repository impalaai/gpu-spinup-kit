#!/usr/bin/env bash
# build_criu_hp.sh — build criu e1b3eb0a3 (premap-threads + premap-hugepage) as /usr/local/bin/criu-hp. Marker in log.
set -u
exec > /tmp/build_criu_hp.log 2>&1
cd /home/spinup && rm -rf criu-hp && tar xzf /tmp/criu-hp-e1b3eb0a3.tar.gz && cd criu-hp && \
  make -j"$(nproc)" >/tmp/criu-hp-make.log 2>&1 && sudo install -m755 criu/criu /usr/local/bin/criu-hp && \
  /usr/local/bin/criu-hp --help 2>&1 | grep -A2 premap-hugepage && echo BUILD_OK || { tail -20 /tmp/criu-hp-make.log; echo BUILD_FAIL; }
/usr/local/bin/criu --version | head -1; /usr/local/bin/criu-hp --version | head -1
