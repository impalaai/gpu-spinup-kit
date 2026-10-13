# gpu-spinup-kit

Scripts, patches and field notes from Impala AI's research into fast vLLM cold starts on
cloud GPUs: a 1×H100 that goes from "create instance" to a serving 27B model in about 64 s
with a GPU-state snapshot, and an 8×B300 node that boots four tensor-parallel replicas of a
433 GB model in about two minutes either way.

Companion write-up: *Don't Build On It: How a Fresh H100 Learned to Wake Up Warm* (Impala AI blog).

| directory | what it is |
|---|---|
| [`replay-kit/`](replay-kit/) | GPU-state snapshot boot replay: CRIU + NVIDIA `cuda-checkpoint` images restored by a systemd unit at boot, for Nebius (shared filesystem) and local-NVMe hosts, single-GPU and multi-GPU with NVLink on; the port gate; the trap ledger |
| [`b300-boot-kit/`](b300-boot-kit/) | Tuned cold boot of GLM-5.2-NVFP4 on an 8×B300 preemptible node (four TP=2 replicas or one TP=8), with the venv patches that make it fast: pinned-staging loader, forked workers, paced prewarm, FlashInfer autotune-cache backport |
| [`patches/`](patches/) | The CRIU patches (`--premap-threads`, `--premap-hugepage`) and vLLM patches (pinned loader, sleep-reload fix) as `git format-patch` series, as filed upstream |

Related: [`spinup-profiler`](https://github.com/impalaai/spinup-profiler), the stage-level vLLM cold-start profiler used throughout.

Upstream status:

* CRIU `--premap-threads`: [checkpoint-restore/criu#3208](https://github.com/checkpoint-restore/criu/pull/3208) (draft)
* CRIU `--premap-hugepage`: [checkpoint-restore/criu#3209](https://github.com/checkpoint-restore/criu/pull/3209) (draft)
* vLLM pinned-staging loader and sleep-reload fix: see `patches/vllm/README.md` for the current PR links

Everything here was written during live experiments and is published as-is: it is a record of
what worked, with the exact flags, not a packaged product. Paths assume a `spinup` user with a
venv at `~/venv`, a shared filesystem at `/mnt/fs`, and images named `<prefix><port>`. Read a
script before you run it.

Licensed under Apache-2.0.
