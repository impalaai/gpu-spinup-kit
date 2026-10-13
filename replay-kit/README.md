# replay-kit: GPU-state snapshot boot replay

Checkpoint a fully warmed vLLM server (CPU state with CRIU, GPU state with NVIDIA
`cuda-checkpoint`) onto a golden disk or shared filesystem, and restore it at boot from a
systemd unit, so a freshly created VM is serving within seconds of the OS coming up. No
orchestrator; one `instance create` and a correct answer at the end.

## Variants

| directory | host | image location | what it does |
|---|---|---|---|
| `nebius-h100/` | 1×H100, Nebius | pulled from object storage into RAM at boot | the original self-restoring golden disk (108.8 s create→health) |
| `verda-nvme/` | 1×H100, local NVMe | on the OS volume | adds the **port gate**, a real `cuInit` readiness probe and retrying toggles; restore with `--premap-threads` |
| `nebius-b300-sleep/` | 8×B300, NVLink on, Nebius shared FS | four TP=2 images on the shared filesystem | four replicas restored in parallel with per-pair NUMA binding, huge pages, pipelined GPU restores; `autorestore_install.sh` is the **sleeping-image** unit (17 GB images, wake + in-place reload), `autorestore_install_full_images.sh` the full-image one (491 GB per pair) |

`tools/` holds the helpers the units rely on:

* `nonloop_socks.py <pid>`: lists sockets bound to a non-loopback IP (they pin the dump-time address and break cross-boot restore).
* `jobenv_patch.py`: re-exports `CUDA_CHECKPOINT_JOB_FILE` inside every vLLM process; the driver consumes the variable at CUDA init, so spawned workers otherwise never join the checkpoint job.
* `precapture_patch.py`: an env-gated pause right before CUDA-graph capture, to checkpoint a worker tree before graphs exist.
* `build_criu_hp.sh`: builds CRIU with the two restore patches from `../patches/criu`.

## How a replay boots (B300 sleeping-image unit)

1. Gate every serving port with `iptables ... ! -i lo -j DROP`. CRIU revives the listening socket seconds before the GPU state is back; a request in that window poisons the rank. Loopback stays open so the unit's own checks work.
2. Reseed the `cuda-checkpoint` job file for each image from the copy saved at dump time. The driver rewrites the job file on every restore attempt, so the saved copy must go back before each restore.
3. Recreate `/dev/shm` ghosts (`psm_*`, `link_remap.*`, `sem.*`) in a private tmpfs per pair, alias the dump-time IP on `lo`, truncate append-mode logs to their dump-time size, push `ns_last_pid` past the image pids.
4. `criu restore --premap-threads N --premap-hugepage` for all four images in parallel, each bound to its GPUs' NUMA node.
5. Wait for `cuInit(0)` to succeed and for the fabric manager to report `Completed` on all GPUs, then `cuda-checkpoint --action restore` + `unlock` per worker, in dump order, as each pair's CRIU restore returns.
6. For sleeping images: prewarm the model into the page cache while the GPU leg runs, then `POST /wake_up` and `POST /collective_rpc reload_weights` per pair.
7. Health check, then a real completion with a known answer; open the port only if the content is right.

## Trap ledger

Each one cost a debugging round. Each has a one-line fix.

* **NVLink fabric.** On VMs whose GPUs sit on the provider's NVLink fabric, `cuda-checkpoint` fails with `NV_ERR_FABRIC_STATE_OUT_OF_SYNC`. Driver 615's job mode handles IPC memory with NVLink on; on older drivers, `NVreg_NvLinkDisable=1` works at the cost of slower tensor-parallel traffic.
* **One vLLM fusion pass is not restorable.** The FlashInfer all-reduce + RMSNorm fusion's workspace cannot be brought back. Turn that pass off (−11 % throughput at TP=2) and keep full CUDA graphs. Eager mode is 11× slower and never an option.
* **Symmetric-memory all-reduce probe.** `VLLM_ALLREDUCE_USE_SYMM_MEM=0`, or TP=2 restore fails with a fake CUDA OOM.
* **Leftover NVML handles.** Ranks leave two `/dev/nvidiactl` descriptors open that nobody manages. Close them before the dump.
* **Job files are consumed.** The first restore from each job file works and every later one fails. Reseed before every restore.
* **In-job restores are sequential** on driver 615. Parallel restores fail roughly one time in three. Pipeline across pairs instead.
* **A restored tree cannot be frozen again** on that boot. Snapshots come from freshly served replicas.
* **Health is not readiness.** Gate the port until the GPU state is back; keep loopback open or your own health check locks you out.
* **`nvidia-smi` is not a CUDA readiness gate.** Toggling before `cuInit(0)` succeeds fails with "OS call failed" and poisons the rank. Use `cuinit-probe`.
* **Small images beat the fabric manager.** A 17 GB image restores before the fabric manager finishes on a fresh boot; wait for `Fabric State: Completed`.
* **Append-mode logs** must be exactly their dump-time size. **`s5cmd` drops file modes**, which CRIU validates. **Boot-time processes grab image pids**: bump `ns_last_pid`.
* **Images are bound to GPU topology.** A 1-GPU image does not restore on an 8-GPU preset.
* **Transparent huge pages never engage** on CRIU's premap path, and the final `mremap` splits any that did; `--premap-hugepage` fixes both. Without it the driver repopulates VRAM at a third of its rate.
* **The driver's PTX JIT cache is warm state.** Sharing `~/.nv/ComputeCache` between two CUDA stacks cost 50–60 s per boot. Use a per-venv `CUDA_CACHE_PATH`.
* **Read your config back.** Three "new" timed runs measured the old path because a `sed` matched nothing.
