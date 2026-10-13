# vLLM patches

`git format-patch` series against vLLM v0.20.2, the tree every number was measured on.

| file | what |
|---|---|
| `0001-Loader-Pinned-staging-*.patch` | 16 GB page-locked staging buffer for the Run:ai streamer path (env-gated) + weight checksum hook |
| `0002-Loader-Per-rank-byte-range-reader-*.patch` | per-rank byte-range reader for TP>1 (opt-in; did not pay, kept for the record) |
| `0003-Reload-Keep-captured-graph-tensors-valid-*.patch` | the sleep-reload fix: derived tensors re-derived in place, exact per-layer reload totals |
| `0004-Loader-Run-ai-streamer-loader-get_all_weights-*.patch` | lets `reload_weights` run through the streamer (and the pinned path) |
| `0005-upstream-57635-*.diff` | vLLM #57635 (per-rank FlashInfer autotune cache files) applied to 0.31.0 |
| `0006-exp6-vllm-0.31.0-*.diff` | the full set of 0.31.0 venv edits from the upgrade probe |

Upstream pull requests against `vllm-project/vllm` `main` (rebased, pending re-measure on hardware):

* pinned-staging loader: see the "What we're open-sourcing" section of the blog post for the current link
* sleep-reload fix: same

Apply the 0.20.2 series with `git am` on tag `v0.20.2`; `0003` depends on `0001`.
