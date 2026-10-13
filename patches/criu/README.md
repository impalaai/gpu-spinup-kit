# CRIU patches

Six commits on top of `criu-dev` (88eae6b1b, 2026-10-13), as filed:

* `0001`–`0003`: `--premap-threads N` + record sizes in the restore log + man page → [checkpoint-restore/criu#3208](https://github.com/checkpoint-restore/criu/pull/3208)
* `0004`–`0006`: `--premap-hugepage` + phase-congruent placement + man page → [checkpoint-restore/criu#3209](https://github.com/checkpoint-restore/criu/pull/3209)

Build: `git clone -b criu-dev https://github.com/checkpoint-restore/criu && cd criu && git am ../patches/criu/*.patch && make -j`.
Use: `criu restore --premap-threads 12 --premap-hugepage ...` on plain, uncompressed images without a parent chain.
