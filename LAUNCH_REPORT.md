# RMA reference batch launch — 2026-09-08

Verified startup: **2 training processes running, 10 tasks queued**. Each
running process completed its first forward/backward/optimizer step, and
SwanLab confirmed cloud run creation. No full experiment has completed yet.

| Dataset | GPU | Requested tasks (one-based) | tmux session | First running SwanLab run |
|---|---:|---|---|---|
| PathMNIST alpha=0.3 | 2 | 2, 3, 4 | `rma-nc-pathmnist-20260908` | [Task 2](https://swanlab.cn/@wangbomin/FedSubMerge-Thesis-RMA/runs/c13lxzbtu67kphpw80t43) |
| Hyper-Kvasir alpha=0.3 | 3 | 2–10 | `rma-nc-hyperkvasir-20260908` | [Task 2](https://swanlab.cn/@wangbomin/FedSubMerge-Thesis-RMA/runs/j5a7g89o61y1f7yxuspld) |

[SwanLab project](https://swanlab.cn/@wangbomin/FedSubMerge-Thesis-RMA).

Remote batch directory:
`/remote-home/wangbomin/FedSubMerge_thesis/runs/rma_nc_20260908_211710`

- Logs: `logs/pathmnist_task02.log`, `logs/hyperkvasir_task02.log`, and
  corresponding files for subsequent tasks.
- Queue state: `queue_pathmnist.json`, `queue_hyperkvasir.json`.
- Per-task output: `{dataset}_task{NN}_seed2025/` (config, metrics, rolling
  checkpoint, final result, SwanLab local logs).
- Runtime code and original implementation snapshot: `code/`, `base_source/`.
- Existing environment: `/root/anaconda3/bin/python`; no packages installed.
- Output mount and write/read probe passed on the actual NFS target.

Configuration: seed 2025, 20 rounds, 3 local epochs, batch 64, ten clients,
full participation of nonempty clients, sample-weighted FedAvg, fresh random
model per task, no PGS/replay. First measured training batches had finite
losses: PathMNIST 3.11417; Hyper-Kvasir 3.83415. These startup losses are
diagnostics, not experimental results.

The synthetic GPU-3 full-batch feasibility check at 64x3x256x256 completed
forward/backward/update with peak tensor allocation 18,841 MiB; the runtime
uses the same cuDNN algorithm-search setting. No existing processes were
changed. Queues survive SSH/client disconnection and stop on errors.

Scientific pairing remains pending: these references use the thesis's
20-round budget and 128/256 input sizes, whereas historical runs vary.
Hyper-Kvasir is resized from an existing 224 cache; existing client maps
retain their local holdouts. See README.md before combining the resulting
denominators with historical accuracy values. No RMA values are claimed.

Full result validation and public GitHub delivery are pending actual
completion. No monitoring automation was created and no ongoing polling is
required for these finite queues to continue.
