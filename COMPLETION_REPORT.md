# Independent RMA references: completed results

All 12 requested independent-task FedAvg runs completed with exit code 0. Each has 20 round records and a nonempty final checkpoint on my-gpu. No training was repeated during closeout.

Both queues started on 2026-09-08 at approximately 21:21 China Standard Time. Hyper-Kvasir finished at 22:37:42; PathMNIST finished at 23:42:33. Total batch wall time was approximately 2 hours 21 minutes, with the queues running in parallel.

| Dataset | Task | Train N | Test N | Test accuracy (%) | Training minutes |
|---|---:|---:|---:|---:|---:|
| pathmnist | 2 | 16609 | 973 | 98.8695 | 52.66 |
| pathmnist | 3 | 16150 | 1627 | 100.0000 | 35.79 |
| pathmnist | 4 | 24138 | 2395 | 88.3925 | 51.43 |
| hyperkvasir | 2 | 1128 | 402 | 98.2587 | 13.73 |
| hyperkvasir | 3 | 1116 | 398 | 96.9849 | 18.59 |
| hyperkvasir | 4 | 950 | 339 | 99.4100 | 15.90 |
| hyperkvasir | 5 | 612 | 218 | 89.4495 | 10.78 |
| hyperkvasir | 6 | 444 | 159 | 100.0000 | 5.53 |
| hyperkvasir | 7 | 260 | 92 | 100.0000 | 3.36 |
| hyperkvasir | 8 | 151 | 53 | 96.2264 | 2.18 |
| hyperkvasir | 9 | 55 | 19 | 52.6316 | 1.21 |
| hyperkvasir | 10 | 37 | 13 | 46.1538 | 2.06 |

## Interpretation and remaining work

- These are the single-task reference accuracies a_i^NC, not final RMA values. Historical continual-learning initial-task accuracies still need matching by split, model, preprocessing, budget, seed and inference semantics before division.
- This batch uses 20 rounds and Hyper-Kvasir 256x256 inputs resized from the existing 224x224 cache; historical exports vary. Do not claim direct comparability to every old thesis table row.
- Hyper-Kvasir tasks 9 and 10 contain only 19 and 13 test images, respectively. Their accuracies are 10/19 and 6/13; report these denominators and do not infer stable performance from these small samples.
- Runs use seed 2025 only. No multi-seed uncertainty estimate is claimed.
- GPU memory and elapsed time were measured on shared GPUs and do not establish isolated hardware efficiency.

## Validation and artifacts

Checked all 12 queue exit codes, all result records, complete round IDs 1 through 20 (240 records total), finite reference accuracies, and checkpoint presence/size. Checkpoints were not reloaded or byte-hashed during this low-cost closeout.

Published scope: experiment runner, queue launcher, small correctness check, protocol/launch/completion notes, sanitized configurations, round metrics and final reference table. Raw image data, exact training sample-index lists, original implementation snapshot, model weights, credentials and full runtime logs are excluded. The reusable original implementation is required as --source-root; see README.md.

Remote artifacts: `/remote-home/wangbomin/FedSubMerge_thesis/runs/rma_nc_20260908_211710/`.

[SwanLab project](https://swanlab.cn/@wangbomin/FedSubMerge-Thesis-RMA); individual run links are in the CSV.
