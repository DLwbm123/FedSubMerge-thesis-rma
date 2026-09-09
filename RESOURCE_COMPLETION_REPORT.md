# Completed resource and metric integration batch

All eight runs completed with exit code 0. Checked 1,120 round records, 56 stage records, finite accuracy matrices, nonempty stage checkpoints, and recomputed MPE, DRR and BWTR. No training was repeated.

RMA is now paired for this controlled batch: seed, training budget, learning rate, data root, task classes, image shape, per-client source indices, split counts and parameter count match the earlier independent-task references. Both runners use the same ResNet18 construction, preprocessing helper, full-output loss and masked evaluation. This does not validate arbitrary historical thesis-table rows.

| Dataset | Method | ACC (%) | RMA | BWTR | MPE | DRR |
|---|---|---:|---:|---:|---:|---:|
| hyperkvasir | Fed-GPM | 70.0686 | 0.986750 | -0.185159 | 0.000000 | 0.239270 |
| hyperkvasir | FedAvg | 64.0567 | 0.936150 | -0.245536 | 0.000000 | 0.000000 |
| hyperkvasir | FedSubMerge | 71.1912 | 0.998177 | -0.199687 | 0.000000 | 0.947698 |
| hyperkvasir | FedSubMerge-AD | 69.6154 | 0.988836 | -0.207921 | 0.000000 | 0.947698 |
| pathmnist | Fed-GPM | 71.9660 | 0.997963 | -0.329832 | 0.000000 | 0.004853 |
| pathmnist | FedAvg | 65.9942 | 0.990817 | -0.408588 | 0.000000 | 0.000000 |
| pathmnist | FedSubMerge | 75.7673 | 1.001651 | -0.281783 | 0.000000 | 0.116853 |
| pathmnist | FedSubMerge-AD | 78.9334 | 0.988636 | -0.228798 | 0.000000 | 0.116853 |

MPE, DRR, RMA and BWTR above are ratios, not percentages. The four methods share a fixed network architecture, so all MPE values are zero. DRR includes source samples used to construct representations; it does not assert raw-image retention. Hyper-Kvasir has small later tasks, making the mean source-task DRR high even with a cap of 256 examples per client.

## What remains for the thesis

- No additional training is necessary to provide a single-seed ACC/RMA/BWTR/MPE/DRR table for these two alpha=0.3 settings and four methods.
- Keep these controlled-protocol results separate from older tables with different rounds, transforms or subspace settings. See RESOURCE_EXPERIMENTS.md for the exact source implementation and controlled overrides.
- Three-seed mean and standard deviation would require two further seeds, including matched NC references. This is an optional robustness extension, not a missing metric definition.
- Shared-GPU wall times are descriptive. An isolated hardware timing comparison would need representative isolated profiling, not necessarily full retraining.
- Final Hyper-Kvasir tasks have only 19 and 13 test images. Report these sample counts and retain taskwise ratios; do not treat single-seed aggregate rankings as statistical significance.

## Published and retained artifacts

Published: collector, sanitized configurations, 1,120 round metrics, 56 stage summaries, per-layer subspace dimensions, accuracy matrices, paired RMA ratios and summary CSV. Existing launch code and protocol are in the same repository. Excluded: raw images, exact source-index lists, weights, third-party implementation snapshot, credentials and full logs. Large artifacts remain under the remote batch directory. Checkpoints were checked for presence/size, not reloaded or byte-hashed.

Remote batch: `/remote-home/wangbomin/FedSubMerge_thesis/runs/resources_20260909`.

[SwanLab](https://swanlab.cn/@wangbomin/FedSubMerge-Thesis-Resources).
