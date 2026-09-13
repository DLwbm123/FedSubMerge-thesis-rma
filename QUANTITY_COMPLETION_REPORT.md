# Hyper-Kvasir quantity-skew completion

All eight jobs completed with exit code 0: four continual sequences and four independently initialized task references. Validated 400 ordered continual round records, 20 ordered stage records, and 80 reference round records. All 24 required checkpoints exist with nonzero size. Recomputed BWTR, MPE and representation-aware DRR; paired RMA using exact source training indices, task classes, seed, training budget, dimensions and parameter count. No training was repeated.

| Method | Measured ACC (%) | RMA | BWTR | MPE | DRR |
|---|---:|---:|---:|---:|---:|
| FedAvg | 36.229897 | 1.006254 | -0.600038 | 0.000000 | 0.000000 |
| Fed-GPM | 49.263593 | 1.056602 | -0.441206 | 0.000000 | 0.086750 |
| FedSubMerge | 42.016791 | 1.051270 | -0.540314 | 0.000000 | 1.000000 |
| FedSubMerge-AD | 41.243331 | 1.055869 | -0.550971 | 0.000000 | 1.000000 |

These are the specific runs described in QUANTITY_EXPERIMENTS.md. Their measured accuracies differ from the historical thesis values; historical accuracy is not replaced and the original thesis rows remain unpaired. They must not be interpreted as one experiment with a mixed accuracy matrix.

MPE is zero because the classifier network is fixed; auxiliary bases are counted separately. FedSubMerge and FedSubMerge-AD DRR is one because all available current-task training samples at each client fit within the 256-sample representation budget for non-final tasks. This does not mean historical original images are retained. The final task has 87 selected training images and 32 global test images; no statistical-significance claim is made from one seed.

Published: collector, reproducible protocol, numerical round/stage records, accuracy matrices, taskwise RMA/DRR ratios, reference results and summary CSV. Excluded: original images, private source-index lists, weights, full logs and credentials. Those artifacts remain in the existing remote-home directory; no files were deleted or relocated. The clean local thesis table was updated only in the four newly available quantity-skew rows, preserving original ACC.

Remote root: `/remote-home/wangbomin/FedSubMerge_thesis/runs/quantity_20260912`.
