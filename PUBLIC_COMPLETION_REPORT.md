# Public baseline batch completion

All six alpha=0.3 sequences completed with exit code 0. Validated 840 ordered round records, 42 ordered stage records, finite task accuracy matrices and nonempty per-task checkpoints. Recomputed MPE, BWTR and the recorded DRR convention. RMA references match seed, training budget, preprocessing dimensions, task classes, client training indices and classifier parameter count. No completed training was rerun.

These are the recorded public-source adaptations in PUBLIC_BASELINES.md; their accuracy values do not replace historical thesis results. Metrics are single-seed ratios except ACC, which is in percent. TARGET's generated-image attribution proxy is retained separately in the CSV/JSON and is excluded from the common DRR column.

| Dataset | Method | ACC (%) | RMA | BWTR | MPE | DRR |
|---|---|---:|---:|---:|---:|---:|
| pathmnist | FOT | 51.118696 | 0.999621 | -0.612666 | 0.000000 | 1.000000 |
| pathmnist | Fed-DER | 91.280159 | 0.973353 | -0.048850 | 0.000000 | 0.074901 |
| hyperkvasir | FOT | 62.969072 | 0.997646 | -0.291210 | 0.000000 | 1.000000 |
| hyperkvasir | TARGET | 66.734484 | 0.876862 | -0.113558 | 0.000000 |  |
| hyperkvasir | Fed-DER | 72.272530 | 0.876268 | -0.078812 | 0.000000 | 0.447313 |
| pathmnist | TARGET | 66.894495 | 0.834015 | -0.221343 | 0.000000 |  |

Published: collector, source adapters and protocol documentation, numerical round/stage records, matrices, paired task ratios and summary CSV. Excluded: images, exact private sample indices, weights, generated pools, full logs, credentials and third-party source snapshots. Large artifacts remain in the original remote-home batch. Checkpoints were checked for presence and nonzero size, not reloaded or hashed.

Remote batch: `/remote-home/wangbomin/FedSubMerge_thesis/runs/public_baselines_20260909`.

The new Hyper-Kvasir quantity-skew queue is a separate, unfinished batch and is not included in these completion counts.
