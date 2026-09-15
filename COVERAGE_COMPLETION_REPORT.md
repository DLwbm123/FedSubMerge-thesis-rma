# Coverage batch completion

All 26 jobs finished with exit code 0: 12 continual sequences and 14 independent references. Validated 1,360 ordered continual round records, 280 reference round records, 68 ordered stage records, and 82 nonempty required checkpoints. Recomputed ACC, BWTR, MPE and DRR and paired RMA with matching seed, budget, data roots, task classes, shapes, selected training indices and classifier parameter counts. Scenario, alpha, task sizes and skin validation-index configuration also match. No training was repeated during collection.

The scope and accepted skin-data difference are documented in COVERAGE_EXPERIMENTS.md. Original thesis accuracy rows remain historical, not paired with these matrices. Source datasets, checkpoints and full logs were neither modified nor deleted. Checkpoint validation was limited to presence and nonzero size; checkpoints were not reloaded or hashed.

| Setting | Method | Measured ACC (%) | RMA | BWTR | MPE | DRR |
|---|---|---:|---:|---:|---:|---:|
| pathmnist alpha=0.1 | FedAvg | 58.576362 | 1.018026 | -0.479838 | 0.000000 | 0.000000 |
| pathmnist alpha=0.1 | Fed-GPM | 69.560611 | 1.013945 | -0.304087 | 0.000000 | 0.003921 |
| pathmnist alpha=0.1 | FedSubMerge | 54.701827 | 1.016567 | -0.530231 | 0.000000 | 0.095602 |
| pathmnist alpha=0.1 | FedSubMerge-AD | 60.125031 | 1.028211 | -0.462454 | 0.000000 | 0.095602 |
| hyperkvasir alpha=0.1 | FedAvg | 73.655017 | 0.960985 | -0.096723 | 0.000000 | 0.000000 |
| hyperkvasir alpha=0.1 | Fed-GPM | 64.642765 | 0.929394 | -0.152300 | 0.000000 | 0.173486 |
| hyperkvasir alpha=0.1 | FedSubMerge | 69.535163 | 0.940519 | -0.105772 | 0.000000 | 0.863750 |
| hyperkvasir alpha=0.1 | FedSubMerge-AD | 68.286341 | 0.948466 | -0.133838 | 0.000000 | 0.863750 |
| skin feature | FedAvg | 74.956393 | 1.003427 | -0.229997 | 0.000000 | 0.000000 |
| skin feature | Fed-GPM | 79.349363 | 0.997350 | -0.143948 | 0.000000 | 0.009568 |
| skin feature | FedSubMerge | 78.565356 | 0.995922 | -0.162428 | 0.000000 | 0.248541 |
| skin feature | FedSubMerge-AD | 76.285497 | 1.004854 | -0.208366 | 0.000000 | 0.248541 |

All measured settings are consolidated in results/all_measured_metrics.csv and results/all_measured_metrics.md. These include 30 completed continual sequences and 30 independent references across five batches. Numerical round/stage records, paired ratios, matrices, source adapters and collection/summarization scripts are public. Excluded: private train/validation index lists, patient/source images, generated images, weights, raw logs and credentials. Raw artifacts remain on remote-home. The local historical-ACC thesis table is kept separate from the public measured-results table.

No new training is required to close the requested four-method, six-setting metric coverage. OrganAMNIST remains explicitly excluded. TARGET's generated-image DRR proxy remains separate from real-source DRR; repeating training does not resolve that definition difference. Missing rows for additional methods require new adaptation/runs only if complete cross-method coverage is made a new objective.

Remote root: `/remote-home/wangbomin/FedSubMerge_thesis/runs/coverage_20260913`.
