# Hyper-Kvasir quantity-skew metric sequence

Authorized on 2026-09-12, physical GPU 3 only, one seed (2025). Four full sequences: FedAvg, Fed-GPM, FedSubMerge and FedSubMerge-AD. Four independent fresh-task references cover tasks 2–5 for RMA. No completed distribution-skew sequence is repeated.

Reuse the existing `indexes_train_hyperkvasir20_n10_t5_cpr0.5_split0.2_seed2025.npy` partition. Each of ten clients contains ten of twenty classes. Task classes are consecutive global IDs in `[4,4,4,4,4]`; AD uses six related clients, as specified in Chapter 5. The 5,957 selected training samples exclude the disjoint local-test holdout. Train counts by task are 2,345 / 2,064 / 1,058 / 403 / 87; global test counts are 838 / 737 / 377 / 145 / 32. The existing global validation and test splits are preserved.

ResNet18 from scratch, 256 input (existing 224 cache resized online), full-output CE, known-task masked evaluation, 20 communication rounds per task, three local epochs, batch 64 and LR 0.01/0.001/0.0001. Other method settings and metric boundaries follow RESOURCE_EXPERIMENTS.md. The same data and task configuration drives independent references. These measurements do not establish correspondence with historical thesis accuracy rows.

Use the existing Python environment and base-source snapshot. A two-task AD smoke run verified training, subspace construction, adaptive merging and training on the following task; it is excluded from formal results. The source SVD emitted a convergence fallback warning and PyTorch completed using its built-in more accurate method. CPU checks verified quantity task boundaries, existing distribution boundaries, weighted aggregation and task-aware evaluation. Data checks verified all ten clients, class coverage and training/holdout disjointness.

Remote root: `/remote-home/wangbomin/FedSubMerge_thesis/runs/quantity_20260912`. Storage is the verified NFS filesystem, with more than 350 TiB available at preparation. Estimated output allowance is 20 GiB, including checkpoints and logs; datasets are reused. Queue: `queue.json`; per-job logs: `logs/00.log` through `logs/07.log`. Private `jobs.json` retains absolute paths; it is not published.

The finite queue executes `python -u entry.py` with its working directory set to the code snapshot. Worker selection and configuration are passed through environment variables, so neither the parent nor worker command line contains project names, method names or storage paths. One job at a time on GPU 3, admission requires 37,000 MiB free. No other process is stopped. A failure stops the queue without automatic retries; existing output directories are never overwritten. SwanLab records formal runs. The tmux session is `q0912`; it continues independently of SSH or the chat.

Completed and verified on 2026-09-13. All eight final results and the paired RMA checks passed. See QUANTITY_COMPLETION_REPORT.md and results/quantity_resources_summary.csv.
