# Resource and subspace experiment protocol

Scope: PathMNIST and Hyper-Kvasir20, alpha=0.3, seed=2025, four complete sequences each: FedAvg, Fed-GPM, FedSubMerge, FedSubMerge-AD. No additional algorithm variants or seed sweep. This batch targets the thesis Chapter 3 benchmark's MPE and DRR, plus explanatory subspace/resource measurements.

## Accounting

- MPE: mean consecutive network parameter increments divided by first-stage network parameter count. Fixed full-output ResNet18 is used; auxiliary bases are excluded and reported separately. Zero MPE does not mean zero auxiliary cost.
- DRR: mean, over non-final source tasks, of unique training images used to construct retained historical representations divided by that source task's participating training-image count. Count actual sampled IDs across disjoint clients. Images are not stored for later replay; no star is attached to the nonzero representation-based DRR. FedAvg uses zero such images.
- Every stage: actual model parameter count, client auxiliary basis/singular-value bytes, per-client/layer rank and ambient dimension, construction and server merge wall seconds, process-local CUDA allocated/reserved peaks.
- Every round: training/aggregation wall seconds, logical model upload plus download payload bytes. Tensor payloads include state-dict buffers but exclude transport framing; this is simulated federated communication, not measured network traffic.
- Subspace upload/download bytes describe tensor payloads at task boundaries. GPM bases remain local and are not transmitted. All-client auxiliary storage counts logically distinct clients even when the uniform result is identical.
- Projection energy: first training batch per client/round, all convolutional layers, before and after the actual applied projection. Raw energies are retained; zero input energy yields a missing ratio. This is a sampled mechanism observation, not an average over every update.

## Shared training protocol

Reuse the previous NC batch's train-only alpha=0.3 client indices and memory-mapped data: ten clients, all nonempty clients participate, sample-weighted aggregation, SGD without momentum or weight decay, full-output cross entropy, 20 rounds/task, three local epochs/round, batch 64, learning rate 0.01/0.001/0.0001 at rounds 1/11/16. PathMNIST input 128; Hyper-Kvasir uses the existing 224 cache resized online to 256. Known-task class-masked evaluation uses unchanged global class IDs. Test results never select checkpoints.

Fed-GPM reuses the source activation extraction and incremental NumPy SVD update: eight unaugmented examples per client sampled from the first at most 100 current-task samples; threshold 0.92 + 0.002 * completed tasks. Gradient projection uses the equivalent factored U U^T operation rather than materializing dense projection matrices.

The source GPM representation extractor retains its spatial-window convention (maximum 32/16/8/4 feature-map widths by layer) and consumes the dataset's native unaugmented image return; for Hyper-Kvasir this is the 224 cache. Its training input remains 256 like the other methods. This implementation detail is part of the recorded baseline, not a new GPM variant.

FedSubMerge and AD reuse the source gradient extraction, incremental subspace update, singular-value weighted merge and peer-selection functions. This controlled batch uses at most 256 distinct current-task images per client, four batches of at most 64, fixed local threshold 0.99 and merge threshold 0.97. AD selects up to four other clients for PathMNIST and five for Hyper-Kvasir. These explicit settings override the source class's dynamic threshold schedule and its default 100-batch sampling; results must be labeled as this controlled resource protocol, not silently substituted into historical tables. Source rank-selection conventions are preserved. Uniform merging broadcasts to all clients; AD updates participating clients and retains inactive-client history. No final-task representation is constructed because there is no later task.

One GPU network is reused serially across clients, with client bases on CPU between clients. GPU peaks therefore describe this simulator, not ten simultaneously resident client models. CUDA synchronization bounds phase timings. Concurrent jobs share GPU resources; wall times must not be presented as isolated hardware speed comparisons. No AMP or reduced training batch is used to fit more runs.

## Execution and artifacts

Remote batch: `/remote-home/wangbomin/FedSubMerge_thesis/runs/resources_20260909`.

Finite tmux queue uses only physical GPUs 2 and 3, at most two PathMNIST jobs or one Hyper-Kvasir job per GPU. Admission requires 18,000 MiB free for PathMNIST or 37,000 MiB for Hyper-Kvasir. Per-process allocator ceilings are 44% and 90% respectively. Existing unrelated processes are never modified. A failed run is recorded and not automatically restarted.

SwanLab project: https://swanlab.cn/@wangbomin/FedSubMerge-Thesis-Resources

Each run retains `config.json`, data/source-index accounting, `rounds.jsonl`, `construction.jsonl`, `projection.jsonl`, `subspaces.jsonl`, `stages.jsonl`, rolling and per-stage checkpoints, and final `result.json`. The queue records PIDs and exit codes in `queue.json`. Test accuracy matrices and BWTR are useful secondary outputs; historical RMA numerator compatibility is not asserted here.

`resource_metrics.py` provides a runnable accounting/projection check. Two-task reduced smoke runs exercise actual activation-GPM and gradient-subspace AD pipelines; smoke results are excluded from scientific comparisons. Formal completion and publication of shareable result summaries require a later completion check. No timed monitoring is created.

## Launch record (2026-09-09)

Accounting/projection self-check passed. Both reduced two-task smoke pipelines reached `smoke_complete`, including the second-task projection; their scores are not experimental results. An unavailable optional SwanLab local dashboard was avoided by using disabled tracking for smoke checks, without installing dependencies. The formal runs use cloud tracking.

tmux session: `thesis-resources-20260909`. Initial queue placement:

| GPU | Dataset | Method | SwanLab run |
| --- | --- | --- | --- |
| 2 | PathMNIST | FedSubMerge | `v1hco79knp7gxksp0gaf7` |
| 2 | PathMNIST | FedSubMerge-AD | `w2p8mndn5zl8ftwn341aj` |
| 3 | PathMNIST | Fed-GPM | `2jbbeuevojuxrjw5v7uhe` |
| 3 | PathMNIST | FedAvg | `dm88ynvml4nd4ev5ewt7c` |

Four Hyper-Kvasir runs remain queued and will start automatically as capacity becomes available. This is a launch report, not a completion or performance claim.

Startup check: all four formal PathMNIST processes wrote finite first-training-batch records and cloud run IDs, with no immediate failure artifact. First-batch allocation is not a full-run peak estimate.
