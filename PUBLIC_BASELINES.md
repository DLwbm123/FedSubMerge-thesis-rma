# Public-source baseline resource measurements

Completion verified on 2026-09-12: all six sequences finished. See PUBLIC_COMPLETION_REPORT.md and results/public_resources_summary.csv.

Approved scope: FOT, Fed-DER and TARGET on PathMNIST and Hyper-Kvasir20, alpha=0.3, seed=2025: six complete sequences. No multi-seed extension. Preserve thesis accuracy tables; supplementary measured accuracies are diagnostics of these adapters, not replacements or fabricated matches to historical rows.

## Sources

| Method | Original source | Pinned revision | Source logic |
|---|---|---|---|
| FOT | https://github.com/duygunuryldz/Federated_Orthogonal_Training | f343e366f393812ccfbce79a505ddcdaa5305bf7 | trainer/resnet_trainer.py; FedAVGAggregator.expand_orth_set and aggregate in fedavg_seq_cont |
| Fed-DER | https://github.com/aimagelab/mammoth | e75a491c69fd729edeb01431afb753d9157d9a81 | models/der.py; reservoir sampling in utils/buffer.py |
| TARGET | https://github.com/zj-jayzhang/Federated-Class-Continual-Learning | 78cb9c05eea2555c33c0b83391811c3bb1ef059c | methods/target.py: Generator, DeepInversionHook, GlobalSynthesizer, KD losses |

`fetch_public_sources.py DESTINATION` retrieves six source/license files from these exact revisions and writes a provenance manifest. Original third-party files remain outside this repository. TARGET's reviewed self-contained class/function declarations are loaded from the fetched file without importing its application or executing its top-level training configuration. DER/Mammoth is MIT-licensed; its copyright and license are retained in the downloaded source bundle. FOT and TARGET original files are not redistributed here.

## Shared protocol and measured scope

Reuse the completed resource batch's source snapshot and train-only client partitions. Ten clients, full nonempty-client participation, sample-weighted FedAvg, ResNet18 with fixed full output, known-task class-masked evaluation. PathMNIST real inputs are 128; Hyper-Kvasir real inputs are resized from the existing 224 cache to 256. SGD without momentum/weight decay, 20 rounds/task, three local epochs, batch 64, LR 0.01/0.001/0.0001 at rounds 1/11/16. These unified choices override different original dataset/training presets and are not claimed as exact original-paper reproductions.

MPE counts classifier parameters only. Fixed allocated heads imply zero expansion; teacher, student, generator, buffers, optimizer state and subspaces are auxiliary resources reported separately. CUDA peaks include transient working tensors. Tensor storage excludes Python object overhead. Communication is logical tensor/file payload, not measured wire traffic or a secure-aggregation protocol. Checkpoint IO is excluded from local training timers. One simulator network is reused serially across clients. CPU-resident client state and logically distributed storage are distinguished from actual single-process CUDA peaks.

## Method adaptations

### FOT

Preserve the server-side projection of the sample-weighted aggregate update, not client gradient projection. Build residual input-activation Gaussian sketches at task boundaries from each client's final local model, then sum sketches, adjust the retained-energy threshold by the sample-weighted residual/original norm ratio, append directions and QR orthogonalize. Use source ResNet epsilon=0.95 and increment=0.001; sketch width is five times the layer input dimension. Convolution inputs, including shortcuts, follow the source 32/16/8/4 spatial-window convention. Channel dimensions follow the thesis ResNet18, rather than the source's narrower nf=20 model.

All current training images contribute to representation construction. Patches and independent Gaussian matrix rows are streamed in chunks, avoiding the original implementation's all-samples activation materialization. Covariance eigendecomposition supplies the left singular subspace without constructing unused right singular vectors; this is a numerical adaptation and timings reflect it. Exact-zero residual directions are skipped to avoid division by zero. Round learning rates and sample weights follow the shared protocol.

DRR is one for full current-source representation use under the benchmark's broad representation-aware convention. This does not mean historical raw images are retained. Report the actually counted non-final source tasks, bases, full sketch payloads and construction/merge time.

### Fed-DER

Federate the original DER local objective: current full-output cross entropy plus alpha=0.5 times MSE to stored logits. This is DER, not DER++: no extra replay-label cross entropy. Each client has its own capacity-256 online reservoir over the observed stream, storing unaugmented float32 images and pre-update logits. Replay batch size is at most 64, without replacement within a batch, and uses the current training image transform. Sequential backward calls accumulate the same two loss gradients while avoiding simultaneous retention of both graphs. Reservoir updates occur after each optimizer step.

Keep source task and original train index as accounting metadata. DRR counts distinct historical real images actually sampled in later tasks, grouped by original source task; separately retain cumulative replay presentations and resident image/logit tensor bytes. Buffer metadata and raw image contents remain private. All clients together may retain up to 2,560 entries; the budget is not 256 shared globally.

### TARGET

Preserve generator-based teacher inversion with BN-statistic, one-hot and adversarial distillation losses, FOMAML generator updates, a student used for adversarial synthesis, and local new-class CE plus old-class KD (weight 25, temperature 2). Classifier outputs remain preallocated and are sliced to the relevant seen/new/old classes. Original generators and BN hooks are loaded directly from the pinned source.

Explicit configuration correction: the source CIFAR preset has `syn_round=10` and `warmup=20`, so its student KD branch is never reached. This adapter uses 30 synthesis rounds, warmup 20, ten generator updates per round, and logical synthesis batch 256, yielding 7,680 images at each non-final boundary. The ten post-warmup student phases each use 400 updates, batch 64, KD temperature 20, SGD LR 0.2/momentum 0.9/weight decay 0.0001 and the source cosine schedule. Generator LR 0.002, latent LR 0.01, BN weight 10, one-hot weight 0.5, adversarial weight 1 and BN momentum 0.9 follow the source CIFAR preset. Retain the public source's distinct one-based synthesis warmup and zero-based student warmup conditions.

Generate at the actual medical input resolution, with identity normalization consistent with classifier inputs, crop/flip augmentation, and logical-batch gradient accumulation in microbatches of 16. Generator and inversion BN statistics are consequently microbatch statistics; exact source-batch equivalence is not claimed. This adapter is a fully recorded resource configuration, not a tuned accuracy reproduction. Randomly sample up to 64 synthetic examples per real-data training batch without shortening the real-data epoch. Save generated PNGs, generator checkpoints and source-task attribution metadata under remote-home. Retain all prior synthetic directories for reproducibility, while only the preceding boundary pool is active for local replay.

TARGET uses no historical real images. Its synthetic DRR counts unique generated images actually replayed, attributed to source tasks by the frozen teacher's predicted class, divided by original source-task training counts. This is a **generated-image proxy**, not verified real-image provenance; report `DRR_kind` and `raw_image_DRR=0` alongside it. Do not silently pool this value with unique-real-image DRR or call synthesis free. Report pool PNG bytes, decoded tensor bytes, teacher/student/generator and optimizer tensor storage, synthesis time and CUDA peaks.

## Checks and execution

`test_public_methods.py PUBLIC_SOURCE` checks reservoir capacity, an actual FOT server projection and loading/forwarding the original TARGET generator. Reduced two-task smoke runs exercise representation construction/server projection, real replay, and TARGET synthesis/student/local distillation. Smoke outputs and their reduced sketch dimensions/budgets are excluded from formal results.

Remote root: `/remote-home/wangbomin/FedSubMerge_thesis/runs/public_baselines_20260909`.

Start with `python launch_resources.py ROOT --public` in tmux. Only GPUs 2 and 3; at most one public-baseline job per GPU, requiring 37,000 MiB available at admission. Each process uses a 90% allocator ceiling, leaving headroom. Six jobs form a finite queue; failures are recorded, not automatically restarted. No scheduled monitoring is created. Existing unrelated GPU processes are not changed.

SwanLab project: https://swanlab.cn/@wangbomin/FedSubMerge-Thesis-Resources

Every run keeps configuration, per-round/stage resource records, diagnostic accuracy matrix, source attribution, rolling and per-task checkpoints, final result or explicit failure. Large artifacts remain on the verified NFS mount. Public completion results will exclude private images/indices, generated images, weights, full logs and third-party files; only code, settings and shareable metric summaries are published.

## Launch record

All three reduced two-task pipelines reached `smoke_complete`; the small reservoir/projection/source-loader check also passed. TARGET's smoke used real 256x256 Hyper-Kvasir input and exercised generator training, student KD and next-task replay. Smoke scores are excluded from scientific results.

tmux session: `thesis-public-baselines-20260909`. Initial formal placement is Hyper-Kvasir TARGET on GPU 2 (SwanLab `fhtn7jl2p9x1i81ylebdl`) and PathMNIST FOT on GPU 3 (`gc2s5rurbjxd7nh1ir103`). Four additional sequences remain in the finite queue. This is startup, not experiment completion. Queue state and PIDs are stored in `queue.json`; full startup logs are in `logs/` under the remote root.
