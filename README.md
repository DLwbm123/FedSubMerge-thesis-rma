# FedSubMerge thesis: independent-task RMA references

Authorized scope: PathMNIST alpha=0.3 tasks 2–4 and Hyper-Kvasir alpha=0.3
tasks 2–10, one seed (2025), with cloud SwanLab logging. These are 12
independent single-task FedAvg runs, not continual-learning sequences.

Remote batch: `/remote-home/wangbomin/FedSubMerge_thesis/runs/rma_nc_20260908_211710`.
GPU 2 runs the three PathMNIST tasks; GPU 3 runs the nine Hyper-Kvasir tasks.
Each finite queue runs inside tmux and survives SSH/client disconnection.
No periodic monitoring or additional experiments are scheduled.

## Protocol

- Fresh random ResNet-18 and fresh SGD for every task, no inherited weights,
  PGS, replay, pretraining, or optimizer state.
- Reuse the existing project ResNet-18 (3x3 stride-1 stem) and memory-mapped
  dataset implementations, captured in the batch's `base_source` snapshot.
- Ten logical clients, all nonempty clients participate, sample-weighted
  FedAvg. Clients are processed sequentially on their assigned GPU.
- 20 rounds, 3 local epochs/round, batch size 64, initial LR 0.01. LR changes
  to 0.001 at round 11 and 0.0001 at round 16. SGD momentum and weight decay
  are zero, matching the existing SGD implementation.
- Original global class IDs: PathMNIST tasks `[0,1]`, `[2,3]`, `[4,5]`,
  `[6,7,8]`; Hyper-Kvasir task i contains classes `2*(i-1), 2*(i-1)+1`.
- Retain the original 9/20-output classifier and full-output cross-entropy
  training. At evaluation, mask to the selected task's classes, matching the
  existing project's known-task accuracy calculation.
- PathMNIST input 128x128; Hyper-Kvasir input 256x256, resized online from
  the existing 224x224 cache, then use the existing flip/colour augmentation.
- Official/prepared global validation is evaluated each round. Only the
  fixed final round's model is evaluated on global test for the reference
  `a_nc_percent`; test accuracy never selects the model.
- Each task saves exact training indices, config, round metrics, a rolling
  checkpoint, final result, and its SwanLab run ID. RMA itself requires a
  separately matched continual-learning numerator and is not produced here.
- Data loading runs in-process: the NFS temporary directory produced worker
  IPC cleanup errors during preparation. All temporary files remain on NFS.
  cuDNN algorithm search is disabled to use the measured memory footprint.

## Data and provenance boundaries

Reuse alpha=0.3 client indexes; do not redraw partitions. The PathMNIST map
comes from `Medmnist_preliminary_fedavg/datasets/client_indexes/` and the
Hyper-Kvasir map comes from `Medmnist_new_local_pgsfedtorch/datasets/client_indexes/`.
The exact origin paths are in the remote `provenance.json`.

The existing maps hold out an additional local-test subset from the global
training split. This run retains the original *training-only* indices;
it does not merge that holdout into training. Across all tasks, these maps
contain 71,998 of 89,996 PathMNIST training images and 5,972 of 7,444 prepared
Hyper-Kvasir training images. Evaluation uses the separate global validation
and test splits. Existing arrays and maps remain untouched.

Important pairing limitation: historical exports contain different budgets,
seeds, partitions and preprocessing; Hyper-Kvasir's existing cache is 224,
whereas the thesis states 256. This batch follows the thesis's 20-round
budget and input size while retaining the available data/cache provenance.
Do not combine its denominator with a historical table value until the
numerator's actual protocol is matched. This is a new measured reference
batch, not verification that the old thesis numbers used this exact setup.

All large files, logs, temporary files and runtime caches are under the
remote batch directory; existing datasets resolve under
`/remote-home/wangbomin/root-migrated/FCL/`.

## Checks and launch

Use the existing `/root/anaconda3/bin/python` environment (PyTorch 2.6.0+cu124,
torchvision 0.21.0+cu124, SwanLab 0.7.15). Do not reinstall dependencies.

`python test_independent.py` checks task indexing, training split isolation,
sample-weighted aggregation, learning-rate boundaries and known-task scoring.
`run_independent.py --check-data ...` checks real target-task data without training.

Start each queue with `launch_queue.py --batch BATCH --dataset DATASET --gpu GPU`.
Queue state is saved to `queue_pathmnist.json` / `queue_hyperkvasir.json`.
An error stops that queue; successful tasks advance to the next requested task.
Output directories and log files are exclusive to prevent overwriting results.

The experiment is complete only when all 12 `result.json` files exist and
the queue states are complete. After completion, validate the results and
publish shareable code, configurations and metric summaries under the user's
GitHub delivery policy; exclude model files, raw images, credentials and
runtime caches. Starting the queues is not experiment completion.
