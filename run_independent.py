"""Independent per-task FedAvg references for the thesis RMA denominator."""

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


TASKS = {"pathmnist": [2, 2, 2, 3], "hyperkvasir": [2] * 10}


def task_counts(dataset, scenario="distribution"):
    if scenario == "quantity":
        if dataset != "hyperkvasir":
            raise ValueError("Only the existing Hyper-Kvasir quantity partition is supported")
        return [4] * 5
    if scenario != "distribution":
        raise ValueError("Unknown scenario")
    return TASKS[dataset]


def task_bounds(dataset, task, scenario="distribution"):
    counts = task_counts(dataset, scenario)
    if not 2 <= task <= len(counts):
        raise ValueError(f"Expected a one-based task in 2..{len(counts)}")
    return sum(counts[:task - 1]), sum(counts[:task])


def task_indices(labels, indices, lo, hi):
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or (len(indices) and
                             (indices.min() < 0 or indices.max() >= len(labels))):
        raise ValueError("Client indices outside the training split")
    return indices[(labels[indices] >= lo) & (labels[indices] < hi)]


def aggregate(states, counts):
    """Sample-weighted FedAvg, preserving state-dict dtypes (including BN)."""
    if not states or len(states) != len(counts) or min(counts) <= 0:
        raise ValueError("FedAvg requires positive sample counts")
    total = sum(counts)
    return {
        key: (sum(state[key] * n for state, n in zip(states, counts)) / total)
        .to(states[0][key].dtype)
        for key in states[0]
    }


def round_lr(initial, round_index, rounds):
    return initial * 0.1 ** (int(round_index >= rounds * 0.5)
                            + int(round_index >= rounds * 0.75))


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


@torch.no_grad()
def evaluate(model, loader, device, lo, hi):
    model.eval()
    correct = total = 0
    for inputs, labels, _ in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        # Keep the original global class IDs; restrict inference to this task.
        predictions = model(inputs)[:, lo:hi].argmax(1) + lo
        correct += int((predictions == labels).sum())
        total += len(labels)
    if not total:
        raise ValueError("Empty evaluation task")
    return 100.0 * correct / total


def load_data(args, lo, hi):
    # Reuse the project's existing memory-mapped datasets without copying images.
    from datasets.seq_medmnist_fedcl_path_effi import PathMNIST
    from datasets.hyperkvasir20_common import HyperKvasir20

    resize = [transforms.Resize((256, 256))] if args.dataset == "hyperkvasir" else []
    augmentation = ([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                     transforms.ColorJitter(brightness=32.0 / 255.0, saturation=0.5)]
                    if args.dataset == "hyperkvasir" else [])
    train_transform = transforms.Compose(resize + augmentation + [transforms.ToTensor()])
    eval_transform = transforms.Compose(resize + [transforms.ToTensor()])
    dataset_class = PathMNIST if args.dataset == "pathmnist" else HyperKvasir20
    kwargs = dict(root=str(args.data_root), mmap_mode="r")
    train = dataset_class(split="train", transform=train_transform, **kwargs)
    val = dataset_class(split="val", transform=eval_transform, **kwargs)
    test = dataset_class(split="test", transform=eval_transform, **kwargs)
    labels = np.asarray(train.targets).reshape(-1)
    # These trusted local .npy maps are existing experiment artifacts.
    clients = np.load(args.client_index, allow_pickle=True).item()
    if set(clients) != set(range(10)):
        raise ValueError("Expected the existing ten-client partition")
    all_indices = np.concatenate([np.asarray(clients[c]) for c in range(10)])
    if len(np.unique(all_indices)) != len(all_indices):
        raise ValueError("Overlapping training client indices")
    loaders, counts, selected_indices = {}, {}, {}
    for client in range(10):
        ids = task_indices(labels, clients[client], lo, hi)
        counts[client] = len(ids)
        selected_indices[str(client)] = ids.tolist()
        if len(ids):
            loaders[client] = DataLoader(Subset(train, ids), args.batch_size, shuffle=True,
                                         num_workers=args.workers, pin_memory=True)
    def eval_loader(data):
        y = np.asarray(data.targets).reshape(-1)
        ids = np.flatnonzero((y >= lo) & (y < hi))
        return DataLoader(Subset(data, ids), args.batch_size, num_workers=args.workers,
                          pin_memory=True)
    if not loaders:
        raise ValueError("No current-task training samples")
    val_loader, test_loader = eval_loader(val), eval_loader(test)
    audit = dict(client_train_counts=counts, train_count=sum(counts.values()),
                 validation_count=len(val_loader.dataset), test_count=len(test_loader.dataset),
                 source_image_shape=list(train.data.shape[1:]),
                 input_size=128 if args.dataset == "pathmnist" else 256,
                 output_classes=sum(task_counts(args.dataset, getattr(args, 'scenario', 'distribution'))), task_class_ids=list(range(lo, hi)),
                 source_train_count=len(train), train_indices=selected_indices)
    return loaders, val_loader, test_loader, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=TASKS, required=True)
    parser.add_argument("--scenario", choices=["distribution", "quantity"], default="distribution")
    parser.add_argument("--task", type=int, required=True, help="One-based original task ID")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--client-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    # ponytail: in-process loading avoids NFS IPC cleanup; add workers if loading limits throughput.
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default="FedSubMerge-Thesis-RMA")
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    lo, hi = task_bounds(args.dataset, args.task, args.scenario)
    if min(args.rounds, args.local_epochs, args.batch_size) <= 0 or args.lr <= 0:
        raise ValueError("Invalid training budget")
    sys.path.insert(0, str(args.source_root.resolve()))
    from backbone.ResNet18 import resnet18

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    train_loaders, val_loader, test_loader, audit = load_data(args, lo, hi)
    if args.check_data:
        inputs, labels, _ = next(iter(train_loaders[min(train_loaders)]))
        if not ((labels >= lo) & (labels < hi)).all():
            raise ValueError("Task filtering failed")
        print(json.dumps({k: v for k, v in audit.items() if k != "train_indices"}, indent=2))
        print("DATA_CHECK_OK", list(inputs.shape), flush=True)
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one of physical GPU 2 or 3")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.manual_seed_all(args.seed)
    # Keep cuDNN search disabled: the measured workspace fits alongside existing jobs.
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")
    model = resnet18(nclasses=sum(task_counts(args.dataset, args.scenario)), in_ch=3).to(device)
    global_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update({key: value for key, value in audit.items() if key != "train_indices"})
    config.update(alpha=0.3 if args.scenario == "distribution" else None,
                  task_sizes=task_counts(args.dataset, args.scenario), num_clients=10, clients_fraction=1.0,
                  initialization="fresh_random_per_task", optimizer="SGD, momentum=0, weight_decay=0",
                  training_loss="cross_entropy over original full output", evaluation="known-task class mask",
                  lr_milestones_rounds=[args.rounds * 0.5, args.rounds * 0.75],
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  checkpoint_selection="final fixed-budget round; test never selects model",
                  physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  torch_version=torch.__version__, cudnn_benchmark=False,
                  data_replay=False, pgs=False,
                  numerator_pairing="PENDING: verify original run data, budget, preprocessing and seed")
    atomic_json(args.output / "config.json", config)
    atomic_json(args.output / "train_indices.json", audit["train_indices"])
    import swanlab
    run = swanlab.init(project=args.project,
                       experiment_name=f"NC-{args.dataset}-{args.scenario}-task{args.task:02d}-seed{args.seed}",
                       config=config, mode="cloud", logdir=str(args.output / "swanlab"))
    run_info = {key: str(getattr(run, key)) for key in ("id", "url") if hasattr(run, key)}
    atomic_json(args.output / "swanlab_run.json", run_info)
    print("STARTED", json.dumps(run_info), flush=True)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        for round_index in range(args.rounds):
            round_start = time.monotonic()
            lr = round_lr(args.lr, round_index, args.rounds)
            states, counts = [], []
            loss_sum = examples = 0
            model.train()
            client_order = random.sample(list(train_loaders), len(train_loaders))
            for client in client_order:
                model.load_state_dict(global_state)
                optimizer = torch.optim.SGD(model.parameters(), lr=lr)
                for _ in range(args.local_epochs):
                    for inputs, labels, _ in train_loaders[client]:
                        inputs = inputs.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        optimizer.zero_grad(set_to_none=True)
                        loss = torch.nn.functional.cross_entropy(model(inputs), labels)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Non-finite training loss")
                        loss.backward()
                        optimizer.step()
                        if round_index == 0 and examples == 0:
                            startup = dict(client=client, first_batch_loss=float(loss.detach()),
                                           batch_shape=list(inputs.shape),
                                           peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
                            atomic_json(args.output / "startup.json", startup)
                            print("FIRST_TRAIN_BATCH_OK", json.dumps(startup), flush=True)
                        loss_sum += float(loss.detach()) * len(labels)
                        examples += len(labels)
                states.append({key: value.detach().cpu().clone()
                               for key, value in model.state_dict().items()})
                counts.append(audit["client_train_counts"][client])
            global_state = aggregate(states, counts)
            del states
            model.load_state_dict(global_state)
            val_accuracy = evaluate(model, val_loader, device, lo, hi)
            metrics = {"round": round_index + 1, "train/loss": loss_sum / examples,
                       "train/lr": lr, "validation/accuracy": val_accuracy,
                       "resources/round_seconds": time.monotonic() - round_start,
                       "resources/elapsed_seconds": time.monotonic() - started,
                       "resources/peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                       "resources/peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
            if round_index == args.rounds - 1:
                metrics["reference/a_nc_percent"] = evaluate(model, test_loader, device, lo, hi)
            swanlab.log(metrics, step=round_index + 1)
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            checkpoint = dict(model_state_dict=global_state, config=config,
                              round=round_index + 1, metrics=metrics)
            torch.save(checkpoint, args.output / "checkpoint.tmp")
            (args.output / "checkpoint.tmp").replace(args.output / "checkpoint.pt")
            print("ROUND", json.dumps(metrics), flush=True)
        atomic_json(args.output / "result.json", dict(status="complete", **metrics,
                                                       checkpoint=str(args.output / "checkpoint.pt")))
        swanlab.finish()
    except BaseException as error:
        atomic_json(args.output / "failure.json", dict(type=type(error).__name__, message=str(error)))
        swanlab.finish(error=str(error))
        raise


if __name__ == "__main__":
    main()
