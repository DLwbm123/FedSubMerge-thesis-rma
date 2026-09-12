"""Controlled resource/mechanism sequences, sharing the NC data protocol."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from run_independent import TASKS, aggregate, atomic_json, evaluate, load_data, round_lr, task_counts
from resource_metrics import benchmark_resources, project_gradient, tensor_bytes


def cpu(values):
    return [torch.as_tensor(v).detach().cpu().clone() for v in values]


def stamp():
    torch.cuda.synchronize()
    return time.monotonic()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=TASKS, required=True)
    p.add_argument('--scenario', choices=['distribution', 'quantity'], default='distribution')
    p.add_argument('--method', choices=['FedAvg', 'Fed-GPM', 'FedSubMerge', 'FedSubMerge-AD'], required=True)
    for key in ['source-root', 'data-root', 'client-index', 'output']:
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--rounds', type=int, default=20)
    p.add_argument('--local-epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=.01)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--smoke', action='store_true', help='Two tasks, one batch/client; never a formal result')
    a = p.parse_args()
    counts_per_task = task_counts(a.dataset, a.scenario)
    sys.path.insert(0, str(a.source_root))
    from backbone.ResNet18 import resnet18
    from models.gpm import get_representation_matrix_ResNet18, update_GPM as update_activation
    from models.pgsfedtorch import get_layer_gradient_for_each_layer_reduced_conv, update_GPM
    from utils.merging_subspace_torch import merging_fed_subspaces_all, merging_adaptive_fed_subspaces_all
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('2', '3') or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose only physical GPU 2 or 3')
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(.44 if a.dataset == 'pathmnist' else .90)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    net = resnet18(nclasses=sum(counts_per_task), in_ch=3).cuda()
    conv = [(n, v) for n, v in net.named_parameters() if v.ndim == 4]
    global_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    model_bytes = tensor_bytes(global_state)
    spaces, singulars = {c: [] for c in range(10)}, {c: [] for c in range(10)}
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(alpha=.3 if a.scenario == 'distribution' else None,
                  task_sizes=counts_per_task, clients=10, participation=1., pgs_samples_per_client=256,
                  pgs_batch_size=64, pgs_local_threshold=.99, pgs_merge_threshold=.97,
                  gpm_examples=8, gpm_threshold='0.92 + 0.002 * completed_tasks',
                  ad_peers=6 if a.scenario == 'quantity' else 4 if a.dataset == 'pathmnist' else 5,
                  DRR_unit='unique training source images used to construct retained representations',
                  MPE_scope='network parameters only; excludes auxiliary bases',
                  task_evaluation='known-task masked global labels',
                  runtime='shared GPU; synchronized wall times, not isolated speed ranking',
                  hyper_input='existing 224 cache resized online to 256',
                  physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                  protocol='20 rounds x 3 epochs; full-output CE; sample-weighted FedAvg',
                  implementation='shared network, client-specific bases stored on CPU between clients')
    atomic_json(a.output / 'config.json', config)
    import swanlab
    run = swanlab.init(project='FedSubMerge-Thesis-Resources',
                       experiment_name=f'{a.method}-{a.dataset}-{a.scenario}-s{a.seed}' + ('-SMOKE' if a.smoke else ''),
                       config=config, mode='disabled' if a.smoke else 'cloud', logdir=str(a.output / 'swanlab'))
    atomic_json(a.output / 'swanlab_run.json', {k: str(getattr(run, k)) for k in ('id', 'url') if hasattr(run, k)})
    def record(kind, data):
        with (a.output / (kind + '.jsonl')).open('a') as f:
            f.write(json.dumps(data) + '\n')
    parameters, used_counts, train_counts, matrix, tests = [], [], [], [], []
    total_tasks = 2 if a.smoke else len(counts_per_task)
    step = 0
    try:
        for t in range(total_tasks):
            lo, hi = sum(counts_per_task[:t]), sum(counts_per_task[:t+1])
            loaders, val, test, audit = load_data(a, lo, hi)
            if a.smoke:
                loaders = {c: DataLoader(Subset(loader.dataset, range(min(8, len(loader.dataset)))), batch_size=8)
                           for c, loader in list(loaders.items())[:2]}
                audit['client_train_counts'] = {c: len(l.dataset) for c, l in loaders.items()}
            tests.append((test, lo, hi))
            atomic_json(a.output / f'task{t+1:02d}_data.json', audit)
            torch.cuda.reset_peak_memory_stats()
            for r in range(1 if a.smoke else a.rounds):
                begin = stamp()
                states, counts = [], []
                loss_sum = samples = 0
                for c in random.sample(list(loaders), len(loaders)):
                    net.load_state_dict(global_state)
                    net.train()
                    bases = [torch.as_tensor(v, dtype=torch.float32, device='cuda') for v in spaces[c]]
                    optimizer = torch.optim.SGD(net.parameters(), lr=round_lr(a.lr, r, a.rounds))
                    for epoch in range(1 if a.smoke else a.local_epochs):
                        for b, (x, y, _) in enumerate(loaders[c]):
                            x, y = x.cuda(), y.cuda()
                            optimizer.zero_grad(set_to_none=True)
                            loss = torch.nn.functional.cross_entropy(net(x), y)
                            if not torch.isfinite(loss):
                                raise FloatingPointError('Non-finite loss')
                            loss.backward()
                            if bases:
                                if len(bases) != len(conv):
                                    raise ValueError('Convolution/basis mismatch')
                                with torch.no_grad():
                                    for (name, weight), basis in zip(conv, bases):
                                        sampled = epoch == 0 and b == 0
                                        before = weight.grad.square().sum() if sampled else None
                                        project_gradient(weight.grad, basis)
                                        if sampled:
                                            after = weight.grad.square().sum()
                                            record('projection', dict(task=t+1, round=r+1, client=c, layer=name,
                                                   rank=basis.shape[1], dimension=basis.shape[0],
                                                   energy_before=float(before), energy_after=float(after),
                                                   retained_ratio=float(after / before) if before > 0 else None))
                            optimizer.step()
                            loss_sum += float(loss.detach()) * len(y)
                            samples += len(y)
                            if not (a.output / 'startup.json').exists():
                                atomic_json(a.output / 'startup.json', dict(first_batch_loss=float(loss.detach()),
                                            allocated_mib=torch.cuda.max_memory_allocated()/2**20))
                                print('FIRST_TRAIN_BATCH_OK', flush=True)
                    states.append({k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
                    counts.append(audit['client_train_counts'][c])
                    del bases, optimizer, x, y, loss
                training_end = stamp()
                global_state = aggregate(states, counts)
                del states
                aggregation_end = stamp()
                net.load_state_dict(global_state)
                val_acc = evaluate(net, val, torch.device('cuda'), lo, hi)
                step += 1
                m = dict(task=t+1, round=r+1, loss=loss_sum/samples, validation_accuracy=val_acc,
                         local_training_seconds=training_end-begin,
                         model_aggregation_seconds=aggregation_end-training_end,
                         model_communication_bytes=2*len(loaders)*model_bytes,
                         peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                         peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
                record('rounds', m)
                swanlab.log(m, step=step)
                torch.save(dict(model=global_state, spaces=spaces, singulars=singulars, task=t+1, round=r+1),
                           a.output / 'checkpoint.tmp')
                (a.output / 'checkpoint.tmp').replace(a.output / 'checkpoint.pt')
                print('ROUND', json.dumps(m), flush=True)
            net.load_state_dict(global_state)
            scores = [evaluate(net, loader, torch.device('cuda'), low, high) for loader, low, high in tests]
            matrix.append(scores)
            parameters.append(sum(v.numel() for v in net.parameters()))
            stage_start = stamp()
            source_count = 0
            upload_bytes = download_bytes = 0
            if t < total_tasks - 1:
                for c, loader in loaders.items():
                    if a.method == 'FedAvg':
                        continue
                    net.load_state_dict(global_state)
                    count = min(8 if a.method == 'Fed-GPM' else 256, len(loader.dataset))
                    # Independent sampling RNG keeps sample selection reproducible across PGS variants.
                    rng = np.random.default_rng(a.seed + 100*t + c)
                    pool = min(100, len(loader.dataset)) if a.method == 'Fed-GPM' else len(loader.dataset)
                    positions = rng.choice(pool, count, replace=False)
                    subset = Subset(loader.dataset, positions.tolist())
                    ids = np.array(loader.dataset.indices)[positions].tolist()
                    record('representation_sources', dict(task=t+1, client=c, train_indices=ids, count=count))
                    source_count += count
                    start = stamp()
                    if a.method == 'Fed-GPM':
                        # Original GPM uses the unaugmented third dataset return.
                        _, _, x = next(iter(DataLoader(subset, batch_size=count)))
                        mats = get_representation_matrix_ResNet18(net, torch.device('cuda'), x, count)
                        updated = update_activation(mats, [.92+.002*(t+1)]*len(conv),
                                                    [v.numpy() for v in spaces[c]])
                        spaces[c] = cpu(updated)
                        del mats, updated, x
                    else:
                        pgs_loader = DataLoader(subset, batch_size=64, shuffle=False)
                        mats = get_layer_gradient_for_each_layer_reduced_conv(net, torch.device('cuda'),
                                   pgs_loader, torch.nn.functional.cross_entropy, max_batches=4)
                        u, s = update_GPM(mats, [.99]*len(conv), torch.device('cuda'),
                                          cpu(spaces[c]), cpu(singulars[c]))
                        spaces[c], singulars[c] = cpu(u), cpu(s)
                        upload_bytes += tensor_bytes((spaces[c], singulars[c]))
                        del mats, u, s
                    record('construction', dict(task=t+1, client=c, seconds=stamp()-start, samples=count))
                    for (name, _), basis in zip(conv, spaces[c]):
                        record('local_subspaces', dict(task=t+1, client=c, layer=name,
                               rank=basis.shape[1], dimension=basis.shape[0], bytes=tensor_bytes(basis)))
                construction_end = stamp()
                if a.method.startswith('FedSubMerge'):
                    clients = list(loaders)
                    layer_spaces = [[spaces[c][k] for c in clients] for k in range(len(conv))]
                    layer_singulars = [[singulars[c][k] for c in clients] for k in range(len(conv))]
                    if a.method == 'FedSubMerge':
                        u, s = merging_fed_subspaces_all(layer_spaces, layer_singulars, .97)
                        for c in range(10):
                            spaces[c], singulars[c] = cpu(u), cpu(s)
                    else:
                        u, s = merging_adaptive_fed_subspaces_all(layer_spaces, layer_singulars,
                                               min(config['ad_peers'], len(clients)-1), .97)
                        for i, c in enumerate(clients):
                            spaces[c], singulars[c] = cpu(u[i]), cpu(s[i])
                    download_bytes = tensor_bytes((spaces, singulars))
                    del layer_spaces, layer_singulars, u, s
                used_counts.append(source_count)
                train_counts.append(sum(audit['client_train_counts'].values()))
            else:
                construction_end = stamp()
            for c in range(10):
                for (name, _), basis in zip(conv, spaces[c]):
                    record('subspaces', dict(task=t+1, client=c, layer=name, rank=basis.shape[1],
                                            dimension=basis.shape[0], bytes=tensor_bytes(basis)))
            stage = dict(task=t+1, parameters=parameters[-1], accuracy=scores,
                         auxiliary_bytes_all_clients=tensor_bytes((spaces, singulars)),
                         representation_source_count=source_count,
                         construction_seconds=construction_end-stage_start,
                         subspace_merge_seconds=stamp()-construction_end,
                         subspace_upload_bytes=upload_bytes, subspace_download_bytes=download_bytes,
                         peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                         peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
            record('stages', stage)
            swanlab.log({k: v for k, v in stage.items() if k != 'accuracy'}, step=step+1)
            step += 1
            torch.save(dict(model=global_state, spaces=spaces, singulars=singulars, stage=stage),
                       a.output / f'task{t+1:02d}.pt')
            print('STAGE_COMPLETE', json.dumps(stage), flush=True)
        result = dict(status='smoke_complete' if a.smoke else 'complete',
                      **benchmark_resources(parameters, used_counts, train_counts),
                      accuracy_matrix=matrix, final_ACC=float(np.mean(matrix[-1])),
                      BWTR=float(np.mean([(matrix[-1][i]-matrix[i][i])/matrix[i][i]
                                           for i in range(total_tasks-1)])) if all(matrix[i][i] > 0 for i in range(total_tasks-1)) else None)
        atomic_json(a.output / 'result.json', result)
        swanlab.log({k: v for k, v in result.items() if isinstance(v, (float, int))}, step=step+1)
        swanlab.finish()
    except BaseException as error:
        atomic_json(a.output / 'failure.json', dict(type=type(error).__name__, message=str(error)))
        swanlab.finish(error=str(error))
        raise


if __name__ == '__main__':
    main()
