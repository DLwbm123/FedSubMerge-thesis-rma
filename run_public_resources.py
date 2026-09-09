"""Public-source baseline resource measurements, separate from thesis accuracy claims."""
import argparse
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from run_independent import TASKS, aggregate, atomic_json, evaluate, load_data, round_lr
from resource_metrics import benchmark_resources, tensor_bytes
from public_methods import DER, FOT, TARGET, IndexedData, sync_time


def cpu_state(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=TASKS, required=True)
    p.add_argument('--method', choices=['FOT', 'Fed-DER', 'TARGET'], required=True)
    for arg in ('source-root', 'public-source', 'data-root', 'client-index', 'output'):
        p.add_argument('--'+arg, type=Path, required=True)
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--rounds', type=int, default=20)
    p.add_argument('--local-epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=.01)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('2', '3') or torch.cuda.device_count() != 1:
        raise RuntimeError('Use only one physical GPU, 2 or 3')
    if min(a.rounds, a.local_epochs, a.batch_size) < 1 or a.lr <= 0:
        raise ValueError('Invalid training budget')
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.source_root))
    from backbone.ResNet18 import resnet18
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(.9)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    net = resnet18(nclasses=sum(TASKS[a.dataset]), in_ch=3).cuda()
    global_state = cpu_state(net)
    model_bytes = tensor_bytes(global_state)
    size = 128 if a.dataset == 'pathmnist' else 256
    method = (DER() if a.method == 'Fed-DER' else FOT(net, smoke=a.smoke) if a.method == 'FOT'
              else TARGET(net, a.public_source/'target.py', a.output, size, a.smoke))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(alpha=.3, input_size=size, clients=10, fixed_full_output=True,
                  accuracy_role='resource-adaptation diagnostic; does not replace thesis accuracy',
                  MPE_scope='classifier parameters; auxiliary generator/teacher/basis reported separately',
                  DER_buffer_capacity_per_client=256, DER_alpha=.5, DER_replay_batch=64,
                  FOT_epsilon=.95, FOT_epsilon_increment=.001, FOT_sketch_multiplier=5,
                  FOT_representation_samples='all current training images',
                  FOT_numerics='streamed independent Gaussian sketches; covariance eigendecomposition',
                  TARGET_synthesis_rounds=30, TARGET_warmup=20, TARGET_g_steps=10,
                  TARGET_synthesis_batch=256, TARGET_synthesis_microbatch=16,
                  TARGET_kd_steps=400, TARGET_kd_batch=64, TARGET_local_kd_weight=25,
                  TARGET_synthetic_samples_per_boundary=7680,
                  TARGET_normalization='identity, matching medical classifier [0,1] inputs',
                  TARGET_DRR='generated unique images attributed by teacher-predicted task; not real-source IDs',
                  runtime='one batch job per GPU; external GPU sharing still possible',
                  physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'])
    atomic_json(a.output/'config.json', config)
    import swanlab
    run = swanlab.init(project='FedSubMerge-Thesis-Resources',
            experiment_name=f'Public-{a.method}-{a.dataset}-a0.3-s{a.seed}'+('-SMOKE' if a.smoke else ''),
            config=config, mode='disabled' if a.smoke else 'cloud', logdir=str(a.output/'swanlab'))
    atomic_json(a.output/'swanlab_run.json', {k: str(getattr(run, k)) for k in ('id','url') if hasattr(run,k)})
    def record(kind, value):
        with (a.output/(kind+'.jsonl')).open('a') as f:
            f.write(json.dumps(value)+'\n')
    def checkpoint(path, extra):
        aux = (dict(buffers=method.buffers, seen=method.seen, used=method.used) if a.method=='Fed-DER'
               else dict(bases=method.bases, epsilon=method.epsilon) if a.method=='FOT'
               else dict(teacher=cpu_state(method.teacher) if method.teacher else None,
                         pool=[(str(x), t) for x,t in method.pool], used=method.used))
        tmp = path.with_suffix('.tmp')
        torch.save(dict(model=global_state, auxiliary=aux, **extra), tmp)
        tmp.replace(path)
    tasks = 2 if a.smoke else len(TASKS[a.dataset])
    parameters, train_counts, matrix, tests = [], [], [], []
    boundaries = np.cumsum(TASKS[a.dataset]).tolist()
    step = 0
    try:
        for t in range(tasks):
            lo = 0 if t == 0 else boundaries[t-1]
            hi = boundaries[t]
            raw_loaders, val, test, audit = load_data(a, lo, hi)
            if a.smoke:
                raw_loaders = {c: DataLoader(Subset(l.dataset.dataset, list(l.dataset.indices[:4])), batch_size=4)
                               for c,l in list(raw_loaders.items())[:2]}
                audit['client_train_counts'] = {c:len(l.dataset) for c,l in raw_loaders.items()}
                audit['train_count'] = sum(audit['client_train_counts'].values())
                val = DataLoader(Subset(val.dataset, range(min(16,len(val.dataset)))), batch_size=8)
                test = DataLoader(Subset(test.dataset, range(min(16,len(test.dataset)))), batch_size=8)
            transform = next(iter(raw_loaders.values())).dataset.dataset.transform
            loaders = {c:DataLoader(IndexedData(l.dataset), batch_size=4 if a.smoke else a.batch_size,
                                    shuffle=True, num_workers=0, pin_memory=True) for c,l in raw_loaders.items()}
            tests.append((test,lo,hi))
            train_counts.append(audit['train_count'])
            atomic_json(a.output/f'task{t+1:02d}_data.json', audit)
            torch.cuda.reset_peak_memory_stats()
            last_states = {}
            for r in range(1 if a.smoke else a.rounds):
                begin = sync_time()
                states, counts, clients = [], [], random.sample(list(loaders),len(loaders))
                loss_sum = examples = 0
                for c in clients:
                    net.load_state_dict(global_state)
                    net.train()
                    optimizer = torch.optim.SGD(net.parameters(), lr=round_lr(a.lr,r,a.rounds))
                    for epoch in range(1 if a.smoke else a.local_epochs):
                        for x,y,native,ids in loaders[c]:
                            x,y = x.cuda(),y.cuda()
                            optimizer.zero_grad(set_to_none=True)
                            logits = net(x)
                            loss = F.cross_entropy(logits[:,lo:hi], y-lo) if a.method=='TARGET' and t>0 else F.cross_entropy(logits,y)
                            if not torch.isfinite(loss):
                                raise FloatingPointError('Non-finite supervised loss')
                            loss.backward()
                            base_loss = float(loss.detach())
                            if a.method=='Fed-DER':
                                extra = method.replay_backward(net,c,t,transform,4 if a.smoke else 64)
                            elif a.method=='TARGET':
                                extra = method.replay_backward(net,t,lo,4 if a.smoke else 64)
                            else:
                                extra = 0.
                            if not np.isfinite(extra):
                                raise FloatingPointError('Non-finite replay loss')
                            optimizer.step()
                            if a.method=='Fed-DER':
                                method.add(c,t,native,logits.detach(),ids)
                            loss_sum += (base_loss+extra)*len(y)
                            examples += len(y)
                            if not (a.output/'startup.json').exists():
                                atomic_json(a.output/'startup.json', dict(task=t+1,loss=base_loss,
                                            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
                                print('FIRST_TRAIN_BATCH_OK',flush=True)
                            del x,y,logits,loss,native
                    states.append(cpu_state(net))
                    counts.append(len(loaders[c].dataset))
                train_end = sync_time()
                new_state = aggregate(states,counts)
                if a.method=='FOT':
                    new_state = method.project_update(global_state,new_state)
                    if r == (0 if a.smoke else a.rounds-1):
                        last_states = dict(zip(clients,states))
                global_state = new_state
                del states
                aggregate_end = sync_time()
                net.load_state_dict(global_state)
                val_acc = evaluate(net,val,torch.device('cuda'),lo,hi)
                m = dict(task=t+1,round=r+1,loss=loss_sum/examples,validation_accuracy=val_acc,
                         local_training_seconds=train_end-begin,model_aggregation_seconds=aggregate_end-train_end,
                         model_communication_bytes=2*len(loaders)*model_bytes,
                         peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                         peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
                step += 1
                record('rounds',m)
                swanlab.log(m,step=step)
                checkpoint(a.output/'checkpoint.pt',dict(task=t+1,round=r+1))
                print('ROUND',json.dumps(m),flush=True)
            net.load_state_dict(global_state)
            scores = [evaluate(net,l,torch.device('cuda'),low,high) for l,low,high in tests]
            matrix.append(scores)
            parameters.append(sum(p.numel() for p in net.parameters()))
            stage = dict(task=t+1,parameters=parameters[-1],accuracy=scores)
            if t < tasks-1:
                if a.method=='FOT':
                    stage.update(method.build(net,loaders,last_states,record,t))
                elif a.method=='TARGET':
                    stage.update(method.generate(net,hi,t,boundaries,record))
            if a.method=='Fed-DER':
                stage.update(auxiliary_bytes_all_clients=method.storage(),
                             buffer_image_bytes=sum(tensor_bytes(item[0]) for b in method.buffers.values() for item in b),
                             buffer_logit_bytes=sum(tensor_bytes(item[1]) for b in method.buffers.values() for item in b),
                             replay_presentations=method.presentations,
                             buffer_examples=sum(len(b) for b in method.buffers.values()))
            if a.method=='FOT':
                stage['server_auxiliary_bytes'] = tensor_bytes(method.bases)
                stage['client_auxiliary_bytes_logical'] = 10*tensor_bytes(method.bases)
            stage.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                         peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
            record('stages',stage)
            step += 1
            swanlab.log({k:v for k,v in stage.items() if k!='accuracy'},step=step)
            checkpoint(a.output/f'task{t+1:02d}.pt',dict(stage=stage))
            print('STAGE_COMPLETE',json.dumps(stage),flush=True)
            del last_states
        used = train_counts[:-1] if a.method=='FOT' else [len(method.used.get(t,set())) for t in range(tasks-1)]
        metrics = benchmark_resources(parameters,used,train_counts[:-1])
        result = dict(status='smoke_complete' if a.smoke else 'complete', **metrics,
                      accuracy_matrix=matrix,final_ACC=float(np.mean(matrix[-1])),
                      DRR_source_counts=used,DRR_train_counts=train_counts[:-1],
                      DRR_kind='synthetic_teacher_label_attribution' if a.method=='TARGET' else 'unique_real_source_images',
                      historical_accuracy_replaced=False)
        if a.method=='TARGET':
            result['raw_image_DRR'] = 0.
        if all(matrix[t][t]>0 for t in range(tasks-1)):
            result['BWTR'] = float(np.mean([(matrix[-1][t]-matrix[t][t])/matrix[t][t] for t in range(tasks-1)]))
        atomic_json(a.output/'result.json',result)
        swanlab.log({k:v for k,v in result.items() if isinstance(v,(int,float))},step=step+1)
        swanlab.finish()
    except BaseException as error:
        atomic_json(a.output/'failure.json',dict(type=type(error).__name__,message=str(error)))
        swanlab.finish(error=str(error))
        raise


if __name__=='__main__':
    main()
