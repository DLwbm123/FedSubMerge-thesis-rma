"""Finite queue: at most two PathMNIST jobs or one HyperKvasir job per GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_independent import atomic_json


def main():
    root = Path(sys.argv[1]).resolve()
    public = '--public' in sys.argv[2:]
    methods = ['FedSubMerge', 'FedSubMerge-AD', 'Fed-GPM', 'FedAvg']
    pending = ([( 'hyperkvasir', 'TARGET'), ('pathmnist', 'FOT'), ('pathmnist', 'Fed-DER'),
                ('hyperkvasir', 'FOT'), ('pathmnist', 'TARGET'), ('hyperkvasir', 'Fed-DER')]
               if public else [(d, m) for d in ('pathmnist', 'hyperkvasir') for m in methods])
    active, finished = [], []
    while pending or active:
        for job in list(active):
            rc = job['process'].poll()
            if rc is not None:
                job['log'].close()
                finished.append(dict(dataset=job['dataset'], method=job['method'], gpu=job['gpu'], exit_code=rc))
                active.remove(job)
        for gpu in (2, 3):
            running = [j for j in active if j['gpu'] == gpu]
            free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
                                               '--format=csv,noheader,nounits'], text=True).strip())
            for dataset, method in list(pending):
                required = 37000 if public else 18000 if dataset == 'pathmnist' else 37000
                if public and running:
                    continue
                if running and (dataset == 'hyperkvasir' or any(j['dataset'] == 'hyperkvasir' for j in running)):
                    continue
                if len(running) >= 2 or free < required:
                    continue
                name = dataset + '_' + method
                data = ('/remote-home/wangbomin/root-migrated/FCL/medminist_data'
                        if dataset == 'pathmnist' else
                        '/remote-home/wangbomin/root-migrated/FCL/Medmnist_new_local_pgsfedtorch/data/hyperkvasir20')
                runner = 'run_public_resources.py' if public else 'run_resources.py'
                cmd = [sys.executable, '-u', str(root / 'code' / runner), '--dataset', dataset,
                       '--method', method, '--source-root', str(root / 'base_source'), '--data-root', data,
                       '--client-index', str(root / 'indexes' / (dataset + '_train_alpha0.3.npy')),
                       '--output', str(root / name)]
                if public:
                    cmd += ['--public-source', str(root / 'public_source')]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TMPDIR=str(root / 'tmp'),
                           XDG_CACHE_HOME=str(root / 'cache'), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                           MKL_NUM_THREADS='4')
                log = (root / 'logs' / (name + '.log')).open('w')
                process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                job = dict(dataset=dataset, method=method, gpu=gpu, process=process, log=log)
                active.append(job)
                running.append(job)
                pending.remove((dataset, method))
                free -= required
        atomic_json(root / 'queue.json', dict(pending=pending, finished=finished,
                    active=[dict(dataset=j['dataset'], method=j['method'], gpu=j['gpu'], pid=j['process'].pid)
                            for j in active]))
        if pending or active:
            time.sleep(30)


if __name__ == '__main__':
    main()
