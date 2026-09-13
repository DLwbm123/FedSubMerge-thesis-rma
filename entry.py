"""Finite GPU queue/worker; configuration is passed through the environment."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time


def main():
    config_path = Path(os.environ.get('JOB_CONFIG', '../jobs.json')).resolve()
    config = json.loads(config_path.read_text())
    root = config_path.parent
    os.environ.update(JOB_CONFIG=str(config_path), CUDA_VISIBLE_DEVICES='3',
                      TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'),
                      OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    if 'JOB_SLOT' in os.environ:
        job = config['jobs'][int(os.environ['JOB_SLOT'])]
        sys.argv = ['worker'] + job['args']
        runpy.run_path(str(root / 'code' / job['runner']), run_name='__main__')
        return
    assert config['gpu'] == 3
    state = {'gpu': 3, 'jobs': [dict(name=j['name'], status='pending') for j in config['jobs']]}

    def save():
        p = root / 'queue.tmp'
        p.write_text(json.dumps(state, indent=2))
        p.replace(root / 'queue.json')

    for job in config['jobs']:
        output = Path(job['output'])
        if output.exists():
            raise FileExistsError(f'Refusing to rerun or overwrite {output}')
    pending, active, failed = list(range(len(config['jobs']))), {}, False
    save()
    while pending or active:
        for index, (process, log) in list(active.items()):
            rc = process.poll()
            if rc is None:
                continue
            log.close()
            result_path = Path(config['jobs'][index]['output']) / 'result.json'
            complete = (rc == 0 and result_path.exists() and
                        json.loads(result_path.read_text()).get('status') == 'complete')
            state['jobs'][index].update(status='complete' if complete else 'failed', exit_code=rc)
            failed |= not complete
            del active[index]
        if failed and not config.get('continue_after_failure', False):
            for index in pending:
                state['jobs'][index]['status'] = 'blocked_by_failure'
            pending.clear()
        elif pending:
            free = int(subprocess.check_output(['nvidia-smi', '-i', '3', '--query-gpu=memory.free',
                                               '--format=csv,noheader,nounits'], text=True).strip())
            reserved = sum(config['jobs'][i].get('memory_mib', 37000) for i in active)
            for index in list(pending):
                job = config['jobs'][index]
                required = job.get('memory_mib', 37000)
                if (len(active) >= config.get('max_concurrent', 1) or free < required or
                        reserved + required > config.get('capacity_mib', 38000)):
                    continue
                env = dict(os.environ, JOB_SLOT=str(index), CUDA_VISIBLE_DEVICES='3')
                log = (root / 'logs' / f'{index:02d}.log').open('x')
                process = subprocess.Popen([sys.executable, '-u', 'entry.py'], cwd=root / 'code',
                                           env=env, stdout=log, stderr=subprocess.STDOUT)
                active[index] = (process, log)
                state['jobs'][index].update(status='running', pid=process.pid)
                pending.remove(index)
                reserved += required
                free -= required
        save()
        if pending or active:
            time.sleep(10)
    if failed:
        raise SystemExit('Queue stopped; see the failed job log')


if __name__ == '__main__':
    main()
