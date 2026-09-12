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

    save()
    for index, job in enumerate(config['jobs']):
        output = Path(job['output'])
        if output.exists():
            raise FileExistsError(f'Refusing to rerun or overwrite {output}')
        while True:
            free = int(subprocess.check_output(['nvidia-smi', '-i', '3', '--query-gpu=memory.free',
                                               '--format=csv,noheader,nounits'], text=True).strip())
            if free >= 37000:
                break
            state['jobs'][index]['status'] = 'waiting_for_memory'
            save()
            time.sleep(30)
        env = dict(os.environ, JOB_SLOT=str(index), CUDA_VISIBLE_DEVICES='3')
        with (root / 'logs' / f'{index:02d}.log').open('x') as log:
            process = subprocess.Popen([sys.executable, '-u', 'entry.py'], cwd=root / 'code',
                                       env=env, stdout=log, stderr=subprocess.STDOUT)
            state['jobs'][index].update(status='running', pid=process.pid)
            save()
            rc = process.wait()
        result_path = output / 'result.json'
        complete = (rc == 0 and result_path.exists() and
                    json.loads(result_path.read_text()).get('status') == 'complete')
        state['jobs'][index].update(status='complete' if complete else 'failed', exit_code=rc)
        save()
        if not complete:
            raise SystemExit('Queue stopped; see the failed job log')


if __name__ == '__main__':
    main()
