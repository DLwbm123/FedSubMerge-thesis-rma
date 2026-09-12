"""Validate completed public adapters, export numerical records without private IDs."""
import json
import math
from pathlib import Path
import statistics
import sys

from collect_resources import read, rows


def collect(batch, references):
    queue = read(batch / 'queue.json')
    assert not queue['active'] and not queue['pending']
    assert len(queue['finished']) == 6 and all(j['exit_code'] == 0 for j in queue['finished'])
    output = []
    for job in queue['finished']:
        dataset, method = job['dataset'], job['method']
        root = batch / f'{dataset}_{method}'
        config, result = read(root / 'config.json'), read(root / 'result.json')
        rounds, stages = rows(root / 'rounds.jsonl'), rows(root / 'stages.jsonl')
        tasks = 4 if dataset == 'pathmnist' else 10
        assert result['status'] == 'complete' and not (root / 'failure.json').exists()
        assert [(r['task'], r['round']) for r in rounds] == [(t, r) for t in range(1, tasks+1) for r in range(1, 21)]
        assert [s['task'] for s in stages] == list(range(1, tasks+1))
        assert all((root / f'task{t:02d}.pt').stat().st_size > 0 for t in range(1, tasks+1))
        matrix = result['accuracy_matrix']
        assert [len(r) for r in matrix] == list(range(1, tasks+1))
        assert all(math.isfinite(v) for r in matrix for v in r)
        ratios = []
        for t in range(2, tasks+1):
            nc = references / f'{dataset}_task{t:02d}_seed2025'
            nc_config, data = read(nc / 'config.json'), read(root / f'task{t:02d}_data.json')
            for k in ('seed', 'rounds', 'local_epochs', 'batch_size', 'lr', 'data_root'):
                assert config[k] == nc_config[k], (dataset, method, k)
            for k in ('client_train_counts', 'train_count', 'validation_count', 'test_count',
                      'source_image_shape', 'input_size', 'output_classes', 'task_class_ids', 'source_train_count'):
                assert data[k] == nc_config[k], (dataset, t, k)
            assert data['train_indices'] == read(nc / 'train_indices.json')
            assert stages[t-1]['parameters'] == nc_config['parameter_count']
            denominator = read(nc / 'result.json')['reference/a_nc_percent']
            assert denominator > 0
            ratios.append(matrix[t-1][t-1] / denominator)
        bwtr = statistics.mean((matrix[-1][t]-matrix[t][t])/matrix[t][t] for t in range(tasks-1))
        mpe = statistics.mean((stages[t]['parameters']-stages[t-1]['parameters'])/stages[0]['parameters'] for t in range(1,tasks))
        drr = statistics.mean(n/d for n,d in zip(result['DRR_source_counts'], result['DRR_train_counts']))
        assert len(result['DRR_source_counts']) == len(result['DRR_train_counts']) == tasks-1
        assert math.isclose(mpe, result['MPE'], abs_tol=1e-12)
        assert math.isclose(drr, result['DRR'], abs_tol=1e-12)
        if 'BWTR' in result:
            assert math.isclose(bwtr, result['BWTR'], abs_tol=1e-12)
        result.update(RMA=statistics.mean(ratios), BWTR=bwtr)
        clean_config = {k:v for k,v in config.items() if k not in ('source_root','data_root','client_index','output','public_source')}
        output.append(dict(dataset=dataset, method=method, result=result, RMA_task_ratios=ratios,
                           config=clean_config, rounds=rounds, stages=stages,
                           swanlab=read(root / 'swanlab_run.json')))
    return dict(status='complete', runs=output, checked_rounds=sum(len(x['rounds']) for x in output),
                checked_stages=sum(len(x['stages']) for x in output))


if __name__ == '__main__':
    print(json.dumps(collect(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
