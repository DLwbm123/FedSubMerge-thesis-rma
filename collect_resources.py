"""Read completed artifacts and emit shareable metrics; never starts training."""
import json
import math
from pathlib import Path
import statistics
import sys


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def collect(batch, references):
    queue = read(batch / 'queue.json')
    quantity = 'jobs' in queue
    references_summary = []
    if quantity:
        assert len(queue['jobs']) == 8
        assert all(j['status'] == 'complete' and j['exit_code'] == 0 for j in queue['jobs'])
        methods = ['FedAvg', 'Fed-GPM', 'FedSubMerge', 'FedSubMerge-AD']
        assert {j['name'] for j in queue['jobs']} == set(methods + [f'NC-task{t}' for t in range(2, 6)])
        jobs = [dict(dataset='hyperkvasir', method=m) for m in methods]
        for t in range(2, 6):
            nc = references / f'nc_task{t:02d}'
            metrics, result = rows(nc / 'metrics.jsonl'), read(nc / 'result.json')
            assert [r['round'] for r in metrics] == list(range(1, 21))
            assert result['status'] == 'complete' and not (nc / 'failure.json').exists()
            assert (nc / 'checkpoint.pt').stat().st_size > 0
            references_summary.append(dict(task=t, result={k:v for k,v in result.items() if k != 'checkpoint'},
                                           rounds=metrics))
    else:
        assert not queue['active'] and not queue['pending']
        assert len(queue['finished']) == 8 and all(j['exit_code'] == 0 for j in queue['finished'])
        jobs = queue['finished']
    output = []
    for job in jobs:
        dataset, method = job['dataset'], job['method']
        root = batch / (method if quantity else f'{dataset}_{method}')
        config, result = read(root / 'config.json'), read(root / 'result.json')
        rounds, stages = rows(root / 'rounds.jsonl'), rows(root / 'stages.jsonl')
        tasks = 5 if quantity else 4 if dataset == 'pathmnist' else 10
        assert result['status'] == 'complete' and not (root / 'failure.json').exists()
        assert [(x['task'], x['round']) for x in rounds] == [(t, r) for t in range(1, tasks+1) for r in range(1, 21)]
        assert [x['task'] for x in stages] == list(range(1, tasks+1))
        assert all((root / f'task{t:02d}.pt').stat().st_size > 0 for t in range(1, tasks+1))
        assert [len(r) for r in result['accuracy_matrix']] == list(range(1, tasks+1))
        assert all(math.isfinite(x) for row in result['accuracy_matrix'] for x in row)
        ratios = []
        for t in range(2, tasks+1):
            nc = references / (f'nc_task{t:02d}' if quantity else f'{dataset}_task{t:02d}_seed2025')
            nc_config = read(nc / 'config.json')
            data = read(root / f'task{t:02d}_data.json')
            for key in ('seed', 'rounds', 'local_epochs', 'batch_size', 'lr', 'data_root'):
                assert config[key] == nc_config[key], (dataset, key)
            if quantity:
                assert config['scenario'] == nc_config['scenario'] == 'quantity'
                assert config['task_sizes'] == nc_config['task_sizes'] == [4] * 5
                assert config['alpha'] is nc_config['alpha'] is None
            for key in ('client_train_counts', 'train_count', 'validation_count', 'test_count',
                        'source_image_shape', 'input_size', 'output_classes', 'task_class_ids', 'source_train_count'):
                assert data[key] == nc_config[key], (dataset, t, key)
            # Semantic pairing of the selected source IDs, not a file-integrity hash.
            assert data['train_indices'] == read(nc / 'train_indices.json')
            assert stages[t-1]['parameters'] == nc_config['parameter_count']
            denominator = read(nc / 'result.json')['reference/a_nc_percent']
            assert denominator > 0
            ratios.append(result['accuracy_matrix'][t-1][t-1] / denominator)
        replay = []
        for t in range(1, tasks):
            data = read(root / f'task{t:02d}_data.json')
            replay.append(stages[t-1]['representation_source_count'] / data['train_count'])
        assert math.isclose(statistics.mean(replay), result['DRR'], abs_tol=1e-12)
        mpe = statistics.mean((stages[t]['parameters']-stages[t-1]['parameters']) / stages[0]['parameters']
                              for t in range(1, tasks))
        assert math.isclose(mpe, result['MPE'], abs_tol=1e-12)
        matrix = result['accuracy_matrix']
        bwtr = statistics.mean((matrix[-1][t]-matrix[t][t])/matrix[t][t] for t in range(tasks-1))
        assert math.isclose(bwtr, result['BWTR'], abs_tol=1e-12)
        summary = dict(dataset=dataset, method=method, seed=config['seed'], ACC=result['final_ACC'],
                       RMA=statistics.mean(ratios), BWTR=bwtr, MPE=mpe, DRR=result['DRR'],
                       peak_allocated_mib=max(x['peak_allocated_mib'] for x in stages),
                       peak_reserved_mib=max(x['peak_reserved_mib'] for x in stages),
                       max_auxiliary_bytes_all_clients=max(x['auxiliary_bytes_all_clients'] for x in stages),
                       local_training_seconds=sum(x['local_training_seconds'] for x in rounds),
                       construction_seconds=sum(x['construction_seconds'] for x in stages),
                       subspace_merge_seconds=sum(x['subspace_merge_seconds'] for x in stages))
        assert all(math.isfinite(v) for v in summary.values() if isinstance(v, (int, float)))
        clean_config = {k: v for k, v in config.items() if k not in ('source_root', 'data_root', 'client_index', 'output')}
        output.append(dict(summary=summary, config=clean_config, accuracy_matrix=matrix,
                           RMA_task_ratios=ratios, DRR_source_task_ratios=replay,
                           swanlab=read(root / 'swanlab_run.json'), rounds=rounds, stages=stages,
                           local_subspaces=rows(root / 'local_subspaces.jsonl'),
                           subspaces=rows(root / 'subspaces.jsonl')))
    return dict(status='complete', runs=output, checked_rounds=sum(len(x['rounds']) for x in output),
                independent_references=references_summary,
                checked_reference_rounds=sum(len(x['rounds']) for x in references_summary),
                checked_stages=sum(len(x['stages']) for x in output),
                pairing='Controlled resource batch only; settings, per-task data and source IDs match NC references. Historical thesis rows remain unpaired.')


if __name__ == '__main__':
    print(json.dumps(collect(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
