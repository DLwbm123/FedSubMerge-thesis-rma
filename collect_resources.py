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
    assert not queue['active'] and not queue['pending']
    assert len(queue['finished']) == 8 and all(j['exit_code'] == 0 for j in queue['finished'])
    output = []
    for job in queue['finished']:
        dataset, method = job['dataset'], job['method']
        root = batch / f'{dataset}_{method}'
        config, result = read(root / 'config.json'), read(root / 'result.json')
        rounds, stages = rows(root / 'rounds.jsonl'), rows(root / 'stages.jsonl')
        tasks = 4 if dataset == 'pathmnist' else 10
        assert result['status'] == 'complete' and not (root / 'failure.json').exists()
        assert [(x['task'], x['round']) for x in rounds] == [(t, r) for t in range(1, tasks+1) for r in range(1, 21)]
        assert [x['task'] for x in stages] == list(range(1, tasks+1))
        assert all((root / f'task{t:02d}.pt').stat().st_size > 0 for t in range(1, tasks+1))
        assert all(math.isfinite(x) for row in result['accuracy_matrix'] for x in row)
        ratios = []
        for t in range(2, tasks+1):
            nc = references / f'{dataset}_task{t:02d}_seed2025'
            nc_config = read(nc / 'config.json')
            data = read(root / f'task{t:02d}_data.json')
            for key in ('seed', 'rounds', 'local_epochs', 'batch_size', 'lr', 'data_root'):
                assert config[key] == nc_config[key], (dataset, key)
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
                pairing='Controlled resource batch only; settings, per-task data and source IDs match NC references. Historical thesis rows remain unpaired.')


if __name__ == '__main__':
    print(json.dumps(collect(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
