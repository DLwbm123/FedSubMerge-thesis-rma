"""Assemble measured results; refresh available thesis metric cells without changing ACC."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
CORE = ['FedAvg', 'Fed-GPM', 'FedSubMerge', 'FedSubMerge-AD']
LABELS = {
    ('pathmnist', 'distribution', 0.3): 'PathMNIST α=0.3',
    ('pathmnist', 'distribution', 0.1): 'PathMNIST α=0.1',
    ('hyperkvasir', 'distribution', 0.3): 'Hyper-Kvasir α=0.3',
    ('hyperkvasir', 'distribution', 0.1): 'Hyper-Kvasir α=0.1',
    ('hyperkvasir', 'quantity', None): 'Hyper-Kvasir quantity-skew',
    ('skin', 'feature', None): '皮肤病变 feature-skew',
}


def read(name):
    return json.loads((RESULTS / name).read_text())


def write_csv(name, data):
    with (RESULTS / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def markdown(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '|---|' + '---:|' * (len(headers)-1)] +
                     ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])


def main():
    all_rows = []
    for name, scenario, alpha in [('resources_complete.json', 'distribution', .3),
                                  ('quantity_resources_complete.json', 'quantity', None),
                                  ('coverage_resources_complete.json', None, None)]:
        batch = read(name)
        if name.startswith('coverage'):
            assert (len(batch['runs']), batch['checked_rounds'], batch['checked_reference_rounds'], batch['checked_stages']) == (12, 1360, 280, 68)
            write_csv('coverage_resources_summary.csv', [r['summary'] for r in batch['runs']])
        for run in batch['runs']:
            s = run['summary']
            c = run['config']
            all_rows.append(dict(dataset=s['dataset'], scenario=scenario or c['scenario'],
                alpha=alpha if scenario else c['alpha'], method=s['method'], seed=s['seed'],
                **{k:s[k] for k in ['ACC', 'RMA', 'BWTR', 'MPE', 'DRR']},
                DRR_kind='unique_real_source_images', synthetic_DRR_proxy=None,
                client_auxiliary_peak_MiB=s['max_auxiliary_bytes_all_clients']/2**20,
                cuda_allocated_peak_GiB=s['peak_allocated_mib']/1024,
                local_training_seconds=s['local_training_seconds'],
                boundary_seconds=s['construction_seconds']+s['subspace_merge_seconds']))
    target_components = []
    for run in read('public_resources_complete.json')['runs']:
        s, stages = run['result'], run['stages']
        method = run['method']
        aux_key = 'client_auxiliary_bytes_logical' if method == 'FOT' else 'auxiliary_bytes_all_clients'
        real = s['DRR_kind'] == 'unique_real_source_images'
        all_rows.append(dict(dataset=run['dataset'], scenario='distribution', alpha=.3,
            method=method, seed=run['config']['seed'], **{k:s[k] for k in ['RMA','BWTR','MPE']},
            ACC=s['final_ACC'], DRR=s['DRR'] if real else None, DRR_kind=s['DRR_kind'],
            synthetic_DRR_proxy=None if real else s['DRR'],
            client_auxiliary_peak_MiB=None if method=='TARGET' else max(t[aux_key] for t in stages)/2**20,
            cuda_allocated_peak_GiB=max(t['peak_allocated_mib'] for t in stages+run['rounds'])/1024,
            local_training_seconds=sum(t['local_training_seconds'] for t in run['rounds']),
            boundary_seconds=sum(t.get('construction_seconds',0)+t.get('merge_seconds',0) for t in stages)))
        if method == 'TARGET':
            row = dict(dataset=run['dataset'], scenario='distribution', alpha=.3)
            for key in ['generator_bytes', 'student_bytes', 'teacher_bytes', 'generator_optimizer_bytes',
                        'student_optimizer_bytes', 'synthetic_disk_bytes', 'synthetic_decoded_bytes']:
                row[key.replace('_bytes','_peak_MiB')] = max(t.get(key,0) for t in stages)/2**20
            target_components.append(row)
    assert len(all_rows) == 30
    assert len({(r['dataset'],r['scenario'],r['alpha'],r['method']) for r in all_rows}) == 30
    all_rows.sort(key=lambda r:(list(LABELS).index((r['dataset'],r['scenario'],r['alpha'])),
                               (CORE+['FOT','Fed-DER','TARGET']).index(r['method'])))
    write_csv('all_measured_metrics.csv', all_rows)
    write_csv('target_auxiliary_components.csv', target_components)
    by_key = {(r['dataset'],r['scenario'],r['alpha'],r['method']):r for r in all_rows}
    table = []
    for r in all_rows:
        table.append([LABELS[r['dataset'],r['scenario'],r['alpha']],r['method']] +
                     ['' if r[k] is None else f'{r[k]:.4f}' for k in ['ACC','RMA','BWTR','MPE','DRR']] +
                     ['' if r[k] is None else f'{r[k]:.2f}' for k in ['client_auxiliary_peak_MiB','cuda_allocated_peak_GiB']])
    notes = '''# 全部实测结果

30 条完整持续学习序列，30 个独立任务参考；统一 seed=2025。ACC 来自本行同一实测准确率矩阵，未混入论文历史 ACC。原始实测记录及逐任务参考见各 complete.json。六个设置均覆盖 FedAvg、Fed-GPM、FedSubMerge、FedSubMerge-AD；两个 α=0.3 设置额外覆盖 FOT、Fed-DER、TARGET。OrganAMNIST 按用户要求未运行。

ACC 单位为百分数；RMA、BWTR、MPE、DRR 为比值。客户端辅助存储为任务末所有客户端辅助张量逻辑总量的最大值，不包含分类网络、服务器状态、临时工作区或 Python 对象开销；FOT 包含十个客户端的逻辑副本。显存是 PyTorch 已分配张量的进程峰值，不是设备总占用。辅助状态内容与精度随实现而异，Fed-GPM 基使用 float64，PGS 基使用 float32。

TARGET 的 DRR 是生成图像按教师预测类别归属的代理统计，不并入真实源样本 DRR；公共 DRR 和客户端辅助总量留空。其教师、生成器、优化器和合成池在 target_auxiliary_components.csv 中分项报告，不将各项跨阶段最大值相加当作同时峰值。DRR 包括表示构建使用的源样本，因此零历史原图回放不等于 DRR=0。并发任务耗时不用于严格速度排名。

皮肤病变使用用户批准的 24,827 张缓存：训练 15,966、验证 3,991、测试 4,870；该版本与论文表中的 24,180 张不同。代码适配和数据差异见各实验协议。单 seed 不支持统计显著性声明。

'''
    (RESULTS/'all_measured_metrics.md').write_text(notes + markdown(
        ['设置','方法','ACC (%) ↑','RMA ↑','BWTR ↑','MPE ↓','DRR ↓','辅助存储 MiB','显存 GiB'],table)+'\n')
    resources=[]
    for key,label in LABELS.items():
        resource=[label]+[by_key[*key,m]['client_auxiliary_peak_MiB'] for m in CORE[1:]]
        resources.append(resource)
    print(markdown(['设置','Fed-GPM MiB','FedSubMerge MiB','AD MiB'],
                   [[r[0]]+[f'{v:.2f}' for v in r[1:]] for r in resources]))
    # The user-requested clean historical table remains separate from measured results.
    csv_path=ROOT/'thesis_tables/main_tables_three_skews.csv'
    tex_path=ROOT/'thesis_tables/FedCL.tex'
    if csv_path.exists() and tex_path.exists():
        data=list(csv.DictReader(csv_path.open()))
        for r in data:
            if r['setting']=='OrganAMNIST':continue
            dataset='pathmnist' if 'PathMNIST' in r['setting'] else 'hyperkvasir' if 'Hyper-Kvasir' in r['setting'] else 'skin'
            scenario='distribution' if r['skew']=='distribution-skew' else 'quantity' if r['skew']=='quantity-skew' else 'feature'
            alpha=(.3 if '0.3' in r['setting'] else .1) if scenario=='distribution' else None
            measured=by_key.get((dataset,scenario,alpha,r['method'].rstrip('*')))
            if measured:
                for k in ['BWTR','RMA','MPE','DRR']:
                    r[k+'_supplement']='' if measured[k] is None else f'{measured[k]:.4f}'
        with csv_path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
        lines=tex_path.read_text().splitlines();setting=None;scenario=None;changed=0
        for i,line in enumerate(lines):
            if line.startswith('\\caption{'):
                scenario='distribution-skew' if 'Distribution-skew' in line else 'quantity-skew' if 'Quantity-skew' in line else 'real-world feature skew'
            if line.startswith('\\multicolumn'):
                setting='PathMNIST' if 'PathMNIST' in line else 'Hyper-Kvasir' if 'Hyper-Kvasir' in line else 'OrganAMNIST' if 'OrganAMNIST' in line else '多来源皮肤病变'
                if scenario=='distribution-skew':setting+=', α='+('0.3' if '0.3' in line else '0.1')
            if ' & ' not in line or not setting:continue
            parts=line.split(' & ');method=parts[0].replace('$^{\\ast}$','*')
            row=next((r for r in data if (r['skew'],r['setting'],r['method'])==(scenario,setting,method)),None)
            if row:
                assert float(parts[1])==float(row['ACC_original_percent'])
                lines[i]=' & '.join(parts[:2]+[row[k+'_supplement'] for k in ['BWTR','RMA','MPE','DRR']])+r' \\'
                changed+=1
        assert changed==98
        tex_path.write_text('\n'.join(lines)+'\n')
        print('Updated 98 historical table rows; 30 measured metric rows available; ACC preserved.')


if __name__ == '__main__':
    main()
