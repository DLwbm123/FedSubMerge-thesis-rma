"""Fetch pinned upstream source declarations without installing upstream environments."""
import base64
import json
from pathlib import Path
import sys
import urllib.request

SOURCES = {
    'FOT': ('duygunuryldz/Federated_Orthogonal_Training', 'f343e366f393812ccfbce79a505ddcdaa5305bf7', {
        'fot_trainer.py': 'trainer/resnet_trainer.py',
        'fot_aggregator.py': 'FedML/fedml_api/distributed/fedavg_seq_cont/FedAVGAggregator.py'}),
    'Fed-DER': ('aimagelab/mammoth', 'e75a491c69fd729edeb01431afb753d9157d9a81', {
        'der.py': 'models/der.py', 'der_buffer.py': 'utils/buffer.py', 'DER_LICENSE': 'LICENSE'}),
    'TARGET': ('zj-jayzhang/Federated-Class-Continual-Learning', '78cb9c05eea2555c33c0b83391811c3bb1ef059c', {
        'target.py': 'methods/target.py'})}


if __name__ == '__main__':
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, (repo, revision, files) in SOURCES.items():
        manifest[name] = dict(repository='https://github.com/'+repo, revision=revision, files=files)
        for destination, source in files.items():
            url = f'https://api.github.com/repos/{repo}/contents/{source}?ref={revision}'
            request = urllib.request.Request(url, headers={'User-Agent':'thesis-resource-adapter'})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.load(response)
            (output/destination).write_bytes(base64.b64decode(data['content']))
    (output/'sources.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Fetched six source/license files and revision manifest')
