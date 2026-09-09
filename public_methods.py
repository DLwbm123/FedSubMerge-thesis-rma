"""Resource adapters for pinned public FOT, DER and TARGET implementations.

See PUBLIC_BASELINES.md for algorithm sources and explicit adaptation choices.
"""
import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from resource_metrics import tensor_bytes


def sync_time():
    import time
    torch.cuda.synchronize()
    return time.monotonic()


class IndexedData:
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        return (*self.subset[index], int(self.subset.indices[index]))


class DER:
    def __init__(self, clients=10, capacity=256, alpha=.5):
        self.capacity, self.alpha = capacity, alpha
        self.buffers = {c: [] for c in range(clients)}
        self.seen = {c: 0 for c in range(clients)}
        self.used = {}
        self.presentations = 0

    def replay_backward(self, net, client, task, transform, batch_size):
        buffer = self.buffers[client]
        if not buffer:
            return 0.
        selected = np.random.choice(len(buffer), min(batch_size, len(buffer)), replace=False)
        x, logits = [], []
        for i in selected:
            image, previous, source_task, source_id = buffer[i]
            x.append(transform(transforms.ToPILImage()(image)))
            logits.append(previous)
            if source_task < task:
                self.used.setdefault(source_task, set()).add(source_id)
                self.presentations += 1
        output = net(torch.stack(x).cuda())
        penalty = self.alpha * F.mse_loss(output, torch.stack(logits).cuda())
        penalty.backward()
        return float(penalty.detach())

    def add(self, client, task, images, logits, ids):
        # Standard reservoir over the observed stream, as in Mammoth DER.
        for image, logit, source_id in zip(images, logits, ids):
            n = self.seen[client]
            index = n if n < self.capacity else np.random.randint(0, n+1)
            self.seen[client] += 1
            if index < self.capacity:
                item = (image.detach().cpu().clone(), logit.detach().cpu().clone(), task, int(source_id))
                if len(self.buffers[client]) < self.capacity:
                    self.buffers[client].append(item)
                else:
                    self.buffers[client][index] = item

    def storage(self):
        return tensor_bytes(self.buffers)


class FOT:
    def __init__(self, net, epsilon=.95, increment=.001, smoke=False):
        self.layers = {name+'.weight': layer for name, layer in net.named_modules() if isinstance(layer, nn.Conv2d)}
        self.bases = {}
        self.epsilon, self.increment, self.smoke = epsilon, increment, smoke

    @torch.no_grad()
    def project_update(self, old, new):
        for key, basis_cpu in self.bases.items():
            delta = (new[key]-old[key]).cuda()
            basis = basis_cpu.cuda()
            g = delta.reshape(delta.shape[0], -1)
            g.sub_((g @ basis) @ basis.T)
            new[key] = old[key] + delta.cpu()
        return new

    @torch.no_grad()
    def build(self, net, loaders, local_states, record, task):
        """Stream the official Gaussian sketch instead of retaining all patches.

        Source ResNet windows are 32/16/8/4 with sketch width 5*input dimension.
        Channel counts follow the thesis ResNet18 instead of source nf=20.
        """
        started = sync_time()
        totals, ratio_lists, counts = {}, {}, []
        upload = 0
        for client, loader in loaders.items():
            net.load_state_dict(local_states[client])
            net.eval()
            sketches, norm_orig, norm_residual = {}, {}, {}
            bases = {k: v.cuda() for k, v in self.bases.items()}
            def hook(key, module, inputs):
                a = inputs[0].detach()
                stage = int(key[5]) if key.startswith('layer') else 1
                # First block input and shortcut enter a downsampling stage at the preceding resolution.
                downsample_input = stage > 1 and '.0.' in key and ('.conv1.' in key or '.shortcut.' in key)
                window = 32 // (2 ** max(0, stage-1-int(downsample_input)))
                if self.smoke:
                    window = min(window, 2)
                window = min(window, a.shape[-1])
                kernel, stride, padding = module.kernel_size[0], module.stride[0], module.padding[0]
                side = (window+2*padding-kernel)//stride+1
                crop = a[:, :, :window+padding, :window+padding]
                patches = F.unfold(crop, kernel, padding=padding, stride=stride)
                full_side = (crop.shape[-1]+2*padding-kernel)//stride+1
                patches = patches.reshape(a.shape[0], -1, full_side, full_side)[:, :, :side, :side]
                mat = patches.permute(1, 0, 2, 3).reshape(patches.shape[1], -1)
                norm_orig[key] = norm_orig.get(key, 0.) + float(mat.square().sum())
                if key in bases:
                    u = bases[key]
                    mat = mat - u @ (u.T @ mat)
                norm_residual[key] = norm_residual.get(key, 0.) + float(mat.square().sum())
                width = mat.shape[0] * 5
                if self.smoke:
                    width = min(width, 8)
                if key not in sketches:
                    sketches[key] = torch.zeros(mat.shape[0], width, device='cuda')
                for start in range(0, mat.shape[1], 512):
                    chunk = mat[:, start:start+512]
                    gaussian = torch.randn(chunk.shape[1], width, device='cuda')
                    sketches[key].addmm_(chunk, gaussian)
            handles = [m.register_forward_pre_hook(lambda mod, ins, key=k: hook(key, mod, ins))
                       for k, m in self.layers.items()]
            try:
                for x, *_ in DataLoader(loader.dataset, batch_size=8, shuffle=False):
                    net(x.cuda())
            finally:
                for h in handles:
                    h.remove()
            counts.append(len(loader.dataset))
            for k, sketch in sketches.items():
                ratio = math.sqrt(norm_residual[k] / norm_orig[k]) if norm_orig[k] > 0 else 0.
                ratio_lists.setdefault(k, []).append(ratio)
                upload += tensor_bytes(sketch)
                if k not in totals:
                    totals[k] = sketch
                else:
                    totals[k].add_(sketch)
            record('fot_clients', dict(task=task+1, client=client, source_count=len(loader.dataset)))
            del sketches, bases
        construction_end = sync_time()
        for k in list(totals):
            sketch = totals.pop(k)
            average_ratio = float(np.average(ratio_lists[k], weights=counts))
            if average_ratio <= 1e-12 or float(sketch.square().sum()) <= 1e-20:
                continue
            threshold = (average_ratio-(1-self.epsilon)) / average_ratio
            # Left singular vectors via covariance: equivalent subspace, avoids unused enormous V.
            if sketch.shape[1] < sketch.shape[0]:
                u, singular, _ = torch.linalg.svd(sketch, full_matrices=False)
                energy = singular.square()
            else:
                covariance = sketch @ sketch.T
                eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
                energy = eigenvalues.flip(0).clamp_min(0)
                u = eigenvectors.flip(1)
                del covariance, eigenvalues, eigenvectors
            rank = min(len(energy), max(1, int((energy.cumsum(0)/energy.sum() <= threshold).sum())+1))
            update = u[:, :rank]
            if k in self.bases:
                update = torch.cat([self.bases[k].cuda(), update], dim=1)
            q, _ = torch.linalg.qr(update, mode='reduced')
            self.bases[k] = q.cpu()
            record('fot_subspaces', dict(task=task+1, layer=k, rank=q.shape[1], dimension=q.shape[0],
                   residual_norm_ratio=average_ratio, effective_threshold=threshold))
            del sketch, energy, u, update, q
        self.epsilon += self.increment
        return dict(construction_seconds=construction_end-started, merge_seconds=sync_time()-construction_end,
                    auxiliary_upload_bytes=upload, auxiliary_download_bytes=len(loaders)*tensor_bytes(self.bases),
                    representation_source_count=sum(counts), server_auxiliary_bytes=tensor_bytes(self.bases))


def target_components(source):
    """Load only reviewed, pinned self-contained declarations; no framework imports."""
    import torch.nn.init as init
    from torch.autograd import Variable
    names = {'Generator', 'DeepInversionHook', 'weight_init', '_KD_loss', 'kldiv'}
    parsed = ast.parse(Path(source).read_text())
    nodes = [n for n in parsed.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError('Pinned TARGET declarations missing')
    scope = dict(torch=torch, nn=nn, F=F, init=init, Variable=Variable)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)
    return SimpleNamespace(**{k: scope[k] for k in names})


class TARGET:
    def __init__(self, net, source, output, input_size, smoke=False):
        self.parts = target_components(source)
        self.output, self.size, self.smoke = Path(output), input_size, smoke
        self.teacher = None
        self.pool = []
        self.used = {}
        self.presentations = 0
        self.last_generator_bytes = 0
        self.generator_parameters = 0

    def replay_backward(self, net, task, lo, batch_size):
        if self.teacher is None or not self.pool:
            return 0.
        chosen = np.random.choice(len(self.pool), min(batch_size, len(self.pool)), replace=False)
        images = []
        for idx in chosen:
            path, source_task = self.pool[idx]
            with Image.open(path) as image:
                images.append(transforms.ToTensor()(image.convert('RGB')))
            self.used.setdefault(source_task, set()).add(str(path))
        self.presentations += len(chosen)
        x = torch.stack(images).cuda()
        with torch.no_grad():
            target = self.teacher(x)[:, :lo]
        penalty = 25 * self.parts._KD_loss(net(x)[:, :lo], target, 2)
        penalty.backward()
        return float(penalty.detach())

    def generate(self, net, seen_classes, task, boundaries, record):
        from kornia import augmentation
        begin = sync_time()
        self.teacher = copy.deepcopy(net).cuda().eval().requires_grad_(False)
        student = copy.deepcopy(net).cuda()
        student.apply(self.parts.weight_init)
        generator = self.parts.Generator(nz=256, ngf=64, img_size=self.size, nc=3).cuda()
        self.generator_parameters = sum(p.numel() for p in generator.parameters())
        self.last_generator_bytes = tensor_bytes(generator.state_dict())
        steps, rounds, warmup, kd_steps = (2, 2, 0, 1) if self.smoke else (10, 30, 20, 400)
        logical_batch = 8 if self.smoke else 256
        microbatch = 8 if self.smoke else 16
        augment = nn.Sequential(augmentation.RandomCrop((self.size, self.size), padding=4),
                                augmentation.RandomHorizontalFlip())
        hooks = [self.parts.DeepInversionHook(m, .9) for m in self.teacher.modules() if isinstance(m, nn.BatchNorm2d)]
        meta = torch.optim.Adam(generator.parameters(), lr=.002*steps, betas=(.5,.999))
        student_opt = torch.optim.SGD(student.parameters(), lr=.2, weight_decay=.0001, momentum=.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(student_opt, 200, eta_min=2e-4)
        pool_dir = self.output / f'synthetic_task{task+1:02d}'
        pool_dir.mkdir()
        self.pool = []
        try:
            for outer in range(rounds):
                fast = copy.deepcopy(generator).cuda().train()
                z = torch.randn(logical_batch, 256, device='cuda', requires_grad=True)
                labels = torch.randint(seen_classes, (logical_batch,), device='cuda')
                opt = torch.optim.Adam([{'params': fast.parameters()}, {'params': [z], 'lr': .01}],
                                       lr=.002, betas=(.5,.999))
                meta.zero_grad(set_to_none=True)
                student.eval().requires_grad_(False)
                best_cost, best_images = math.inf, None
                for inner in range(steps):
                    opt.zero_grad(set_to_none=True)
                    cost, candidates = 0., []
                    for start in range(0, logical_batch, microbatch):
                        stop = min(start+microbatch, logical_batch)
                        images = fast(z[start:stop])
                        x = augment(images)
                        target = self.teacher(x)[:, :seen_classes]
                        bn = sum(h.r_feature for h in hooks)
                        oh = F.cross_entropy(target, labels[start:stop])
                        adv = 0.
                        if outer+1 >= warmup:
                            prediction = student(x)[:, :seen_classes]
                            mask = (prediction.argmax(1) == target.argmax(1)).float()
                            adv = -(self.parts.kldiv(prediction, target, reduction='none').sum(1)*mask).mean()
                        loss = 10*bn + .5*oh + adv
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Non-finite TARGET synthesis loss')
                        (loss*((stop-start)/logical_batch)).backward()
                        cost += float(loss.detach())*((stop-start)/logical_batch)
                        candidates.append(images.detach().cpu())
                        del images, x, target, bn, oh, adv, loss
                    for p, q in zip(generator.parameters(), fast.parameters()):
                        if q.grad is not None:
                            if p.grad is None:
                                p.grad = q.grad.detach().clone()
                            else:
                                p.grad.add_(q.grad.detach())
                    opt.step()
                    if cost < best_cost:
                        best_cost, best_images = cost, torch.cat(candidates)
                meta.step()
                for hook in hooks:
                    hook.update_mmt()
                with torch.no_grad():
                    for start in range(0, logical_batch, microbatch):
                        x = best_images[start:start+microbatch]
                        predicted = self.teacher(x.cuda())[:, :seen_classes].argmax(1).cpu().tolist()
                        for j, (image, label) in enumerate(zip(x, predicted)):
                            path = pool_dir / f'{outer:03d}_{start+j:03d}.png'
                            transforms.ToPILImage()(image.clamp(0,1)).save(path)
                            source_task = next(i for i, end in enumerate(boundaries) if label < end)
                            self.pool.append((path, source_task))
                del fast, z, opt, best_images, candidates
                student.train().requires_grad_(True)
                if outer >= warmup:
                    for _ in range(kd_steps):
                        chosen = np.random.choice(len(self.pool), min(64, len(self.pool)), replace=False)
                        images = []
                        for idx in chosen:
                            with Image.open(self.pool[idx][0]) as image:
                                images.append(transforms.ToTensor()(image.convert('RGB')))
                        x = torch.stack(images).cuda()
                        with torch.no_grad():
                            target = self.teacher(x)[:, :seen_classes]
                        student_opt.zero_grad(set_to_none=True)
                        loss = self.parts.kldiv(student(x)[:, :seen_classes], target, T=20.)
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Non-finite TARGET student loss')
                        loss.backward()
                        student_opt.step()
                    scheduler.step()
                record('target_synthesis', dict(task=task+1, synthesis_round=outer+1,
                       synthesis_loss=best_cost, generated_count=len(self.pool),
                       elapsed_seconds=sync_time()-begin))
                print('SYNTHESIS', task+1, outer+1, len(self.pool), flush=True)
            torch.save(generator.state_dict(), self.output / f'generator_task{task+1:02d}.pt')
        finally:
            for hook in hooks:
                hook.remove()
        disk_bytes = sum(path.stat().st_size for path, _ in self.pool)
        record('synthetic_provenance', dict(task=task+1,
               samples=[dict(file=path.name, source_task=t+1) for path, t in self.pool]))
        return dict(construction_seconds=sync_time()-begin, merge_seconds=0.,
                    generator_parameters=self.generator_parameters, generator_bytes=self.last_generator_bytes,
                    student_bytes=tensor_bytes(student.state_dict()),
                    generator_optimizer_bytes=tensor_bytes(meta.state_dict()),
                    student_optimizer_bytes=tensor_bytes(student_opt.state_dict()),
                    teacher_bytes=tensor_bytes(self.teacher.state_dict()), generated_count=len(self.pool),
                    synthetic_disk_bytes=disk_bytes, synthetic_decoded_bytes=len(self.pool)*3*self.size*self.size*4,
                    auxiliary_upload_bytes=0, auxiliary_download_bytes=10*(disk_bytes+tensor_bytes(self.teacher.state_dict())))
