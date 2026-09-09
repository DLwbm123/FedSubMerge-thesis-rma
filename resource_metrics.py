"""Benchmark resource accounting; subspace storage is not model expansion."""
import numpy as np
import torch


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def benchmark_resources(parameters, source_counts, train_counts):
    if len(parameters) < 2 or len(source_counts) != len(parameters) - 1:
        raise ValueError('Need all stages and all non-final source tasks')
    if len(train_counts) != len(source_counts) or min(train_counts) <= 0:
        raise ValueError('Invalid source-task denominators')
    return dict(MPE=float(np.mean(np.diff(parameters) / parameters[0])),
                DRR=float(np.mean(np.array(source_counts) / train_counts)))


def project_gradient(gradient, basis):
    matrix = gradient.reshape(gradient.shape[0], -1)
    matrix.sub_((matrix @ basis) @ basis.T)


if __name__ == '__main__':
    result = benchmark_resources([100, 110, 130], [2, 3], [10, 20])
    assert np.allclose([result['MPE'], result['DRR']], [.15, .175])
    g = torch.tensor([[3., 4.]])
    project_gradient(g, torch.tensor([[1.], [0.]]))
    assert torch.equal(g, torch.tensor([[0., 4.]]))
    assert tensor_bytes({'g': g}) == 8
    print('RESOURCE_CHECK_OK')
