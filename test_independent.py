"""Run with the existing server Python: python test_independent.py."""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from run_independent import aggregate, evaluate, round_lr, task_bounds, task_indices


def main():
    assert task_bounds("pathmnist", 2) == (2, 4)
    assert task_bounds("pathmnist", 4) == (6, 9)
    assert task_bounds("hyperkvasir", 10) == (18, 20)
    assert task_bounds("skin", 2, "feature") == (2, 4)
    assert task_bounds("skin", 3, "feature") == (4, 6)
    assert task_bounds("hyperkvasir", 2, "quantity") == (4, 8)
    assert task_bounds("hyperkvasir", 5, "quantity") == (16, 20)
    assert task_bounds("hyperkvasir", 10) == (18, 20)
    try:
        task_bounds("pathmnist", 2, "quantity")
    except ValueError:
        pass
    else:
        raise AssertionError("Unsupported quantity partition accepted")
    labels = np.array([3, 1, 2, 3, 8])
    assert task_indices(labels, [4, 0, 2, 1], 2, 4).tolist() == [0, 2]
    try:
        task_indices(labels, [5], 2, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid client index accepted")
    states = [{"w": torch.tensor([1.0, 4.0]), "n": torch.tensor(2)},
              {"w": torch.tensor([5.0, 0.0]), "n": torch.tensor(6)}]
    out = aggregate(states, [1, 3])
    assert torch.equal(out["w"], torch.tensor([4.0, 1.0]))
    assert out["n"].dtype == torch.int64 and out["n"].item() == 5
    assert states[0]["w"].tolist() == [1.0, 4.0]
    assert [round_lr(0.01, r, 20) for r in (0, 9, 10, 14, 15, 19)] == [
        0.01, 0.01, 0.001, 0.001, 0.00010000000000000002, 0.00010000000000000002]
    # The largest logits are outside this task; inference must retain global IDs.
    logits = torch.tensor([[99.0, 98.0, 2.0, 1.0], [99.0, 98.0, 1.0, 2.0]])
    loader = DataLoader(TensorDataset(logits, torch.tensor([2, 3]), logits), batch_size=2)
    assert evaluate(torch.nn.Identity(), loader, torch.device("cpu"), 2, 4) == 100.0
    print("PASS: task boundaries, index isolation, weighted FedAvg, schedule, task-aware evaluation")


if __name__ == "__main__":
    main()
