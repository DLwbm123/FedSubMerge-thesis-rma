"""Small checks of reservoir accounting, actual server projection, and TARGET loading."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from public_methods import DER, FOT, target_components


if __name__=='__main__':
    np.random.seed(1)
    replay = DER(clients=1, capacity=2)
    replay.add(0,0,torch.zeros(3,3,4,4),torch.zeros(3,2),[11,12,13])
    assert replay.seen[0]==3 and len(replay.buffers[0])==2
    net = nn.Sequential(nn.Conv2d(1,1,(1,2),bias=False))
    fot = FOT(net)
    fot.bases = {'0.weight':torch.tensor([[1.],[0.]])}
    old = {'0.weight':torch.zeros(1,1,1,2)}
    new = {'0.weight':torch.tensor([[[[3.,4.]]]])}
    projected = fot.project_update(old,new)
    assert torch.equal(projected['0.weight'],torch.tensor([[[[0.,4.]]]]))
    target = target_components(Path(sys.argv[1])/'target.py')
    generator = target.Generator(nz=8,ngf=4,img_size=16,nc=3)
    assert generator(torch.zeros(2,8)).shape==(2,3,16,16)
    print('PUBLIC_METHOD_CHECK_OK')
