import torch
import math

class AdjacencyMatrixDataset(torch.utils.data.Dataset):
    def __init__(self, srcs, targets):
        super(AdjacencyMatrixDataset).__init__()

        assert len(srcs) == len(targets)
        
        self.srcs = srcs
        self.targets = targets
        self.l = len(srcs)

    def __len__(self):
        return self.l
    
    def __getitem__(self, idx):
        src = self.srcs[idx]
        target = self.targets[idx]
        return src, target
