
from typing import List
import torch
from pdb import set_trace
from torchvision import transforms as t
from Utils.utils import set_seed
from torchvision.transforms.transforms import ToPILImage

class TensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):            
        return self.tensors[0].size(0)