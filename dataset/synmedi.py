import os
import torch.utils.data as data
import torch
from PIL import Image
import numpy as np

from .task import task_get_label_transformation_lambda

class SynMediIncremental(data.Dataset):

    def __init__(self, mode, transform=None, task=None, _root=None, _version=None, proposal=False, save_pred=None, skip_step=1):

        return

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        return

    def __len__(self):
        return self.dataset.shape[0] // self.skip_step
    


