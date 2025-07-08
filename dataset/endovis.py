import glob
import os
import json
import re
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from albumentations.pytorch import ToTensorV2

from .task import task_get_label_transformation_lambda

class EndovisIncremental(data.Dataset):

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

