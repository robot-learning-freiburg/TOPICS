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

def get_camera_id(path):
    match = re.search(r"camera(\d+)", path)
    return int(match.group(1)) if match else -1


class MMORIncremental(data.Dataset):

    def __init__(self, mode, transform=None, task=None, _root=None, _version=None, proposal=False, save_pred=None, skip_step=1):

        self.prep_label = True

        if mode == 'train':
            imagesets = "ImageSets/train_200_3_azure.json"
        elif mode == 'val': 
            imagesets = "ImageSets/val_200_3_azure.json"
        else:
            mode = 'val'  
            imagesets = "ImageSets/test_200_3_azure.json"

        with open(os.path.join(_root, imagesets)) as f:
            dataset = json.load(f)
        print('length of dataset', len(dataset))

        if proposal:
            dataset = [(x, x.replace('.jpg', '.png').replace('colorimage/', f'segmentation_export_{get_camera_id(x)}/'), 
                                  x.replace("/0", "/proposal_100/0")) for x in dataset]
        else:
            dataset = [(x, x.replace('.jpg', '.png').replace('colorimage/', f'segmentation_export_{get_camera_id(x)}/')) for x in dataset]

        # self.compute_statistics(dataset)

        train = True if mode == 'train' else False
        dataset = task.filter_images(dataset, train=train)
        self.dataset = np.asarray(dataset)
        self.transform = transform
        self.label_transform = task_get_label_transformation_lambda(task, mode)
        if mode != 'train':
            self.label_transform_train = task_get_label_transformation_lambda(task, 'train')
        else:
            self.label_transform_train = None
        
        self.internal_masking_value = task.internal_masking_value
        self.counter = 0
        self.mode = mode
        self.proposal = proposal
        self.skip_step = skip_step
        print("Init MMOR Dataset!")

    def compute_statistics(self, dataset):

        pixel_count = {}

        for file in dataset:
            img = Image.open(file[1]).convert('L')  # Convert to grayscale ('L' mode)
            folder = Path(file[1]).parts[4]
            if folder not in pixel_count:
                pixel_count[folder] = {}
            img_array = np.array(img)
            unique_values, counts = np.unique(img_array, return_counts=True)

            for value, count in zip(unique_values, counts):
                if value in pixel_count[folder]:
                    pixel_count[folder][value] += count
                else:
                    pixel_count[folder][value] = count
        
        # Print the pixel count dictionary
        for folder, dicts in pixel_count.items():
            requirements = list(range(0,19)) + list(range(24,30))
            print(f'{folder}:{list(dicts.keys())}')
            for value, count in sorted(dicts.items()):
                if value in requirements:
                    requirements.remove(value)
            print(f'missing {folder}:', requirements)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        index *= self.skip_step
        if self.prep_label:
            img = np.array(Image.open(self.dataset[index][0]).convert('RGB'))
            target = np.array(Image.open(self.dataset[index][1]))
            target_torch = self.label_transform(torch.tensor(target))

            if self.label_transform_train is not None:
                target_torch_train = self.label_transform_train(torch.tensor(target))
                targets = [target_torch_train.numpy(), target_torch.numpy()]
                if self.proposal:
                    proposal = np.array(Image.open(self.dataset[index][2]))
                    targets.append(proposal)
                augmented = self.transform(image=img, masks=targets, mask=target_torch.numpy())
                img_t = augmented["image"]
                target_t = augmented['masks']
                target_check = augmented['mask']

            else:
                if self.proposal:
                    proposal = np.array(Image.open(self.dataset[index][2]))
                    targets = [target_torch.numpy(), proposal]
                    augmented = self.transform(image=img, masks=targets, mask=target_torch.numpy())
                    target_t = augmented['masks']
                    target_check = augmented['mask']
                else:
                    augmented = self.transform(image=img, mask=target_torch.numpy())
                    target_t = augmented['mask']
                    target_check = target_t
                img_t = augmented["image"]

            return img_t, target_t
        else:
            img = np.array(Image.open(self.dataset[index]).convert('RGB'))
            if self.transform is not None:
                img = self.transform(image=img)["image"]
            return img, self.dataset[index].split('/')[-1]

    def __len__(self):
        return self.dataset.shape[0] // self.skip_step

