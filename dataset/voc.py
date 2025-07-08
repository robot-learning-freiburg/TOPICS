import os
import torch.utils.data as data
import torch
from PIL import Image
import numpy as np

from .task import task_get_label_transformation_lambda


class PascalVOCIncremental(data.Dataset):

    def __init__(self, mode, transform=None, task=None, _root=None, _version=None, proposal=False, save_pred=None, skip_step=1):

        if mode == 'train':
            split = 'training'
            self.prep_label = True
            imagesets = "ImageSets/train_aug.txt"

        else:
            mode = 'val'
            split = 'validation'
            self.prep_label = True
            imagesets = "ImageSets/val.txt"

        self.split = split
        with open(os.path.join(_root, imagesets)) as f:
            dataset = [x[:-1].split(' ') for x in f.readlines()]

        if proposal:
            dataset = [(os.path.join(_root, x[0][1:]), os.path.join(_root, x[1][1:]),
                        os.path.join(_root, x[1][1:].replace('SegmentationClassAug', 'proposal_100'))) for x in dataset]
        else:
            dataset = [(os.path.join(_root, x[0][1:]), os.path.join(_root, x[1][1:])) for x in dataset]
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
        print("Init PascalVOC Dataset!")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        index *= self.skip_step
        img = np.array(Image.open(self.dataset[index][0]).convert('RGB'))
        target = np.array(Image.open(self.dataset[index][1]))
        target_torch = self.label_transform(torch.tensor(target))
        if (torch.count_nonzero(target_torch != self.internal_masking_value) == 0) and (self.mode == 'train'):
            print("ERROR, wrong dataset selected! Target doesn't contain image!")
            exit()

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
            if torch.count_nonzero(target_check != target_t[1]) > 0:
                print("Albumentations didn't augment masks synchronized!", torch.unique(target_check),
                      torch.unique(target_t[1]))
        else:
            if self.proposal:
                proposal = np.array(Image.open(self.dataset[index][2]))
                targets = [target_torch.numpy(), proposal]
                augmented = self.transform(image=img, masks=targets, mask=target_torch.numpy())
                target_t = augmented['masks']
            else:
                augmented = self.transform(image=img, mask=target_torch.numpy())
                target_t = augmented['mask']
            img_t = augmented["image"]
            target_check = augmented['mask']

        if torch.count_nonzero(target_check != self.internal_masking_value) == 0:
            self.counter += 1
            if self.counter % 50 == 0:
                print(f"removed {self.counter} images due to wrong augmentations!")

        return img_t, target_t

    def __len__(self):
        return self.dataset.shape[0] // self.skip_step