import glob
import json
import os

import albumentations as A
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from albumentations.pytorch import ToTensorV2

from .task import task_get_label_transformation_lambda


class EndovisIncremental(data.Dataset):

    def __init__(self, mode, transform=None, task=None, _root=None, _version=None, proposal=False, save_pred=None, skip_step=1):
        self.prep_label = True

        if mode == 'train' or mode == 'val': #validation data defined in Task in npy file last column
            split = 'train'
        else:
            split = 'test'
            mode = 'val' 

        self.split = split
        self.mode = mode
        self.skip_step = skip_step
        self.proposal = proposal
        self.counter = 0
        self.internal_masking_value = task.internal_masking_value

        dataset = self.get_endovis18_dataset(_root, task, proposal)
        self.dataset = np.asarray(dataset)
        self.transform = transform
        self.label_transform = task_get_label_transformation_lambda(task, mode)
        if mode != 'train':
            self.label_transform_train = task_get_label_transformation_lambda(task, 'train')
        else:
            self.label_transform_train = None
        print("Init Medical Dataset!")


    def get_endovis18_dataset(self, root_dir, task, proposal):
        if self.split == 'train':
            all_folders = [f"miccai_challenge_release_{i}" for i in range(1, 5)]

            dataset = []
            for folder in all_folders:
                folder_path = os.path.join(root_dir, folder)
                sequence_paths = sorted(glob.glob(os.path.join(folder_path, "*")))

                for sequence in sequence_paths:
                    images = sorted(glob.glob(os.path.join(sequence, "left_frames", "*.png")))
                    for img_path in images:
                        img_name = os.path.basename(img_path)
                        label_path = os.path.join(sequence, "labels_class", img_name)
                        if proposal:
                            proposal_path = label_path.replace('labels_class', 'proposal_100')
                            label_path = (label_path, proposal_path)

                        dataset.append({
                            'image': img_path,
                            'label': label_path
                        })
            train = True if self.mode == 'train' else False
            dataset = task.filter_images(dataset, train=train)

        elif self.split == 'test':
            test_path = os.path.join(root_dir, "test_data")
        
            dataset = []
            sequence_paths = sorted(glob.glob(os.path.join(test_path, "*")))
            for sequence in sequence_paths:
                images = sorted(glob.glob(os.path.join(sequence, "left_frames", "*.png")))
                for img_path in images:
                    img_name = os.path.basename(img_path)
                    label_path = os.path.join(sequence, "labels_class", img_name)
                    dataset.append({
                        'image': img_path,
                        'label': label_path
                    })
        else:
            raise ValueError(f"Unsupported split: {self.split}")

        return dataset
    
    def __len__(self):
        return self.dataset.shape[0] // self.skip_step

    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        index *= self.skip_step
        img = np.array(Image.open(self.dataset[index]['image']).convert('RGB'))
        if self.proposal and self.split != 'test':
            target = np.array(Image.open(self.dataset[index]['label'][0]))
        else:
            target = np.array(Image.open(self.dataset[index]['label']))
        target_torch = self.label_transform(torch.tensor(target))


        if self.label_transform_train is not None:
            target_torch_train = self.label_transform_train(torch.tensor(target))
            targets = [target_torch_train.numpy(), target_torch.numpy()]
            if self.proposal:
                proposal = np.array(Image.open(self.dataset[index]['label'][1]))
                targets.append(proposal)
            augmented = self.transform(image=img, masks=targets, mask=target_torch.numpy())
            img_t = augmented["image"]
            target_t = augmented['masks']
            target_check = augmented['mask']

        else:
            if self.proposal:
                proposal = np.array(Image.open(self.dataset[index]['label'][1]))
                targets = [target_torch.numpy(), proposal]
                augmented = self.transform(image=img, masks=targets, mask=target_torch.numpy())
                target_t = augmented['masks']
            else:
                augmented = self.transform(image=img, mask=target_torch.numpy())
                target_t = augmented['mask']
            img_t = augmented["image"]
            target_check = augmented['mask']

        return img_t, target_t

