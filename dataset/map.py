import os
import torch.utils.data as data
import torch
from PIL import Image
import numpy as np

from .task import task_get_label_transformation_lambda


class MapillarySegmentationIncremental(data.Dataset):

    def __init__(self, mode, transform=None, task=None, _root=None, _version=None, proposal=False, save_pred=None, skip_step=1):

        self.prep_label = True

        if mode == 'train':
            split = 'training'
            image_folder = os.path.join(_root, split, 'images')
        elif mode == 'val':
            split = 'training'
            image_folder = os.path.join(_root, split, 'images')
        elif save_pred is not None and ("demoVideo" in save_pred):
            split = 'test'
            self.prep_label = False
            mode = 'val'  # used for label translation
            image_folder = save_pred
        else:
            split = 'validation'
            mode = 'val' # used for label translation
            image_folder = os.path.join(_root, split, 'images')
        self.split = split

        dataset = sorted(os.listdir(image_folder))
        dataset = [os.path.join(image_folder, image) for image in dataset]
        if split != 'test':
            train = True if mode == 'train' else False
            dataset = task.filter_images(dataset, train=train)
        if self.prep_label:
            if proposal:
                self.dataset = np.asarray([(x, x.replace('.jpg', '.png').replace('images', f'{_version[7:-5]}/labels'), x.replace('images/', 'proposal_100/')) for x in dataset])
            else:
                self.dataset = np.asarray([(x, x.replace('.jpg', '.png').replace('images', f'{_version[7:-5]}/labels')) for x in dataset])
        else:
            self.dataset = np.asarray(dataset)
        self.transform = transform
        self.label_transform = task_get_label_transformation_lambda(task, mode)
        if mode != 'train':
            self.label_transform_train = task_get_label_transformation_lambda(task, 'train')
        else:
            self.label_transform_train = None
        self.internal_masking_value = task.internal_masking_value
        print("Init Map Dataset!")
        self.mode = mode
        self.proposal = proposal
        self.skip_step = skip_step

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
            if (torch.count_nonzero(target_torch != self.internal_masking_value) == 0) and (self.mode == "train"):
                print("ERROR, wrong dataset selected! Target doesn't contain image!")
                print(self.dataset[index][0])
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
                    target_check = augmented['mask']
                else:
                    augmented = self.transform(image=img, mask=target_torch.numpy())
                    target_t = augmented['mask']
                    target_check = target_t
                img_t = augmented["image"]

            if torch.count_nonzero(target_check != self.internal_masking_value) == 0:
                print("Albumentation approx. gone wrong!", self.internal_masking_value, torch.unique(target_check))
                
            return img_t, target_t
        else:

            img = np.array(Image.open(self.dataset[index]).convert('RGB'))
            if self.transform is not None:
                img = self.transform(image=img)["image"]
            return img, self.dataset[index].split('/')[-1]

    def __len__(self):
        return self.dataset.shape[0] // self.skip_step

