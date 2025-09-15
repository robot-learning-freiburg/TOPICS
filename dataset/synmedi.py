import os
import torch.utils.data as data
import torch
from PIL import Image
import numpy as np

from .task import task_get_label_transformation_lambda

class SynMediIncremental(data.Dataset):

    def __init__(self, mode, transform=None, task=None, _root=None, _version=None, proposal=False, save_pred=None, skip_step=1):

        self.prep_label = True
        image_folder = os.path.join(_root, 'img', mode)
        
        self.split = mode

        dataset = sorted(os.listdir(image_folder))
        dataset = [os.path.join(image_folder, image) for image in dataset]

        if self.split == 'train':
            dataset = task.filter_images(dataset, train=mode)

        if self.prep_label:
            if proposal:
                self.dataset = np.asarray([(x, x.replace('.jpg', '.png').replace('img', 'semantic'), x.replace('img/', 'proposal_100/')) for x in dataset])
            else:
                self.dataset = np.asarray([(x, x.replace('.jpg', '.png').replace('img', 'semantic')) for x in dataset])
        else:
            self.dataset = np.asarray(dataset)
        self.transform = transform
        mode = "train" if mode == 'train' else "val"
        self.label_transform = task_get_label_transformation_lambda(task, mode)
        if mode != 'train':
            self.label_transform_train = task_get_label_transformation_lambda(task, 'train')
        else:
            self.label_transform_train = None
        self.internal_masking_value = task.internal_masking_value
        print("Init SynMedi Dataset!")
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
            # print(img.shape, target.shape)
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


