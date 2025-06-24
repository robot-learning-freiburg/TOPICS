import numpy as np
import random
import os
import cv2
import wandb

from torchvision.transforms.functional import normalize



class Denormalize:
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean / std
        self._std = 1 / std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1, 1, 1)) / self._std.reshape(-1, 1, 1)
        return normalize(tensor, self._mean, self._std)


class Label2Color:
    def __init__(self, dataset, task):
        self.cmap = {}
        if dataset == 'map' or dataset == 'city' or dataset == 'synmedi':
            self.cmap = {255: [0, 0, 0]}
        elif dataset == 'voc' or dataset == 'mmor':
            self.cmap = {255: [255, 255, 255]}
        else:
            print("CMAP not defined!")
        org_colors = task.meta["stuff_colors"]
        task_classes = task.gt_id_list()  # gt_id
        for id, cl in enumerate(task_classes):
            if len(cl) == 0:
                continue
            gt_id = random.sample(cl, 1)[0]
            if gt_id == 255 and dataset == 'synmedi':
                self.cmap[0] = [255, 255, 255]
                continue
            self.cmap[id] = org_colors[gt_id]

    def __call__(self, lbls):
        colored_img = np.zeros((*lbls.shape, 3))
        for k, v in self.cmap.items():
            colored_img[lbls == int(k)] = v
        return colored_img


class PredictionVis:
    def __init__(self, logger, dataset, task, type, opts):
        self.logger = logger
        self.label2color = Label2Color(dataset, task)  # convert labels to images
        self.denorm = Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.logdir = opts.logdir_full
        self.save_pred = opts.save_pred
        self.logdir_full = opts.logdir_full

        if self.save_pred is not None:
            self.dataset = self.save_pred.split("/")[-1]
            if os.path.isdir(f"{self.logdir_full}/{self.dataset}_gt_rgb"):
                print(
                    f"Pred samples already exist! Avoiding overwriting! {self.logdir_full}/vis/gt_mod")
                exit()
            os.makedirs(f"{self.logdir_full}/vis/{self.dataset}_gt_rgb")
            os.makedirs(f"{self.logdir_full}/vis/{self.dataset}_pred_rgb")
            os.makedirs(f"{self.logdir_full}/vis/{self.dataset}_pred")
            os.makedirs(f"{self.logdir_full}/vis/{self.dataset}_gt")
            os.makedirs(f"{self.logdir_full}/vis/{self.dataset}_img")

    def __call__(self, images, labels, pseudo_label, prediction, outputs, i, file_names=None):

        if self.save_pred is not None:
            for j in range(images.shape[0]):
                if not isinstance(file_names, list):
                    file_name = f"sample_{i}_{j}.png"
                else:
                    file_name = file_names[j]
                if labels is not None:
                    cv2.imwrite(f"{self.logdir_full}/vis/{self.dataset}_gt/{file_name}", labels[j].astype(np.uint8))
                    target = self.label2color(labels[j]).astype(np.uint8)
                    cv2.imwrite(f"{self.logdir_full}/vis/{self.dataset}_gt_rgb/{file_name}", target[..., ::-1])
                    img = (self.denorm(images[j].detach().cpu().numpy()) * 255).astype(np.uint8).transpose(1, 2, 0)
                    cv2.imwrite(f"{self.logdir_full}/vis/{self.dataset}_img/{file_name}", img[..., ::-1])
                lbl = self.label2color(prediction[j]).astype(np.uint8)
                cv2.imwrite(f"{self.logdir_full}/vis/{self.dataset}_pred/{file_name}", prediction[j].astype(np.uint8))
                cv2.imwrite(f"{self.logdir_full}/vis/{self.dataset}_pred_rgb/{file_name}", lbl[..., ::-1])
            return None, None
        else:
            j = 0
            img = (self.denorm(images[j].detach().cpu().numpy()) * 255).astype(np.uint8)
            target = self.label2color(labels[j]).transpose(2, 0, 1).astype(np.uint8)
            target2 = self.label2color(pseudo_label[j]).transpose(2, 0, 1).astype(np.uint8)
            lbl = self.label2color(prediction[j]).transpose(2, 0, 1).astype(np.uint8)
            concat_img = np.concatenate((img, target, target2, lbl), axis=2)  # concat along width
            self.logger.add_image(f'Sample_{i}_{j}', concat_img)
            img = wandb.Image(concat_img.transpose(1, 2, 0), caption=f'Sample_{i}_{j}')
            return img

