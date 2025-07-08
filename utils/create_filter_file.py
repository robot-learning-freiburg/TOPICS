import json
import os
import pandas as pd
import random
import numpy as np
from multiprocessing.pool import ThreadPool
import multiprocessing
from functools import partial
from ast import literal_eval
from PIL import Image
import glob

from dataset.task import Task
import dataset.dataset_meta as datasetmeta

def filter_dataset(task_file, dataset, base_images, max_class_images, meta):
    task = Task(file=task_file, step=0, meta=meta, other_class=True)
    ov = "-ov" if task.overlap else ""
    filter_file = task_file + ov + ".npy"
    if os.path.exists(filter_file):
        print("Filter file exists!")
        exit()

    val_ids = None
    # check for validation ids (these images will be excluded from the training set)
    # automatically created for cityscapes and mapillary datasets
    if ("map" in task_file) or ("city" in task_file):
        val_id_file = f"{'/'.join(filter_file.split('/')[:-1])}/map_val_ids"
        if not os.path.exists(val_id_file):
            validation_held_out = int(len(dataset) * 0.2)
            val_ids = random.sample(range(0, len(dataset)), validation_held_out)
            with open(val_id_file, "w") as fp:
                json.dump(val_ids, fp)
                print("computed val ids from scratch!")
        else:
            with open(val_id_file, "r") as fp:
                val_ids = json.load(fp)

    print(f'Computing for {task.incr_tasks} with {base_images}.')

    filter_train = filter_images(task, dataset, val_ids, task.incr_tasks, base_images, max_class_images, task.overlap)
    np.save(filter_file, filter_train)
    print('images per column', np.sum(filter_train, axis=0))
        
def filter_images(task, paths, val_ids, incr_tasks, base_images, max_class_images, overlap, internal_masking_value=255):
        """
        
        
        Filter images for incremental tasks based on the task configuration.
            
        :param task: Task object containing task configuration.
        :param paths: List of paths to images.
        :param val_ids: List of validation image indices to be excluded from training.
        :param incr_tasks: Number of incremental tasks.
        :param base_images: Fraction of images to be used for the base task.
        :param max_class_images: Maximum number of images per class for incremental tasks.
        :param overlap: Boolean indicating whether to allow overlapped setting.
        :param internal_masking_value: Value used for internal masking (default is 255).
        :return: A numpy array indicating which images are selected for each task.
        """

        t = np.zeros((len(paths), incr_tasks + 2))

        # remove validation ids from training set
        if val_ids is not None:
            validation_held_out = len(val_ids)
            t[val_ids, -1] = 1
            for i in val_ids:
                paths[i] = {}
        else:
            validation_held_out = 0

        valid_paths = len(paths) - validation_held_out
        print(f"Valid Paths {valid_paths}, val {validation_held_out}, all {len(paths)}")
        # define number of base and incremental images
        if base_images > 0:
            max_base_images = int(base_images * valid_paths)
            max_incr_images = int(valid_paths * (1 - base_images)) // incr_tasks
        else:
            max_base_images, max_incr_images = -1, -1

        inc_class_id_list = task.classes(incr_tasks)

        # iterate over all tasks in reverse order
        for task_id in range(incr_tasks, -1, -1):
            if task_id == 0:
                max_class_images_incr = -1
                max_task_images = max_base_images
            else:
                max_class_images_incr = max_class_images
                max_task_images = max_incr_images

            labels = inc_class_id_list[task_id]
            all_labels = [xs for x in inc_class_id_list[0:task_id + 1] for xs in x] + [0, 255]
            img_class_counter = {}
            img_count = 0
            print(task_id, labels)

            indexes_to_classes = []
            _, inverted_order = task.get_label_transformation(mode='train', task_index=task_id)
            print(inverted_order)
            find_classes = partial(_find_classes,
                                   transform=lambda v: inverted_order.get(v, internal_masking_value))
            # check occurances of classes in images in parallel
            with ThreadPool(min(8, multiprocessing.cpu_count())) as pool:
                for i, classes in enumerate(pool.imap(find_classes, paths), start=1):
                    indexes_to_classes.append(classes)

            for index, classes in enumerate(indexes_to_classes):
                # skip as image used for other task / not valid labels
                if len(classes) == 0:
                    continue
                
                if np.count_nonzero(t[index, :]) > 0 and (max_task_images > 0):
                    print("Using images in multiple increments is disabled. ERROR!")
                    continue
                overlap_labels = [c for c in classes if c in labels]
                if len(overlap_labels) > 0:
                    # check if image contains classes of this task
                    if overlap or (not overlap and all(c in all_labels for c in classes)):
                        t[index, task_id] = 1
                        img_count += 1
                        # mark image as invalid for further tasks
                        if max_task_images > 0:
                            paths[index] = {}
                        if max_class_images_incr > 0:
                            for cl in overlap_labels:
                                img_class_counter[cl] = img_class_counter.get(cl, 0) + 1
                                if img_class_counter[cl] > max_class_images_incr:
                                    labels.remove(cl)
                if ((img_count >= max_task_images) and (max_task_images > 0)) or (len(labels) == 0):
                    break
            print(f"{np.count_nonzero(t[:, task_id])} images selected for {task_id}.")

        return t

def _find_classes(path, transform=None) -> set:
    """Open a ground-truth segmentation map image and returns all unique classes
    contained.

    :param path: Path to the image.
    :return: Unique classes.
    """
    if path:
        if transform:
            return set(np.vectorize(transform)(np.unique(np.array(Image.open(path)).reshape(-1))))
        return set(np.unique(np.array(Image.open(path)).reshape(-1)))
    return {}

if __name__ == '__main__':
    # set to select challenge
    challenge = "mmor"
    _root='/home/shared/MM-OR_processed'

    if challenge == "map":
        _version = "config_v2.0.json"
        image_folder = os.path.join(_root, 'training', 'images')
        dataset = sorted(os.listdir(image_folder))
        dataset = [os.path.join(image_folder, image) for image in dataset]
        labels = [x.replace('.jpg', '.png').replace('images', f'{_version[7:-5]}/labels') for x in dataset]
        task_file = 'configs/map/task_files/l2_increment_5tasks'
        meta = datasetmeta.get_mapillary_vistas_meta(_root, _version)
    # voc
    elif challenge == "voc":
        _version = "pascal_hier.txt"
        if not os.path.exists(_root):
            _root = "/home/hindel/Documents/032_HCIL/data/PascalVOC12"
        label_folder = os.path.join(_root, 'SegmentationClassAug')
        with open(os.path.join(_root, "ImageSets/train_aug.txt")) as f:
            dataset = [x[:-1].split(' ') for x in f.readlines()]
        task_file = 'configs/voc/task_files/l2_increment_5tasks'
        meta = datasetmeta.pascal_voc_meta(_root, _version)

    elif challenge == "city":
        _version = "city_labels.json"
        annotation_folder = os.path.join(_root, 'gtFine')
        image_folder = os.path.join(_root, 'leftImg8bit')
        labels = [
            os.path.join(
                annotation_folder,
                "train",
                path.split("/")[-2],
                path.split("/")[-1][:-15] + "gtFine_labelIds.png"
            ) for path in sorted(glob.glob(os.path.join(image_folder, "train/*/*.png")))
        ]
        task_file = 'configs/city/task_files/l2_increment_5tasks'
        meta = datasetmeta.city_meta(_root, _version)
    elif challenge == "mmor":
        import re
        def get_camera_id(path):
            match = re.search(r"camera(\d+)", path)
            return int(match.group(1)) if match else -1 
        
        _version='ImageSets/meta.json'
        train_config = "ImageSets/train_200_3_azure.json"
        with open(os.path.join(_root, train_config)) as f:
            dataset = json.load(f)
        print('length of dataset', len(dataset))
        labels = [x.replace('.jpg', '.png').replace('colorimage/', f'segmentation_export_{get_camera_id(x)}/') for x in dataset]
        task_file = 'configs/mmor/task_files/l24_increment_4tasks'
        meta = datasetmeta.mmor_meta(_root, _version)

    elif challenge == "synmedi":
        _version='semantic_class_mapping.json'
        image_folder = os.path.join(_root, 'img', 'train')
        dataset = sorted(os.listdir(image_folder))
        dataset = [os.path.join(image_folder, image) for image in dataset]
        labels = [x.replace('.jpg', '.png').replace('img', f'semantic') for x in dataset]
        task_file = 'configs/synmedi/task_files/l24_increment_6tasks'
        meta = datasetmeta.synmedi_meta(_root, _version)

    filter_dataset(task_file, labels, base_images=0.5, max_class_images=-1, meta=meta)
