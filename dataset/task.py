import os
import pandas as pd
import numpy as np
import torch
import torchvision
from ast import literal_eval
from PIL import Image
import json


class Task:
    def __init__(self, file, step, bg_shift=True, incremental_level=-1,
                 debug=False, overlap=True, other_class=False, no_bg_class=False, hyp_hier=False, meta=None):
        self.adj_matrix = None
        self.masking_value = 255
        self.no_bg_class = no_bg_class
        self.hyp_hier = hyp_hier
        if other_class or self.no_bg_class:
            self.internal_masking_value = 255
            print(f"Other class is mapped to 255.")
        else:
            self.internal_masking_value = 0
            print(
                f"Other class is collapsed into background (0) which is trainable. Please verify that this behavior is desirable.")
        self.transform = {'train': {}, 'val': {}}
        self.inverted_order = {'train': {}, 'val': {}}
        self.background_shift = bg_shift
        self.step = step

        self.overlap = overlap
        self.debug = debug
        ov = "-ov" if overlap else ""
        self.filter_file = file + ov + ".npy"
        self.joined_training = True if incremental_level == 0 else False
        if self.joined_training and file:
            file = file + "_jn.csv"
        else:
            file += ".csv"

        self.meta = meta

        print(file)
        if os.path.exists(file):
            self.all_class_matrix = pd.read_csv(file)[["class", "gt_id", "start", "end"]]
            self.all_class_matrix.gt_id = self.all_class_matrix.gt_id.apply(lambda x: literal_eval(x))
            self.incr_tasks = self.all_class_matrix["start"].nunique() - 1
            if "city" in file or "synmedi" in file: # or "ignore" in self.all_class_matrix["class"]
                self.incr_tasks -= 1  # remove 255
            print(f"Loaded file {file}!")
            self.incremental_parent = True if (file.split('/')[-1][2]).isdigit() else False
        else:
            print("Creating mapper files should be pursued in create_csv.py!")
            exit()

        if self.no_bg_class:
            self.all_class_matrix.drop(0, inplace=True)
            self.all_class_matrix.reset_index(drop=True, inplace=True)
            print(
                f"Removed void class. Void and other class are mapped to masking value {self.internal_masking_value}.")

        if os.path.exists(self.filter_file):
            print(f"Loading previously saved indexes ({self.filter_file}).")
            self.filter_train = np.load(self.filter_file)
            if self.joined_training:
                    start = 0 if file.split('/')[-1][2] == "_" else 1
                    end = self.filter_train.shape[1] - 1
                    self.filter_train[:, 0] = np.sum(self.filter_train[:, start:end], axis=1, keepdims=True)[:, 0]
                    print(np.count_nonzero(self.filter_train[-1]))
                    print(f"Loading incremental images {start}-{end} for joined training!",
                          np.count_nonzero(self.filter_train))
        elif self.joined_training:
            print('Not using filter-file for JT training.')
        else:
            print(f"No dataset pre-filtered! Create with create_filter_file.py before training!")

    def classes(self, task_index):
        if task_index > self.incr_tasks:
            print("Task index larger than maximum!", task_index, self.incr_tasks)
            exit()
        incr_list_id = []
        for i in range(task_index + 1):
            list_task_i = list(self.all_class_matrix[self.all_class_matrix["start"] == i + 1].index)
            incr_list_id.append(list_task_i)
        return incr_list_id

    def sibling_ancestor_matrix(self, task_index):
        if task_index is None:
            task_index = self.step

        ids = self.classes(task_index)
        flatten = lambda l: sum(map(flatten, l), []) if isinstance(l, list) else [l]
        no_classes = len(flatten(ids))

        sibling_matrix = torch.zeros((no_classes, no_classes))
        ancestor_matrix = torch.zeros((no_classes, no_classes))

        limited_class_matrix = self.all_class_matrix[self.all_class_matrix.index < no_classes]

        if not self.hyp_hier:
            print("didn't define class hierarchy!")
            exit()

        for i, row in limited_class_matrix.iterrows():
            splits = row['class'].split('--')
            ancestors = ['--'.join(splits[0:i]) for i in range(1, len(splits))]
            if len(ancestors) == 0:
                sibling_matrix[i, i] = 1
                ancestor_matrix[i, i] = 1
                continue
            common_trunk = f"{ancestors[-1]}--"
            lvl = len(splits)
            sibling_matrix[i, list(set([i for i, v in enumerate(limited_class_matrix["class"]) if
                                        ((row["class"] == v) or (
                                                (common_trunk in str(v)) and (len(v.split('--')) == lvl)))]))] = 1
            ancestor_matrix[i, list(set([i for i, v in enumerate(limited_class_matrix["class"]) if
                                         ((row["class"] == v) or (str(v) in ancestors))]))] = 1

        non_leaf_classes = list(limited_class_matrix[limited_class_matrix["gt_id"] == {}].index)
        if self.incremental_parent:
            non_leaf_classes += list(limited_class_matrix[limited_class_matrix["end"] <= task_index + 1].index)
        return sibling_matrix, ancestor_matrix, non_leaf_classes

    def entail_matrix(self, task_index, level):
        if not task_index:
            task_index = self.step

        ids = self.classes(task_index)
        flatten = lambda l: sum(map(flatten, l), []) if isinstance(l, list) else [l]
        no_classes = len(flatten(ids))
        adj_dict = []

        limited_class_matrix = self.all_class_matrix[self.all_class_matrix.index < no_classes]

        if not self.hyp_hier:
            print("didn't define class hierarchy!")
            exit()

        for k, row in limited_class_matrix.iterrows():
            splits = row['class'].split('--')
            parent = '--'.join(splits[0:len(splits)-1])
            parent_id = [i for i, v in enumerate(limited_class_matrix["class"]) if (v == parent)]
            if len(parent_id) == 0:
                continue
            parent_id = parent_id[0]
            while len(adj_dict) < len(splits):
                adj_dict.append({})
            if not parent_id in adj_dict[len(splits)-1]:
                adj_dict[len(splits)-1][parent_id] = []
            adj_dict[len(splits)-1][parent_id].append(k)


        if level != -1:
            return adj_dict[level]
        return adj_dict

    def class_list(self):
        return list(self.all_class_matrix["class"])

    def gt_id_list(self):
        return list(self.all_class_matrix["gt_id"])

    def get_label_transformation(self, mode, task_index=None):
        if not task_index:
            task_index = self.step
        task_index += 1
        if task_index in self.transform:
            return self.transform[mode][task_index], self.inverted_order[mode][task_index]
        elif self.background_shift and mode == 'train':
            inverted_order = {}
            task_classes = self.all_class_matrix[self.all_class_matrix["start"] == task_index]["gt_id"]
            for index, gt_ids in task_classes.items():
                for gt_class in gt_ids:
                    if gt_class in inverted_order:
                        print("ERROR: mapping to two parents, multi-hierarchy is incorrect! Exit recommended!", gt_class)
                    inverted_order[gt_class] = index
            task_classes = self.all_class_matrix[self.all_class_matrix["start"] == 0]["gt_id"]
            for index, gt_ids in task_classes.items():
                for gt_class in gt_ids:
                    inverted_order[gt_class] = self.masking_value
        elif (task_index > 0) and ((task_index - 1) in self.inverted_order[mode]):
            inverted_order = self.inverted_order[mode][task_index - 1]
            task_classes = self.all_class_matrix[self.all_class_matrix["start"] == task_index]["gt_id"]
            for index, gt_ids in task_classes.items():
                for gt_class in gt_ids:
                    inverted_order[gt_class] = index
        else:
            print(f"Building label transformation from scratch for {task_index}, {mode}.")
            inverted_order = {}
            task_classes = self.all_class_matrix[(self.all_class_matrix["start"] <= task_index) &
                                                 (self.all_class_matrix["end"] > task_index)][
                ["gt_id", "start"]]

            for index, row in task_classes.iterrows():
                if row["start"] == 0:
                    index = 255
                for gt_class in row["gt_id"]:
                    if gt_class in inverted_order:
                        id = inverted_order[gt_class]
                        task_classes_element_start = task_classes[task_classes.index == id]["start"]
                        if int(row["start"]) > int(task_classes_element_start):
                            inverted_order[gt_class] = index
                    else:
                        inverted_order[gt_class] = index
        if mode == 'train':
            masking_value = self.internal_masking_value
        else:
            masking_value = self.masking_value
        transform = torchvision.transforms.Lambda(Transform(inverted_order, masking_value))
        self.transform[mode][task_index] = transform
        self.inverted_order[mode][task_index] = inverted_order
        print(f"mode {mode}-masked:{masking_value}", inverted_order)
        return transform, inverted_order

    def filter_images(self, dataset, train=True):
        if train and self.filter_train is not None:
            print(len(dataset), len(self.filter_train))
            assert len(dataset) == len(self.filter_train) and len(self.filter_train) > 0
            indices = np.where(self.filter_train[:, self.step])[0]
            return list(np.asarray(dataset)[indices])
        elif not train and self.filter_train is not None:
            if len(dataset) != len(self.filter_train):
                return dataset
            print("Using held-out train set for validation!")
            indices = np.where(self.filter_train[:, -1])[0]
            return list(np.asarray(dataset)[indices])
        elif self.joined_training:  # skipped if condition above is filled
            return dataset
        else:
            print('exit case, l.356 task.py')

def task_get_label_transformation_lambda(task, mode):
    return task.get_label_transformation(mode)[0]


class Transform:
    def __init__(self, inverted_order, internal_masking_value):
        self.inverted_order = inverted_order
        self.internal_masking_value = internal_masking_value

    def __call__(self, item):
        return item.apply_(self.helper_fct)

    def helper_fct(self, item):
        return self.inverted_order.get(item, self.internal_masking_value)
