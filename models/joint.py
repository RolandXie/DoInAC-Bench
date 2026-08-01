# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import ipdb
import numpy as np
import torch
from datasets.utils.validation import ValidationDataset
from torch.optim import SGD
from torchvision import transforms

from datasets import get_dataset
from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_backbone_args, add_scheduler_args,ArgumentParser
from utils.status import progress_bar

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data import RandomSampler



def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Joint training: a strong, simple baseline.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_backbone_args(parser)
    add_scheduler_args(parser)
    return parser


class OversampledDataset(Dataset):
    """
    Repeat dataset to target length.
    """

    def __init__(self, dataset, target_size):
        self.dataset = dataset
        self.target_size = target_size

    def __len__(self):
        return self.target_size

    def __getitem__(self, idx):
        # repeat by random sampling
        real_idx = idx % len(self.dataset)
        return self.dataset[real_idx]


class Joint(ContinualModel):
    NAME = 'joint'
    current_task = 0

    def __init__(self, backbone, loss, args, transform):
        super(Joint, self).__init__(backbone, loss, args, transform)


    def begin_task(self, cur_train_loader, next_train_loader):
        self.current_task += 1

    def setup_joint_loader(self, dataset, args):
        """
        Build joint loader using current task and all previous tasks.
        """
        joint_train_loaders = []
        joint_test_loaders = []
        for i in range(dataset.N_TASKS):
            # cumulative train datasets
            trainsets = [
                dataset.train_loaders[j].dataset
                for j in range(i + 1)
            ]

            # 获取当前所有任务中最大的样本数量
            max_samples = max(
                len(ds) for ds in trainsets
            )

            balanced_trainsets = [
                OversampledDataset(ds, max_samples)
                if len(ds) < max_samples else ds
                for ds in trainsets
            ]

            joint_train_dataset = torch.utils.data.ConcatDataset(balanced_trainsets)
            joint_train_loader = DataLoader(
                joint_train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers
            )

            # cumulative test datasets
            testsets = [
                dataset.test_loaders[j].dataset
                for j in range(i + 1)
            ]
            joint_test_dataset = torch.utils.data.ConcatDataset(testsets)
            joint_test_loader = DataLoader(
                joint_test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers
            )
            joint_train_loaders.append(joint_train_loader)
            joint_test_loaders.append(joint_test_loader)

        dataset.train_loaders = joint_train_loaders
        dataset.test_loaders = joint_test_loaders


    def observe(self, cur_data, next_data):
        x, y, idx = cur_data
        x, y = x.to(self.device), y.to(self.device)
        self.opt.zero_grad()
        outputs = self.net(x)
        loss = self.loss(outputs, y)
        loss.backward()
        self.opt.step()

        return loss.item()