# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import sys
import warnings
from argparse import Namespace
from contextlib import suppress
from typing import List

import torch
import torch.nn as nn
import numpy as np
from torch.optim import SGD, Adam

from utils.conf import get_device
from utils.magic import persistent_locals

with suppress(ImportError):
    import wandb

optimizer_dict = {
    'sgd': SGD,
    'adam': Adam
}

class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME: str

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                 args: Namespace, transform: nn.Module) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.opt = optimizer_dict[args.opt](self.net.parameters(), lr=self.args.lr) # opt created. 
        self.device = get_device()

        if not self.NAME:
            raise NotImplementedError('Please specify the name and the compatibility of the model.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    def meta_observe(self, *args, **kwargs):
        if 'wandb' in sys.modules and not self.args.nowand:
            pl = persistent_locals(self.observe)
            ret = pl(*args, **kwargs)
            self.autolog_wandb(pl.locals)
        else:
            ret = self.observe(*args, **kwargs)
        return ret

    def observe(self, cur_data, next_data) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        raise NotImplementedError

    def autolog_wandb(self, locals):
        """
        All variables starting with "_wandb_" or "loss" in the observe function
        are automatically logged to wandb upon return if wandb is installed.
        """
        if not self.args.nowand and not self.args.debug_mode:
            wandb.log({k: (v.item() if isinstance(v, torch.Tensor) and v.dim() == 0 else v)
                      for k, v in locals.items() if k.startswith('_wandb_') or k.startswith('loss')})

    def save(self, save_path, scheduler=None):
        """
        Checkpoint the model parameters on the local file system. 
        """
        torch.save(self.net.state_dict(), save_path)

    def load(self, load_path, scheduler=None):
        """
        Load the model parameters on the local file system. 
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        if isinstance(checkpoint, dict) and checkpoint.get('evaluation_only'):
            self._load_evaluation_checkpoint(checkpoint)
            return
        self.net.load_state_dict(checkpoint)

    def _load_evaluation_checkpoint(self, checkpoint):
        """Load only compatible network weights from an evaluation checkpoint."""
        state = checkpoint['net_state_dict']
        current = self.net.state_dict()
        compatible = {
            name: value for name, value in state.items()
            if name in current
            and torch.is_tensor(value)
            and current[name].shape == value.shape
        }
        skipped = sorted(set(state) - set(compatible))
        self.net.load_state_dict(compatible, strict=False)
        if 'current_task' in checkpoint and hasattr(self, 'current_task'):
            self.current_task = checkpoint['current_task']
        if hasattr(self.net, 'current_task') and 'current_task' in checkpoint:
            self.net.current_task = max(0, checkpoint['current_task'] - 1)
        if skipped:
            warnings.warn(
                'Skipped incompatible evaluation-only parameters: '
                + ', '.join(skipped),
                RuntimeWarning,
            )
        return True

    def _load_optimizer_state(self, optimizer, state_dict):
        """Restore optimizer tensors without moving Adam's step counter to CUDA."""
        if state_dict is None:
            return

        optimizer.load_state_dict(state_dict)
        for parameter, state in optimizer.state.items():
            for name, value in state.items():
                if not torch.is_tensor(value):
                    continue
                state[name] = value.cpu() if name == 'step' else value.to(parameter.device)

    @staticmethod
    def _rng_state():
        return {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    @staticmethod
    def _restore_rng_state(state):
        if state is None:
            return
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
        if state.get('cuda') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['cuda'])
