import numpy as np
import torch
import warnings

from datasets import get_dataset
from models.utils.continual_model import ContinualModel, optimizer_dict
from utils.args import (
    ArgumentParser,
    add_backbone_args,
    add_experiment_args,
    add_management_args,
    add_rehearsal_args,
    add_scheduler_args,
)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Finetuning baseline.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_backbone_args(parser)
    add_scheduler_args(parser)
    parser.add_argument(
        '--freeze-incremental-classifier',
        action='store_true',
        help=(
            'Freeze the classifier together with the shared backbone after '
            'D1, so that incremental domains train only their task-specific '
            'batch-normalization parameters. This option is ADIL-specific.'
        ),
    )
    parser.add_argument(
        '--incremental-bn-init',
        choices=('fresh', 'd1'),
        default='fresh',
        help=(
            'Initialization for a newly activated ADIL domain-specific BN: '
            'keep its fresh initialization, or copy the learned D1 affine '
            'parameters and running statistics. This option is ADIL-specific.'
        ),
    )
    return parser


class ADIL(ContinualModel):
    NAME = 'adil'

    def __init__(self, backbone, loss, args, transform):
        super(ADIL, self).__init__(backbone, loss, args, transform)
        self.current_task = 0
        dataset = get_dataset(args)
        self.cpt = dataset.N_CLASSES_PER_TASK

        self.to(self.device)

    def begin_task(self, cur_train_loader, next_train_loader):
        self.current_task += 1
        self.net.current_task = self.current_task - 1

        if self.current_task > 1:
            task = self.current_task - 1
            if getattr(self.args, 'incremental_bn_init', 'fresh') == 'd1':
                self._initialize_batch_norm_from_d1(task)
            self.net.freeze_other_share(task)
            if getattr(self.args, 'freeze_incremental_classifier', False):
                classifier = getattr(self.net, 'fc', None)
                if classifier is None:
                    raise AttributeError(
                        '--freeze-incremental-classifier requires the '
                        'backbone to expose its classifier as `fc`.'
                    )
                classifier.eval()
                for parameter in classifier.parameters():
                    parameter.requires_grad = False

    def _initialize_batch_norm_from_d1(self, task):
        """Warm-start a new domain's BN state from the learned D1 state."""
        if not hasattr(self.net, '_task_batch_norms'):
            raise AttributeError(
                '--incremental-bn-init=d1 requires the backbone to expose '
                '`_task_batch_norms(task)`.'
            )

        source_bns = list(self.net._task_batch_norms(0))
        target_bns = list(self.net._task_batch_norms(task))
        if len(source_bns) != len(target_bns):
            raise ValueError(
                'D1 and incremental domains expose different numbers of BN layers: '
                f'{len(source_bns)} vs {len(target_bns)}.'
            )

        for source, target in zip(source_bns, target_bns):
            target.load_state_dict(source.state_dict())

    def end_task(self, cur_train_loader, next_train_loader):
        # setup_buffer(self, cur_train_loader, next_train_loader)
        pass

    def observe(self, cur_data, next_data):
        inputs, labels, _, = cur_data

        self.opt.zero_grad()

        outputs = self.net(inputs, 'logits', self.current_task - 1)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()

    def save(self, save_path, scheduler=None):
        """Save all task-specific heads/BNs plus ADIL's active task."""
        torch.save({
            'format_version': 2,
            'algorithm': self.NAME,
            'net_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'current_task': self.current_task,
            'rng_state': self._rng_state(),
            'scheduler_state_dict': (
                scheduler.state_dict() if scheduler is not None else None
            ),
        }, save_path)

    def load(self, load_path, scheduler=None):
        checkpoint = torch.load(load_path, map_location='cpu')
        if isinstance(checkpoint, dict) and checkpoint.get('evaluation_only'):
            self._load_evaluation_checkpoint(checkpoint)
            return
        if 'net_state_dict' not in checkpoint:
            self.net.load_state_dict(checkpoint)
            return
        if checkpoint.get('algorithm') != self.NAME:
            raise ValueError(f"Cannot load {checkpoint.get('algorithm')} checkpoint into {self.NAME}.")
        net_state = checkpoint['net_state_dict']
        if checkpoint.get('evaluation_only', False):
            current_state = self.net.state_dict()
            compatible = {
                key: value for key, value in net_state.items()
                if key in current_state and current_state[key].shape == value.shape
            }
            skipped = sorted(set(net_state) - set(compatible))
            self.net.load_state_dict(compatible, strict=False)
            if skipped:
                warnings.warn(
                    'Skipped incompatible evaluation-only ADIL parameters: '
                    + ', '.join(skipped),
                    RuntimeWarning,
                )
        else:
            self.net.load_state_dict(net_state)
        self.current_task = checkpoint.get('current_task', 1)
        self.net.current_task = max(0, self.current_task - 1)
        if self.current_task > 1:
            self.net.freeze_other_share(self.net.current_task)
        self._load_optimizer_state(self.opt, checkpoint.get('optimizer_state_dict'))
        if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self._restore_rng_state(checkpoint.get('rng_state'))
