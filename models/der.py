import torch
import numpy as np

import torch.nn.functional as F

from datasets import get_dataset
from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, add_backbone_args,add_scheduler_args, ArgumentParser
from utils.buffer import Buffer

from models.er import setup_buffer

import ipdb


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_backbone_args(parser)
    add_scheduler_args(parser)

    parser.add_argument('--alpha', type=float, default=0.5,
                        help='weight of the logits l2-loss.')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='weight of the memory replay.')
    return parser


class DER(ContinualModel):
    """DER++ model with slight modification applied to the logits mse loss."""
    NAME = 'der'

    def __init__(self, backbone, loss, args, transform):
        super(DER, self).__init__(backbone, loss, args, transform)
        # self.buffer = Buffer(self.args.buffer_size, self.device)
        self.current_task = 0

        dataset = get_dataset(args)
        self.cpt = dataset.N_CLASSES_PER_TASK
        
        # set up the memory buffer 
        self.memory = Buffer(
            buffer_size=args.buffer_size, 
            device=self.device, 
            input_size=dataset.INDIM, 
            num_classes=dataset.N_CLASSES_PER_TASK,
            batch_size=args.buffer_batch_size,
            domain_buffers=None
        ).to(self.device)
        
        self.alpha = args.alpha
        self.beta = args.beta

        self.to(self.device)

    def begin_task(self, cur_train_loader, next_train_loader):
        self.current_task += 1

        if self.current_task > 1 and not self.memory.is_empty():
            self.memory = iter(self.memory)

    def end_task(self, cur_train_loader, next_train_loader):
        setup_buffer(self, cur_train_loader, next_train_loader)

    def get_past_data(self):
        # get past data, past pseudo-labels, past domain ids.
        try: 
            past_data = next(self.memory)
        except:
            self.memory = iter(self.memory)
            past_data = next(self.memory)
        return past_data

    def observe(self, cur_data, next_data):
        inputs, labels, _, = cur_data
        
        bs1 = inputs.shape[0]
        
        has_memory = self.current_task > 1 and not self.memory.is_empty()
        if has_memory:
            buf_inputs, buf_labels, buf_logits, _ = self.get_past_data()
            bs2 = buf_inputs.shape[0]
            inputs = torch.cat((inputs, buf_inputs))

        logits = self.net(inputs)

        if has_memory:
            logits, logits_past = torch.split(logits, [bs1, bs2])

        # normal cross-entropy loss at current domain.
        loss = self.loss(logits, labels)
        assert not torch.isnan(loss)

        if has_memory:
            loss_replay = self.beta * self.loss(logits_past, buf_labels)
            loss += loss_replay
            # since we only stored the predictions instead of direct logits, need to log back.
            loss_mse = self.alpha * F.mse_loss(logits_past, buf_logits)
            loss += loss_mse
            assert not torch.isnan(loss_replay) and not torch.isnan(loss_mse)
            # ipdb.set_trace()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()

    def save(self, save_path, scheduler=None):
        """Save DER++ state, including stored examples and historical logits."""
        torch.save({
            'format_version': 2,
            'algorithm': self.NAME,
            'net_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'current_task': self.current_task,
            'memory_state_dict': self.memory.state_dict(),
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
        self.net.load_state_dict(checkpoint['net_state_dict'])
        self.current_task = checkpoint['current_task']
        self.memory.load_state_dict(checkpoint['memory_state_dict']).to(self.device)
        self._load_optimizer_state(self.opt, checkpoint.get('optimizer_state_dict'))
        if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self._restore_rng_state(checkpoint.get('rng_state'))
