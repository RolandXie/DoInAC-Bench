# Task-IL/Domain-IL Learning without Forgetting, 

import torch
from datasets import get_dataset
from torch.optim import SGD

from models.utils.continual_model import ContinualModel

from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, add_backbone_args,add_scheduler_args, ArgumentParser


## Model Specific Argument Parsing
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='A simple baseline LwF.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_backbone_args(parser)
    add_scheduler_args(parser)
    # for model-specific arguments.
    parser.add_argument('--alpha', type=float, default=1.,
                        help='base weight of the kd loss')
    parser.add_argument('--scale', type=str, default='linear', choices=['linear', 'const'],
                        help='scale with the task number.')
    return parser


def soft_ce(logits_p, logits_t, temp, reduction=True):
    res = -(torch.softmax(logits_t/temp, 1) * torch.log_softmax(logits_p/temp, 1)).sum(1)
    if reduction:
        return torch.mean(res)
    return res


class Lwf(ContinualModel):
    """Learning without Forgetting for DIL&TIL, without Rehearsal (Memory Bank)."""
    NAME = 'lwf-cl'

    def __init__(self, backbone, loss, args, transform):
        super(Lwf, self).__init__(backbone, loss, args, transform) # optimizer self.opt created here
        self.old_net = None
        self.soft = torch.nn.Softmax(dim=1)
        self.logsoft = torch.nn.LogSoftmax(dim=1)
        self.current_task = 0
        self.cpt = get_dataset(args).N_CLASSES_PER_TASK

        # lwf-specific parameters 
        self.scale = args.scale
        self.alpha = args.alpha

        # the buffer is registered using the maximum length of the dataset
        # TODO: might cause too much memory in the GPU.
        self.register_buffer("logits", torch.randn(get_dataset(args).MAX_N_SAMPLES_PER_TASK, self.cpt))
        
        # put self to the correct device.
        self.to(self.device)

    def begin_task(self, cur_train_loader, next_train_loader):
        self.net.eval()
        if self.current_task > 0:
            with torch.no_grad():
                for _, (x, _, idx) in enumerate(cur_train_loader):
                    # set the logits produced by the previous model
                    self.logits[idx] = self.net(x.to(self.device))
        self.net.train()

        # reset the parameters, this is usually bad.
        # self.net.reset_parameters()

        self.current_task += 1

    def observe(self, cur_data, next_data):
        cur_x, cur_y, cur_idx = cur_data
        next_x, next_y, next_idx = next_data # lwf-cl doesn't use the next domain's data.

        self.opt.zero_grad()
        outputs = self.net(cur_x)
        loss_ce = self.loss(outputs, cur_y)

        loss_kd = 0.
        if self.current_task > 1:
            loss_kd = soft_ce(logits_p=outputs, logits_t=self.logits[cur_idx, :].to(self.device), temp=2, reduction=True)
        
        # naive way of preserving the past knowledge.
        loss = self.combine_loss(loss_ce=loss_ce, loss_kd=loss_kd)

        loss.backward()
        self.opt.step()

        return loss.item()

    def combine_loss(self, loss_ce, loss_kd):
        if self.scale == 'const':
            return loss_ce + self.alpha * loss_kd
        elif self.scale == 'linear':
            return loss_ce + self.alpha * (self.current_task-1) * loss_kd
            
        raise NotImplementedError('Not supported combination of the Cross Entropy and Knowledge Distillation.')

    def save(self, save_path, scheduler=None):
        """Save LwF's network and the teacher logits used by the next stage."""
        torch.save({
            'format_version': 2,
            'algorithm': self.NAME,
            'net_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'current_task': self.current_task,
            'logits': self.logits.detach().cpu(),
            'rng_state': self._rng_state(),
            'scheduler_state_dict': (
                scheduler.state_dict() if scheduler is not None else None
            ),
        }, save_path)

    def load(self, load_path, scheduler=None):
        checkpoint = torch.load(load_path, map_location='cpu')
        if isinstance(checkpoint, dict) and checkpoint.get('evaluation_only'):
            self._load_evaluation_checkpoint(checkpoint)
            if checkpoint.get('logits') is not None:
                self.logits.copy_(checkpoint['logits'].to(self.logits.device))
            return
        if 'net_state_dict' not in checkpoint:
            self.net.load_state_dict(checkpoint)
            return
        if checkpoint.get('algorithm') != self.NAME:
            raise ValueError(f"Cannot load {checkpoint.get('algorithm')} checkpoint into {self.NAME}.")
        self.net.load_state_dict(checkpoint['net_state_dict'])
        self.current_task = checkpoint['current_task']
        self.logits.copy_(checkpoint['logits'].to(self.logits.device))
        self._load_optimizer_state(self.opt, checkpoint.get('optimizer_state_dict'))
        if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self._restore_rng_state(checkpoint.get('rng_state'))
