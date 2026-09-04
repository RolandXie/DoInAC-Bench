import torch
import numpy as np

from datasets import get_dataset
from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, add_backbone_args, add_scheduler_args, ArgumentParser
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_backbone_args(parser)
    add_scheduler_args(parser)
    return parser

############ Important #############
### shared across the rehearsal-based methods.
############ 
def setup_buffer(self, cur_train_loader, next_train_loader):
    """
    Select the examples from the training set and update the memory.
    Note: self must have attribute 'memory' to represent the replay buffer.
    """
    if self.memory.buffer_size == 0:
        return

    # by default.
    sample_num = self.memory.buffer_size // self.current_task
    samples_per_class = sample_num // self.cpt
    
    examples = [[] for _ in range(self.cpt)]
    labels = [[] for _ in range(self.cpt)]
    preds = [[] for _ in range(self.cpt)]
    sizes = np.zeros((self.cpt, ))

    self.net.eval() # this network is trained on the cur_train_loader.
    with torch.no_grad():
        for _, (xs, ys, __) in enumerate(cur_train_loader):
            xs, ys = xs.to(self.device), ys.to(self.device)
            pred = self.net(xs) # logits stored in the memory.
            for label in range(self.cpt):
                examples[label].append(xs[ys==label])
                labels[label].append(ys[ys==label])
                preds[label].append(pred[ys==label])
                sizes[label] += xs[ys==label].shape[0]
            # no need to iterate through the whole dataset.
            if all(sizes > samples_per_class): 
                break
        # stack the preds 
        examples = torch.vstack([torch.vstack(m)[:samples_per_class] for m in examples])
        labels = torch.hstack([torch.hstack(m)[:samples_per_class] for m in labels])
        preds = torch.vstack([torch.vstack(m)[:samples_per_class] for m in preds])
        
        assert examples.shape[0] == labels.shape[0] == preds.shape[0], 'Mismatched shape of the added data.'
    self.net.train()

    # update the memory bank with the selected examples.
    #  example shape = [N x C x H x W]
    #  labels shape  = [N x C x H x W]
    #  preds shape   = [N x C x H x W]
    self.memory.update(examples, labels=labels, preds=preds)

class Er(ContinualModel):
    NAME = 'er'

    def __init__(self, backbone, loss, args, transform):
        super(Er, self).__init__(backbone, loss, args, transform)
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

        self.to(self.device)

    def begin_task(self, cur_train_loader, next_train_loader):
        self.current_task += 1

        if self.current_task > 1 and not self.memory.is_empty():
            self.memory = iter(self.memory)
        
        self.reset_opt()

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

        self.opt.zero_grad()
        if self.current_task > 1 and not self.memory.is_empty():
            buf_inputs, buf_labels, _, _ = self.get_past_data()
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()

    def reset_opt(self):
        # Keep the optimizer object so an attached scheduler remains connected.
        self.opt.state.clear()
        for group in self.opt.param_groups:
            group['lr'] = self.args.lr

    def save(self, save_path, scheduler=None):
        """Save ER's network, replay memory, task position, and optimizer."""
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
