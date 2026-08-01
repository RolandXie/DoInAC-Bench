import torch
import numpy as np
from datasets import get_dataset
from models.utils.continual_model import ContinualModel, optimizer_dict
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, add_backbone_args, add_scheduler_args, ArgumentParser

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Finetuning baseline.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_backbone_args(parser)
    add_scheduler_args(parser)
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
            self.net.freeze_other_share(self.current_task-1)
        

    def end_task(self, cur_train_loader, next_train_loader):
        # setup_buffer(self, cur_train_loader, next_train_loader)
        pass


    def observe(self, cur_data, next_data):
        inputs, labels, _, = cur_data

        self.opt.zero_grad()

        outputs = self.net(inputs,'logits', self.current_task-1)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()
