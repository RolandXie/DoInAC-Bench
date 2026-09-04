import torch
from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_backbone_args, add_scheduler_args,ArgumentParser


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Finetuning baseline.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_backbone_args(parser)
    add_scheduler_args(parser)
    return parser


class Finetune(ContinualModel):
    NAME = 'finetune'

    def __init__(self, backbone, loss, args, transform):
        super(Finetune, self).__init__(backbone, loss, args, transform)

    def observe(self, cur_data, next_data):

        x, y, idx = cur_data
        self.opt.zero_grad()
        outputs = self.net(x)

        loss = self.loss(outputs, y)
        loss.backward()
        self.opt.step()
        return loss.item()
