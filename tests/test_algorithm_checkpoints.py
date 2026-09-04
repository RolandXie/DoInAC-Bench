from argparse import Namespace
import os
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from models.der import DER
from models.er import Er
from models.adil import ADIL
from models.joint import Joint
from models.lwf import Lwf
from models.udil import UDIL
from models.utils.continual_model import ContinualModel
from main.training import _load_stage_checkpoint, _save_stage_checkpoints
from utils.buffer import Buffer as ReplayBuffer
from utils.buffer_feature import Buffer as FeatureBuffer


def make_shell(model_type):
    model = object.__new__(model_type)
    torch.nn.Module.__init__(model)
    model.device = torch.device('cpu')
    model.net = torch.nn.Linear(2, 2)
    model.opt = torch.optim.Adam(model.net.parameters(), lr=1e-3)
    model.current_task = 1
    return model


def populate_optimizer(model):
    loss = model.net(torch.ones(1, 2)).sum()
    loss.backward()
    model.opt.step()
    model.opt.zero_grad()


class AlgorithmCheckpointTest(unittest.TestCase):
    def round_trip(self, source, target):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'checkpoint.pt')
            source.save(path)
            target.load(path)
            return torch.load(path, map_location='cpu')

    def test_er_restores_replay_memory_and_optimizer(self):
        source = make_shell(Er)
        source.memory = ReplayBuffer(8, source.device, (2,), 2, batch_size=2)
        source.memory.update(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([0, 1]),
            torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
        )
        populate_optimizer(source)

        target = make_shell(Er)
        target.memory = ReplayBuffer(8, target.device, (2,), 2, batch_size=2)
        checkpoint = self.round_trip(source, target)

        self.assertNotIn('args', checkpoint)
        self.assertIsInstance(checkpoint['memory_state_dict'], dict)
        self.assertEqual(target.current_task, 1)
        self.assertEqual(len(target.memory), 2)
        self.assertEqual(target.memory.domain_buffers[0].examples.shape[0], 2)
        examples, labels, preds, _ = next(iter(target.memory))
        self.assertEqual(examples.shape, (2, 2))
        torch.testing.assert_close(labels.sort().values, torch.tensor([0, 1]))
        self.assertEqual(preds.shape, (2, 2))
        self.assertTrue(target.opt.state)

    def test_der_restores_examples_labels_and_historical_logits(self):
        source = make_shell(DER)
        source.memory = ReplayBuffer(8, source.device, (2,), 2, batch_size=0)
        expected_logits = torch.tensor([[0.2, 0.8], [0.7, 0.3]])
        source.memory.update(
            torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
            torch.tensor([0, 1]),
            expected_logits,
        )

        target = make_shell(DER)
        target.memory = ReplayBuffer(8, target.device, (2,), 2, batch_size=0)
        self.round_trip(source, target)

        self.assertEqual(target.memory.domain_buffers[0].examples.shape[0], 2)
        examples, labels, logits, _ = next(iter(target.memory))
        order = labels.argsort()
        torch.testing.assert_close(logits[order], expected_logits)
        self.assertEqual(examples.shape, (2, 2))

    def test_udil_restores_feature_memory_and_auxiliary_learners(self):
        source = make_shell(UDIL)
        source.args = Namespace(task_weight_lr=2e-3)
        source.disc = torch.nn.Linear(3, 2)
        source.disc_opt = torch.optim.Adam(source.disc.parameters(), lr=3e-3)
        source.register_buffer('logits', torch.randn(5, 2))
        source.memory = FeatureBuffer(8, source.device, (2,), 2, batch_size=0)
        expected_features = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        source.memory.update(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([0, 1]),
            torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
            expected_features,
        )
        source.current_task = 2
        source.Nt = 20
        source.past_errs = [0.25, 0.5]
        source.task_logits = torch.randn(3, 1, requires_grad=True)
        source.task_weight_opt = torch.optim.Adam(
            [source.task_logits], lr=source.args.task_weight_lr
        )

        target = make_shell(UDIL)
        target.args = Namespace(task_weight_lr=2e-3)
        target.disc = torch.nn.Linear(3, 2)
        target.disc_opt = torch.optim.Adam(target.disc.parameters(), lr=3e-3)
        target.register_buffer('logits', torch.zeros(5, 2))
        target.memory = FeatureBuffer(8, target.device, (2,), 2, batch_size=0)
        self.round_trip(source, target)

        self.assertEqual(target.memory.domain_buffers[0].examples.shape[0], 2)
        _, labels, _, features, _ = next(iter(target.memory))
        torch.testing.assert_close(features[labels.argsort()], expected_features)
        self.assertEqual(target.current_task, 2)
        self.assertEqual(target.Nt, 20)
        self.assertEqual(target.past_errs, [0.25, 0.5])
        torch.testing.assert_close(target.task_logits, source.task_logits)
        self.assertTrue(target.task_logits.requires_grad)

    def test_lwf_restores_distillation_logits_and_task_position(self):
        source = make_shell(Lwf)
        source.register_buffer('logits', torch.randn(5, 2))
        source.current_task = 3
        target = make_shell(Lwf)
        target.register_buffer('logits', torch.zeros(5, 2))

        self.round_trip(source, target)

        self.assertEqual(target.current_task, 3)
        torch.testing.assert_close(target.logits, source.logits)

    def test_adil_restores_active_task_and_task_specific_network_state(self):
        class TinyADILNet(torch.nn.Linear):
            def freeze_other_share(self, task):
                self.frozen_for_task = task

        source = make_shell(ADIL)
        source.net = TinyADILNet(2, 2)
        source.opt = torch.optim.Adam(source.net.parameters(), lr=1e-3)
        source.current_task = 3
        source.net.current_task = 2
        target = make_shell(ADIL)
        target.net = TinyADILNet(2, 2)
        target.opt = torch.optim.Adam(target.net.parameters(), lr=1e-3)

        self.round_trip(source, target)

        self.assertEqual(target.current_task, 3)
        self.assertEqual(target.net.current_task, 2)
        self.assertEqual(target.net.frozen_for_task, 2)
        for expected, actual in zip(source.net.parameters(), target.net.parameters()):
            torch.testing.assert_close(actual, expected)

    def test_joint_restores_domain_position(self):
        source = make_shell(Joint)
        source.current_task = 4
        target = make_shell(Joint)

        self.round_trip(source, target)

        self.assertEqual(target.current_task, 4)

    def test_base_checkpoint_remains_network_only(self):
        model = make_shell(Er)
        model.save = ContinualModel.save.__get__(model, Er)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'weights.pt')
            model.save(path)
            checkpoint = torch.load(path, map_location='cpu')
        self.assertEqual(set(checkpoint), set(model.net.state_dict()))

    def test_checkpoint_restores_all_random_generators(self):
        source = make_shell(DER)
        source.memory = ReplayBuffer(8, source.device, (2,), 2)
        target = make_shell(DER)
        target.memory = ReplayBuffer(8, target.device, (2,), 2)

        random.seed(11)
        np.random.seed(12)
        torch.manual_seed(13)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'checkpoint.pt')
            source.save(path)
            expected = (random.random(), np.random.rand(), torch.rand(1))
            target.load(path)
            actual = (random.random(), np.random.rand(), torch.rand(1))

        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2])

    def test_training_layer_writes_weight_and_algorithm_checkpoints(self):
        source = make_shell(DER)
        source.memory = ReplayBuffer(8, source.device, (2,), 2)
        target = make_shell(DER)
        target.memory = ReplayBuffer(8, target.device, (2,), 2)
        source_scheduler = torch.optim.lr_scheduler.StepLR(
            source.opt, step_size=1, gamma=0.5
        )
        target_scheduler = torch.optim.lr_scheduler.StepLR(
            target.opt, step_size=1, gamma=0.5
        )
        source.opt.step()
        source_scheduler.step()
        args = Namespace()

        with tempfile.TemporaryDirectory() as folder:
            with patch('main.training._checkpoint_folder', return_value=folder):
                _save_stage_checkpoints(
                    source, args, stage=1, scheduler=source_scheduler
                )
                weights = torch.load(
                    os.path.join(folder, 'domain-1.pt'), map_location='cpu'
                )
                resume = torch.load(
                    os.path.join(folder, 'domain-1-resume.pt'),
                    map_location='cpu',
                )
                _load_stage_checkpoint(
                    target, args, completed_task=0,
                    scheduler=target_scheduler,
                )

        self.assertNotIn('algorithm', weights)
        self.assertEqual(resume['algorithm'], 'der')
        self.assertEqual(target_scheduler.last_epoch, source_scheduler.last_epoch)
        for expected, actual in zip(source.net.parameters(), target.net.parameters()):
            torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
