from argparse import Namespace
import unittest

import torch
import torch.nn.functional as F

from backbones.cnn14 import CNN14
from datasets.tut_europe_korea import (
    TASK_CLASS_COUNTS,
    TASK_CLASS_INDICES,
)
from main.training import (
    _evaluation_labels,
    maybe_reset_training_state,
    reset_optimizer,
    reset_scheduler,
    scheduler_enabled_for_task,
)
from utils.loggers import Logger
from utils.metrics import forward_recovery
from models.adil import ADIL


def make_model():
    return CNN14(
        in_dim=(1, 64, 64),
        hidden_dim=100,
        classes_num=10,
        args=Namespace(
            n_task=5,
            is_audio=False,
            task_class_counts=TASK_CLASS_COUNTS,
            task_class_indices=TASK_CLASS_INDICES,
        ),
    )


class TUTEuropeKoreaReproductionTest(unittest.TestCase):
    def test_adil_d1_bn_warm_start_copies_affine_and_running_state(self):
        model = make_model()
        source_bns = list(model._task_batch_norms(0))
        target_bns = list(model._task_batch_norms(1))
        with torch.no_grad():
            for index, source in enumerate(source_bns):
                source.weight.fill_(1.5 + index)
                source.bias.fill_(-0.25 - index)
                source.running_mean.fill_(0.5 + index)
                source.running_var.fill_(2.0 + index)
                source.num_batches_tracked.fill_(3 + index)

        wrapper = object.__new__(ADIL)
        torch.nn.Module.__init__(wrapper)
        wrapper.net = model
        wrapper._initialize_batch_norm_from_d1(1)

        for source, target in zip(source_bns, target_bns):
            for name, value in source.state_dict().items():
                torch.testing.assert_close(target.state_dict()[name], value)

    def test_adil_bn_warm_start_does_not_change_shared_or_classifier_parameters(self):
        model = make_model()
        before_shared = model.conv_block1.conv1.weight.detach().clone()
        before_head = model.fc[1].weight.detach().clone()
        wrapper = object.__new__(ADIL)
        torch.nn.Module.__init__(wrapper)
        wrapper.net = model

        wrapper._initialize_batch_norm_from_d1(1)

        torch.testing.assert_close(model.conv_block1.conv1.weight, before_shared)
        torch.testing.assert_close(model.fc[1].weight, before_head)

    def test_task_heads_and_residual_classifier(self):
        model = make_model()
        features = torch.randn(3, 2048)

        self.assertEqual(model._classify(features, 0).shape, (3, 10))
        self.assertEqual(model._classify(features, 4).shape, (3, 4))

        expected = (
            model.fc[0](features)[:, list(TASK_CLASS_INDICES[4])]
            + model.fc[4](features)
        )
        torch.testing.assert_close(model._classify(features, 4), expected)

    def test_incremental_task_trains_all_current_bn_and_head_only(self):
        torch.manual_seed(7)
        model = make_model()
        model.train()
        model.current_task = 1
        model.freeze_other_share(1)

        inputs = torch.randn(2, 1, 64, 64)
        labels = torch.tensor([0, 1])
        loss = F.cross_entropy(model(inputs, task=1), labels)
        loss.backward()

        self.assertIsNotNone(model.fc[1].weight.grad)
        self.assertGreater(model.fc[1].weight.grad.abs().sum().item(), 0)
        self.assertIsNone(model.fc[0].weight.grad)
        self.assertIsNone(model.conv_block1.conv1.weight.grad)

        for bn in model._task_batch_norms(1):
            self.assertTrue(bn.weight.requires_grad)
            self.assertIsNotNone(bn.weight.grad)
        for bn in model._task_batch_norms(0):
            self.assertFalse(bn.weight.requires_grad)
            self.assertIsNone(bn.weight.grad)

    def test_incremental_bn_only_strategy_freezes_classifier(self):
        torch.manual_seed(7)
        model = make_model()
        model.train()
        model.current_task = 1
        model.freeze_other_share(1)
        model.fc.eval()
        for parameter in model.fc.parameters():
            parameter.requires_grad = False

        inputs = torch.randn(2, 1, 64, 64)
        labels = torch.tensor([0, 1])
        loss = F.cross_entropy(model(inputs, task=1), labels)
        loss.backward()

        trainable = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(all('bn' in name.lower() for name in trainable))
        self.assertTrue(all(parameter.grad is None for parameter in model.fc.parameters()))
        for bn in model._task_batch_norms(1):
            self.assertTrue(bn.weight.requires_grad)
            self.assertIsNotNone(bn.weight.grad)

    def test_domain_aware_and_agnostic_output_shapes(self):
        model = make_model()
        model.current_task = 4
        model.eval()
        inputs = torch.randn(2, 1, 64, 64)

        self.assertEqual(model(inputs, task=4).shape, (2, 4))
        self.assertEqual(model(inputs).shape, (2, 10))

    def test_korea_labels_map_to_global_space_for_agnostic_eval(self):
        dataset = Namespace(TASK_CLASS_INDICES=TASK_CLASS_INDICES)
        labels = torch.tensor([0, 1, 2, 3])
        mapped = _evaluation_labels(
            dataset, labels, task=4, protocol='domain-agnostic'
        )
        torch.testing.assert_close(mapped, torch.tensor([1, 2, 3, 4]))

    def test_optimizer_and_cosine_scheduler_restart_at_incremental_lr(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.Adam([parameter], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=120, eta_min=1e-5
        )

        reset_optimizer(optimizer, 1e-4)
        reset_scheduler(scheduler, 1e-4)
        self.assertEqual(optimizer.param_groups[0]['lr'], 1e-4)
        optimizer.step()
        scheduler.step()
        self.assertLess(optimizer.param_groups[0]['lr'], 1e-4)
        self.assertGreater(optimizer.param_groups[0]['lr'], 1e-5)

    def test_training_state_restart_is_opt_in(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.Adam([parameter], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=120, eta_min=1e-5
        )
        optimizer.state[parameter]['marker'] = torch.tensor(1.0)
        optimizer.step()
        scheduler.step()
        original_lr = optimizer.param_groups[0]['lr']
        original_epoch = scheduler.last_epoch

        disabled = Namespace(
            lr=1e-3,
            incremental_lr=1e-4,
        )
        self.assertFalse(
            maybe_reset_training_state(disabled, optimizer, scheduler, task=1)
        )
        self.assertIn(parameter, optimizer.state)
        self.assertEqual(optimizer.param_groups[0]['lr'], original_lr)
        self.assertEqual(scheduler.last_epoch, original_epoch)

        enabled = Namespace(
            lr=1e-3,
            incremental_lr=1e-4,
            incremental_scheduler='none',
            reset_training_state_per_task=True,
        )
        self.assertTrue(
            maybe_reset_training_state(enabled, optimizer, scheduler, task=1)
        )
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual(optimizer.param_groups[0]['lr'], 1e-4)
        self.assertEqual(scheduler.last_epoch, original_epoch)

    def test_scheduler_can_be_disabled_only_after_first_task(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.Adam([parameter], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=120, eta_min=1e-5
        )
        args = Namespace(
            lr=1e-3,
            incremental_lr=1e-4,
            incremental_scheduler='none',
            reset_training_state_per_task=True,
        )

        self.assertTrue(scheduler_enabled_for_task(args, task=0))
        self.assertFalse(scheduler_enabled_for_task(args, task=1))
        self.assertTrue(
            maybe_reset_training_state(args, optimizer, scheduler, task=0)
        )
        self.assertEqual(optimizer.param_groups[0]['lr'], 1e-3)
        self.assertEqual(scheduler.eta_min, 1e-5)

    def test_forward_recovery_metric_uses_initial_and_final_accuracies(self):
        initial_results = [20.0, 10.0, 40.0]
        results = [
            [25.0, 12.0, 45.0],
            [30.0, 15.0, 42.0],
            [28.0, 14.0, 41.0],
        ]
        value = forward_recovery(results, initial_results)
        expected = (
            (30.0 - 28.0) / (30.0 - 20.0)
            + (15.0 - 14.0) / (15.0 - 10.0)
            + (45.0 - 41.0) / (45.0 - 40.0)
        ) / 3.0
        self.assertAlmostEqual(value, expected)

    def test_logger_prepends_initial_results_to_acc_matrix(self):
        logger = Logger('dummy_dataset', 'dummy_model')
        initial_results = [1.0, 2.0]
        results = [[3.0, 4.0], [5.0, 6.0]]

        acc_matrix = logger.add_acc_matrix(
            results=results,
            initial_results=initial_results,
        )

        self.assertEqual(acc_matrix.shape, (3, 2))
        self.assertEqual(acc_matrix.tolist()[0], initial_results)
        self.assertEqual(acc_matrix.tolist()[1:], results)


if __name__ == '__main__':
    unittest.main()
