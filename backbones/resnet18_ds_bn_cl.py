import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace


try:
    from torchlibrosa.stft import Spectrogram, LogmelFilterBank
    from torchlibrosa.augmentation import SpecAugmentation
    TORCHLIBROSA_AVAILABLE = True
except ImportError:
    TORCHLIBROSA_AVAILABLE = False
    print("Warning: torchlibrosa not available. Audio features will not work.")

class DomainBatchNorm2d(nn.Module):
    """
    Domain-specific BatchNorm.
    Each task/domain owns an independent BN.
    """
    def __init__(self, num_features, nb_tasks):
        super().__init__()
        self.bn = nn.ModuleList(
            [nn.BatchNorm2d(num_features) for _ in range(nb_tasks)]
        )

    def forward(self, x, task):
        return self.bn[task](x)


class BasicBlockDSBN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, nb_tasks=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            inplanes, planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = DomainBatchNorm2d(planes, nb_tasks)

        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = DomainBatchNorm2d(planes, nb_tasks)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x, task):

        identity = x
        out = self.conv1(x)
        out = self.bn1(out, task)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, task)
        if self.downsample is not None:
            identity = self.downsample(x, task)
        out += identity
        out = self.relu(out)
        return out


class DownsampleDSBN(nn.Module):
    def __init__(self, inplanes, planes, stride, nb_tasks):
        super().__init__()

        self.conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=1,
            stride=stride,
            bias=False
        )
        self.bn = DomainBatchNorm2d(
            planes,
            nb_tasks
        )

    def forward(self, x, task):
        x = self.conv(x)
        x = self.bn(x, task)
        return x


class ResNet18_DS_BN_CL(nn.Module):
    NAME = 'resnet18_ds_bn_cl'

    def __init__(
        self,
        indim,
        hiddim,
        classes_num,
        args
    ):
        super().__init__()
        self.current_task = 0
        self.nb_tasks = getattr(args, 'n_task', getattr(args, 'n_tasks', 6))
        if self.nb_tasks < 1:
            raise ValueError(f'n_task must be positive, got {self.nb_tasks}')

        default_class_counts = (classes_num,) * self.nb_tasks
        self.task_class_counts = tuple(
            getattr(args, 'task_class_counts', default_class_counts)
        )
        if len(self.task_class_counts) != self.nb_tasks:
            raise ValueError(
                'task_class_counts must contain one value per task: '
                f'{self.task_class_counts} vs {self.nb_tasks} tasks'
            )
        default_class_indices = tuple(
            tuple(range(class_count)) for class_count in self.task_class_counts
        )
        self.task_class_indices = tuple(
            tuple(indices) for indices in getattr(
                args, 'task_class_indices', default_class_indices
            )
        )
        if len(self.task_class_indices) != self.nb_tasks:
            raise ValueError('task_class_indices must contain one entry per task')
        for class_count, indices in zip(
            self.task_class_counts, self.task_class_indices
        ):
            if len(indices) != class_count:
                raise ValueError(
                    'Each task_class_indices entry must match its class count: '
                    f'{len(indices)} vs {class_count}'
                )

        self.inplanes = 64

        # Check if this is audio mode
        self.is_audio = getattr(args, 'is_audio', False)
        self.input_channel = getattr(args, 'input_channel', 3)

        # Audio feature extractors (will be initialized if audio mode)
        self.spectrogram_extractor = None
        self.logmel_extractor = None

        self.conv1 = nn.Conv2d(self.input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = DomainBatchNorm2d(64, self.nb_tasks)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2, self.nb_tasks)
        self.layer2 = self._make_layer(128, 2, self.nb_tasks, stride=2)
        self.layer3 = self._make_layer(256, 2, self.nb_tasks, stride=2)
        self.layer4 = self._make_layer(512, 2, self.nb_tasks, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # self.fc = nn.Linear(512, classes_num)
        self.fc = nn.ModuleList(
            [
                nn.Linear(512, class_count)
                for class_count in self.task_class_counts
            ]
        )

        if self.is_audio:
            # Audio mode: initialize with 1 channel for spectrogram
            self.audio_init()
            # Modify first conv layer for single channel input (spectrogram)
            self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def _make_layer(
        self,
        planes,
        blocks,
        nb_tasks,
        stride=1
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = DownsampleDSBN(
                self.inplanes,
                planes,
                stride,
                nb_tasks
            )
        layers = []
        layers.append(BasicBlockDSBN(self.inplanes, planes, stride, downsample,nb_tasks))

        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                BasicBlockDSBN(
                    self.inplanes,
                    planes,
                    nb_tasks=nb_tasks
                )
            )
        return nn.ModuleList(layers)

    def _forward_layer(self, layer, x, task):
        for block in layer:
            x = block(x, task)
        return x

    def audio_init(
            self, 
            window_size=512,
            hop_size=160, 
            win_length=None,
            window='hann',
            center=True,
            pad_mode='reflect',
            sample_rate=16000,
            mel_bins=64,
            fmin=50,
            fmax=8000,
            ref=1.0,
            amin=1e-10,
            top_db=None
        ):
        """
        Initialize audio feature extractors.
        Call this method when using audio mode.
        """
        if not TORCHLIBROSA_AVAILABLE:
            raise ImportError("torchlibrosa is required for audio features. Please install it: pip install torchlibrosa")
            
        if win_length is None:
            win_length = window_size
            
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size, 
            hop_length=hop_size,
            win_length=win_length, 
            window=window, 
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate, 
            n_fft=window_size,
            n_mels=mel_bins, 
            fmin=fmin, 
            fmax=fmax, 
            ref=ref, 
            amin=amin,
            top_db=top_db,
            freeze_parameters=True
        )

    def forward(
        self,
        x,
        returnt="logits",
        task=None
    ):  
        if self.training:
            if task is None:
                task = self.current_task
            self._validate_task(task)
            return self._forward_train(x,returnt,task)
        if task is not None:
            self._validate_task(task)
            # Domain-aware evaluation must use the caller-provided domain.
            return self._forward_train(x, returnt, task)
        return self._forward_eval(x, returnt)

    def _validate_task(self, task):
        if not isinstance(task, int) or not 0 <= task < self.nb_tasks:
            raise ValueError(
                f'task must be an integer in [0, {self.nb_tasks}), got {task}'
            )

    def _forward_train(
        self,
        x,
        returnt='logits',
        task=0
    ):
        if self.is_audio:
            x = self.spectrogram_extractor(x)
            x = self.logmel_extractor(x)

        x = self.conv1(x)
        x = self.bn1(x, task)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self._forward_layer(self.layer1, x, task)
        x = self._forward_layer(self.layer2, x, task)
        x = self._forward_layer(self.layer3, x, task)
        x = self._forward_layer(self.layer4, x, task)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self._classify(features, task)

        fwd_outputs = {
            'features': features,
            'logits': logits
        }
        if returnt == 'features':
            return fwd_outputs['features']
        elif returnt == 'logits':
            return fwd_outputs['logits']
        elif returnt == 'prob':
            return torch.softmax(
                fwd_outputs['logits'],
                dim=1
            )
        elif returnt == 'all':
            logits = fwd_outputs['logits']
            probs = torch.softmax(logits, dim=1)
            return logits, probs, fwd_outputs['features']

    def _classify(self, features, task):
        """Apply ADIL's shared-base plus domain-specific residual head."""
        base_logits = self.fc[0](features)
        if task == 0:
            return base_logits

        indices = torch.as_tensor(
            self.task_class_indices[task], device=features.device
        )
        if torch.any(indices >= base_logits.size(1)) or torch.any(indices < 0):
            raise ValueError(
                'task_class_indices must refer to columns in the base '
                f'classifier (size {base_logits.size(1)}), got '
                f'{self.task_class_indices[task]}'
            )
        return base_logits.index_select(1, indices) + self.fc[task](features)

    def _forward_eval(
        self,
        x,
        returnt='logits',
        in_task=0
    ):
        max_task = min(self.current_task, self.nb_tasks - 1)
        task_outputs = []

        with torch.no_grad():
            for task in range(max_task + 1):
                probs = torch.softmax(self._forward_train(x, "logits", task), dim=1)
                task_outputs.append(probs)
        entropy = torch.stack([
            -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            for probs in task_outputs
        ], dim=1)
        task_ids = torch.argmin(entropy, dim=1)
        global_classes = self.fc[0].out_features
        outputs = x.new_zeros((x.size(0), global_classes))
        features = torch.empty(
            x.size(0),
            self.fc[0].in_features,
            device=x.device,
            dtype=x.dtype
        )
        with torch.no_grad():
            for task in torch.unique(task_ids):
                task = int(task.item())
                mask = task_ids == task
                _, probs, features_sub = self._forward_train(x[mask], "all", task)
                indices = torch.as_tensor(self.task_class_indices[task], device=x.device)
                selected = outputs[mask]
                selected[:, indices] = probs
                outputs[mask] = selected
                features[mask] = features_sub

        logits = torch.log(outputs + 1e-8)
        if returnt == 'features':
            return features
        elif returnt == 'logits':
            return logits
        elif returnt == 'prob':
            return outputs
        elif returnt == 'all':
            return logits, outputs, features

    def freeze_other_share(self, task):
        self._validate_task(task)
        # freeze everything
        for p in self.parameters():
            p.requires_grad = False

        # enable current task BN
        for module in self.modules():
            if isinstance(module, DomainBatchNorm2d):
                for i, bn in enumerate(module.bn):
                    if i == task:
                        bn.train()
                        for p in bn.parameters():
                            p.requires_grad = True
                    else:
                        bn.eval()

        # enable current task classifier
        if isinstance(self.fc, nn.ModuleList):
            for i, fc in enumerate(self.fc):
                if i == task:
                    fc.train()
                    for p in fc.parameters():
                        p.requires_grad = True
                else:
                    fc.eval()

    def _task_batch_norms(self, task):
        """Yield the task-specific BN modules for ADIL warm-starting."""
        self._validate_task(task)
        for module in self.modules():
            if isinstance(module, DomainBatchNorm2d):
                yield module.bn[task]

def print_model_structure(model):
    """
    Print model structure and parameter information.
    """
    print("\n" + "=" * 80)
    print("Model Structure")
    print("=" * 80)

    for name, module in model.named_modules():
        # skip root module
        if name == "":
            continue
        
        print(
            f"{name:<40} : {module.__class__.__name__}"
        )

    print("\n" + "=" * 80)
    print("Parameter Information")
    print("=" * 80)

    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():

        num_params = param.numel()

        total_params += num_params

        if param.requires_grad:
            trainable_params += num_params
        print(
            f"{name:<60} "
            f"shape={str(tuple(param.shape)):<25} "
            f"trainable={param.requires_grad}"
        )

    print("-" * 80)

    print(
        f"Total parameters: {total_params:,}"
    )

    print(
        f"Trainable parameters: {trainable_params:,}"
    )

    print("=" * 80)

def test_resnet18_dsbn():
    """
    Simple test for ResNet18DSBN.
    """
    nb_tasks = 6
    classes_num = 10

    model = ResNet18_DS_BN_CL(
        indim=(3, 128, 128),
        hiddim=100,
        classes_num=classes_num,
        args=Namespace(
            n_task=nb_tasks,
            is_audio=False,
            input_channel=3,
        ),
    )
    model.eval()

    # fake spectrogram input:
    # [batch, channel, time, freq]
    x = torch.randn(
        4,
        3,
        128,
        128
    )
    print("=" * 50)
    print("Model test start")
    print("=" * 50)
    print(
        "Total parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    # test each domain/task BN
    for task in range(nb_tasks):
        with torch.no_grad():
            output = model(
                x,
                task=task
            )
        print(
            f"Task {task}: output shape = {output.shape}"
        )

    # test freeze BN
    model.freeze_other_share(task=2)
    print("\nBN freeze test:")
    for name, module in model.named_modules():
        if isinstance(module, DomainBatchNorm2d):
            status = []
            for i, bn in enumerate(module.bn):
                trainable = any(
                    p.requires_grad
                    for p in bn.parameters()
                )
                status.append(trainable)
            print(
                name,
                status
            )
    print("=" * 50)
    print("Test finished successfully")
    print("=" * 50)
    print_model_structure(model)


if __name__ == "__main__":
    test_resnet18_dsbn()
