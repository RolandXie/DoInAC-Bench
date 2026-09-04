'''
Taken and modified  PANN's CNN14 model architecture written by Qiuqiang Kong
from https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py
'''
try:
    from torchlibrosa.stft import Spectrogram, LogmelFilterBank
    from torchlibrosa.augmentation import SpecAugmentation
    TORCHLIBROSA_AVAILABLE = True
except ImportError:
    TORCHLIBROSA_AVAILABLE = False
    print("Warning: torchlibrosa not available for CNN14. Audio features will not work.")

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

try:
    from utils.autoaugment_audio import NsynthPolicy
except ImportError:
    NsynthPolicy = None

try:
    from utils.mixup import Mixup 
except ImportError:
    Mixup = None

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_tasks):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        # self.bn1 = nn.BatchNorm2d(out_channels)
        # self.bn2 = nn.BatchNorm2d(out_channels)

        self.bnF = nn.ModuleList([nn.BatchNorm2d(out_channels) for i in range(nb_tasks)])
        self.bnS = nn.ModuleList([nn.BatchNorm2d(out_channels) for i in range(nb_tasks)])

        # self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)


    def forward(self, input, pool_size=(2, 2), pool_type='avg', task=1):

        x = input
        x = F.relu_(self.bnF[task](self.conv1(x)))
        x = F.relu_(self.bnS[task](self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class CNN14(nn.Module):

    NAME = 'cnn14'

    def __init__(
            self,
            in_dim,
            hidden_dim,
            classes_num,
            args
        ):

        super(CNN14, self).__init__()

        self.current_task = 0
        self.nb_tasks = getattr(args, 'n_task', getattr(args, 'n_tasks', 6))
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

        # Check if this is audio mode
        self.is_audio = getattr(args, 'is_audio', False)

        # Audio feature extractors (will be initialized if audio mode)
        self.spectrogram_extractor = None
        self.logmel_extractor = None

        if NsynthPolicy is not None:
            self.trans = NsynthPolicy()
        else:
            self.trans = None

        self.bn0 = nn.ModuleList(
            [nn.BatchNorm2d(64) for _ in range(self.nb_tasks)]
        )
        self.conv_block1 = ConvBlock(1, 64, self.nb_tasks)
        self.conv_block2 = ConvBlock(64, 128, self.nb_tasks)
        self.conv_block3 = ConvBlock(128, 256, self.nb_tasks)
        self.conv_block4 = ConvBlock(256, 512, self.nb_tasks)
        self.conv_block5 = ConvBlock(512, 1024, self.nb_tasks)
        self.conv_block6 = ConvBlock(1024, 2048, self.nb_tasks)

        self.fc = nn.ModuleList(
            [
                nn.Linear(2048, class_count)
                for class_count in self.task_class_counts
            ]
        )

        self.reset_parameters()

        if self.is_audio:
            self.audio_init()

    def audio_init(
            self,
            window_size=1024,
            hop_size=320,
            win_length=None,
            window='hann',
            center=True,
            pad_mode='reflect',
            sample_rate=32000,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            ref=1.0,
            amin=1e-10,
            top_db=None
        ):
        """Initialize the audio feature extractors."""
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
                                        sr=sample_rate, n_fft=window_size,
                                        n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
                                        top_db=top_db,
                                        freeze_parameters=True
                                    )

    def get_output_dim(self):
        return max(head.out_features for head in self.fc)

    ##
    # Utilized in the incremental learning setting to change
    #   the output dimension of the model.
    ##
    def change_output_dim(self, new_dim, second_iter=False):
        old_out_features = self.fc[0].out_features
        if new_dim < old_out_features:
            raise ValueError("new_dim must not be smaller than the current output dimension")

        new_heads = nn.ModuleList()
        for old_head in self.fc:
            new_head = nn.Linear(old_head.in_features, new_dim)
            new_head = new_head.to(device=old_head.weight.device, dtype=old_head.weight.dtype)
            with torch.no_grad():
                new_head.weight[:old_out_features].copy_(old_head.weight)
                new_head.bias[:old_out_features].copy_(old_head.bias)
            new_heads.append(new_head)

        self.fc = new_heads
        self.n_classes = new_dim


    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias'):
                    if m.bias is not None:
                        m.bias.data.fill_(0.)
            elif isinstance(m, nn.Linear):
                init_layer(m)
            elif isinstance(m, nn.BatchNorm2d):
                init_bn(m)

    def spectrum(self, input, augmentation=False):
        if not TORCHLIBROSA_AVAILABLE:
            raise ImportError("torchlibrosa is required for audio features. Please install it: pip install torchlibrosa")
            
        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)

        if augmentation and self.trans is not None:
            x = self.trans(x)
        return x 

    def _extract_features(self, input, task):
        x = input.transpose(1, 3)
        x = self.bn0[task](x)
        x = x.transpose(1, 3)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        return x1 + x2

    def spectrum_forward(self, input, task):
        features = self._extract_features(input, task)
        return self._classify(features, task)

    def _classify(self, features, task):
        """Apply the paper's base-plus-domain residual classifier."""
        base_logits = self.fc[0](features)
        if task == 0:
            return base_logits
        indices = torch.as_tensor(
            self.task_class_indices[task], device=features.device
        )
        return base_logits.index_select(1, indices) + self.fc[task](features)

    def _task_batch_norms(self, task):
        yield self.bn0[task]
        for block in (
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
            self.conv_block6,
        ):
            yield block.bnF[task]
            yield block.bnS[task]

    def freeze_other_share(self, task):
        """Freeze shared/old parameters and train the new domain modules."""
        for p in self.parameters():
            p.requires_grad = False

        for domain in range(self.nb_tasks):
            for bn in self._task_batch_norms(domain):
                bn.train(domain == task)
                for p in bn.parameters():
                    p.requires_grad = domain == task

        for domain, head in enumerate(self.fc):
            head.train(domain == task)
            for p in head.parameters():
                p.requires_grad = domain == task

    def forward(
        self,
        x,
        returnt="logits",
        task=None
    ):
        if self.training:
            if task is None:
                task = self.current_task
            return self._forward_train(x, returnt, task)
        if task is not None:
            return self._forward_train(x, returnt, task)
        return self._forward_eval(x, returnt)

    def _forward_train(
        self,
        x,
        returnt='logits',
        task=0
    ):
        if self.is_audio:
            x = self.spectrogram_extractor(x)
            x = self.logmel_extractor(x)

        features = self._extract_features(x, task)
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

    def _forward_eval(
        self,
        x,
        returnt='logits'
    ):
        task_outputs = []
        task_features = []
        with torch.no_grad():
            for task in range(self.current_task + 1):
                logits, probs, features = self._forward_train(x, "all", task)
                task_outputs.append(probs)
                task_features.append(features)

        # Heads may have different sizes (the Korea head has four classes), so
        # calculate entropy before stacking.
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
                indices = torch.as_tensor(
                    self.task_class_indices[task], device=x.device
                )
                selected = outputs[mask]
                selected[:, indices] = task_outputs[task][mask]
                outputs[mask] = selected
                features[mask] = task_features[task][mask]

        # The generic evaluator expects logits. Log-probabilities preserve the
        # argmax while allowing heads with different class counts to share a
        # ten-class tensor in the domain-agnostic protocol.
        logits = torch.log(outputs + 1e-8)

        if returnt == 'features':
            return features
        elif returnt == 'logits':
            return logits
        elif returnt == 'prob':
            return outputs
        elif returnt == 'all':
            return logits, outputs, features
