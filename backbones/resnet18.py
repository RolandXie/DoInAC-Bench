import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights

from backbones.utils.continual_backbone import FwdContinualBackbone
from backbones.utils.modules import xavier

try:
    from torchlibrosa.stft import Spectrogram, LogmelFilterBank
    from torchlibrosa.augmentation import SpecAugmentation
    TORCHLIBROSA_AVAILABLE = True
except ImportError:
    TORCHLIBROSA_AVAILABLE = False
    print("Warning: torchlibrosa not available. Audio features will not work.")


class Resnet18(FwdContinualBackbone):
    NAME = 'resnet18'
    def __init__(self, indim, hiddim, outdim, args):
        super(Resnet18, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)

        # dictionary for the output and the intermediate layers
        self.fwd_outputs = {}

        def get_feature(name):
            def hook(model, input, output):
                self.fwd_outputs[name] = output
            return hook

        # Check if this is audio mode
        self.is_audio = getattr(args, 'is_audio', False)
        hop_size = getattr(args, 'hop_size', 160)

        # Audio feature extractors (will be initialized if audio mode)
        self.spectrogram_extractor = None
        self.logmel_extractor = None
        
        if self.is_audio:
            # Audio mode: initialize with 1 channel for spectrogram
            self.audio_init(hop_size=hop_size)
            self.net = resnet18(num_classes=outdim)
            # Modify first conv layer for single channel input (spectrogram)
            self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            # Image mode: use default resnet18
            self.net = resnet18(num_classes=outdim)

        # forward hook to store the intermediate results
        self.net.avgpool.register_forward_hook(get_feature('features'))
        self.net.fc.register_forward_hook(get_feature('logits'))
        

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

    def forward(self, x, returnt='logits', task=None):
        # If audio mode and feature extractors are initialized
        if self.is_audio and self.spectrogram_extractor is not None and self.logmel_extractor is not None:
            # x is raw audio waveform: (batch_size, data_length)
            x = self.spectrogram_extractor(x)  # (batch_size, 1, time_steps, freq_bins)
            x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
            
            # Transpose to match ResNet input format: (batch_size, channels, height, width)
            # Spectrogram output is (batch_size, 1, time_steps, mel_bins)
            # ResNet expects (batch_size, channels, height, width)
            # We can keep it as is since channel=1

        self.net(x)

        if returnt == 'features':
            shape = self.fwd_outputs['features'].shape
            return self.fwd_outputs['features'].view(*shape[:2])
        elif returnt == 'logits':
            return self.fwd_outputs['logits']
        elif returnt == 'prob':
            return self.softmax(self.fwd_outputs['logits'])
        elif returnt == 'all':
            logits = self.fwd_outputs['logits']
            probs = self.softmax(self.fwd_outputs['logits'])

            shape = self.fwd_outputs['features'].shape
            features = self.fwd_outputs['features'].view(*shape[:2])
            return logits, probs, features

    def get_output_dim(self):
        return self.net.fc.out_features

    def change_output_dim(self, new_dim, second_iter=False):
        in_features = self.net.fc.in_features
        out_features = self.net.fc.out_features

        new_out_features = new_dim
        num_new_classes = new_dim - out_features
        new_fc = nn.Linear(in_features, out_features + num_new_classes)

        new_fc.weight.data[:out_features] = self.net.fc.weight.data
        new_fc.bias.data[:out_features] = self.net.fc.bias.data
        self.net.fc = new_fc


if __name__ == '__main__':
    from argparse import Namespace
    
    # Test image mode (default)
    print("Testing image mode...")
    args = Namespace(is_audio=False)
    model = Resnet18(indim=(3, 128, 128), hiddim=256, outdim=10, args=args)
    x = torch.ones([2, 3, 128, 128])  # batch_size=2, channels=3, height=128, width=128
    output = model(x, returnt='all')
    print("Image mode - Logits shape:", output[0].shape)
    print("Image mode - Features shape:", output[2].shape)
    
    # Test audio mode (only if torchlibrosa is available)
    if TORCHLIBROSA_AVAILABLE:
        print("\nTesting audio mode...")
        args = Namespace(is_audio=True)
        model = Resnet18(indim=16000, hiddim=256, outdim=10, args=args)
        model.audio_init()
        
        x = torch.ones([1,16000])  # batch_size=2, data_length=16000
        output = model(x, returnt='all')
        print("Audio mode - Logits shape:", output[0].shape)
        print("Audio mode - Features shape:", output[2].shape)
    else:
        print("\nSkipping audio mode test (torchlibrosa not available)")
    
    print("\nAll tests passed!")
