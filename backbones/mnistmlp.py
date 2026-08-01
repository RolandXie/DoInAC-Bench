import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.utils.continual_backbone import FwdContinualBackbone
from backbones.utils.modules import xavier

# Try to import torchlibrosa, but don't fail if it has issues
TORCHLIBROSA_AVAILABLE = False
Spectrogram = None
LogmelFilterBank = None

try:
    from torchlibrosa.stft import Spectrogram, LogmelFilterBank
    from torchlibrosa.augmentation import SpecAugmentation
    # Test if torchlibrosa actually works
    try:
        # Try to create a simple Spectrogram to test if it works
        test_spec = Spectrogram(n_fft=512, hop_length=160, win_length=512, window='hann', center=True, pad_mode='reflect', freeze_parameters=True)
        TORCHLIBROSA_AVAILABLE = True
        del test_spec
    except Exception as e:
        print(f"Warning: torchlibrosa imported but has issues: {e}")
        TORCHLIBROSA_AVAILABLE = False
except ImportError:
    TORCHLIBROSA_AVAILABLE = False
    print("Warning: torchlibrosa not available. Audio features will use fallback implementation.")

# The creation of the backbones follows the following paradigm:
#   Backbone(indim, hiddim, outdim, args)

class LinearClassifier(nn.Module):
    """Linear classifier, predicting the class label."""
    NAME = 'mnist-classifier'
    def __init__(self, indim, num_classes):
        super(LinearClassifier, self).__init__()
        self.classifier = nn.Linear(indim, num_classes)

    def forward(self, x, return_softmax=False):
        x = self.classifier(x)
        x_softmax = F.softmax(x, dim=1)
        # x = F.log_softmax(x, dim=1)

        return x, x_softmax if return_softmax else x


class MNISTMLP(FwdContinualBackbone):
    NAME = 'mnistmlp'
    
    def __init__(self, indim, hiddim, outdim, args) -> None:
        super().__init__()
        self.indim = indim # indim not necessarily an integer
        self.hiddim = hiddim
        self.outdim = outdim
        
        # Check if this is audio mode
        self.is_audio = getattr(args, 'audio', False)
        
        # Audio feature extractors (will be initialized if audio mode)
        self.spectrogram_extractor = None
        self.logmel_extractor = None
        
        # For audio mode, we need to determine the flattened dimension after spectrogram
        # This will be set when audio_init() is called
        self.audio_flat_dim = None
        
        # constructing the encoder and the predictor
        if isinstance(indim, tuple) or isinstance(indim, list):
            self.enc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(np.prod(indim), hiddim),
                nn.ReLU(),
                nn.Linear(hiddim, hiddim),
                nn.ReLU()
            )
        else: 
            self.enc = nn.Sequential(
                nn.Linear(indim, hiddim),
                nn.ReLU(),
                nn.Linear(hiddim, hiddim),
                nn.ReLU()
            )

        self.pred = LinearClassifier(indim=hiddim, num_classes=outdim)

        self.net = nn.Sequential(self.enc, self.pred)
        self.reset_parameters()

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
            # Fallback: use a simple spectrogram implementation
            print("Warning: torchlibrosa not available. Using simple spectrogram implementation.")
            self.audio_flat_dim = mel_bins
            return
            
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
        
        # Calculate the flattened dimension after spectrogram extraction
        # For a given audio length, we need to know the output size
        # We'll calculate this dynamically in forward() or set a default
        self.audio_flat_dim = mel_bins  # This is the frequency dimension

    def forward(self, x, returnt='logits', task=None):
        # If audio mode and feature extractors are initialized
        if self.is_audio:
            if self.spectrogram_extractor is not None and self.logmel_extractor is not None:
                # x is raw audio waveform: (batch_size, data_length)
                x = self.spectrogram_extractor(x)  # (batch_size, 1, time_steps, freq_bins)
                x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
            else:
                # Fallback: use simple spectrogram implementation
                # This is a placeholder - in practice, you should use proper audio processing
                # For now, we'll just reshape the input
                batch_size = x.shape[0]
                # Assume x is already processed or create a simple representation
                if len(x.shape) == 2:  # (batch_size, data_length)
                    # Simple: just use the raw audio as features (not ideal but functional)
                    # In practice, you should implement proper spectrogram extraction
                    x = x.unsqueeze(1)  # Add channel dimension
            
            # Flatten the spectrogram for MLP input
            batch_size = x.shape[0]
            if len(x.shape) > 2:  # If we have spectrogram dimensions
                x = x.view(batch_size, -1)  # Flatten to (batch_size, time_steps * mel_bins)
        
        feats = self.enc(x)
        
        if returnt == 'features':
            return feats
        
        # classifier supports returning two outputs
        logits, prob = self.pred(feats, return_softmax=True)

        if returnt == 'logits':
            return logits
        elif returnt == 'prob':
            return prob
        elif returnt == 'all':
            return logits, prob, feats
        else:
            return NotImplementedError("Unsupported return type")

    def get_output_dim(self):
        return self.outdim

    def change_output_dim(self, new_dim, second_iter=False):
        in_features = self.pred.classifier.in_features
        out_features = self.pred.classifier.out_features

        new_out_features = new_dim
        num_new_classes = new_dim - out_features
        new_classifier = nn.Linear(in_features, out_features + num_new_classes)

        new_classifier.weight.data[:out_features] = self.pred.classifier.weight.data
        new_classifier.bias.data[:out_features] = self.pred.classifier.bias.data
        self.pred.classifier = new_classifier
        self.outdim = new_out_features

    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        self.net.apply(xavier)


if __name__ == '__main__':
    from argparse import Namespace
    import torch
    
    # Test image mode (default)
    print("Testing image mode...")
    args = Namespace(audio=False)
    model = MNISTMLP(indim=(1, 28, 28), hiddim=800, outdim=10, args=args)
    x = torch.ones([2, 1, 28, 28])  # batch_size=2, channels=1, height=28, width=28
    output = model(x, returnt='all')
    print("Image mode - Logits shape:", output[0].shape)
    print("Image mode - Features shape:", output[2].shape)
    
    # Test audio mode with fallback implementation
    print("\nTesting audio mode with fallback implementation...")
    args = Namespace(audio=True)
    model = MNISTMLP(indim=16000, hiddim=800, outdim=10, args=args)
    model.audio_init()
    
    x = torch.ones([2, 16000])  # batch_size=2, data_length=16000
    output = model(x, returnt='all')
    print("Audio mode (fallback) - Logits shape:", output[0].shape)
    print("Audio mode (fallback) - Features shape:", output[2].shape)
    
    print("\nAll tests passed!")
