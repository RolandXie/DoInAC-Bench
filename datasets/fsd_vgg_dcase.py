"""Four-domain audio dataset: VGG, FSD50K, DCASE-D2 and DCASE-D3.

The four domains share one ten-class label space and are exposed as four
continual-learning tasks in this order:

    0. VGG
    1. FSD50K
    2. DCASE D2
    3. DCASE D3

Each item is returned as ``(waveform, target, index)``, matching
``librispeech.py``.
"""

from argparse import Namespace
from pathlib import Path
from typing import List, Tuple, Union
import warnings

import librosa
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from backbones.mnistmlp import MNISTMLP
from datasets.utils.continual_dataset import ContinualDataset

try:
    import config_task7 as config
except ImportError:
    class config:
        sample_rate = 16000
        clip_samples = 32000


# This order agrees with the integer targets in DCASE/evaluation_setup/*.tsv.
CLASS_NAMES = (
    "alarm",
    "baby_cry",
    "dog_bark",
    "engine",
    "fire",
    "footsteps",
    "knock",
    "phone_ringbell",
    "piano",
    "speech",
)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Dataset-specific names mapped into the common ten-class label space.
VGG_LABEL_MAP = {name: name for name in CLASS_NAMES}
FSD50K_LABEL_MAP = {
    "Alarm": "alarm",
    "Baby_cry": "baby_cry",
    "Bark": "dog_bark",
    "Engine": "engine",
    "Fire": "fire",
    "Footsteps": "footsteps",
    "Knock": "knock",
    "Phone": "phone_ringbell",
    "Piano": "piano",
    "Speech": "speech",
}
DCASE_LABEL_MAP = {
    "alarm": "alarm",
    "baby": "baby_cry",
    "dog": "dog_bark",
    "engine": "engine",
    "fire": "fire",
    "footsteps": "footsteps",
    "knock": "knock",
    "phone": "phone_ringbell",
    "piano": "piano",
    "speech": "speech",
}

DOMAINS_1 = ["VGG", "FSD50K", "D2", "D3"]

DOMAINS_2 = ["FSD50K", "VGG", "D3", "D2"]

def pad_sequence(audio: np.ndarray, max_len: int) -> np.ndarray:
    """Zero-pad or truncate a mono waveform to exactly ``max_len`` samples."""
    if len(audio) < max_len:
        return np.pad(audio, (0, max_len - len(audio)))
    return audio[:max_len]


class DILDatasetInc_VGGFSDDCASE(Dataset):
    """Lazy audio dataset for one split of one of the four domains."""

    # DOMAINS = ("VGG", "FSD50K", "D2", "D3")
    DOMAINS = [
        DOMAINS_1,
        DOMAINS_2,
    ]

    def __init__(
        self,
        train: bool,
        domain_name: Union[int, str],
        domain_group: int = 0,
        data_root: Union[str, Path, None] = None,
        classes_num: int = 10,
    ) -> None:
        if classes_num != len(CLASS_NAMES):
            raise ValueError(
                f"This dataset has {len(CLASS_NAMES)} shared classes, "
                f"but classes_num={classes_num}."
            )

        self.train = train
        self.domain_name = self._resolve_domain(domain_name, domain_group)
        self.data_root = Path(data_root or Path(__file__).resolve().parent)
        self.classes_num = classes_num

        samples = self._read_samples()
        self.file_list: List[Path] = []
        self.label_list: List[int] = []
        missing_files: List[Path] = []

        for file_path, target in samples:
            if file_path.is_file():
                self.file_list.append(file_path)
                self.label_list.append(target)
            else:
                missing_files.append(file_path)

        if missing_files:
            preview = ", ".join(str(path) for path in missing_files[:3])
            warnings.warn(
                f"{len(missing_files)} files listed for {self.domain_name} "
                f"were not found (first entries: {preview}).",
                RuntimeWarning,
            )

        split = "train" if train else "test"
        print(
            f"Loaded {len(self.file_list)} {split} samples "
            f"from domain {self.domain_name}"
        )

    @classmethod
    def _resolve_domain(cls, domain_name: Union[int, str], domain_group=0) -> str:
        if isinstance(domain_name, int):
            if not 0 <= domain_name < len(cls.DOMAINS[domain_group]):
                raise ValueError(
                    f"domain_name must be in [0, {len(cls.DOMAINS) - 1}]"
                )
            return cls.DOMAINS[domain_group][domain_name]

        normalized = domain_name.strip()
        for valid_name in cls.DOMAINS[domain_group]:
            if normalized.casefold() == valid_name.casefold():
                return valid_name
        raise ValueError(
            f"Unknown domain {domain_name!r}; expected one of {cls.DOMAINS}."
        )

    def _read_samples(self) -> List[Tuple[Path, int]]:
        if self.domain_name == "VGG":
            return self._read_vgg()
        if self.domain_name == "FSD50K":
            return self._read_fsd50k()
        return self._read_dcase()

    @staticmethod
    def _target(raw_label: str, label_map: dict) -> int:
        try:
            canonical_label = label_map[raw_label]
        except KeyError as exc:
            raise ValueError(f"Unknown audio label: {raw_label!r}") from exc
        return CLASS_TO_IDX[canonical_label]

    def _read_vgg(self) -> List[Tuple[Path, int]]:
        vgg_root = self.data_root / "VGG"
        metadata = pd.read_csv(vgg_root / "manifest.csv")
        split = "train" if self.train else "test"
        metadata = metadata.loc[metadata["split"] == split]

        return [
            (
                vgg_root / row.wav_path,
                self._target(row.mapped_label, VGG_LABEL_MAP),
            )
            for row in metadata.itertuples(index=False)
        ]

    def _read_fsd50k(self) -> List[Tuple[Path, int]]:
        fsd_root = self.data_root / "FSD50K"
        split = "dev" if self.train else "eval"
        metadata = pd.read_csv(
            fsd_root / "collection" / f"collection_{split}.csv"
        )

        return [
            (
                fsd_root / row.fname,
                self._target(row.coarse_category, FSD50K_LABEL_MAP),
            )
            for row in metadata.itertuples(index=False)
        ]

    def _read_dcase(self) -> List[Tuple[Path, int]]:
        dcase_root = self.data_root / "DCASE"
        split = "train" if self.train else "test"
        metadata = pd.read_csv(
            dcase_root / "evaluation_setup" / f"{split}.tsv",
            sep="\t",
            header=None,
            names=("audio_path", "label", "domain", "target"),
            dtype={"audio_path": str, "label": str, "domain": str},
        )
        metadata = metadata.loc[metadata["domain"] == self.domain_name]

        return [
            (
                dcase_root / row.audio_path,
                self._target(row.label, DCASE_LABEL_MAP),
            )
            for row in metadata.itertuples(index=False)
        ]

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        file_path = self.file_list[idx]
        target = self.label_list[idx]

        try:
            audio, _ = librosa.load(
                file_path,
                sr=config.sample_rate,
                mono=True,
            )
        except Exception as exc:
            warnings.warn(
                f"Could not load {file_path}: {exc}; returning silence.",
                RuntimeWarning,
            )
            audio = np.zeros(config.clip_samples, dtype=np.float32)

        audio = pad_sequence(audio, config.clip_samples)
        return audio.astype(np.float32), target, idx


class VGGFSD50KDCASE(ContinualDataset):
    """Continual dataset whose tasks are the four source domains."""

    NAME = "vgg-fsd50k-dcase"
    N_CLASSES_PER_TASK = len(CLASS_NAMES)
    N_TASKS = len(DOMAINS_1)
    INDIM = (32000,)
    INDIM_SPEC = (1, 201, 64)
    MAX_N_SAMPLES_PER_TASK = 60000


    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.setup_loaders()

    def get_data_loaders(
        self,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        current_train = self.train_loaders[self.i]
        current_test = self.test_loaders[self.i]

        next_train, next_test = None, None
        if self.i + 1 < self.N_TASKS:
            next_train = self.train_loaders[self.i + 1]
            next_test = self.test_loaders[self.i + 1]

        return current_train, current_test, next_train, next_test

    def setup_loaders(self) -> None:
        self.train_loaders, self.test_loaders = [], []
        data_root = getattr(self.args, "data_root", None)

        for domain_idx in range(self.N_TASKS):
            train_dataset = DILDatasetInc_VGGFSDDCASE(
                train=True,
                domain_name=domain_idx,
                data_root=data_root,
                domain_group=self.args.domain_group,
                classes_num=self.N_CLASSES_PER_TASK,
            )
            test_dataset = DILDatasetInc_VGGFSDDCASE(
                train=False,
                domain_name=domain_idx,
                data_root=data_root,
                domain_group=self.args.domain_group,
                classes_num=self.N_CLASSES_PER_TASK,
            )

            self.train_loaders.append(
                DataLoader(
                    train_dataset,
                    batch_size=self.args.batch_size,
                    shuffle=True,
                    num_workers=self.args.num_workers,
                )
            )
            self.test_loaders.append(
                DataLoader(
                    test_dataset,
                    batch_size=self.args.batch_size,
                    shuffle=False,
                    num_workers=self.args.num_workers,
                )
            )

    @staticmethod
    def get_backbone():
        backbone = MNISTMLP(
            VGGFSD50KDCASE.INDIM_SPEC,
            100,
            VGGFSD50KDCASE.N_CLASSES_PER_TASK,
            Namespace(is_audio=True),
        )
        backbone.audio_init()
        return backbone

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_epochs():
        return 1

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_batch_size() -> int:
        return 128

    @staticmethod
    def get_minibatch_size() -> int:
        return VGGFSD50KDCASE.get_batch_size()


def get_datasets(
    domain_name: Union[int, str],
    domain_group: Union[int, str],
    data_root: Union[str, Path, None] = None,
) -> Tuple[DILDatasetInc_VGGFSDDCASE, DILDatasetInc_VGGFSDDCASE]:
    """Create the train/test pair for a domain index or domain name."""
    train_dataset = DILDatasetInc_VGGFSDDCASE(
        train=True,
        domain_name=domain_name,
        domain_group=domain_group,
        data_root=data_root,
    )
    test_dataset = DILDatasetInc_VGGFSDDCASE(
        train=False,
        domain_name=domain_name,
        domain_group=domain_group,
        data_root=data_root,
    )
    return train_dataset, test_dataset

if __name__ == "__main__":
    for domain in DILDatasetInc_VGGFSDDCASE.DOMAINS[0]:
        train_dataset, test_dataset = get_datasets(domain, 1, "/home/wakamatsu/DataSets2/FSD_VGG_DCASE")
        print(
            f"{domain}: train={len(train_dataset)}, test={len(test_dataset)}"
        )
