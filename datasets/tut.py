"""TUT Urban Acoustic Scenes 2018 dataset for domain-incremental learning.

The six recording cities are exposed as continual-learning tasks.  All tasks
share the same ten acoustic-scene classes and use the official fold-1 split.
Each item is returned as ``(waveform, target, index)``, consistently with the
other audio datasets in this directory.
"""

from argparse import Namespace
from pathlib import Path
from typing import List, Tuple, Union
import csv
import warnings

import librosa
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from backbones.mnistmlp import MNISTMLP
from datasets.utils.continual_dataset import ContinualDataset

try:
    import config_task7 as config
except ImportError:
    class config:
        sample_rate = 16000


# TUT Urban Acoustic Scenes 2018 recordings are 10 seconds long.  Keep the
# complete recording after resampling instead of cropping it to two seconds.
CLIP_SECONDS = 10
FULL_CLIP_SAMPLES = config.sample_rate * CLIP_SECONDS


CLASS_NAMES = (
    "airport",
    "bus",
    "metro",
    "metro_station",
    "park",
    "public_square",
    "shopping_mall",
    "street_pedestrian",
    "street_traffic",
    "tram",
)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

CITIES = (
    "barcelona",
    "helsinki",
    "london",
    "paris",
    "stockholm",
    "vienna",
)
CITIES_1 = (
    "barcelona",
)
CITIES_2 = (
    "helsinki",
)
CITIES_3 = (
    "london",
)
CITIES_4 = (
    "paris",
)
CITIES_5 = (
    "stockholm",
)
CITIES_6 = (
    "vienna",
)

def pad_sequence(audio: np.ndarray, max_len: int) -> np.ndarray:
    """Zero-pad short waveforms while preserving complete longer recordings."""
    if len(audio) < max_len:
        return np.pad(audio, (0, max_len - len(audio)))
    return audio


class DILDatasetInc_TUT(Dataset):
    """One train/test split for one city domain of TUT Acoustic Scenes 2018."""

    DOMAINS = CITIES
    DATASET_DIR = "TUT-urban-acoustic-scenes-2018-development"

    def __init__(
        self,
        train: bool,
        domain_name: Union[int, str],
        data_root: Union[str, Path, None] = None,
        classes_num: int = 10,
    ) -> None:
        if classes_num != len(CLASS_NAMES):
            raise ValueError(
                f"TUT 2018 has {len(CLASS_NAMES)} scene classes, "
                f"but classes_num={classes_num}."
            )

        self.train = train

        # Return the domain name;
        self.domain_name = self._resolve_domain(domain_name)

        # Return the data root Path;
        self.data_root = self._resolve_data_root(data_root)

        self.classes_num = classes_num

        split_file = self.data_root / "evaluation_setup" / (
            "fold1_train.txt" if train else "fold1_evaluate.txt"
        )
        if not split_file.is_file():
            raise FileNotFoundError(f"TUT split file not found: {split_file}")

        self.file_list: List[Path] = []
        self.label_list: List[int] = []
        missing_files: List[Path] = []

        with split_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row_number, row in enumerate(reader, start=1):
                if not row or not row[0].strip():
                    continue
                if len(row) < 2:
                    raise ValueError(
                        f"Missing label in {split_file} at line {row_number}."
                    )

                relative_path = Path(row[0].strip())
                label = row[1].strip()
                city = self._city_from_filename(relative_path.name)
                if city != self.domain_name:
                    continue
                if label not in CLASS_TO_IDX:
                    raise ValueError(
                        f"Unknown scene label {label!r} in {split_file} "
                        f"at line {row_number}."
                    )

                file_path = self.data_root / relative_path
                if file_path.is_file():
                    self.file_list.append(file_path)
                    self.label_list.append(CLASS_TO_IDX[label])
                else:
                    missing_files.append(file_path)

        if missing_files:
            preview = ", ".join(str(path) for path in missing_files[:3])
            warnings.warn(
                f"{len(missing_files)} listed audio files were not found "
                f"(first entries: {preview}).",
                RuntimeWarning,
            )

        split = "train" if train else "test"
        print(
            f"Loaded {len(self.file_list)} {split} samples "
            f"from TUT city {self.domain_name}"
        )

    @classmethod
    def _resolve_domain(cls, domain_name: Union[int, str]) -> str:
        if isinstance(domain_name, int):
            if not 0 <= domain_name < len(cls.DOMAINS):
                raise ValueError(
                    f"domain_name must be in [0, {len(cls.DOMAINS) - 1}]"
                )
            return cls.DOMAINS[domain_name]

        normalized = domain_name.strip().casefold()
        for city in cls.DOMAINS:
            if normalized == city.casefold():
                return city
        raise ValueError(
            f"Unknown domain {domain_name!r}; expected one of {cls.DOMAINS}."
        )

    @classmethod
    def _resolve_data_root(
        cls, data_root: Union[str, Path, None]
    ) -> Path:
        """Accept either the dataset directory itself or its parent directory."""
        root = Path(data_root) if data_root is not None else Path(__file__).parent
        root = root.expanduser().resolve()
        if (root / "evaluation_setup").is_dir():
            return root
        return root / cls.DATASET_DIR

    @staticmethod
    def _city_from_filename(filename: str) -> str:
        # Official convention: scene-city-location-segment-device.wav
        parts = filename.split("-")
        if len(parts) < 5:
            raise ValueError(f"Unexpected TUT audio filename: {filename!r}")
        return parts[1].casefold()

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
            audio = np.zeros(FULL_CLIP_SAMPLES, dtype=np.float32)

        audio = pad_sequence(audio, FULL_CLIP_SAMPLES)
        return audio.astype(np.float32), target, idx


class TUTUrbanAcousticScenes2018(ContinualDataset):
    """Six-city domain-incremental version of TUT Acoustic Scenes 2018."""

    NAME = "tut-urban-acoustic-scenes-2018"
    N_CLASSES_PER_TASK = len(CLASS_NAMES)
    INDIM = (FULL_CLIP_SAMPLES,)
    INDIM_SPEC = (1, FULL_CLIP_SAMPLES // 160 + 1, 64)
    MAX_N_SAMPLES_PER_TASK = 60000

    domain_group=[
        CITIES,
        CITIES_1,
        CITIES_2,
        CITIES_3,
        CITIES_4,
        CITIES_5,
        CITIES_6
    ]

    def __init__(self, args: Namespace) -> None:
        self.N_TASKS = len(self.domain_group[args.domain_group])
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


        for domain_idx in self.domain_group[self.args.domain_group]:
            train_dataset = DILDatasetInc_TUT(
                train=True,
                domain_name=domain_idx,
                data_root=data_root,
                classes_num=self.N_CLASSES_PER_TASK,
            )
            test_dataset = DILDatasetInc_TUT(
                train=False,
                domain_name=domain_idx,
                data_root=data_root,
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
            TUTUrbanAcousticScenes2018.INDIM_SPEC,
            100,
            TUTUrbanAcousticScenes2018.N_CLASSES_PER_TASK,
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
        return TUTUrbanAcousticScenes2018.get_batch_size()


def get_datasets(
    domain_name: Union[int, str],
    data_root: Union[str, Path, None] = None,
) -> Tuple[DILDatasetInc_TUT, DILDatasetInc_TUT]:
    """Create the official fold-1 train/test pair for a city domain."""
    train_dataset = DILDatasetInc_TUT(
        train=True,
        domain_name=domain_name,
        data_root=data_root,
    )
    test_dataset = DILDatasetInc_TUT(
        train=False,
        domain_name=domain_name,
        data_root=data_root,
    )
    return train_dataset, test_dataset


if __name__ == "__main__":
    for domain in CITIES:
        train_dataset, test_dataset = get_datasets(domain)
        print(f"{domain}: train={len(train_dataset)}, test={len(test_dataset)}")
