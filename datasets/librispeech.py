from argparse import Namespace
from typing import Tuple

from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pandas as pd
import os
import librosa
import numpy as np

from backbones.mnistmlp import MNISTMLP
from datasets.utils.continual_dataset import ContinualDataset

try:
    import config_task7 as config
except ImportError:
    class config:
        sample_rate = 16000
        clip_samples = 32000 

# | Task  | Factor | domains_list_0 | domains_list_1 | domains_list_2 |
# | ----- | ------ | -------------- | -------------- | -------------- |
# | Task0 | clean  | none_noise     | none_noise     | none_noise     |
# | Task1 | noise  | 10-20          | 5-10           | 0-5            |
# | Task2 | scene  | indoor         | outdoor        | transportation |
# | Task3 | encode | 1.5kHz         | 3kHz           | 6kHz           |
# | Task4 | space  | large_hall     | office_meeting | small_room     |
# | Task5 | mic    | condenser      | dynamic        | ribbon         |

domains_list_0 = [
    "libri_none_noise",
    "MUSAN/libri_noise_10_20",
    "TUT/libri_scene_indoor",
    "RIR/libri_none_noise_small_room",
    "MIC/libri_none_noise_mic_condenser",
    "EncodeC/libri_none_noise_encode_bd_6khz",
]

domains_list_1 = [
    "libri_none_noise",
    "MUSAN/libri_noise_5_10",
    "TUT/libri_scene_outdoor",
    "RIR/libri_none_noise_office_meeting",
    "MIC/libri_none_noise_mic_ribbon",
    "EncodeC/libri_none_noise_encode_bd_3khz",
]

domains_list_2 = [
    "libri_none_noise",
    "MUSAN/libri_noise_0_5",
    "TUT/libri_scene_transportation",
    "RIR/libri_none_noise_large_hall",
    "MIC/libri_none_noise_mic_dynamic",
    "EncodeC/libri_none_noise_encode_bd_1_5khz",
]

domains_list_3 = [
    "libri_none_noise",
    "MUSAN/libri_noise_0_5",
    "MUSAN_TUT/transportation",
    "MUSAN_TUT_RIR/Large_Hall",
    "MUSAN_TUT_RIR_MIC/mic_dynamic",
    "MUSAN_TUT_RIR_MIC_EncodeC/1_5",
]

domains_list_4 = [
    "libri_none_noise",
    "MUSAN/libri_noise_5_10",
    "MUSAN_TUT/outdoor",
    "MUSAN_TUT_RIR/Office_Meeting",
    "MUSAN_TUT_RIR_MIC/mic_ribbon",
    "MUSAN_TUT_RIR_MIC_EncodeC/3",
]

domains_list_5 = [
    "libri_none_noise",
    "MUSAN/libri_noise_10_20",
    "MUSAN_TUT/indoor",
    "MUSAN_TUT_RIR/Small_Room",
    "MUSAN_TUT_RIR_MIC/mic_condenser",
    "MUSAN_TUT_RIR_MIC_EncodeC/6",
]

def to_one_hot(k, classes_num):
    target = np.zeros(classes_num)
    target[k] = 1
    return target


def pad_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0:max_len]  # 截断到 max_len


class DILDatasetInc_Libri(Dataset):
    # 可选 domain 列表
    domains = [
        domains_list_0,
        domains_list_1,
        domains_list_2,
        domains_list_3,
        domains_list_4,
        domains_list_5,
    ]

    audio_folder = "/home/wakamatsu/DataSets2/librispeech_domain_3"
    #train_meta = "/home/wakamatsu/DataSets2/librispeech_domain_3/librispeech_fscil_train_sppr_sub_2.csv"
    #test_meta = "/home/wakamatsu/DataSets2/librispeech_domain_3/librispeech_fscil_test_sppr_sub_2.csv"
    train_meta = "/home/wakamatsu/DataSets2/librispeech_domain_3/librispeech_fscil_train_sppr.csv"
    test_meta = "/home/wakamatsu/DataSets2/librispeech_domain_3/librispeech_fscil_test_sppr.csv"

    def __init__(self, train, domain_group:int=0, domain_name:int=0, label_col="label", classes_num=10):
        """
        Args:
            meta_file (string): Path to the CSV metadata file (train.csv or test.csv)
            audio_folder (string): Base directory containing all domain folders
            domain_name (int): Index of the domain folder to load (e.g., 0 for "libri_none_noise")
            label_col (string): Column name for labels in CSV (default: "label")
            classes_num (int): Number of classes for one-hot encoding
            domain_group (int): Group identifier for the domain
        """
        if train:
            self.meta_file = self.train_meta
        else:
            self.meta_file = self.test_meta
        self.domain_name = self.domains[domain_group][domain_name]
        self.label_col = label_col
        self.classes_num = classes_num

        # 构建完整的音频路径
        self.audio_path = os.path.join(self.audio_folder, self.domain_name)
        
        # 加载元数据 (只读 CSV，不加载音频)
        self.df = pd.read_csv(self.meta_file)
        
        # 只保存文件列表和标签，不加载音频
        self.file_list = []
        self.label_list = []
        
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            file_name = row["filename"]
            label = row[self.label_col]
            
            # 构建文件路径（但不加载）
            file_path = os.path.join(self.audio_path, file_name)
            
            # 只记录存在的文件
            if os.path.exists(file_path):
                self.file_list.append(file_path)
                self.label_list.append(label)
            else:
                print(f"Warning: File not found: {file_path}")
        
        print(f"Loaded {len(self.file_list)} samples from {self.audio_path}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        延迟加载：只在被调用时才加载音频
        """
        # 获取文件路径和标签
        file_path = self.file_list[idx]
        label = self.label_list[idx]
        
        # 提取文件名（用于调试）
        file_name = os.path.basename(file_path)
        
        # 加载音频 (librosa 默认 mono=True)
        try:
            audio, _ = librosa.core.load(file_path, sr=config.sample_rate, mono=True)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            audio = np.zeros(config.clip_samples, dtype=np.float32)
        
        audio = pad_sequence(audio, config.clip_samples)
        
        target = int(label)

        return audio.astype(np.float32), target, idx


class LibriSpeech(ContinualDataset):
    NAME = 'librispeech'
    N_CLASSES_PER_TASK = 100
    N_TASKS = 6
    INDIM = (32000,)
    INDIM_SPEC = (1, 201, 64)
    # INDIM = config.clip_samples
    MAX_N_SAMPLES_PER_TASK = 60000

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.setup_loaders()

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        current_train = self.train_loaders[self.i]
        current_test = self.test_loaders[self.i]

        next_train, next_test = None, None
        if self.i + 1 < self.N_TASKS:
            next_train = self.train_loaders[self.i + 1]
            next_test = self.test_loaders[self.i + 1]

        return current_train, current_test, next_train, next_test

    def setup_loaders(self):
        self.test_loaders, self.train_loaders = [], []
        domain_group = getattr(self.args, 'domain_group', 0)

        for i in range(self.N_TASKS):
            train_dataset = DILDatasetInc_Libri(
                train=True,
                domain_group=domain_group,
                domain_name=i,
                classes_num=self.N_CLASSES_PER_TASK
            )
            test_dataset = DILDatasetInc_Libri(
                train=False,
                domain_group=domain_group,
                domain_name=i,
                classes_num=self.N_CLASSES_PER_TASK
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers
            )

            self.test_loaders.append(test_loader)
            self.train_loaders.append(train_loader)

    @staticmethod
    def get_backbone():
        disc = MNISTMLP(
                   LibriSpeech.INDIM_SPEC,
                   100,
                   LibriSpeech.N_CLASSES_PER_TASK,
                   Namespace({"is_audio":True})
                )
        disc.audio_init()
        return disc

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
        return LibriSpeech.get_batch_size()


def get_datasets(train_meta, test_meta, audio_folder, domain_name, label_col="label", classes_num=10):
    """
    创建训练和测试数据集。
    
    Args:
        train_meta (string): Path to training metadata CSV
        test_meta (string): Path to test metadata CSV
        audio_folder (string): Base directory containing all domain folders
        domain_name (string): Domain folder name (e.g., "libri_none_noise")
        label_col (string): Column name for labels
        classes_num (int): Number of classes
        domain_group (int): Group identifier for the domain
    
    Returns:
        train_dataset, test_dataset: DILDatasetInc instances
    """
    train_dataset = DILDatasetInc_Libri(
        train=True,
        domain_group=0,
        domain_name=domain_name,
        label_col=label_col,
        classes_num=classes_num
    )
    
    test_dataset = DILDatasetInc_Libri(
        train=False,
        domain_group=0,
        domain_name=domain_name,
        label_col=label_col,
        classes_num=classes_num
    )
    
    return train_dataset, test_dataset

# ========== 使用示例 ==========
if __name__ == "__main__":
    # 定义路径
    AUDIO_BASE = "/home/wakamatsu/DataSets2/librispeech_domain_3"
    TRAIN_META = "/home/wakamatsu/DataSets2/librispeech_domain_3/librispeech_fscil_train_sppr_sub.csv"
    TEST_META = "/home/wakamatsu/DataSets2/librispeech_domain_3/librispeech_fscil_test_sppr_sub.csv"
    
    # 选择 domain (例如: "libri_none_noise")
    selected_domain = 0
    
    # 创建训练和测试数据集
    train_dataset, test_dataset = get_datasets(
        train_meta=TRAIN_META,
        test_meta=TEST_META,
        audio_folder=AUDIO_BASE,
        domain_name=selected_domain,
        label_col="label"
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # 测试加载一个样本
    if len(train_dataset) > 0:
        data, label, audio_file = train_dataset[0]
        print(f"Sample: {audio_file}, shape: {data.shape}, label: {label}")
