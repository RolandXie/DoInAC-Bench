# Data Preparation

This document provides instructions for preparing the datasets used in **DoInAC-Bench**. The subdirectories in this folder contain the corresponding **metadata files** for each dataset and benchmark scenario.

## Single-Factor & Multi-Factor Scenarios

The **Single-Factor** and **Multi-Factor** scenarios are constructed based on **LibriSpeech**.

### Original Dataset

LibriSpeech can be downloaded from [here](https://www.openslr.org/12)

We use the data organization and preprocessing pipeline from [here](https://github.com/vinceasvp/FCAC)

The processed dataset used in our experiments is available at [here](https://github.com/vinceasvp/FCAC)

After preprocessing, the dataset should be organized as follows:
```text
Single-Factor & Multi-Factor/
    └── LibriSpeech/
        ├── libri_none_noise/                         # Clean speech, 75,464 WAV files
        │   └── <speaker>_<index>.wav
        │
        ├── MUSAN/                                   # Independent noise factor
        │   ├── libri_noise_0_5/                     # 75,464
        │   ├── libri_noise_5_10/                    # 75,464
        │   └── libri_noise_10_20/                   # 75,464
        │
        ├── TUT/                                     # Independent acoustic-scene factor
        │   ├── libri_scene_indoor/                  # 75,464
        │   ├── libri_scene_outdoor/                 # 75,464
        │   └── libri_scene_transportation/          # 75,464
        │
        ├── RIR/                                     # Independent room-response factor
        │   ├── libri_none_noise_corridor_stairway/
        │   ├── libri_none_noise_large_hall/
        │   ├── libri_none_noise_lecture_classroom/
        │   ├── libri_none_noise_office_meeting/
        │   └── libri_none_noise_small_room/
        │
        ├── MIC/                                     # Independent microphone factor
        │   ├── libri_none_noise_mic_condenser/
        │   ├── libri_none_noise_mic_dynamic/
        │   └── libri_none_noise_mic_ribbon/
        │
        ├── EncodeC/                                 # Independent codec/bandwidth factor
        │   ├── libri_none_noise_encode_bd_1_5khz/
        │   ├── libri_none_noise_encode_bd_3khz/
        │   └── libri_none_noise_encode_bd_6khz/
        │
        ├── MUSAN_TUT/                               # Noise + acoustic scene
        │   ├── indoor/
        │   ├── outdoor/
        │   └── transportation/
        │
        ├── MUSAN_TUT_RIR/                           # Noise + acoustic scene + room response
        │   ├── Large_Hall/
        │   ├── Office_Meeting/
        │   └── Small_Room/
        │
        ├── MUSAN_TUT_RIR_MIC/                       # Noise + scene + room response + microphone
        │   ├── mic_condenser/
        │   ├── mic_dynamic/
        │   └── mic_ribbon/
        │
        └── MUSAN_TUT_RIR_MIC_EncodeC/               # Noise + scene + room response + microphone + codec
            ├── 1_5/
            ├── 3/
            └── 6/
```

The independently transformed subsets are used to construct the **Single-Factor** scenario, while the progressively accumulated transformations are used to construct the **Multi-Factor** scenario.

---

## Intra-Dataset Scenario

The **Intra-Dataset** scenario is constructed using the **TUT Urban Acoustic Scenes 2018** dataset.

The original dataset can be downloaded from [here](https://zenodo.org/records/1228142)

After downloading and extracting the dataset, the expected directory structure is:
```text
Intra-Dataset/
    └── TUT-urban-acoustic-scenes-2018/
        └── audio/                                   # 8,640 WAV files stored in a flat directory
```

The metadata files provided in this repository specify the samples and domain assignments used by DoInAC-Bench.

---

## Inter-Dataset Scenario

The **Inter-Dataset** scenario is constructed from four data sources: **VGGSound**, **FSD50K**, **DCASE_D2**, and **DCASE_D3**.

### VGGSound

VGGSound can be obtained from [here](https://huggingface.co/datasets/Loie/VGGSound/tree/main)

The metadata provided in this repository can be used to select the samples required by DoInAC-Bench.

### FSD50K

FSD50K can be downloaded [from here](https://zenodo.org/records/4060432)

### DCASE_D2 / DCASE_D3

The DCASE_D2 and DCASE_D3 datasets can be obtained from the [DCASE 2026 challenge task 7 page](https://dcase.community/challenge2026/task-domain-agnostic-incremental-learning-for-audio-classification)

The metadata provided with DoInAC-Bench defines the class mapping and sample selection used to align these datasets to the common label space.

After data preparation, the **Inter-Dataset** scenario should follow the directory structure below:

<img width="1063" height="774" alt="Inter-Dataset directory structure" src="https://github.com/user-attachments/assets/6f98f20b-06dc-4b4f-8a74-057d4ccda4e3" />

```text
Inter-Dataset/
    ├── VGG/
    │   ├── train_chunk/                             # 46,799 WAV files
    │   └── test_chunk/                              # 3,390 WAV files
    │
    ├── FSD50K/
    │   ├── dev_chunk/                               # 7,627 WAV files
    │   └── eval_chunk/                              # 6,238 WAV files
    │
    └── DCASE/
        └── audio/
            ├── train/
            │   └── output/                          # 4,763 WAV files
            └── test/
                └── output/                          # 2,051 WAV files
```

Please ensure that the prepared datasets follow the directory structures above before running the corresponding training or evaluation scripts.
