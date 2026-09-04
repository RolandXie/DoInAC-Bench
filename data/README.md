



TUT2018在此处下载
https://zenodo.org/records/1228142

VGGSound在此处下载
https://huggingface.co/datasets/Loie/VGGSound/tree/main
可以依据metafile选出对应的数据


librispeech 在此处下载: "https://www.openslr.org/12"
同时我们使用: "https://github.com/vinceasvp/FCAC 对数据进行组织"


```text
AudioDatasets/
  ├── scenario_01_librispeech/
  │   └── LibriSpeech/
  │       ├── libri_none_noise/                         # 干净语音，75,464 WAV
  │       │   └── <speaker>_<index>.wav
  │       │
  │       ├── MUSAN/                                   # 独立噪声因素
  │       │   ├── libri_noise_0_5/                     # 75,464
  │       │   ├── libri_noise_5_10/                    # 75,464
  │       │   └── libri_noise_10_20/                   # 75,464
  │       │
  │       ├── TUT/                                     # 独立声景因素
  │       │   ├── libri_scene_indoor/                  # 75,464
  │       │   ├── libri_scene_outdoor/                 # 75,464
  │       │   └── libri_scene_transportation/          # 75,464
  │       │
  │       ├── RIR/                                     # 独立空间响应因素
  │       │   ├── libri_none_noise_corridor_stairway/
  │       │   ├── libri_none_noise_large_hall/
  │       │   ├── libri_none_noise_lecture_classroom/
  │       │   ├── libri_none_noise_office_meeting/
  │       │   └── libri_none_noise_small_room/
  │       │
  │       ├── MIC/                                     # 独立麦克风因素
  │       │   ├── libri_none_noise_mic_condenser/
  │       │   ├── libri_none_noise_mic_dynamic/
  │       │   └── libri_none_noise_mic_ribbon/
  │       │
  │       ├── EncodeC/                                 # 独立编码带宽因素
  │       │   ├── libri_none_noise_encode_bd_1_5khz/
  │       │   ├── libri_none_noise_encode_bd_3khz/
  │       │   └── libri_none_noise_encode_bd_6khz/
  │       │
  │       ├── MUSAN_TUT/                               # 相关：噪声 + 声景
  │       │   ├── indoor/
  │       │   ├── outdoor/
  │       │   └── transportation/
  │       │
  │       ├── MUSAN_TUT_RIR/                           # 相关：噪声 + 声景 + 空间
  │       │   ├── Large_Hall/
  │       │   ├── Office_Meeting/
  │       │   └── Small_Room/
  │       │
  │       ├── MUSAN_TUT_RIR_MIC/                       # 相关：噪声 + 声景 + 空间 + 麦克风
  │       │   ├── mic_condenser/
  │       │   ├── mic_dynamic/
  │       │   └── mic_ribbon/
  │       │
  │       └── MUSAN_TUT_RIR_MIC_EncodeC/               # 相关：噪声 + 声景 + 空间 + 麦克风 + 编码带宽
  │           ├── 1_5/
  │           ├── 3/
  │           └── 6/
  │
  ├── scenario_02_tut_2018/
  │   └── TUT-urban-acoustic-scenes-2018/
  │       └── audio/                                   # 8,640 WAV，文件扁平存放
  │
  └── scenario_03_vgg_fsd50k_dcase/
      ├── VGG/
      │   ├── train_chunk/                             # 46,799 WAV
      │   └── test_chunk/                              # 3,390 WAV
      │
      ├── FSD50K/
      │   ├── dev_chunk/                               # 7,627 WAV
      │   └── eval_chunk/                              # 6,238 WAV
      │
      └── DCASE/
          └── audio/
              ├── train/
              │   └── output/                          # 4,763 WAV
              └── test/
                  └── output/                          # 2,051 WAV
```
