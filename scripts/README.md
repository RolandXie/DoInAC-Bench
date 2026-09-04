```text
scripts/
  ├── librispeech/                          # Standard LibriSpeech experiments
  │   ├── librispeech_adil.sh               # ADIL training
  │   ├── librispeech_udil.sh               # UDIL training
  │   ├── librispeech_der.sh                # Dark Experience Replay baseline
  │   ├── librispeech_er.sh                 # Experience Replay baseline
  │   ├── librispeech_finetune.sh           # Sequential fine-tuning baseline
  │   ├── librispeech_joint.sh              # Joint-training baseline
  │   └── librispeech_lwf*.sh               # Learning without Forgetting variants
  │
  ├── librispeech_4/                        # LibriSpeech experiments with domain group 4
  ├── librispeech_6/                        # LibriSpeech experiments with domain group 6
  ├── librispeech_7/                        # LibriSpeech experiments with domain group 7
  ├── librispeech_8/                        # LibriSpeech experiments with domain group 8
  ├── librispeech_9/                        # LibriSpeech experiments with domain group 9
  ├── librispeech_10/                       # LibriSpeech experiments with domain group 10
  │   ├── librispeech_adil.sh               # ADIL configuration
  │   ├── librispeech_udil.sh               # UDIL configuration
  │   ├── librispeech_der.sh                # DER configuration
  │   ├── librispeech_er.sh                 # ER configuration
  │   ├── librispeech_finetune.sh           # Fine-tuning configuration
  │   ├── librispeech_joint.sh              # Joint-training configuration
  │   └── librispeech_lwf.sh                # LwF configuration
  │
  ├── librispeech_direct_ft/                # Direct fine-tuning experiments
  │   └── librispeech_finetune_12–20.sh     # Fine-tuning across domain groups 12–20
  │
  ├── tut/                                  # TUT Urban Acoustic Scenes experiments
  │   ├── tut_adil.sh                       # ADIL training
  │   ├── tut_finetune.sh                   # Sequential fine-tuning
  │   ├── tut_joint.sh                      # Joint training
  │   ├── tut_der_*.sh                      # DER buffer-size and coefficient sweeps
  │   ├── tut_er_*.sh                       # ER buffer-size sweeps
  │   ├── tut_lwf_*.sh                      # LwF hyperparameter sweeps
  │   ├── tut_udil_*.sh                     # UDIL buffer-size sweeps
  │   └── README.md                         # TUT experiment instructions
  │
  ├── tut_uppper/                           # Extended TUT fine-tuning experiments
  │   ├── tut_finetune_1–6.sh               # Per-domain-group fine-tuning scripts
  │   └── README.md                         # Experiment instructions
  │
  ├── vgg_fsd_dcase/                        # VGGSound–FSD50K–DCASE experiments
  │   ├── vgg_fsd_dcase_adil.sh             # ADIL training
  │   ├── vgg_fsd_dcase_udil.sh             # UDIL training
  │   ├── vgg_fsd_dcase_der.sh              # DER baseline
  │   ├── vgg_fsd_dcase_er.sh               # ER baseline
  │   ├── vgg_fsd_dcase_lwf.sh              # LwF baseline
  │   ├── vgg_fsd_dcase_joint.sh            # Joint-training baseline
  │   └── vgg_fsd_dcase_finetune.sh         # Sequential fine-tuning baseline
  │
  └── vgg_fsd_dcase_upper/                  # Extended VGG–FSD–DCASE fine-tuning runs
      └── vgg_fsd_dcase_finetune_3–6.sh     # Fine-tuning for selected domain groups
```
