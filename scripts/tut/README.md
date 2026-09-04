# TUT Urban Acoustic Scenes 2018 experiments

These scripts run the six-city domain-incremental dataset implemented in
`datasets/tut.py` with the project's supported continual-learning methods.

The scripts use the dataset at:

```text
/home/wakamatsu/DataSets3/TUT/TUT-urban-acoustic-scenes-2018-development/
├── audio/
└── evaluation_setup/
    ├── fold1_train.txt
    └── fold1_evaluate.txt
```

Each script contains one direct `python -m main.main` command, following the
same format as the scripts for the other datasets. To change an experiment,
edit the corresponding command directly. For example:

```bash
scripts/tut/tut_finetune.sh
```
