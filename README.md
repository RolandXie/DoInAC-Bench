# DoInAC-Bench

Official implementation of **DoInAC-Bench: A Domain-agnostic Incremental Audio Classification Benchmark**.

## Overview

DoInAC-Bench is a unified benchmark and evaluation framework for domain-incremental audio classification (DIAC).

The benchmark contains four complementary domain-shift scenarios:

* **Single-Factor**: individual acoustic transformations are independently applied to construct different domains.
* **Multi-Factor**: multiple acoustic transformations are progressively accumulated across domains.
* **Inter-Dataset**: domains are constructed from different audio datasets with a shared label space.
* **Intra-Dataset**: domains are constructed from different subsets of the same dataset.

## Repository Structure

```text
• UDIL/
  ├── backbones/      # Backbone architectures (CNN14, ResNet18, MNIST-MLP, etc.)
  ├── datasets/       # Dataset definitions and data-loading utilities
  ├── main/           # Program entry point and continual-learning training pipeline
  ├── models/         # Baselines: UDIL、ADIL、ER、DER、LwF、Joint、Fine-tune
  ├── scheduler/      # Learning-rate schedulers
  ├── scripts/        # Training and evaluation scripts for different experiments
  ├── utils/          # Common utilities for arguments, replay buffers, losses, logging, and metrics
  ├── tests/          # Checkpoint validation and experiment reproduction tests
  ├── data/           # Dataset Meta files
  ├── fig/            # Documentation and experimental-result figures
  ├── eval*.sh        # Model evaluation launcher scripts
  └── README.md       # Project overview, installation, and usage instructions
```
  
## Installation

Clone the repository:

```bash
git clone https://github.com/RolandXie/DoInAC-Bench.git
cd DoInAC-Bench
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

The benchmark is constructed from publicly available audio datasets.

Metadata, data splits, label mappings, and preprocessing instructions will be provided for reproducing the four benchmark scenarios.

Detailed instructions can be found in:

```text
datasets/
```

## Training

Example:

```bash
python train.py --config configs/example.yaml
```

More experimental configurations and scripts will be provided in the `configs/` and `scripts/` directories.

## Evaluation

DoInAC-Bench evaluates domain-incremental learning methods using metrics including:

* Average Accuracy (AA)
* Relative Forgetting Rate (rFR)
* Knowledge Transfer (KT)

## Supported Methods

The current implementation includes representative DIAC and continual-learning baselines, including:

* Sequential Fine-Tuning (SeqFT)
* Experience Replay (ER)
* DER++
* Learning without Forgetting (LwF)
* ADIL
* UDIL
* Joint Training

## License

Please refer to the licenses of the original datasets when using the benchmark data.

The source code in this repository is released under the LICENSE provided in this repository.
