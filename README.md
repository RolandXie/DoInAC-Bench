<div align="center">
  
# DoInAC-Bench: A Domain-agnostic Incremental Audio Classification Benchmark
<img width="563" height="90" alt="image" src="https://github.com/user-attachments/assets/ec1d76bc-4c3f-40f0-b0a8-5b76fe915fe1" />
</div>

## Overview
<img width="1030" height="401" alt="image" src="https://github.com/user-attachments/assets/16f08e66-3264-4af4-affc-169923246714" />

DoInAC-Bench is a unified benchmark and evaluation framework for domain-incremental audio classification (DIAC).

The benchmark contains four complementary domain-shift scenarios:

* **Single-Factor**: individual acoustic transformations are independently applied to construct different domains.
* **Multi-Factor**: multiple acoustic transformations are progressively accumulated across domains.
* **Inter-Dataset**: domains are constructed from different audio datasets with a shared label space.
* **Intra-Dataset**: domains are constructed from different subsets of the same dataset.

## Repository Structure

```text
DoInAC-Bench/
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

Metadata, data splits, label mappings, and preprocessing instructions are provided for reproducing the four benchmark scenarios.

Detailed data preparation instructions can be found [here](./data/README.md).


## Training

Example:

```bash
bash scripts/librispeech_adil.sh
```

More experimental configurations and scripts will be provided in the [scripts](./scripts) directories.

## Evaluation
<img width="1063" height="307" alt="image" src="https://github.com/user-attachments/assets/ed2c3d08-2090-4997-ad60-2843d23e8d71" />

<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/f3b94919-97f5-417c-a336-b7ef4868a991" />
<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/13dd407a-41fd-46f9-a9fc-4e29e2f9b5de" />


DoInAC-Bench evaluates domain-incremental learning methods using metrics including:

* Average Accuracy (AA)
* Relative Forgetting Rate (rFR)
* Knowledge Transfer (KT)

## Supported Methods

The current implementation includes representative DIAC and continual-learning baselines, including:

* SeqFT
* ER
* DER++
* LwF
* ADIL
* UDIL
* Joint Training

## License

Please refer to the licenses of the original datasets when using the benchmark data.

The source code in this repository is released under the LICENSE provided in this repository.
