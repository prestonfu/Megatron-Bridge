# Nemotron 3 Nano Recipes with Megatron Bridge

This guide shows you how to pretrain and finetune Nemotron 3 Nano (30B-A3B MoE) using Megatron Bridge.

## Quickstart

The fastest way to get started with Megatron Bridge pretraining:

```bash
torchrun --nproc_per_node=8 00_quickstart_pretrain.py
```

This runs Nemotron 3 Nano pretraining with mock data. The model requires TP=4, EP=8 by default.

For finetuning, you first need a checkpoint in Megatron format. Convert from HuggingFace using the `AutoBridge`:

> **Note:** You must be authenticated with Hugging Face to download the model. Run `hf auth login --token $HF_TOKEN` if needed.

```bash
python ../../conversion/convert_checkpoints.py import \
    --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --megatron-path ./checkpoints/nemotron_3_nano
```

Then run finetuning:

```bash
torchrun --nproc_per_node=8 01_quickstart_finetune.py \
    --pretrained-checkpoint ./checkpoints/nemotron_3_nano
```

The [01_quickstart_finetune.py](01_quickstart_finetune.py) recipe finetunes Nemotron 3 Nano using LoRA on the SQuAD dataset by default.

To plug in your own JSONL dataset, swap the dataset config in that script:

```python
from megatron.bridge.training.config import FinetuningDatasetConfig

config.dataset = FinetuningDatasetConfig(
    dataset_root="/path/to/dataset_dir",  # contains training/validation/test jsonl files
    seq_length=config.model.seq_length,
)
```

## Configuration

Megatron Bridge recipes are standard Python scripts, giving you full flexibility in how you configure your training. You can:
1.  Modify the Python scripts directly
2.  Use the framework's YAML-based configuration system
3.  Implement your own configuration management (ArgParse, Hydra, etc.)

### Using Framework YAML Configs

The recipes include optional support for YAML configuration and dot-notation overrides via `ConfigContainer`. This is just one way to manage config; you are free to use other methods.

To use the provided YAML system:

```bash
torchrun --nproc_per_node=8 02_pretrain_with_yaml.py \
    --config-file conf/nemotron_3_nano_pretrain.yaml
```

Understanding the YAML Structure:

YAML files mirror the `ConfigContainer` structure. Each top-level key corresponds to a configuration section (e.g., `dataset`, `train`, `model`, `optimizer`).

Example YAML (`conf/nemotron_3_nano_pretrain.yaml`):

```yaml
# Each section maps to a ConfigContainer field
dataset:                           # GPTDatasetConfig
  data_path: /path/to/training/data
  sequence_length: 8192

train:                             # TrainingConfig
  train_iters: 100
  global_batch_size: 3072

checkpoint:                        # CheckpointConfig
  save: ./checkpoints/nemotron_3_nano
  save_interval: 200

model:                             # Model Provider
  seq_length: 8192                 # Must match data.sequence_length
  tensor_model_parallel_size: 4

optimizer:                         # OptimizerConfig
  lr: 0.0016
```

Command-Line Overrides:

You can override values using dot notation (`section.field=value`):

```bash
torchrun --nproc_per_node=8 02_pretrain_with_yaml.py \
    --config-file conf/nemotron_3_nano_pretrain.yaml \
    train.train_iters=5000 \
    train.global_batch_size=512 \
    optimizer.lr=0.0002
```

Priority order (highest to lowest):
1.  Command-line overrides
2.  YAML config file
3.  Base recipe defaults

### Finetuning Configuration

For more complex finetuning configurations:

```bash
torchrun --nproc_per_node=8 03_finetune_with_yaml.py \
    --config-file conf/nemotron_3_nano_finetune.yaml
```

Example YAML (`conf/nemotron_3_nano_finetune.yaml`):

```yaml
# Each section maps to a ConfigContainer field
dataset:                           # FinetuningDatasetConfig
  data_path: /path/to/finetuning_dataset.jsonl
  seq_length: 2048

train:                             # TrainingConfig
  train_iters: 100
  global_batch_size: 128

checkpoint:                        # CheckpointConfig
  pretrained_checkpoint: /path/to/pretrained/checkpoint
  save: ./checkpoints/nemotron_3_nano_finetuned
  save_interval: 200

peft:                             # PEFT (LoRA config)
  dim: 32     # LoRA rank
  alpha: 32   # LoRA alpha

model:                            # Model Provider
  seq_length: 2048                # Must match data.seq_length

optimizer:                        # OptimizerConfig
  lr: 0.0001
```

Full Finetuning (No LoRA)

```bash
torchrun --nproc_per_node=8 03_finetune_with_yaml.py \
    --peft none \
    train.train_iters=1000
```

## Multi-Node Training

### Direct Slurm with sbatch

For traditional HPC workflows without NeMo-Run:

```bash
# 1. Configure launch_with_sbatch.sh
# Edit SBATCH directives and script variables at the top

# 2. Submit job
sbatch launch_with_sbatch.sh
```

The `launch_with_sbatch.sh` script shows how to:
- Configure Slurm job parameters
- Set up multi-node torchrun
- Use containers (optional)
- Pass arguments to training scripts

### NeMo-Run

For job management and remote launching capabilities:

Prerequisites:

```bash
pip install nemo-run
```

From the Slurm cluster login node:

```bash
python 04_launch_slurm_with_nemo_run.py \
    --script 00_quickstart_pretrain.py \
    --nodes 2 \
    --devices 8 \
    --partition gpu \
    --account my_account
```

From your local machine (SSHTunnel):

```bash
python 04_launch_slurm_with_nemo_run.py \
    --script 00_quickstart_pretrain.py \
    --nodes 2 \
    --devices 8 \
    --partition gpu \
    --account my_account \
    --ssh-tunnel \
    --host my-cluster.example.com \
    --user myusername \
    --remote-job-dir /home/myusername/nemo-runs
```

With custom config:

```bash
python 04_launch_slurm_with_nemo_run.py \
    --script 03_finetune_with_yaml.py \
    --nodes 1 \
    --devices 8 \
    --partition gpu \
    --account my_account \
    --config-file conf/nemotron_3_nano_finetune.yaml
```
