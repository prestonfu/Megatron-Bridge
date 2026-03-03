#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pretrain with YAML Configuration and CLI Overrides

This script demonstrates how to use YAML configuration files and command-line
overrides for more complex configuration overrides.

Supports a custom `data_sources` YAML section for specifying dataset paths with
glob patterns and per-source weights. Example YAML:

    data_sources:
      - path: "/mnt/pretrain_data/fineweb_edu/processed_data_*_text_document"
        weight: 0.7
      - path: "/mnt/pretrain_data/other_dataset/data_text_document"
        weight: 0.3

Each `path` can be a glob pattern (matched against .bin files) or a direct prefix.
Weights are distributed equally across all shards within a source.

Usage:
    With default config file:
        torchrun --nproc_per_node=8 02_pretrain_with_yaml.py

    With custom config file:
        torchrun --nproc_per_node=8 02_pretrain_with_yaml.py \
            --config-file conf/nemotron_3_nano_pretrain.yaml

    With command-line overrides:
        torchrun --nproc_per_node=8 02_pretrain_with_yaml.py \
            train.train_iters=5000 \
            train.global_batch_size=256

    Combining YAML and CLI (CLI takes precedence):
        torchrun --nproc_per_node=8 02_pretrain_with_yaml.py \
            --config-file conf/nemotron_3_nano_pretrain.yaml \
            train.train_iters=10000

Configuration Priority (highest to lowest):
    1. Command-line overrides (highest)
    2. YAML config file
    3. Base recipe defaults (lowest)

See conf/ directory for example YAML configurations.
For a pure Python usage see 00_quickstart_pretrain.py.
"""

import argparse
import glob
import logging
import math
from pathlib import Path
from typing import Tuple

import yaml

from megatron.bridge.recipes.nemotronh import nemotron_3_nano_pretrain_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides


logger = logging.getLogger(__name__)

logging.getLogger("megatron.core.datasets.indexed_dataset").setLevel(logging.WARNING)
logging.getLogger("megatron.core.datasets.gpt_dataset").setLevel(logging.WARNING)
logging.getLogger("megatron.core.datasets.blended_megatron_dataset_builder").setLevel(logging.WARNING)
logging.getLogger("megatron.core.distributed").setLevel(logging.WARNING)

# Default config file location
SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_CONFIG_FILE = SCRIPT_DIR / "conf" / "nemotron_3_nano_pretrain.yaml"


def resolve_data_sources(data_sources: list[dict]) -> list[tuple[str, float]]:
    """Resolve data_sources entries (with glob support) into a flat blend list.

    Each entry in data_sources should have:
        - path: A prefix path or glob pattern (matched against .bin files).
        - weight: Weight for this source. Distributed equally across matched shards.

    Returns:
        List of (prefix, per_shard_weight) tuples suitable for cfg.dataset.blend.
    """
    blend = []
    for source in data_sources:
        pattern = source["path"]
        weight = float(source.get("weight", 1.0))

        # If pattern contains glob characters, resolve against .bin files
        if any(c in pattern for c in "*?["):
            bin_pattern = pattern if pattern.endswith(".bin") else f"{pattern}.bin"
            matched = sorted(glob.glob(bin_pattern))
            if not matched:
                raise FileNotFoundError(f"Glob pattern matched no .bin files: {bin_pattern}")
            prefixes = [f.removesuffix(".bin") for f in matched]
        else:
            # Direct prefix — verify the .bin file exists
            prefix = pattern.removesuffix(".bin")
            if not Path(f"{prefix}.bin").exists():
                raise FileNotFoundError(f"Dataset file not found: {prefix}.bin")
            prefixes = [prefix]

        per_shard_weight = weight / len(prefixes)
        blend.extend((p, per_shard_weight) for p in prefixes)

    return blend


def parse_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Pretrain with YAML configuration and CLI overrides",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help=f"Path to YAML config file (optional). Default: {DEFAULT_CONFIG_FILE}",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Separate known args from CLI overrides
    args, cli_overrides = parser.parse_known_args()
    return args, cli_overrides


def extract_custom_keys(raw_yaml: dict) -> dict:
    """Pop custom keys that aren't part of the Megatron ConfigContainer.

    Returns a dict of extracted values. Modifies raw_yaml in place.
    """
    custom = {}

    # data_sources (top-level)
    custom["data_sources"] = raw_yaml.pop("data_sources", None)

    # checkpoint.enabled / checkpoint.save_on_last_step
    ckpt = raw_yaml.get("checkpoint", {})
    if isinstance(ckpt, dict):
        custom["checkpoint_enabled"] = ckpt.pop("enabled", True)
        custom["save_on_last_step"] = ckpt.pop("save_on_last_step", True)

    return custom


def main() -> None:
    """Run pretraining with YAML configuration and CLI overrides."""
    args, cli_overrides = parse_args()

    # Load base configuration from recipe
    config: ConfigContainer = nemotron_3_nano_pretrain_config()

    custom = {}
    config_filepath = args.config_file
    if config_filepath:
        with open(config_filepath) as f:
            raw_yaml = yaml.safe_load(f)

        # Extract custom keys before handing YAML to process_config_with_overrides
        custom = extract_custom_keys(raw_yaml)

        if custom["data_sources"]:
            resolved = resolve_data_sources(custom["data_sources"])
            prefixes, weights = zip(*resolved)
            config.dataset.blend = (list(prefixes), list(weights))
            logger.info(f"Resolved {len(prefixes)} dataset shards from data_sources")

        if not custom.get("checkpoint_enabled", True):
            config.checkpoint.save = None
            config.checkpoint.save_interval = None
            logger.info("Checkpointing disabled")

        # Write stripped YAML so process_config_with_overrides doesn't see unknown keys
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(raw_yaml, tmp)
            config_filepath = tmp.name

    config = process_config_with_overrides(
        config,
        config_filepath=config_filepath,
        cli_overrides=cli_overrides or None,
    )

    # Save on last step: set save_interval to land exactly on train_iters
    if custom.get("save_on_last_step", True) and custom.get("checkpoint_enabled", True):
        train_iters = config.train.train_iters
        save_interval = config.checkpoint.save_interval
        if save_interval and train_iters and (train_iters % save_interval != 0):
            config.checkpoint.save_interval = math.gcd(save_interval, train_iters)
            logger.info(
                f"Adjusted save_interval to {config.checkpoint.save_interval} "
                f"to checkpoint on last step ({train_iters})"
            )

    # Start pretraining
    pretrain(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
