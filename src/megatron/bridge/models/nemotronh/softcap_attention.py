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

"""Attention with softcap support via flash_attn_func for Nemotron-H / Nemotron 3 Nano.

TransformerEngine 2.9.0 wraps flash_attn internally but does not expose the `softcap`
argument.  This module bypasses TEDotProductAttention and calls flash_attn_func directly,
enabling attention logit softcapping (Gemma 2 / nanogpt speedrun record #9/#18 style).

Tradeoffs vs TEDotProductAttention:
  - No context parallelism (CP) support
  - No FP8 support
  - No packed-sequence (THD) support
When attn_logit_softcap is None/0, the caller should keep using TEDotProductAttention.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from flash_attn import flash_attn_func
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec as _base_mamba_stack_spec
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.extensions.transformer_engine import (
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add


class SoftcapFlashAttention(nn.Module):
    """Drop-in replacement for TEDotProductAttention that adds softcap support.

    Calls flash_attn_func directly with softcap=attn_logit_softcap from config.
    Q/K/V arrive in Megatron's sbhd format [sq, b, np, hn] and are transposed
    to flash_attn's bshd format [b, sq, np, hn] internally.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.softcap = getattr(config, 'attn_logit_softcap', None) or 0.0
        self.attention_dropout = (
            attention_dropout if attention_dropout is not None else config.attention_dropout
        )
        head_dim = config.kv_channels
        self.softmax_scale = softmax_scale if softmax_scale is not None else head_dim ** -0.5
        self.log_max_attention_logit = getattr(config, 'log_max_attention_logit', False)
        # Approximation of per-head max logit via log-sum-exp upper bound.
        # True max <= LSE <= true max + log(seq_len).
        # Set to None between steps; populated in forward when log_max_attention_logit is enabled.
        self.current_max_attn_logits = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        attn_mask_type: AttnMaskType,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward. Inputs in sbhd [sq, b, np, hn], output in [sq, b, np*hn] (heads flattened)."""
        assert packed_seq_params is None, (
            "SoftcapFlashAttention does not support packed sequences. "
            "Disable attn_logit_softcap or use TEDotProductAttention."
        )
        # sbhd -> bshd for flash_attn
        q = query.transpose(0, 1).contiguous()
        k = key.transpose(0, 1).contiguous()
        v = value.transpose(0, 1).contiguous()

        out, softmax_lse, _ = flash_attn_func(
            q, k, v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            softmax_scale=self.softmax_scale,
            causal=True,
            softcap=self.softcap,
            return_attn_probs=True,
        )

        if self.log_max_attention_logit:
            # softmax_lse: [b, np, sq] — upper bound on per-head max logit (lse ≥ true max)
            lse_max = softmax_lse.max(dim=-1).values.max(dim=0).values  # [np]
            if self.current_max_attn_logits is None:
                self.current_max_attn_logits = lse_max
            else:
                self.current_max_attn_logits = torch.max(self.current_max_attn_logits, lse_max)

        # bshd -> sbhd, then flatten heads: [sq, b, np*hn] to match TEDotProductAttention output
        b, sq, np, hn = out.shape
        return out.transpose(0, 1).reshape(sq, b, np * hn).contiguous()


def get_softcap_mamba_stack_spec(config) -> ModuleSpec:
    """Return a mamba stack spec with SoftcapFlashAttention swapped in for attention layers.

    Falls back to the standard TEDotProductAttention spec when attn_logit_softcap is not set.
    """
    if not getattr(config, 'attn_logit_softcap', None):
        return _base_mamba_stack_spec

    import copy
    spec = copy.deepcopy(_base_mamba_stack_spec)
    # Swap TEDotProductAttention -> SoftcapFlashAttention in the attention_layer submodules
    spec.submodules.attention_layer.submodules.self_attention.submodules.core_attention = (
        SoftcapFlashAttention
    )
    return spec
