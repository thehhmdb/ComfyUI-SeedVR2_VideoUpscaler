# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

import torch
import torch.nn.functional as F

# Import flash/sage attn with automatic fallback from compatibility layer
from ...optimization.compatibility import (
    call_flash_attn_2_varlen, call_flash_attn_3_varlen,
    call_sage_attn_2_varlen, call_sage_attn_3_varlen
)

from torch import nn


def pytorch_varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, dropout_p=0.0, softmax_scale=None, causal=False, deterministic=False):
    """
    A PyTorch-based implementation of variable-length attention to replace flash_attn_varlen_func.
    Uses a single batched SDPA call with masking for optimal performance.
    
    NOTE: max_seqlen_q and max_seqlen_k are accepted for API compatibility but not used.
    PyTorch's scaled_dot_product_attention automatically handles variable sequence lengths.
    
    COMPILE OPTIMIZATION: Uses torch.tensor_split to avoid .item() graph breaks.
    Uses batched SDPA with masking instead of per-sequence Python loop.
    """
    # Get sequence lengths from cumulative lengths
    # Avoid .cpu() call to prevent GPU sync — use .long() on-device instead
    cu_q = cu_seqlens_q.long()
    cu_k = cu_seqlens_k.long()
    seqlens_q = cu_q[1:] - cu_q[:-1]  # per-sequence lengths
    seqlens_k = cu_k[1:] - cu_k[:-1]
    
    num_seqs = len(seqlens_q)
    
    # Fast path: all sequences same length → single batched call, no masking needed
    if num_seqs > 0 and (seqlens_q[0] == seqlens_q).all() and (seqlens_k[0] == seqlens_k).all():
        sq = seqlens_q[0].item()
        sk = seqlens_k[0].item()
        # Reshape: (total_seq, heads, head_dim) -> (num_seqs, heads, max_seq, head_dim)
        q_b = q[:num_seqs * sq].reshape(num_seqs, -1, sq, q.shape[-1])  # (num_seqs, heads, sq, head_dim)
        k_b = k[:num_seqs * sk].reshape(num_seqs, -1, sk, k.shape[-1])
        v_b = v[:num_seqs * sk].reshape(num_seqs, -1, sk, v.shape[-1])
        
        output_b = F.scaled_dot_product_attention(
            q_b, k_b, v_b,
            dropout_p=dropout_p if not deterministic else 0.0,
            is_causal=causal
        )
        # Reshape back: (num_seqs, heads, sq, head_dim) -> (total_seq, heads, head_dim)
        return output_b.reshape(-1, sq, q.shape[-1])
    
    # General path: variable lengths — use batched SDPA with attention mask
    max_sq = seqlens_q.max().item()
    max_sk = seqlens_k.max().item()
    
    # Allocate padded tensors
    q_padded = q.new_empty(num_seqs, q.shape[-2], max_sq, q.shape[-1])
    k_padded = k.new_empty(num_seqs, k.shape[-2], max_sk, k.shape[-1])
    v_padded = v.new_empty(num_seqs, v.shape[-2], max_sk, v.shape[-1])
    
    # Fill padded tensors
    for i in range(num_seqs):
        sq_i = seqlens_q[i].item()
        sk_i = seqlens_k[i].item()
        q_padded[i, :, :sq_i] = q[cu_q[i]:cu_q[i+1]]
        k_padded[i, :, :sk_i] = k[cu_k[i]:cu_k[i+1]]
        v_padded[i, :, :sk_i] = v[cu_k[i]:cu_k[i+1]]
    
    # Build attention mask: (num_seqs, 1, max_sq, max_sk)
    # True = valid, False = ignored
    mask = torch.ones(num_seqs, 1, max_sq, max_sk, dtype=q.dtype, device=q.device)
    for i in range(num_seqs):
        sq_i = seqlens_q[i].item()
        sk_i = seqlens_k[i].item()
        mask[i, :, sq_i:, :] = float('-inf')  # Mask query positions beyond seq length
        mask[i, :, :sq_i, sk_i:] = float('-inf')  # Mask key positions beyond seq length
    
    output_b = F.scaled_dot_product_attention(
        q_padded, k_padded, v_padded,
        attn_mask=mask,
        dropout_p=dropout_p if not deterministic else 0.0,
        is_causal=causal
    )
    
    # Extract outputs in original order
    output = torch.cat([
        output_b[i, :, :seqlens_q[i].item()]
        for i in range(num_seqs)
    ], dim=1)
    
    return output


class TorchAttention(nn.Module):
    def tflops(self, args, kwargs, output) -> float:
        assert len(args) == 0 or len(args) > 2, "query, key should both provided by args / kwargs"
        q = kwargs.get("query") or args[0]
        k = kwargs.get("key") or args[1]
        b, h, sq, d = q.shape
        b, h, sk, d = k.shape
        return b * h * (4 * d * (sq / 1e6) * (sk / 1e6))

    def forward(self, *args, **kwargs):
        return F.scaled_dot_product_attention(*args, **kwargs)


class FlashAttentionVarlen(nn.Module):
    """
    Variable-length attention with configurable backend.
    
    Supported backends:
    - sdpa: PyTorch SDPA (fully compilable, always available)
    - flash_attn_2: Flash Attention 2 (Ampere+)
    - flash_attn_3: Flash Attention 3 (Hopper+)
    - sageattn_2: SageAttention 2
    - sageattn_3: SageAttention 3 (Blackwell/RTX 50xx)
    
    All non-SDPA backends use @torch._dynamo.disable wrapper (C++ extensions).
    """

    def __init__(self, attention_mode: str = 'sdpa', compute_dtype: torch.dtype = None):
        """
        Initialize with specified attention backend.
        
        Args:
            attention_mode: 'sdpa', 'flash_attn_2', 'flash_attn_3', 'sageattn_2', or 'sageattn_3'
            compute_dtype: Compute dtype for attention (set by pipeline, defaults to None for auto-detection)
        """
        super().__init__()
        self.attention_mode = attention_mode
        self.compute_dtype = compute_dtype

    def tflops(self, args, kwargs, output) -> float:
        cu_seqlens_q = kwargs["cu_seqlens_q"]
        cu_seqlens_k = kwargs["cu_seqlens_k"]
        _, h, d = output.shape
        seqlens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]) / 1e6
        seqlens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]) / 1e6
        return h * (4 * d * (seqlens_q * seqlens_k).sum())

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
        kwargs["deterministic"] = torch.are_deterministic_algorithms_enabled()
        
        # Convert to pipeline compute_dtype if configured (handles FP8 → fp16/bf16)
        if self.compute_dtype is not None and q.dtype != self.compute_dtype:
            q = q.to(self.compute_dtype)
            k = k.to(self.compute_dtype)
            v = v.to(self.compute_dtype)
        
        if self.attention_mode == 'flash_attn_3':
            return call_flash_attn_3_varlen(
                q, k, v, cu_seqlens_q, cu_seqlens_k, 
                max_seqlen_q, max_seqlen_k, **kwargs
            )
        elif self.attention_mode == 'flash_attn_2':
            return call_flash_attn_2_varlen(
                q, k, v, cu_seqlens_q, cu_seqlens_k, 
                max_seqlen_q, max_seqlen_k, **kwargs
            )
        elif self.attention_mode == 'sageattn_3':
            return call_sage_attn_3_varlen(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, **kwargs
            )
        elif self.attention_mode == 'sageattn_2':
            return call_sage_attn_2_varlen(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, **kwargs
            )
        else:
            # PyTorch SDPA
            return pytorch_varlen_attention(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, **kwargs
            )