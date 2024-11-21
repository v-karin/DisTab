# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


@dataclass
class ModelArgs:
    dim: int = 4096
    hid_dim: Optional[int] = None
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
    rel_pos_bins: int = 32
    cls_token: bool = True
    embedding_dim_per_char: int = 1


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // 1
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)


    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        attn_bias : Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        keys = xk
        values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attn_bias is not None:
            scores = scores + attn_bias
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        if args.hid_dim is None:
            args.hid_dim = 4 * args.dim
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim= args.hid_dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        attn_bias: Optional[torch.Tensor]
    ):
        h = x + self.attention.forward(
            self.attention_norm(x), mask, attn_bias
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        if params.cls_token:
            self.cls_token = nn.Parameter(torch.Tensor(1, 1, params.dim))
            nn.init.normal_(self.cls_token, std=np.sqrt(0.5))
            self.has_cls_token = 1
        else:
            self.has_cls_token = 0


    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        # masking away padded items in each sequence
        # mask is 1D-array of actual_seq_len
        _bsz, seqlen, _ = tokens.shape
        seqlen += self.has_cls_token # accounting for cls token
        h = tokens

        sample_ind = torch.arange(seqlen)[None].repeat((seqlen, 1)).to(h.device)
        batch_ind = torch.tile(sample_ind[None], (_bsz, 1, 1))
        if mask is None:
            bool_mask = torch.full((_bsz, seqlen, seqlen), float(0.), device=h.device)
        else:
            mask = mask + self.has_cls_token
            mask = mask[:, None, None]
            row_end = batch_ind < mask
            bool_mask = torch.full((_bsz, seqlen, seqlen), float("-inf"), device=h.device)
            bool_mask[row_end] = 0

        if self.has_cls_token:
            h = torch.concat([self.cls_token.expand(h.shape[0], -1, -1), h], dim=1)
            for layer in self.layers:
                h = layer(h, bool_mask[:, None], None)
        else:
            for layer in self.layers:
                h = layer(h, bool_mask[:, None], None)
        h = self.norm(h)
        return h


def build_transformer(dim, n_layer, n_heads, cls_token=False, hid_dim=None):
    model_args = ModelArgs(dim=dim, n_layers=n_layer, n_heads=n_heads, cls_token=cls_token, hid_dim=hid_dim)
    return Transformer(model_args)