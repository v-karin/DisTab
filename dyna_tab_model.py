import torch
import torch.nn.functional as F
import torch.nn as nn
from llama_transformer import build_transformer
import numpy as np
from data import PeriodicOptions, PeriodicMultiBandwidth

class LinearNumEmbed(nn.Module):
    def __init__(self, num_feat, emb_size, std=1, bias=True) -> None:
        super().__init__()

        self.embd = nn.Parameter(torch.Tensor(1, num_feat, emb_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, num_feat, emb_size))
            nn.init.zeros_(self.bias)
        else:
            self.bias = 0.
        nn.init.normal_(self.embd, std=std)

    def forward(self, x):
        return x[..., None] * self.embd + self.bias

def build_model(dim, tab_in_dim, tab_n_head, tab_n_layer, out_dim, num_ind, 
                 lm_tokens=False, cls_token=False, cat_size=0, cat_embed_size=0, raw_pos=True, lm_pos=False):
    raw_tokenizer = RawTokens(tab_in_dim, cat_embed_size, cat_size, len(num_ind), pos_encoder=raw_pos)
    if lm_tokens:
        lm_tokenizer = LMTokens(dim, tab_in_dim, pos_encoder=lm_pos)
    else:
        lm_tokenizer = None

    model = LmTabPredictNew(raw_tokenizer, lm_tokenizer, tab_in_dim, tab_n_head, tab_n_layer, out_dim, cls_token=cls_token)
    return model

class RawTokens(nn.Module):
    def __init__(self, dim, cat_embed_size, cat_count, num_count, pos_encoder=True):
        super().__init__()

        if pos_encoder:
            self.pos_encoder = nn.Parameter(torch.randn((cat_count + num_count, dim)))
            nn.init.normal_(self.pos_encoder, std=np.sqrt(0.5))
        else:
            self.pos_encoder = False
        
        if cat_embed_size > 0:
            self.cat_embed = nn.Embedding(cat_embed_size, dim)
            torch.nn.init.normal_(self.cat_embed.weight, std=np.sqrt(0.5))

        if num_count > 0:
            self.num_embed = LinearNumEmbed(num_count, dim, std=np.sqrt(0.5), bias=False)       
            opt = PeriodicOptions(dim//2, [0.1, 1., 10], False, "normal")
            self.periodic = PeriodicMultiBandwidth(num_count, opt)
    

    def forward(self, x):
        if len(x) == 3:
            x_cat, x_num, x_num_normed = x
            cat_embed = self.cat_embed(x_cat)
            num_embed = self.num_embed(x_num_normed) + self.periodic(x_num)
            embed = torch.cat([cat_embed, num_embed], dim=1)
        elif len(x) == 2:
            x_num, x_num_normed = x
            embed = self.num_embed(x_num_normed) + self.periodic(x_num)
        else:
            x_cat = x[0]
            embed = self.cat_embed(x_cat)
        if isinstance(self.pos_encoder, torch.Tensor):
            embed += self.pos_encoder[None]
        return embed


class LMTokens(nn.Module):
    def __init__(self, lm_dim, dim, pos_encoder=True):
        super().__init__()

        self.token_reduce =  nn.Sequential(
            nn.Linear(lm_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False)
        )

        if pos_encoder:
            self.pos_encoder = nn.Sequential(
                nn.Linear(lm_dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim, bias=False)
            )
        else:
            self.pos_encoder = False
    

    def forward(self, cat_info, num_info, x, cat_header_embed):
        if len(x) == 3:
            x_cat, _, x_num_normed = x
            cat_embed = self._get_cat_embed(cat_info, x_cat)
            num_embed = self._get_num_embed(num_info, x_num_normed)
            embed = torch.cat([cat_embed, num_embed], dim=1)
        elif num_info.shape[0] > 0:
            _, x_num_normed = x
            embed = self._get_num_embed(num_info, x_num_normed)
        else:
            x_cat = x[0]
            embed = self._get_cat_embed(cat_info, x_cat)
        
        if self.pos_encoder:
            tmp = torch.concatenate([cat_header_embed, num_info], dim=0)
            pos_encode = self.pos_encoder(tmp)
            embed += pos_encode[None]
        return embed

    def _get_cat_embed(self, cat_info, x_cat):
        cat_embed = self.token_reduce(cat_info)
        return F.embedding(x_cat, cat_embed)


    def _get_num_embed(self, num_info, x_num_normed):
        tmp = self.token_reduce(num_info) #num_embed * tab_in_dim
        scaled_num_embed = x_num_normed[..., None] * tmp
        return scaled_num_embed


class LmTabPredictNew(nn.Module):
    def __init__(self, raw_tokenizer, lm_tokenier, tab_in_dim, tab_n_head, tab_n_layer, out_dim,
                 cls_token=False) -> None:
        super().__init__()

        self.raw_tokenizer = raw_tokenizer
        self.lm_tokenizer = lm_tokenier

        self.out_linear = nn.Linear(tab_in_dim, out_dim)

        self.linear_f = nn.Sequential(
            nn.Linear(tab_in_dim, tab_in_dim),
            nn.ReLU(),
            nn.Linear(tab_in_dim, tab_in_dim)
        )

        self.f = build_transformer(tab_in_dim, n_layer=tab_n_layer, n_heads=tab_n_head, cls_token=cls_token)

        self.cls_token = cls_token

        
    def forward(self, cat_embed, cat_header_embed, num_header_embed, x):
        embed = self.raw_tokenizer(x)

        if self.lm_tokenizer:
            lm_embed = self.lm_tokenizer(cat_embed, num_header_embed, x, cat_header_embed)
            embed += lm_embed

        feat = self.f(embed, None)
        if self.cls_token:
            feat = feat[:, 0]
        else:
            feat = feat.mean(1)

        mean_feat = embed.mean(1)
        linear_feat = mean_feat + self.linear_f(mean_feat)
        

        out = self.out_linear(feat + linear_feat)
        return out