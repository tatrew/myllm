import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from functools import lru_cache
from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist




world_size = 1
rank = 0
block_size = 128
fp4_block_size = 32
default_dtype = torch.bfloat16
scale_fmt = None
scale_dtype = torch.float32


@contextmanager
def set_dtype(dtype):
    """Temporarily override torch default dtype, restoring it on exit (even if an exception occurs)."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)

@dataclass
class ModelArgs:
    max_batch_size: int = 8
    max_seq_len: int = 2048
    # vse v bf16
    #dtype: str = "bf16"
    #scale_fmt: None
    #expert_dtype: str = None
    #scale_dtype: str = "fp32"

    vocab_size: int = 32000
    dim: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4 
    ffn_dim: int = 2816
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    dropout: float = 0.0

   #moe ya kodanibyd sdelay no tyt eto ne nuzhno potomy chta ya delau densetransformer pod yzkospealizirovanniy zadachu или ну точнее не или а сделаю прост когда буду делать крупнее модельку вот кактатак
#а это ниже я взял у дипсика оно короч ну поймут те кто поймет а так эт если я обчуать буду на нескольких видеокартах а я нищий и буду в гуглколабе делать
    """class ParallelEmbedding(nn.Module):
    Embedding sharded along the vocab dimension. Each rank holds vocab_size // world_size rows.
    Out-of-range indices are zero-masked before all_reduce to combine partial embeddings.
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y
"""
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype: Optional[torch.dtype] = None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        dtype = torch.float32 if dtype is None else dtype

        if dtype not in (
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float64
        ):
            raise ValueError(
                f"непподерживаемый dtype ебень: {dtype}"
            )

        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                dtype=dtype
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=dtype)
            ) 
        else:
            self.register_parameter("bias", None)

        self.register_parameter("scale", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
            nn.init.kaiming_uniform_(
                self.weight,
                a=math.sqrt(5)
            )

            if self.bias is not None:
                bound = 1 / math.sqrt(self.in_features)
                nn.init.uniform_(
                    self.bias,
                    - bound,
                    bound
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.linear(x, self.weight, self.bias)  #eto govno zanilo 40min ya vahue kakoy ya huevi 

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # rmsnorm in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for convenient.
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


@lru_cache(2)
def precompute_freqs_cis(dim: int, seqlen: int, base: float = 10000.0) -> torch.Tensor:
     freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))#skorost vrashenia vrashency derevnya v rossii
     t = torch.arange(seqlen, dtype=torch.float32)
     angles = torch.outer(t, freqs)#pos x speed for every pos & pair   
     return torch.polar(torch.ones_like(angles), angles)   


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
     x_c = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
     freqs = freqs_cis.view(1, x_c.size(1), 1, x_c.size(-1))
     out = torch.view_as_real(x_c * freqs).flatten(-2)
     return out.type_as(x)


class Attention(nn.Module):
     def __init__(self, args: ModelArgs):
         super().__init__()
         self.n_heads = args.n_heads#16
         self.n_kv_heads = args.n_kv_heads#4
         self.head_dim = args.dim // args.n_heads#64
         self.n_rep = self.n_heads // self.n_kv_heads#4 head q for one k/v

         self.wq = Linear(args.dim, self.n_heads * self.head_dim)
         self.wk = Linear(args.dim, self.n_kv_heads * self.head_dim)
         self.wv = Linear(args.dim, self.n_kv_heads * self.head_dim)
         self.wo = Linear(self.n_heads * self.head_dim, args.dim)

     def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
          b, s, _= x.shape

          q = self.wq(x).view(b, s, self.n_heads, self.head_dim)
          k = self.wk(x).view(b, s, self.n_kv_heads, self.head_dim)
          v = self.wv(x).view(b, s, self.n_kv_heads, self.head_dim)  
          
          q = apply_rotary_emb(q, freqs_cis)
          k = apply_rotary_emb(k, freqs_cis)

          q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

          k = k.repeat_interleave(self.n_rep, dim=1)
          v = v.repeat_interleave(self.n_rep, dim=1)

          o = F.scaled_dot_product_attention(q, k, v, is_causal=True)

          o = o.transpose(1,2).reshape(b, s, -1)
          return self.wo(o)

class FeedForward(nn.Module):
    """SwiGLU FFN: w2( silu(w1(x)) * w3(x) )"""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = Linear(args.dim, args.ffn_dim)   # вентиль
        self.w3 = Linear(args.dim, args.ffn_dim)   # содержимое
        self.w2 = Linear(args.ffn_dim, args.dim)   # обратно в 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    """Один слой: внимание + FFN, pre-norm, residual."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.attn = Attention(args)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn = FeedForward(args)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    """Вся модель: embed -> N блоков -> norm -> head."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.embed = nn.Embedding(args.vocab_size, args.dim)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

        self.layers = nn.ModuleList([Block(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, args.norm_eps)

        self.head = Linear(args.dim, args.vocab_size)
        if args.tie_embeddings:
            self.head.weight = self.embed.weight      # одна таблица на вход и выход

        head_dim = args.dim // args.n_heads
        freqs_cis = precompute_freqs_cis(head_dim, args.max_seq_len, args.rope_theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None):
        # tokens: [batch, seq] — номера токенов
        b, s = tokens.shape
        h = self.embed(tokens)                        # [b, s, dim]
        freqs = self.freqs_cis[:s]
        for layer in self.layers:
            h = layer(h, freqs)
        h = self.norm(h)
        logits = self.head(h)                         # [b, s, vocab]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                targets.reshape(-1),
            )
        return logits, loss