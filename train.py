import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from functools import lru_cache
from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist



@dataclass
class TrainArgs:
    total_tokens: int = 6_000_000_000
    seq_len: int = 2048

    #nizhe batch
    micro_batch_size: int = 16
    grad_accum: int = 16


    lr_max: float = 6e-4
    lr_min: float = 6e-5 #ny tyt roflany balansnie mb potom try to sdelat 1e-4 a v lr_max 1e-3 чтоб короч на агрессии на характере сделать жостко да?
    warmup_steps: int = 200 #tyt tipa 2% OT 11450 
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    dtype: str = "bf16"
    compile: bool = True
    checkpoint_every: int = 500
    eval_every: int = 250
    log_every: int = 10
