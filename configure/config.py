from torch.optim import Adam
import torch.functional as F
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainerArgs:
    learning_rate = 1e-4
    optim_params = {}
    optimizer = Adam
    loss_func = F
    num_epochs = 5
    batch_size = 1


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
