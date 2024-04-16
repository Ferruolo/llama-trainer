import torch
from typing import Callable, Optional
import torch.nn.functional as F
import torch.nn.init as inits
from torch.nn.parameter import Parameter


class Embedding(torch.nn.Module):
    """Embedding parallelized in the embedding dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        keep_master_weight_for_test: bool = False,
    ) -> None:
        super(Embedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = scale_grad_by_freq
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._weight = None

        # Allocate weights.
        self.weight = Parameter(torch.Tensor(self.num_embeddings, self.embedding_dim))
        # And initialize.


    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
        output = F.embedding(
            input_,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return output
