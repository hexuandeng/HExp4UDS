import torch
from torch import nn


class Biaffine(nn.Module):
    def __init__(self,
                 left_dim: int,
                 right_dim: int = None,
                 out_dim: int = 128,
                 use_linear: bool = True,
                 use_bilinear: bool = True,
                 bias: bool = True) -> None:
        super(Biaffine, self).__init__()
        self._use_linear = use_linear
        self._use_bilinear = use_bilinear

        if use_linear:
            self.left_proj = nn.Linear(left_dim, out_dim)
            self.right_proj = nn.Linear(right_dim, out_dim)
        if use_bilinear:
            self.bilinear = nn.Bilinear(left_dim, right_dim, out_dim, bias)

    def forward(self,
                left: torch.Tensor,
                right: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [batch_size, query_length, query_vector_dim]
            key: [batch_size, key_length, key_vector_dim]
            query_mask: None or [batch_size, query_length]
            key_mask: None or [batch_size, key_length]
        Returns:
            the energy tensor with shape = [batch_size, num_labels, query_length, key_length]
        """
        # Input: [num_labels, query_vector_dim] * [batch_size, query_vector_dim, query_length]
        # Output: [batch_size, num_labels, query_length, 1]
        left_output = self.left_proj(left)

        # Input: [num_labels, key_vector_dim] * [batch_size, key_vector_dim, key_length]
        # Output: [batch_size, num_labels, 1, key_length]
        right_output = self.right_proj(right)

        attn_weights = left_output + right_output
        if self._use_bilinear:
            # Output: [batch_size, num_labels, query_length, key_length]
            bilinear_output = self.bilinear(left, right)
            attn_weights = bilinear_output + attn_weights

        return attn_weights
