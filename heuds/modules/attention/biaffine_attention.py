import torch
from torch import nn
from heuds.modules.attention.base_attention import BaseAttention


class BiaffineAttention(BaseAttention):
    def __init__(self,
                 embed_dim: int,
                 kdim: int = None,
                 num_labels: int = 1,
                 use_linear: bool = True,
                 use_bilinear: bool = True,
                 bias: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.num_labels = num_labels
        self._use_linear = use_linear
        self._use_bilinear = use_bilinear

        if use_linear:
            self.q_proj = nn.Parameter(torch.Tensor(num_labels, embed_dim))
            self.k_proj = nn.Parameter(torch.Tensor(num_labels, self.kdim))
        if use_bilinear:
            self.U = nn.Parameter(torch.Tensor(
                num_labels, embed_dim, self.kdim))
        # else:
        #     self.register_parameter('U', None)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_labels, 1, 1))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self._use_linear:
            nn.init.xavier_normal_(self.q_proj)
            nn.init.xavier_normal_(self.k_proj)
        if self._use_bilinear:
            nn.init.xavier_uniform_(self.U)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                batch_first: bool = False) -> torch.Tensor:
        """
        Args:
            query: [batch_size, query_length, query_vector_dim]
            key: [batch_size, key_length, key_vector_dim]
            query_mask: None or [batch_size, query_length]
            key_mask: None or [batch_size, key_length]
        Returns:
            the energy tensor with shape = [batch_size, num_labels, query_length, key_length]
        """
        if not batch_first:
            query = query.transpose(0, 1)
            if key is not None:
                key = key.transpose(0, 1)

        bsz, _, _ = query.size()

        # Input: [num_labels, query_vector_dim] * [batch_size, query_vector_dim, query_length]
        # Output: [batch_size, num_labels, query_length, 1]
        query_linear_output = torch.matmul(
            self.q_proj, query.transpose(1, 2)).unsqueeze(3)

        # Input: [num_labels, key_vector_dim] * [batch_size, key_vector_dim, key_length]
        # Output: [batch_size, num_labels, 1, key_length]
        key_linear_output = torch.matmul(
            self.k_proj, key.transpose(1, 2)).unsqueeze(2)

        if self._use_bilinear:
            # Input: [batch_size, 1, query_length, query_vector_dim] * [num_labels, query_vector_dim, key_vector_dim]
            # Output: [batch_size, num_labels, query_length, key_vector_dim]
            bilinear_output = torch.matmul(query.unsqueeze(1), self.U)
            # Input: [batch_size, num_labels, query_length, key_vector_dim]*[batch_size, 1, key_vector_dim, key_length]
            # Output: [batch_size, num_labels, query_length, key_length]
            blinear_output = torch.matmul(
                bilinear_output, key.unsqueeze(1).transpose(2, 3))

            attn_weights = blinear_output + query_linear_output + key_linear_output
        else:
            attn_weights = query_linear_output + key_linear_output
        if self.bias is not None:
            attn_weights += self.bias.unsqueeze(0)

        return attn_weights.permute((0,2,3,1)).contiguous()
