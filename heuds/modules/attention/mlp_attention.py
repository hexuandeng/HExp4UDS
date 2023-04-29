import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Optional, Tuple
from heuds.modules.attention.base_attention import BaseAttention


class MLPAttention(BaseAttention):
    def __init__(self,
                 embed_dim: int,
                 kdim: int = None,
                 vdim: int = None,
                 attn_hidden_dim: int = 128,
                 output_dim: int = None,
                 use_coverage: bool = True,
                 bias: bool = True,
                 self_attention=False) -> None:
        super(MLPAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else self.kdim
        self.output_dim = embed_dim if output_dim is None else output_dim
        self.self_attention = self_attention

        self.q_proj = nn.Linear(embed_dim, attn_hidden_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, attn_hidden_dim, bias=False)
        self.attn_proj = nn.Linear(attn_hidden_dim, 1, bias=False)
        self.out_proj = nn.Linear(embed_dim + self.kdim, self.output_dim, bias=True)

        if use_coverage:
            self.coverage_proj = torch.nn.Linear(
                1, attn_hidden_dim, bias=False)

    def forward(
        self,
        query,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        coverage: torch.Tensor = None,
        key_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        batch_first: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if not batch_first:
            query = query.transpose(0, 1)
            if key is not None:
                key = key.transpose(0, 1)
            if value is not None:
                value = value.transpose(0, 1)
        bsz, tgt_len, embed_dim = query.size()
        assert (
            embed_dim == self.embed_dim
        ), f"query dim {embed_dim} != {self.embed_dim}"
        if self.self_attention:
            value = query

        if not batch_first:
            query = query.transpose(0, 1)
            if key is not None:
                key = key.transpose(0, 1)

        if self.self_attention:
            key = query

        # Output: [batch_size, query_seq_length, 1, hidden_vector_dim]
        query_linear_output = self.q_proj(query).unsqueeze(2)
        # Output: [batch_size, 1, key_seq_length, hidden_vector_dim]
        key_linear_output = self.k_proj(key).unsqueeze(1)
        # Output: [batch_size, query_seq_length, key_seq_length, hidden_vector_dim]
        activation_input = query_linear_output + key_linear_output

        if hasattr(self, "coverage_proj"):
            # Output: [batch_size, 1, key_seq_length, hidden_vector_dim]
            coverage_linear_output = self.coverage_proj(
                coverage.unsqueeze(-1))
            activation_input = activation_input + coverage_linear_output

        attn_weights = self.attn_proj(torch.tanh(activation_input)).squeeze(-1)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_mask is not None:
            attn_weights = attn_weights.masked_fill(
                ~key_mask.unsqueeze(1)
                .to(torch.bool),
                float("-inf"),
            )

        if before_softmax:
            return attn_weights

        attn_weights_float = F.softmax(
            attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights_float.type_as(attn_weights)
        # attn_probs = self.dropout_module(attn_weights)

        attn = torch.bmm(attn_weights, value)
        assert list(attn.size()) == [bsz, tgt_len, embed_dim]

        attn = torch.tanh(self.out_proj(torch.cat([attn, query], 2)))

        if coverage is not None:
            coverage = coverage + attn_weights_float

        if not batch_first:
            attn = attn.transpose(0, 1)

        return {
            "attentional": attn,
            "attention_weights": attn_weights,
            "coverage": coverage
        }
