import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Tuple, Dict
from heuds.modules.attention.biaffine_attention import BiaffineAttention
from heuds.config.base_config import BaseConfig
from heuds.utils import combine_initial_dims, fill_with_neg_inf, pad_to_tensor
from heuds.modules.base.mlp import MLP
from heuds.scoring.attachment_score import AttachmentScores
from heuds.utils import safe_softmax

@dataclass
class SynEdgeParserConfig(BaseConfig):
    dropout: float = field(
        default=0.3, metadata={"help": "dropout for SynEdgeParser"}
    )
    edge_head_vector_dim: int = field(
        default=256, metadata={"help": "edge_head_vector_dim for SynEdgeParser"}
    )
    edge_type_vector_dim: int = field(
        default=256, metadata={"help": "edge_type_vector_dim for SynEdgeParser"}
    )
    label_smoothing: float = field(
        default=0.0, metadata={"help": "using label_smoothing"}
    )
    output_dim: int = field(
        default=256, metadata={"help": "output_dim for UDSEdgeAttrParser"}
    )
    gcn: bool = field(
        default=False, metadata={"help": "activation for UDSEdgeAttrParser"}
    )
    loss_multiplier: int = field(
        default=1, metadata={"help": "n_layers for UDSEdgeAttrParser"}
    )

class SynEdgeParser(nn.Module):
    def __init__(self, cfg,
                 query_vector_dim: int,
                 key_vector_dim: int,
                 num_labels: int = None) -> None:
        super().__init__()
        self.edge_head_dim = cfg.edge_head_vector_dim
        self.edge_type_dim = cfg.edge_type_vector_dim
        self.edge_head_query_linear = nn.Linear(
            query_vector_dim, cfg.edge_head_vector_dim)
        self.edge_head_key_linear = nn.Linear(
            key_vector_dim, cfg.edge_head_vector_dim)
        self.edge_type_query_linear = nn.Linear(
            query_vector_dim, cfg.edge_type_vector_dim)
        self.edge_type_key_linear = nn.Linear(
            key_vector_dim, cfg.edge_type_vector_dim)
        
        self.bias_k = nn.Parameter(torch.randn([1, 1, key_vector_dim]))
        self.dropout = nn.Dropout(cfg.dropout)
        self.loss_multiplier = cfg.loss_multiplier

        self.output_dim = cfg.output_dim
        self.head_attention = BiaffineAttention(cfg.edge_head_vector_dim, bias=False)

        if num_labels:
            self.edge_type_bilinear = nn.Bilinear(
                cfg.edge_type_vector_dim, cfg.edge_type_vector_dim, num_labels)
        else:
            self.edge_type_bilinear = None

        self._minus_inf = -1e8
        self._query_vector_dim = query_vector_dim
        self._key_vector_dim = key_vector_dim
        self._edge_type_vector_dim = cfg.edge_type_vector_dim

        self.head_criterion = nn.CrossEntropyLoss(
            label_smoothing=cfg.label_smoothing, ignore_index=-100)
        self.type_criterion = nn.CrossEntropyLoss(
            label_smoothing=cfg.label_smoothing, ignore_index=-100)

        self.gcn = cfg.gcn
        if self.gcn:
            self.type_emb = nn.Linear(num_labels, cfg.edge_type_vector_dim * 2, bias=False)
            self.head_proj11 = nn.Linear(query_vector_dim, cfg.edge_head_vector_dim * 2)
            self.head_proj12 = nn.Linear(query_vector_dim, cfg.edge_head_vector_dim * 2)
            self.head_proj21 = nn.Linear(query_vector_dim, cfg.edge_head_vector_dim * 2)
            self.head_proj22 = nn.Linear(query_vector_dim, cfg.edge_head_vector_dim * 2)
            self.type_proj1 = nn.Linear(cfg.edge_type_vector_dim * 2, cfg.edge_type_vector_dim)
            self.type_proj2 = nn.Linear(cfg.edge_type_vector_dim * 2, cfg.edge_type_vector_dim)
            self.out_proj1 = nn.Linear(cfg.edge_head_vector_dim * 4 + cfg.edge_type_vector_dim, query_vector_dim)
            self.out_proj2 = nn.Linear(cfg.edge_head_vector_dim * 4 + cfg.edge_type_vector_dim, self.output_dim)
        else:
            self.out_proj = nn.Linear((cfg.edge_head_vector_dim + cfg.edge_type_vector_dim) * 2, self.output_dim)

        self.metric = AttachmentScores()

    def reset_edge_type_bilinear(self, num_labels: int) -> None:
        self.edge_type_bilinear = nn.Bilinear(
            self._edge_type_vector_dim, self._edge_type_vector_dim, num_labels)

    def forward(self,
                query: torch.FloatTensor,
                key: torch.FloatTensor,
                edge_mask: torch.ByteTensor = None,
                gold_edge_heads: torch.Tensor = None,
                attention_mask: torch.Tensor = None
                ) -> Dict:
        """
        :param query: [batch_size, query_length, query_vector_dim]
        :param key: [batch_size, key_length, key_vector_dim]
        :param edge_head_mask: [batch_size, query_length, key_length]
                        1 indicates a valid position; otherwise, 0.
        :param gold_edge_heads: None or [batch_size, query_length].
                        head indices start from 1.
        :return:
            edge_heads: [batch_size, query_length].
            edge_types: [batch_size, query_length].
            edge_head_ll: [batch_size, query_length, key_length + 1(sentinel)].
            edge_type_ll: [batch_size, query_length, num_labels] (based on gold_edge_head) or None.
        """
        # if not self.is_syntax:
        bsz, src_len = query.shape[: 2]
        _, tgt_len = key.shape[: 2]
        attn_mask = query.new_zeros((bsz, src_len, tgt_len))
        if attention_mask is not None:
            attn_mask = attn_mask.masked_fill(~attention_mask.bool(), float('-inf'))
        key, edge_mask, attn_mask = self._add_bias(key, edge_mask, attn_mask)
        attn_mask = attn_mask.masked_fill(~edge_mask.unsqueeze(1).bool(), float('-inf'))

        edge_head_query = self.dropout(F.relu(self.edge_head_query_linear(query)))
        edge_head_key = self.dropout(F.relu(self.edge_head_key_linear(key)))
        edge_type_query = self.dropout(F.relu(self.edge_type_query_linear(query)))
        edge_type_key = self.dropout(F.elu(self.edge_type_key_linear(key)))

        # [batch_size, query_length, key_length + 1]
        edge_head_score = self.head_attention(edge_head_query, edge_head_key, batch_first=True)
        edge_head_score = edge_head_score.squeeze(-1) + attn_mask
        _, edge_heads = edge_head_score.max(dim=-1)

        if self.training:
            assert gold_edge_heads is not None
            if len(gold_edge_heads.shape) == 3:
                _, gold_edge_heads = gold_edge_heads.max(dim=-1)
            gold_edge_heads = gold_edge_heads.long()
            edge_type_score = self._get_edge_type_score(
                edge_type_query, edge_type_key, gold_edge_heads)
        else:
            edge_type_score = self._get_edge_type_score(
                edge_type_query, edge_type_key, edge_heads)

        _, edge_types = edge_type_score.max(dim=2)
        if self.gcn:
            edge_reps = self._gcn(query, edge_head_score, edge_type_score)
        else:
            edge_reps = self._get_representations(edge_head_score, torch.cat([edge_head_query, edge_head_key[:, : -1]], dim=2),
                                                  torch.cat([edge_type_query, edge_type_key[:, : -1]], dim=2))

        return dict(
            # Note: head indices start from 1.
            edge_head_query=edge_head_query,
            edge_head_key=edge_head_key,
            edge_type_query=edge_type_query,
            edge_type_key=edge_type_key,
            edge_heads=edge_heads,
            edge_types=edge_types,
            edge_reps=edge_reps,
            # Log-Likelihood.
            edge_head_ll=edge_head_score,
            edge_type_ll=edge_type_score
        )

    def _gcn(self, query, edge_head_score, edge_type_score):
        head_score = safe_softmax(edge_head_score[:, :, : -1], dim=2)
        trans_head_score = safe_softmax(torch.transpose(edge_head_score[:, :, : -1], 1, 2), dim=2)
        type_score = safe_softmax(edge_type_score, dim=2)

        out1 = self.head_proj11(torch.matmul(head_score, query))
        out2 = self.head_proj12(torch.matmul(trans_head_score, query))
        out3 = self.type_proj1(self.type_emb(type_score))
        out = torch.relu(torch.cat([out1, out2, out3], dim=2))
        out = self.out_proj1(out)
        out = self.dropout(out)

        out1 = self.head_proj21(torch.matmul(head_score, out))
        out2 = self.head_proj22(torch.matmul(trans_head_score, out))
        out3 = self.type_proj2(self.type_emb(type_score))
        out = torch.relu(torch.cat([out1, out2, out3], dim=2))
        out = self.out_proj2(out)
        out = self.dropout(out)
        return out

    def _pad_masks(self, key_mask=None, attn_mask=None):
        if attn_mask is not None:
            shape = attn_mask.size()[:-1] + torch.Size([1])
            attn_mask = torch.cat(
                [attn_mask, attn_mask.new_zeros(shape)], dim=-1)
        if key_mask is not None:
            shape = key_mask.size()[:-1] + torch.Size([1])
            key_mask = torch.cat(
                [
                    key_mask,
                    key_mask.new_ones(shape),
                ],
                dim=-1,
            )
        return key_mask, attn_mask

    def _add_bias(self, k, key_mask=None, attn_mask=None):
        assert self.bias_k is not None
        bsz = k.size(0)
        k = torch.cat([k, self.bias_k.repeat(bsz, 1, 1)], dim=1)
        key_mask, attn_mask = self._pad_masks(
            key_mask=key_mask, attn_mask=attn_mask
        )
        return k, key_mask, attn_mask

    def _get_edge_type_score(self,
                             query: torch.FloatTensor,
                             key: torch.FloatTensor,
                             edge_head: torch.Tensor) -> torch.Tensor:
        """
        Compute the edge type scores.
        :param query:  [batch_size, query_length, query_vector_dim]
        :param key: [batch_size, key_length, key_vector_dim]
        :param edge_head: [batch_size, query_length]
        :return:
            label_score: None or [batch_size, query_length, num_labels]
        """
        batch_size = key.size(0)
        batch_index = torch.arange(0, batch_size).view(
            batch_size, 1).type_as(edge_head)
        # [batch_size, query_length, hidden_size]
        selected_key = key[batch_index, edge_head].contiguous()
        query = query.contiguous()
        edge_type_score = self.edge_type_bilinear(query, selected_key)

        return edge_type_score

    def _get_representations(self,
                             edge_head_scores: torch.FloatTensor,
                             edge_head_key: torch.FloatTensor,
                             edge_type_key: torch.FloatTensor):
        """
        Compute the weighted representations of the head and type keys 
        :param edge_head_scores:  [batch_size, query_length, key_length]
        :param edge_head_key: [batch_size, key_length, key_head_dim]
        :param edge_type_key: [batch_size, key_length, key_type_dim]
        :return:
            reps: [batch_size, query_length, key_head_dim + key_type_dim]
        """
        edge_head_scores = safe_softmax(edge_head_scores[:, :, : -1], dim=2)

        # [batch_size, query_length, key_length] x [batch_size, key_length, key_head_dim]
        # -> [batch_size, query_length, key_head_dim]
        weighted_head_key = edge_head_scores @ edge_head_key
        # [batch_size, query_length, key_length] x [batch_size, key_length, key_type_dim]
        # -> [batch_size, query_length, key_head_dim]
        weighted_type_key = edge_head_scores @ edge_type_key

        keys_and_types = torch.cat([weighted_head_key, weighted_type_key], dim=2)
        keys_and_types = self.dropout(self.out_proj(keys_and_types))

        return keys_and_types

    def loss(self, output,
             gold_edge_heads: torch.Tensor,
             gold_edge_types: torch.Tensor,
             edge_mask: torch.Tensor,
             metric=None) -> Dict:
        """
        Compute the edge prediction loss.

        :param edge_head_ll: [batch_size, target_length, target_length + 1 (for sentinel)].
        :param edge_type_ll: [batch_size, target_length, num_labels].
        :param pred_edge_heads: [batch_size, target_length].
        :param pred_edge_types: [batch_size, target_length].
        :param gold_edge_heads: [batch_size, target_length].
        :param gold_edge_types: [batch_size, target_length].
        :param valid_node_mask: [batch_size, target_length].
        """
        gold_edge_heads = pad_to_tensor(gold_edge_heads, output['edge_heads'])
        gold_edge_types = pad_to_tensor(gold_edge_types, output['edge_types'])
        edge_head_mask = (gold_edge_heads != -1) & edge_mask.bool()
        padding_mask = ~edge_head_mask.bool()

        # Compute loss.
        gold_edge_heads.masked_fill_(
            padding_mask, self.head_criterion.ignore_index)
        edge_head = combine_initial_dims(output["edge_head_ll"])
        gold_head = combine_initial_dims(gold_edge_heads, keep_back=0)
        loss = self.head_criterion(edge_head, gold_head.long())

        gold_edge_types.masked_fill_(
            padding_mask, self.type_criterion.ignore_index)
        edge_type = combine_initial_dims(output["edge_type_ll"])
        gold_type = combine_initial_dims(gold_edge_types, keep_back=0)
        loss += self.type_criterion(edge_type, gold_type.long())

        if self.metric is not None:
            self.metric(output['edge_heads'], output['edge_types'],
                   gold_edge_heads, gold_edge_types, edge_head_mask)

        return loss * self.loss_multiplier

    def soft_loss(self, output,
             gold_edge_heads: torch.Tensor,
             gold_edge_types: torch.Tensor,
             edge_mask: torch.Tensor) -> Dict:
        """
        Compute the edge prediction loss.

        :param edge_head_ll: [batch_size, target_length, target_length + 1 (for sentinel)].
        :param edge_type_ll: [batch_size, target_length, num_labels].
        :param pred_edge_heads: [batch_size, target_length].
        :param pred_edge_types: [batch_size, target_length].
        :param gold_edge_heads: [batch_size, target_length].
        :param gold_edge_types: [batch_size, target_length].
        :param valid_node_mask: [batch_size, target_length].
        """
        gold_edge_heads = pad_to_tensor(gold_edge_heads, output['edge_heads'])
        gold_edge_types = pad_to_tensor(gold_edge_types, output['edge_types'])
        edge_head_mask = edge_mask.bool()
        padding_mask = ~edge_head_mask.bool()

        # Compute loss.
        edge_head = output["edge_head_ll"].masked_fill(padding_mask.unsqueeze(-1), 0)
        edge_head = combine_initial_dims(edge_head)
        gold_head = gold_edge_heads.masked_fill(padding_mask.unsqueeze(-1), 0)
        gold_head = combine_initial_dims(gold_head).softmax(dim=-1)
        loss = self.head_criterion(edge_head, gold_head)

        edge_type = output["edge_type_ll"].masked_fill(padding_mask.unsqueeze(-1), 0)
        edge_type = combine_initial_dims(edge_type)
        gold_type = gold_edge_types.masked_fill(padding_mask.unsqueeze(-1), 0)
        gold_type = combine_initial_dims(gold_type).softmax(dim=-1)
        loss += self.type_criterion(edge_type, gold_type)

        return loss * self.loss_multiplier

    def get_metric(self, reset: bool = False):
        return self.metric.get_metric(reset)

    def reset(self):
        self.metric.reset()
