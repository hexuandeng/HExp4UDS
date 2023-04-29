import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from heuds.modules.base.mlp import MLP
from heuds.modules.base.biaffine import Biaffine
from heuds.config.base_config import BaseConfig
from heuds.utils import pad_to_tensor, combine_initial_dims
from heuds.scoring.precision import Precision
from heuds.modules.embeddings.word_embedding import WordEmbedding, WordEmbeddingConfig
from torch.nn import Parameter
from heuds.modules.attention.biaffine_attention import BiaffineAttention
from heuds.utils import batch_index_select, batch_mask_diagonal, pad_to_tensor

@dataclass
class SemEdgeParserConfig(BaseConfig):
    edge_vector_dim: int = field(
        default=512, metadata={"help": "edge_head_vector_dim for SynEdgeParser"}
    )
    dropout: float = field(
        default=0.3, metadata={"help": "dropout for SynEdgeParser"}
    )
    loss_multiplier: int = field(
        default=1, metadata={"help": "hidden_dim for UDSEdgeAttrParser"}
    )

class SemEdgeParser(nn.Module):
    def __init__(self, cfg, input_dim, out_dim):
        super(SemEdgeParser, self).__init__()
        self.input_dim = input_dim
        self.edge_vector_dim = cfg.edge_vector_dim
        self.edge_query_linear = nn.Linear(input_dim, cfg.edge_vector_dim)
        self.edge_key_linear = nn.Linear(input_dim, cfg.edge_vector_dim)
        self.dropout = nn.Dropout(p=cfg.dropout)

        self.sem_out = BiaffineAttention(cfg.edge_vector_dim, num_labels=out_dim)
        self.ignore_index = -100
        self.attr_loss_function = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.metric = Precision()
        self.loss_multiplier = cfg.loss_multiplier

    def forward(self, sem_embedding, sem_mask):
        bsz = sem_embedding.shape[0]
        query_embedding = self.dropout(F.relu(self.edge_query_linear(sem_embedding)))
        key_embedding = self.dropout(F.relu(self.edge_key_linear(sem_embedding)))

        output = self.sem_out(query_embedding, key_embedding, batch_first=True)
        sem_mask = torch.cat([sem_mask.new_ones((bsz, 1)), sem_mask], dim=-1).bool()
        attn_mask = output.new_zeros(output.shape)
        if attn_mask.shape[-1] > 1:
            attn_mask[:, :, :, 1] = attn_mask[:, :, :, 1].masked_fill(~sem_mask.unsqueeze(1), float('-inf'))
            attn_mask[:, :, :, 1] = attn_mask[:, :, :, 1].masked_fill(~sem_mask.unsqueeze(2), float('-inf'))
            for i in range(2, attn_mask.shape[-1]):
                attn_mask[:, :, :, i] = attn_mask[:, :, :, 1]
        output += attn_mask

        return output.contiguous()

    def loss(self, output, golds, sem_masks):
        output = pad_to_tensor(output, golds)
        gold = pad_to_tensor(golds, output)
        sem_mask = pad_to_tensor(sem_masks, gold.new_ones((gold.shape[0], gold.shape[1] - 1)))

        # mask out non-predicted stuff
        bsz, tgt_len = gold.shape[: 2]
        sem_mask = torch.cat([sem_mask.new_ones((bsz, 1)), sem_mask], dim=-1)
        mask = output.new_ones((bsz, tgt_len, tgt_len))
        mask = mask.masked_fill(~sem_mask.unsqueeze(1).bool(), 0)
        mask = mask.masked_fill(~sem_mask.unsqueeze(2).bool(), 0)
        mask = batch_mask_diagonal(mask)
        self.metric(output.max(dim=-1)[1], gold, mask)

        if self.training:
            mask = mask.bool()
            loss = self.attr_loss_function(output[mask], gold[mask].long())
            return loss * self.loss_multiplier
        return 0.0

    def soft_loss(self, output, golds, sem_masks):
        output = pad_to_tensor(output, golds)
        gold = pad_to_tensor(golds, output)
        sem_mask = pad_to_tensor(sem_masks, gold.new_ones((gold.shape[0], gold.shape[1] - 1)))

        # mask out non-predicted stuff
        bsz, tgt_len = gold.shape[: 2]
        sem_mask = torch.cat([sem_mask.new_ones((bsz, 1)), sem_mask], dim=-1)
        mask = output.new_ones((bsz, tgt_len, tgt_len))
        mask = mask.masked_fill(~sem_mask.unsqueeze(1).bool(), 0)
        mask = mask.masked_fill(~sem_mask.unsqueeze(2).bool(), 0)
        mask = batch_mask_diagonal(mask)

        if self.training:
            out = output.masked_fill(~mask.bool().unsqueeze(-1), 0)
            out = combine_initial_dims(out)
            go = gold.masked_fill(~mask.bool().unsqueeze(-1), 0)
            go = combine_initial_dims(go).softmax(dim=-1)
            return self.attr_loss_function(out, go) * self.loss_multiplier
        return 0.0

    def get_metric(self, reset: bool = False):
        return self.metric.get_metric(reset)

    def reset(self):
        self.metric.reset()
