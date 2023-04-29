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
from heuds.utils import batch_index_select, pad_to_tensor
@dataclass
class SemSpanParserConfig(BaseConfig):
    dropout: float = field(
        default=0.3, metadata={"help": "dropout for SynEdgeParser"}
    )
    edge_vector_dim: int = field(
        default=512, metadata={"help": "edge_head_vector_dim for SynEdgeParser"}
    )
    loss_multiplier: int = field(
        default=2, metadata={"help": "n_layers for UDSEdgeAttrParser"}
    )

class SemSpanParser(nn.Module):

    def __init__(self, cfg, input_dim):
        super(SemSpanParser, self).__init__()

        self.input_dim = input_dim
        self.bias = Parameter(torch.Tensor(1, 1, input_dim))
        nn.init.xavier_normal_(self.bias)

        self.edge_query_linear = nn.Linear(input_dim, cfg.edge_vector_dim)
        self.edge_key_linear = nn.Linear(input_dim, cfg.edge_vector_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.syn_out = BiaffineAttention(cfg.edge_vector_dim)

        self.ignore_index = -100
        self.attr_loss_function = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.metric = Precision()
        self.loss_multiplier = cfg.loss_multiplier

    def forward(self, encoder_outs, embedding_sem, syn_mask, sem_mask):
        bsz = encoder_outs.shape[0]
        # encoder_outs = encoder_outs.masked_fill(~syn_mask.unsqueeze(2).bool(), 0)
        embedding = torch.cat([self.bias.repeat(bsz, 1, 1), embedding_sem], dim=1)
        query_embedding = self.dropout(F.relu(self.edge_query_linear(encoder_outs)))
        key_embedding = self.dropout(F.relu(self.edge_key_linear(embedding)))

        output = self.syn_out(query_embedding, key_embedding, batch_first=True)
        output = output.squeeze(-1)
        
        sem_mask = torch.cat([sem_mask.new_ones((bsz, 1)), sem_mask], dim=-1)
        attn_mask = output.new_zeros(output.shape)
        attn_mask = attn_mask.masked_fill(~sem_mask.bool().unsqueeze(1), float('-inf'))
        attn_mask[:, :, 1:] = attn_mask[:, :, 1:].masked_fill(~syn_mask.bool().unsqueeze(2), float('-inf'))
        output += attn_mask

        return output

    def loss(self, outputs, golds, syn_masks):
        # mask out non-predicted stuff
        output = pad_to_tensor(outputs, golds)
        gold = pad_to_tensor(golds, outputs)
        syn_mask = pad_to_tensor(syn_masks, gold)
        self.metric(output.max(dim=-1)[1], gold, syn_mask)

        if self.training:
            out = combine_initial_dims(output)
            go = gold.masked_fill(~syn_mask.bool(), self.ignore_index).long()
            go = combine_initial_dims(go, keep_back=0)
            return self.attr_loss_function(out, go) * self.loss_multiplier
        return 0.0

    def soft_loss(self, outputs, golds, syn_masks):
        # mask out non-predicted stuff
        output = pad_to_tensor(outputs, golds)
        gold = pad_to_tensor(golds, outputs)
        syn_mask = pad_to_tensor(syn_masks, gold)

        if self.training:
            out = output.masked_fill(~syn_mask.bool().unsqueeze(-1), 0)
            out = out.masked_fill(out.isinf(), -1e16)
            out = combine_initial_dims(out)
            go = gold.masked_fill(~syn_mask.bool().unsqueeze(-1), 0)
            go = combine_initial_dims(go).softmax(dim=-1)
            return self.attr_loss_function(out, go) * self.loss_multiplier
        return 0.0

    def get_metric(self, reset: bool = False):
        return self.metric.get_metric(reset)

    def reset(self):
        self.metric.reset()
