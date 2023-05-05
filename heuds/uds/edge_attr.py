import torch
from torch import nn
from dataclasses import dataclass, field
from heuds.modules.base.mlp import MLP
import torch.nn.functional as F
from heuds.modules.base.biaffine import Biaffine
from heuds.base.base_config import BaseConfig
from heuds.utils import pad_to_tensor
from heuds.modules.attention.biaffine_attention import BiaffineAttention
from heuds.modules.base.mlp import MLP
from heuds.scoring.mse_cross_entropy import MSECrossEntropyLoss

@dataclass
class EdgeAttrParserConfig(BaseConfig):
    dropout: float = field(
        default=0.3, metadata={"help": "dropout for SynEdgeParser"}
    )
    attr_vector_dim: int = field(
        default=128, metadata={"help": "edge_head_vector_dim for SynEdgeParser"}
    )
    mask_vector_dim: int = field(
        default=128, metadata={"help": "edge_head_vector_dim for SynEdgeParser"}
    )
    hidden_dim: int = field(
        default=256, metadata={"help": "hidden_dim for UDSEdgeAttrParser"}
    )
    ffn_dim: float = field(
        default=2048, metadata={"help": "hidden_dim for UDSEdgeAttrParser"}
    )
    n_layers: int = field(
        default=3, metadata={"help": "n_layers for UDSEdgeAttrParser"}
    )
    activation: str = field(
        default='relu', metadata={"help": "activation for UDSEdgeAttrParser"}
    )
    binary: bool = field(
        default=True, metadata={"help": "use binary for UDSEdgeAttrParser loss instead of confidence"}
    )
    loss_multiplier: int = field(
        default=1, metadata={"help": "n_layers for UDSEdgeAttrParser"}
    )


class EdgeAttrParser(nn.Module):

    def __init__(self, cfg, h_input_dim, output_dim):
        super(EdgeAttrParser, self).__init__()

        self.h_input_dim = h_input_dim
        self.m_input_dim = h_input_dim
        self.hidden_dim = cfg.hidden_dim
        self.output_dim = output_dim
        self.binary = cfg.binary

        self.attr_query_linear = nn.Linear(self.h_input_dim, cfg.attr_vector_dim)
        self.attr_key_linear = nn.Linear(self.h_input_dim, cfg.attr_vector_dim)
        self.mask_query_linear = nn.Linear(self.m_input_dim, cfg.mask_vector_dim)
        self.mask_key_linear = nn.Linear(self.m_input_dim, cfg.mask_vector_dim)
        self.attr_U = nn.Bilinear(cfg.attr_vector_dim, cfg.attr_vector_dim, cfg.hidden_dim)
        self.mask_U = nn.Bilinear(cfg.mask_vector_dim, cfg.mask_vector_dim, cfg.hidden_dim)

        self.attr_proj = MLP(cfg.hidden_dim + 2 * cfg.attr_vector_dim, cfg.ffn_dim, 
                             output_dim, cfg.n_layers, cfg.activation, cfg.dropout)
        self.mask_proj = MLP(cfg.hidden_dim + 2 * cfg.mask_vector_dim, cfg.ffn_dim, 
                             output_dim, cfg.n_layers, cfg.activation, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

        self.attr_loss_function = MSECrossEntropyLoss()
        self.mask_loss_function = nn.BCEWithLogitsLoss()
        self.loss_multiplier = cfg.loss_multiplier


    def forward(self, query, key, edge):
        # do bilinear
        edge = torch.nonzero(edge > 1)
        query = query[edge[:, 0], edge[:, 1]]
        key = key[edge[:, 0], edge[:, 2]]
        
        attr_query = self.dropout(F.relu(self.attr_query_linear(query)))
        attr_key = self.dropout(F.relu(self.attr_key_linear(key)))
        mask_query = self.dropout(F.relu(self.mask_query_linear(query)))
        mask_key = self.dropout(F.relu(self.mask_key_linear(key)))

        attr_output = self.dropout(self.attr_U(attr_query, attr_key))
        attr_output = torch.cat([attr_output, attr_query, attr_key], dim=-1)
        attr_output = self.attr_proj(attr_output)

        mask_output = self.dropout(self.mask_U(mask_query, mask_key))
        mask_output = torch.cat([mask_output, mask_query, mask_key], dim=-1)
        mask_output = self.mask_proj(self.dropout(mask_output))
        # cat in the original as well
        # attr_output = torch.cat([attr_output, query, key],  2)
        # mask_output = torch.cat([mask_output, query, key],  2)
        # feedforward
        # attr_output = self.attr_MLP(attr_output)
        # mask_output = self.mask_MLP(mask_output)

        return dict(pred_attributes=attr_output,
                    pred_mask=mask_output)

    def loss(self, output, attr_mult, confidences, edge_mask, metric=None):
        # mask out non-predicted stuff
        pred_attributes = output['pred_attributes']
        pred_mask = output['pred_mask']
        edge_mask = torch.nonzero(edge_mask)
        attr_mult = attr_mult[edge_mask[:, 0], edge_mask[:, 1], edge_mask[:, 2]]
        confidences = confidences[edge_mask[:, 0], edge_mask[:, 1], edge_mask[:, 2]]

        if metric is not None:
            metric(pred_attributes, pred_mask, attr_mult, confidences, 'edge')

        attr_loss = torch.tensor(0).float().to(edge_mask.device)
        mask_loss = torch.tensor(0).float().to(edge_mask.device)
        if self.training:
            to_mult = confidences
            mask_binary = torch.gt(confidences, 0).float()
            if self.binary:
                to_mult = mask_binary

            pred_attr = pred_attributes * to_mult
            attr_mult = attr_mult * to_mult

            attr_loss = self.attr_loss_function(pred_attr, attr_mult)
            # see if annotated at all; don't model annotator confidence, already modeled above
            mask_loss = self.mask_loss_function(pred_mask, confidences)

        return (attr_loss + mask_loss / 2) * self.loss_multiplier
