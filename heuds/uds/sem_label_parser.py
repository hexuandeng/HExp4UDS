import torch
from torch import nn
from dataclasses import dataclass, field
from heuds.modules.base.mlp import MLP
from heuds.modules.base.biaffine import Biaffine
from heuds.base.base_config import BaseConfig
from heuds.utils import pad_to_tensor, combine_initial_dims
from heuds.scoring.precision import Precision

@dataclass
class NodeClassificationConfig(BaseConfig):
    dropout: float = field(
        default=0.3, metadata={"help": "dropout for UDSEdgeAttrParser"}
    )
    hidden_dim: int = field(
        default=2048, metadata={"help": "hidden_dim for UDSEdgeAttrParser"}
    )
    n_layers: int = field(
        default=3, metadata={"help": "n_layers for UDSEdgeAttrParser"}
    )
    activation: str = field(
        default='relu', metadata={"help": "activation for UDSEdgeAttrParser"}
    )
    loss_multiplier: int = field(
        default=1, metadata={"help": "n_layers for UDSEdgeAttrParser"}
    )

class NodeClassification(nn.Module):

    def __init__(self, cfg, input_dim, output_dim):
        super(NodeClassification, self).__init__()
        self.hidden_dim = cfg.hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.MLP = MLP(self.input_dim, cfg.hidden_dim, self.output_dim, cfg.n_layers, dropout=cfg.dropout, activation=cfg.activation)
        self.ignore_index = -100
        self.attr_loss_function = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.metric = Precision()
        self.loss_multiplier = cfg.loss_multiplier

    def forward(self, encoder_outs):
        return self.MLP(encoder_outs)

    def loss(self, output, gold, mask):
        # mask out non-predicted stuff
        out = combine_initial_dims(output)
        go = gold.masked_fill(~mask.bool(), self.ignore_index).long()
        go = combine_initial_dims(go, keep_back=0)
        predict = output.max(dim=2)[1]
        self.metric(predict, gold, mask)

        return self.attr_loss_function(out, go) * self.loss_multiplier

    def soft_loss(self, output, gold, mask):
        # mask out non-predicted stuff
        out = output.masked_fill(~mask.bool().unsqueeze(-1), 0)
        out = combine_initial_dims(out)
        go = gold.masked_fill(~mask.bool().unsqueeze(-1), 0)
        go = combine_initial_dims(go).softmax(dim=-1)

        return self.attr_loss_function(out, go) * self.loss_multiplier

    def get_metric(self, reset: bool = False):
        return self.metric.get_metric(reset)

    def reset(self):
        self.metric.reset()
