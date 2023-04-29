import numpy as np
import torch
from torch import nn
import logging

from scipy.stats import pearsonr
import torch
from torch import nn
import torch.nn.functional as F
from heuds.utils import fill_with_neg_inf
from heuds.modules.attention.mlp_attention import MLPAttention
import torch
from torch import nn
from heuds.modules.embeddings.char_embedding import TokenCharactersEncoder
from heuds.modules.embeddings.word_embedding import WordEmbedding
from heuds.modules.seq2vec_encoders.cnn_seq2vec import CnnEncoder
from heuds.models.transformer.transformer_encoder import TransformerEncoder
from heuds.models.transformer.transformer_config import EncDecBaseConfig
from heuds.models.uds.encoder_embedding import EncoderEmbedding, EncoderEmbeddingConfig
from heuds.config.base_config import BaseConfig
from dataclasses import dataclass, field, fields
from typing import List, Optional
from heuds.constant import register_model
from heuds.constant import DEFAULT_PAD_IDX
from heuds.utils import get_activation_nn
from heuds.modules.base.mlp import MLP
from heuds.utils import pad_to_tensor
from heuds.scoring.mse_cross_entropy import MSECrossEntropyLoss

@dataclass
class NodeAttrParserConfig(BaseConfig):
    dropout: float = field(
        default=0.3, metadata={"help": "dropout for NodeAttrParser"}
    )
    hidden_dim: int = field(
        default=2048, metadata={"help": "hidden_dim for NodeAttrParser"}
    )
    n_layers: int = field(
        default=4, metadata={"help": "n_layers for NodeAttrParser"}
    )
    activation: str = field(
        default='relu', metadata={"help": "activation for NodeAttrParser"}
    )
    binary: bool = field(
        default=True, metadata={"help": "use binary for NodeAttrParser loss instead of confidence"}
    )
    loss_multiplier: int = field(
        default=1, metadata={"help": "n_layers for UDSEdgeAttrParser"}
    )


class NodeAttrParser(nn.Module):
    def __init__(self, cfg,
                 input_dim,
                 output_dim):
        super(NodeAttrParser, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = cfg.hidden_dim
        self.output_dim = output_dim
        self.binary = cfg.binary

        self.attr_proj = MLP(input_dim, cfg.hidden_dim, output_dim,
                             cfg.n_layers, cfg.activation, cfg.dropout)
        self.mask_proj = MLP(input_dim, cfg.hidden_dim, output_dim,
                             cfg.n_layers, cfg.activation, cfg.dropout)

        self.attr_loss_function = MSECrossEntropyLoss()
        self.mask_loss_function = nn.BCEWithLogitsLoss()
        self.loss_multiplier = cfg.loss_multiplier

    def forward(self,
                decoder_output):
        """
        decoder_output: batch, target_len, input_dim
        """
        boolean_output = self.mask_proj(decoder_output)
        attr_output = self.attr_proj(decoder_output)
        #pred_mask = torch.gt(boolean_output, 0)
        #prod = attr_output[0] * pred_mask[0]
        #print(f"pred attr {prod[0:6, 0:8]}")
        return dict(
            pred_attributes=attr_output,
            pred_mask=boolean_output
        )

    def loss(self, output, attrs, confidences, node_mask, metric=None):
        # mask out non-predicted stuff
        pred_attributes = output['pred_attributes']
        pred_mask = output['pred_mask']
        attrs = pad_to_tensor(attrs, pred_attributes)
        pred_attributes = pad_to_tensor(pred_attributes, attrs)
        confidences = pad_to_tensor(confidences, pred_mask)
        pred_mask = pad_to_tensor(pred_mask, confidences)
        node_mask = pad_to_tensor(node_mask, confidences)

        if metric is not None:
            metric(pred_attributes, pred_mask, attrs, confidences, 'node')

        pred_attributes = pred_attributes.masked_select(node_mask.bool().unsqueeze(-1))
        pred_mask = pred_mask.masked_select(node_mask.bool().unsqueeze(-1))
        attrs = attrs.masked_select(node_mask.bool().unsqueeze(-1))
        confidences = confidences.masked_select(node_mask.bool().unsqueeze(-1))

        attr_loss = torch.tensor(0).float().to(node_mask.device)
        mask_loss = torch.tensor(0).float().to(node_mask.device)
        if self.training:
            to_mult = confidences
            mask_binary = torch.gt(confidences, 0).float()
            if self.binary:
                to_mult = mask_binary

            pred_attr = pred_attributes * to_mult
            attr_mult = attrs * to_mult

            attr_loss = self.attr_loss_function(pred_attr, attr_mult)
            # see if annotated at all; don't model annotator confidence, already modeled above
            mask_loss = self.mask_loss_function(pred_mask, confidences)

        return (attr_loss + mask_loss) * self.loss_multiplier
