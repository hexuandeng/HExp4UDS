# Adopt from fairseq https://github.com/facebookresearch/fairseq

import math
from typing import Dict, List

import torch
from torch import nn
from torch import Tensor
from heuds.modules.embeddings.positional_embedding import PositionalEmbedding
from heuds.models.transformer.transformer_layer import TransformerEncoderLayer
from heuds.modules.layer_drop import LayerDropModuleList
from heuds.modules.quant_noise import quant_noise


class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *cfg.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, cfg, input_embed_dim, return_fc=False):
        super().__init__()
        self.cfg = cfg
        self.dropout_module = nn.Dropout(cfg.dropout)
        self.encoder_layerdrop = cfg.layerdrop
        self.return_fc = return_fc

        embed_dim = cfg.embed_dim if cfg.embed_dim > 0 else None
        self.padding_idx = cfg.padding_idx
        self.max_source_positions = cfg.max_source_positions

        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(
            embed_dim)
        self.embed_positions = (
            PositionalEmbedding(
                cfg.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.learned_pos
            )
            if not cfg.no_token_positional_embeddings
            else None
        )
        if cfg.layernorm_embedding:
            self.layernorm_embedding = nn.LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        if cfg.quant_noise > 0:
            self.quant_noise = quant_noise(
                nn.Linear(embed_dim, embed_dim, bias=False),
                cfg.quant_noise,
                cfg.quant_noise_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [TransformerEncoderLayer(cfg, return_fc=self.return_fc)
             for _ in range(cfg.layers)]
        )

        self.num_layers = cfg.layers

        self.project_in_dim = (
            nn.Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.output_dim = cfg.embed_dim
        if cfg.output_dim > 0:
            self.output_dim = cfg.output_dim
            self.project_out_dim = nn.Linear(embed_dim, cfg.output_dim, bias=False)

        if cfg.normalize_before:
            self.layer_norm = nn.LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(
        self,
        token_embeddings: torch.Tensor,
        encoder_mask: torch.Tensor,
        return_all_hiddens: bool = False
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # The mask of padding is set to one, else zero
        encoder_padding_mask = (~encoder_mask.to(bool)).to(torch.uint8)
        has_pads = encoder_padding_mask.any()

        x = encoder_embedding = self.embed_scale * token_embeddings

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        # account for padding while computing the representation
        if self.embed_positions is not None:
            x = x + self.embed_positions(encoder_mask)

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []
        fc_results = []

        # if return_all_hiddens:
        #     encoder_states.append(x)

        # encoder layers
        for layer in self.layers:
            x, fc_result = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)
                fc_results.append(fc_result)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        if hasattr(self, 'project_out_dim'):
            x = self.project_out_dim(x)

        return {
            "encoder_out": x,  # B x T x C
            "encoder_padding_mask": encoder_padding_mask,  # B x T
            "encoder_embedding": encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            "mask": encoder_mask
        }

    @property
    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def post_process(self, x, *args):
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        if hasattr(self, 'project_out_dim'):
            x = self.project_out_dim(x)
        return x
        