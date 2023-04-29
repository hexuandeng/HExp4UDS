# Adopt from fairseq https://github.com/facebookresearch/fairseq

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from heuds.modules.embeddings.positional_embedding import PositionalEmbedding
from heuds.models.transformer.transformer_layer import TransformerDecoderLayer
from heuds.utils import fill_with_neg_inf
from heuds.modules.layer_drop import LayerDropModuleList
from heuds.modules.quant_noise import quant_noise


class TransformerDecoder(nn.Module):
    """
    Transformer decoder consisting of *cfg.layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, cfg, input_embed_dim, no_encoder_attn=False):
        super().__init__()
        self.cfg = cfg
        self._future_mask = torch.empty(0)

        self.dropout_module = nn.Dropout(cfg.dropout)
        self.decoder_layerdrop = cfg.layerdrop

        self.embed_dim = cfg.embed_dim
        self.output_embed_dim = cfg.output_dim

        self.padding_idx = 0
        self.max_target_positions = cfg.max_target_positions

        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(
            cfg.embed_dim)

        if cfg.quant_noise > 0:
            self.quant_noise = quant_noise(
                nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False),
                cfg.quant_noise,
                cfg.quant_noise_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            nn.Linear(input_embed_dim, cfg.embed_dim, bias=False)
            if cfg.embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                cfg.embed_dim,
                self.padding_idx,
                learned=cfg.learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )
        if cfg.layernorm_embedding:
            self.layernorm_embedding = nn.LayerNorm(cfg.embed_dim)
        else:
            self.layernorm_embedding = None

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerDecoderLayer(cfg, no_encoder_attn)
                for _ in range(cfg.layers)
            ]
        )
        self.num_layers = len(self.layers)

        if cfg.normalize_before and not cfg.no_decoder_final_norm:
            self.layer_norm = nn.LayerNorm(cfg.embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            nn.Linear(cfg.embed_dim, self.output_embed_dim, bias=False)
            if cfg.embed_dim != self.output_embed_dim
            else None
        )

    @property
    def output_dim(self):
        return self.output_embed_dim

    def forward(
        self,
        prev_output_embeddings,
        decoder_mask: Optional[Tensor] = None,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str,
                                         Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        return_all_hiddens: bool = False
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        decoder_padding_mask = (~decoder_mask.to(bool)).to(torch.uint8)
        has_pads = decoder_padding_mask.any()

        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"].transpose(0, 1)
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                decoder_mask, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_embeddings = prev_output_embeddings[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # Prevent torchscript exporting issue for dynamic quant embedding
        prev_output_embeddings = prev_output_embeddings.contiguous()
        # embed tokens and positions
        x = self.embed_scale * prev_output_embeddings

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)
        if has_pads:
            x = x * (1 - decoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = []
        if return_all_hiddens:
            inner_states.append(x)
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=decoder_padding_mask,
                need_attn=bool(idx == alignment_layer),
                need_head_weights=bool(idx == alignment_layer),
            )
            if return_all_hiddens:
                inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return {
            "decoder_out": x,
            "decoder_states": inner_states,
            "attention": attn,
            "mask": decoder_mask
            }

    @property
    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]
