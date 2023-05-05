# Adopt from fairseq https://github.com/facebookresearch/fairseq

from typing import Optional
from dataclasses import dataclass, field
from heuds.base.base_config import BaseConfig

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@dataclass
class EncDecBaseConfig(BaseConfig):
    embed_dim: Optional[int] = field(
        default=512, metadata={"help": "embedding dimension"}
    )
    ffn_embed_dim: int = field(
        default=2048, metadata={"help": "embedding dimension for FFN"}
    )
    activation_fn: str = field(
        default='relu', metadata={"help": "activation_fn for encdec"}
    )
    layers: int = field(default=6, metadata={"help": "number of layers"})
    attention_heads: int = field(
        default=8, metadata={"help": "number of attention heads"}
    )
    normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each block"}
    )
    no_token_positional_embeddings: bool = field(
        default=False, metadata={"help": "use token positional embeddings"}
    )
    learned_pos: bool = field(
        default=False, metadata={"help": "use learned positional embeddings"}
    )
    dropout: float = field(
        default=0.3, metadata={"help": "dropout probablity"}
    )
    attention_dropout: float = field(
        default=0.3, metadata={"help": "dropout probablity"}
    )
    relu_dropout: float = field(
        default=0.0, metadata={"help": "dropout probablity"}
    )
    layerdrop: float = field(
        default=0.0, metadata={"help": "layerdrop probablity"}
    )
    activation_dropout: float = field(
        default=0.3, metadata={"help": "dropout probablity"}
    )
    padding_idx: int = field(
        default=0, metadata={"help": "padding index"}
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "max source positions"}
    )
    layernorm_embedding: bool = field(
        default=False, metadata={"help": "use layernorm after embedding layer"}
    )
    no_scale_embedding: bool = field(
        default=False, metadata={"help": "use scale_embedding"}
    )
    output_dim: int = field(
        default=-1, metadata={"help": "decoder output dimension (extra linear layer if different from decoder embed dim)"},
    )
    quant_noise: float = field(
        default=0.0, metadata={"help": "quant_noise probablity"}
    )
    quant_noise_block_size: int = field(
        default=8, metadata={"help": "quant_noise probablity"}
    )
    

@dataclass
class DecoderConfig(EncDecBaseConfig):
    input_dim: int = field(
        default=-1, metadata={"help": "decoder input dimension"}
    )
    max_target_positions: int = field(
        default=DEFAULT_MAX_TARGET_POSITIONS,
        metadata={"help": "Maximum output length supported by the decoder"},
    )
    cross_self_attention: bool = field(
        default=False, metadata={"help": "use cross_self_attention in decoder"}
    )

    def __post_init__(self):
        if self.input_dim == -1:
            self.input_dim = self.embed_dim
        if self.output_dim == -1:
            self.output_dim = self.embed_dim


@dataclass
class QuantNoiseConfig(BaseConfig):
    pq: float = field(
        default=0.0,
        metadata={"help": "iterative PQ quantization noise at training time"},
    )
    pq_block_size: int = field(
        default=8,
        metadata={"help": "block size of quantization noise at training time"},
    )
    scalar: float = field(
        default=0.0,
        metadata={
            "help": "scalar quantization noise and scalar quantization at training time"
        },
    )
    no_decoder_final_norm: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )


@dataclass
class TransformerConfig(BaseConfig):
    activation_fn: str = field(
        default="relu", metadata={"help": "activation function to use"},
    )
    encoder: EncDecBaseConfig = EncDecBaseConfig()
    decoder: DecoderConfig = DecoderConfig()
    max_target_positions: int = field(
        default=DEFAULT_MAX_TARGET_POSITIONS,
        metadata={"help": "Maximum output length supported by the decoder"},
    )
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    share_all_embeddings: bool = field(
        default=False,
        metadata={
            "help": "share encoder, decoder and output embeddings (requires shared dictionary and embed dim)"
        },
    )
    merge_src_tgt_embed: bool = field(
        default=False,
        metadata={
            "help": "if true then the source and target embedding table is "
            "merged into one table. This is going to make the model smaller but "
            "it might hurt performance."
        },
    )

    # config for "BASE Layers: Simplifying Training of Large, Sparse Models"
    base_layers: Optional[int] = field(
        default=0, metadata={"help": "number of BASE layers in total"}
    )
    base_sublayers: Optional[int] = field(
        default=1, metadata={"help": "number of sublayers in each BASE layer"}
    )
    base_shuffle: Optional[int] = field(
        default=1,
        metadata={
            "help": "shuffle tokens between workers before computing assignment"},
    )
