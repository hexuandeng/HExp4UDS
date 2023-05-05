import torch
from torch import nn
from typing import Optional, Tuple
from dataclasses import dataclass, field
from heuds.base.base_config import BaseConfig
from heuds.modules.seq2vec_encoders.base_seq2vec import BaseSeq2VecEncoder
from heuds.utils import combine_initial_dims, uncombine_initial_dims, get_activation_fn


@dataclass
class CnnEncoderConfig(BaseConfig):
    num_filters: int = field(
        default=50, metadata={"help": "num_filters for char embedding"}
    )
    ngram_filter_sizes: Tuple[int, ...] = field(
        default=(3,), metadata={"help": "ngram_filter_sizes for char embedding"}
    )
    layer_activation: str = field(
        default='relu', metadata={"help": "conv_layer_activation for char embedding"}
    )
    output_dim: Optional[int] = field(
        default=None, metadata={"help": "whether using proj layer for char embedding"}
    )


class CnnEncoder(BaseSeq2VecEncoder):
    """
    A ``CnnEncoder`` is a combination of multiple convolution layers and max pooling layers.  As a
    :class:`Seq2VecEncoder`, the input to this module is of shape ``(batch_size, num_tokens,
    input_dim)``, and the output is of shape ``(batch_size, output_dim)``.

    The CNN has one convolution layer for each ngram filter size. Each convolution operation gives
    out a vector of size num_filters. The number of times a convolution layer will be used
    is ``num_tokens - ngram_size + 1``. The corresponding maxpooling layer aggregates all these
    outputs from the convolution layer and outputs the max.

    This operation is repeated for every ngram size passed, and consequently the dimensionality of
    the output after maxpooling is ``len(ngram_filter_sizes) * num_filters``.  This then gets
    (optionally) projected down to a lower dimensional output, specified by ``output_dim``.

    We then use a fully connected layer to project in back to the desired output_dim.  For more
    details, refer to "A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural
    Networks for Sentence Classification", Zhang and Wallace 2016, particularly Figure 1.

    Parameters
    ----------
    embedding_dim : ``int``
        This is the input dimension to the encoder.  We need this because we can't do shape
        inference in pytorch, and we need to know what size filters to construct in the CNN.
    num_filters: ``int``
        This is the output dim for each convolutional layer, which is the number of "filters"
        learned by that layer.
    ngram_filter_sizes: ``Tuple[int]``, optional (default=``(2, 3, 4, 5)``)
        This specifies both the number of convolutional layers we will create and their sizes.  The
        default of ``(2, 3, 4, 5)`` will have four convolutional layers, corresponding to encoding
        ngrams of size 2 to 5 with some number of filters.
    conv_layer_activation: ``Activation``, optional (default=``torch.nn.ReLU``)
        Activation to use after the convolution layers.
    output_dim : ``Optional[int]``, optional (default=``None``)
        After doing convolutions and pooling, we'll project the collected features into a vector of
        this size.  If this value is ``None``, we will just return the result of the max pooling,
        giving an output of shape ``len(ngram_filter_sizes) * num_filters``.
    """

    def __init__(self, cfg,
                 embedding_dim: int) -> None:
        super(CnnEncoder, self).__init__()
        self.cfg = cfg
        self._embedding_dim = embedding_dim
        self._activation = get_activation_fn(self.cfg.layer_activation)

        self._convolution_layers = nn.ModuleList([nn.Conv1d(in_channels=self._embedding_dim,
                                                            out_channels=self.cfg.num_filters,
                                                            kernel_size=ngram_size,
                                                            padding='same')
                                                  for ngram_size in self.cfg.ngram_filter_sizes])

        maxpool_output_dim = self.cfg.num_filters * \
            len(self.cfg.ngram_filter_sizes)
        if self.cfg.output_dim:
            self.projection_layer = nn.Linear(
                maxpool_output_dim, self.cfg.output_dim)
        else:
            self.projection_layer = None
            self.cfg.output_dim = maxpool_output_dim

    @property
    def input_dim(self) -> int:
        return self._embedding_dim

    @property
    def output_dim(self) -> int:
        return self.cfg.output_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):  # pylint: disable=arguments-differ
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()

        original_size = tokens.size()[:-2]
        tokens = combine_initial_dims(tokens, keep_back=2)

        # Our input is expected to have shape `(batch_size, num_tokens, embedding_dim)`.  The
        # convolution layers expect input of shape `(batch_size, in_channels, sequence_length)`,
        # where the conv layer `in_channels` is our `embedding_dim`.  We thus need to transpose the
        # tensor first.
        tokens = torch.transpose(tokens, 1, 2)
        # Each convolution layer returns output of size `(batch_size, num_filters, pool_length)`,
        # where `pool_length = num_tokens - ngram_size + 1`.  We then do an activation function,
        # then do max pooling over each filter for the whole input sequence.  Because our max
        # pooling is simple, we just use `torch.max`.  The resultant tensor of has shape
        # `(batch_size, num_conv_layers * num_filters)`, which then gets projected using the
        # projection layer, if requested.

        filter_outputs = []
        for convolution_layer in self._convolution_layers:
            filter_outputs.append(
                self._activation(convolution_layer(tokens)).max(dim=2)[0]
            )

        # Now we have a list of `num_conv_layers` tensors of shape `(batch_size, num_filters)`.
        # Concatenating them gives us a tensor of shape `(batch_size, num_filters * num_conv_layers)`.
        maxpool_output = torch.cat(filter_outputs, dim=1) if len(
            filter_outputs) > 1 else filter_outputs[0]

        if self.projection_layer:
            result = self.projection_layer(maxpool_output)
        else:
            result = maxpool_output

        result = uncombine_initial_dims(result, original_size)
        return result
