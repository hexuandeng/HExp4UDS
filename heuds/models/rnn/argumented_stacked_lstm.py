from typing import Optional, Tuple
import torch
from torch.nn.utils.rnn import PackedSequence
from heuds.models.rnn.argumented_lstm import LstmConfig, AugmentedLstm
from dataclasses import dataclass, field
from heuds.base.base_config import BaseConfig
from heuds.models.fairseq_incremental_decoder import FairseqIncrementalDecoder
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from heuds.models.fairseq_incremental_decoder import FairseqIncrementalDecoder

@dataclass
class StackedLstmConfig(LstmConfig):
    hidden_size: int = field(
        default=1024, metadata={"help": "how many subprocesses to use for data loading"}
    )
    layers: int = field(
        default=2, metadata={"help": "how many subprocesses to use for data loading"}
    )
    
class StackedLstm(FairseqIncrementalDecoder):
    """
    A standard stacked LSTM where the LSTM layers
    are concatenated between each layer. The only difference between
    this and a regular LSTM is the application of
    variational dropout to the hidden states of the LSTM.
    Note that this will be slower, as it doesn't use CUDNN.
    Parameters
    ----------
    input_size : int, required
        The dimension of the inputs to the LSTM.
    hidden_size : int, required
        The dimension of the outputs of the LSTM.
    layers : int, required
        The number of stacked Bidirectional LSTMs to use.
    recurrent_dropout_probability: float, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ .
    """
    def __init__(self, cfg, input_size: int) -> None:
        super(StackedLstm, self).__init__()

        # Required to be wrapped with a :class:`PytorchSeq2SeqWrapper`.
        self.input_size = input_size
        self.hidden_size = self.output_dim = cfg.hidden_size
        self.layers = cfg.layers
        self.bidirectional = False

        layers = []
        lstm_input_size = input_size
        for layer_index in range(cfg.layers):

            layer = AugmentedLstm(cfg, lstm_input_size, go_forward=True)
            lstm_input_size = cfg.hidden_size
            self.add_module('layer_{}'.format(layer_index), layer)
            layers.append(layer)

        self.lstm_layers = layers

    def forward(self,  # pylint: disable=arguments-differ
                output_sequence: PackedSequence,
                mask=None, 
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]]=None,
                incremental_state=None):
        """
        Parameters
        ----------
        inputs : ``PackedSequence``, required.
            A batch first ``PackedSequence`` to run the stacked LSTM over.
        initial_state : Tuple[torch.Tensor, torch.Tensor], optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. Each tensor has shape (1, batch_size, output_dimension).
        Returns
        -------
        output_sequence : PackedSequence
            The encoded sequence of shape (batch_size, sequence_length, hidden_size)
        final_states: torch.Tensor
            The per-layer final (state, memory) states of the LSTM, each with shape
            (layers, batch_size, hidden_size).
        """
        if incremental_state is not None:
            prev_decoder_out = self.get_incremental_state(incremental_state, "decoder_out")
            prev_initial_state = self.get_incremental_state(incremental_state, "hidden_state")
            output_sequence = output_sequence[:, -1:]
            if prev_initial_state is not None:
                initial_state = prev_initial_state

        if not initial_state:
            hidden_states = [None] * len(self.lstm_layers)
        elif initial_state[0].size()[1] != len(self.lstm_layers):
            raise ValueError("Initial states were passed to forward() but the number of "
                                     "initial states does not match the number of layers.")
        else:
            hidden_states = list(zip(initial_state[0].split(1, 1),
                                     initial_state[1].split(1, 1)))

        final_h = []
        final_c = []
        for i, state in enumerate(hidden_states):
            forward_layer = getattr(self, 'layer_{}'.format(i))
            # The state is duplicated to mirror the Pytorch API for LSTMs.
            output_sequence, final_state = forward_layer(output_sequence, state)
            final_h.append(final_state[0])
            final_c.append(final_state[1])

        output_sequence, lengths = pad_packed_sequence(output_sequence, batch_first=True)
        final_h = torch.cat(final_h, dim=1)
        final_c = torch.cat(final_c, dim=1)

        if incremental_state is not None:
            if prev_decoder_out is not None:
                output_sequence = torch.cat((prev_decoder_out, output_sequence), dim=1)
            self.set_incremental_state(incremental_state, "decoder_out", output_sequence)
            self.set_incremental_state(incremental_state, "hidden_state", (final_h, final_c))

        return {
            "decoder_out": output_sequence,
            "hidden_state": (final_h, final_c),
            "mask": mask
        }

    def reorder_incremental_state(
        self,
        incremental_state,
        new_order
    ):
        return super().reorder_incremental_state(incremental_state, new_order, ["decoder_out", "hidden_state"])
