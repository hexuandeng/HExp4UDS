from typing import Optional, Tuple
import torch
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from heuds.models.rnn.argumented_lstm import AugmentedLstm
from heuds.modules.input_variational_dropout import InputVariationalDropout
from heuds.models.rnn.argumented_lstm import LstmConfig, AugmentedLstm
from dataclasses import dataclass, field
from heuds.base.base_config import BaseConfig
from heuds.models.fairseq_incremental_decoder import FairseqIncrementalDecoder
from loguru import logger
@dataclass
class StackedBiLstmConfig(LstmConfig):
    layers: int = field(
        default=2, metadata={"help": "how many subprocesses to use for data loading"}
    )
    dropout: float = field(
        default=0, metadata={"help": "how many subprocesses to use for data loading"}
    )

class StackedBidirectionalLstm(torch.nn.Module):
    """
    A standard stacked Bidirectional LSTM where the LSTM layers
    are concatenated between each layer. The only difference between
    this and a regular bidirectional LSTM is the application of
    variational dropout to the hidden states and outputs of each layer apart
    from the last layer of the LSTM. Note that this will be slower, as it
    doesn't use CUDNN.

    Parameters
    ----------
    input_size : int, required
        The dimension of the inputs to the LSTM.
    hidden_size : int, required
        The dimension of the outputs of the LSTM.
    layers : int, required
        The number of stacked Bidirectional LSTMs to use.
    layer_dropout_probability: float, optional (default = 0.0)
        The layer wise dropout probability to be used in a dropout scheme as
        stated in  `A Theoretically Grounded Application of Dropout in
        Recurrent Neural Networks <https://arxiv.org/abs/1512.05287>`_ .
    use_highway: bool, optional (default = True)
        Whether or not to use highway connections between layers. This effectively involves
        reparameterising the normal output of an LSTM as::

            gate = sigmoid(W_x1 * x_t + W_h * h_t)
            output = gate * h_t  + (1 - gate) * (W_x2 * x_t)
    """
    def __init__(self, cfg, input_size: int) -> None:
        super(StackedBidirectionalLstm, self).__init__()

        # Required to be wrapped with a :class:`PytorchSeq2SeqWrapper`.
        self.input_size = input_size
        self.hidden_size = cfg.hidden_size
        self.output_dim = cfg.hidden_size * 2
        self.layers = cfg.layers
        self.bidirectional = True

        layers = []
        lstm_input_size = input_size
        for layer_index in range(cfg.layers):

            forward_layer = AugmentedLstm(cfg, lstm_input_size, go_forward=True)
            backward_layer = AugmentedLstm(cfg, lstm_input_size, go_forward=False)

            lstm_input_size = cfg.hidden_size * 2
            self.add_module('forward_layer_{}'.format(layer_index), forward_layer)
            self.add_module('backward_layer_{}'.format(layer_index), backward_layer)
            layers.append([forward_layer, backward_layer])
        self.lstm_layers = layers
        self.layer_dropout = InputVariationalDropout(cfg.dropout)

    def forward(self,  # pylint: disable=arguments-differ
                inputs: PackedSequence,
                mask=None, 
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                return_all_hiddens: bool = False
               ) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs : ``PackedSequence``, required.
            A batch first ``PackedSequence`` to run the stacked LSTM over.
        initial_state : Tuple[torch.Tensor, torch.Tensor], optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. Each tensor has shape (layers, batch_size, output_dimension * 2).

        Returns
        -------
        output_sequence : PackedSequence
            The encoded sequence of shape (batch_size, sequence_length, hidden_size * 2)
        final_states: torch.Tensor
            The per-layer final (state, memory) states of the LSTM, each with shape
            (layers * 2, batch_size, hidden_size * 2).
        """
        if not initial_state:
            hidden_states = [None] * len(self.lstm_layers)
        elif initial_state[0].size()[0] != len(self.lstm_layers):
            raise ValueError("Initial states were passed to forward() but the number of "
                                     "initial states does not match the number of layers.")
        else:
            hidden_states = list(zip(initial_state[0].split(1, 1),
                                     initial_state[1].split(1, 1)))

        output_sequence = self.layer_dropout(inputs)

        encoder_states = []
        final_h = []
        final_c = []
        for i, state in enumerate(hidden_states):
            forward_layer = getattr(self, 'forward_layer_{}'.format(i))
            backward_layer = getattr(self, 'backward_layer_{}'.format(i))
            # The state is duplicated to mirror the Pytorch API for LSTMs.
            forward_output, final_forward_state = forward_layer(output_sequence, state)
            backward_output, final_backward_state = backward_layer(output_sequence, state)
            forward_output, lengths = pad_packed_sequence(forward_output, batch_first=True)
            backward_output, _ = pad_packed_sequence(backward_output, batch_first=True)

            output_sequence = torch.cat([forward_output, backward_output], -1)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(output_sequence)

            # Apply layer wise dropout on each output sequence apart from the
            # first (input) and last
            if i < (self.layers - 1):
                output_sequence = self.layer_dropout(output_sequence)

            final_h.append(torch.cat([final_forward_state[0], final_backward_state[0]], dim=-1))
            final_c.append(torch.cat([final_forward_state[1], final_backward_state[1]], dim=-1))

        final_h = torch.cat(final_h, dim=1)
        final_c = torch.cat(final_c, dim=1)
        
        return {
            "encoder_out": output_sequence, 
            "hidden_state": (final_h, final_c),
            "encoder_states": encoder_states,  # List[T x B x C]
            "mask": mask
        }

    def post_process(self, output_sequence, *args):
        return self.layer_dropout(output_sequence)
        