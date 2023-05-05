"""
An LSTM with Recurrent Dropout and the option to use highway
connections between layers.
"""

from typing import Optional, Tuple

import torch

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
import itertools
from typing import Callable, List, Tuple, Type, Dict
from heuds.utils import get_dropout_mask
from dataclasses import dataclass, field
from heuds.base.base_config import BaseConfig
from heuds.models.fairseq_incremental_decoder import FairseqIncrementalDecoder
from loguru import logger
@dataclass
class LstmConfig(BaseConfig):
    hidden_size: int = field(
        default=512, metadata={"help": "how many subprocesses to use for data loading"}
    )
    recurrent_dropout: float = field(
        default=0.3, metadata={"help": "how many subprocesses to use for data loading"}
    )
    use_highway: bool = field(
        default=False, metadata={"help": "how many subprocesses to use for data loading"}
    )
    use_input_projection_bias: bool = field(
        default=False, metadata={"help": "how many subprocesses to use for data loading"}
    )


class AugmentedLstm(torch.nn.Module):
    """
    An LSTM with Recurrent Dropout and the option to use highway
    connections between layers. Note: this implementation is slower
    than the native Pytorch LSTM because it cannot make use of CUDNN
    optimizations for stacked RNNs due to the highway layers and
    variational dropout.

    Parameters
    ----------
    input_size : int, required.
        The dimension of the inputs to the LSTM.
    hidden_size : int, required.
        The dimension of the outputs of the LSTM.
    go_forward: bool, optional (default = True)
        The direction in which the LSTM is applied to the sequence.
        Forwards by default, or backwards if False.
    recurrent_dropout_probability: float, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ . Implementation wise, this simply
        applies a fixed dropout mask per sequence to the recurrent connection of the
        LSTM. Dropout is not applied to the output sequence nor the last hidden
        state that is returned, it is only applied to all previous hidden states.
    use_highway: bool, optional (default = True)
        Whether or not to use highway connections between layers. This effectively involves
        reparameterising the normal output of an LSTM as::

            gate = sigmoid(W_x1 * x_t + W_h * h_t)
            output = gate * h_t  + (1 - gate) * (W_x2 * x_t)
    use_input_projection_bias : bool, optional (default = True)
        Whether or not to use a bias on the input projection layer. This is mainly here
        for backwards compatibility reasons and will be removed (and set to False)
        in future releases.

    Returns
    -------
    output_accumulator : PackedSequence
        The outputs of the LSTM for each timestep. A tensor of shape
        (batch_size, max_timesteps, hidden_size) where for a given batch
        element, all outputs past the sequence length for that batch are
        zero tensors.
    """
    def __init__(self, cfg,
                 input_size: int,
                 go_forward: bool = True) -> None:
        super(AugmentedLstm, self).__init__()
        # Required to be wrapped with a :class:`PytorchSeq2SeqWrapper`.
        self.input_size = input_size
        self.hidden_size = hidden_size = self.output_dim = cfg.hidden_size

        self.go_forward = go_forward
        self.use_highway = cfg.use_highway
        self.recurrent_dropout_probability = cfg.recurrent_dropout

        # We do the projections for all the gates all at once, so if we are
        # using highway layers, we need some extra projections, which is
        # why the sizes of the Linear layers change here depending on this flag.
        if cfg.use_highway:
            self.input_linearity = torch.nn.Linear(input_size, 6 * hidden_size, bias=cfg.use_input_projection_bias)
            self.state_linearity = torch.nn.Linear(hidden_size, 5 * hidden_size, bias=True)
        else:
            self.input_linearity = torch.nn.Linear(input_size, 4 * hidden_size, bias=cfg.use_input_projection_bias)
            self.state_linearity = torch.nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        # Use sensible default initializations for parameters.
        block_orthogonal(self.input_linearity.weight.data, [self.hidden_size, self.input_size])
        block_orthogonal(self.state_linearity.weight.data, [self.hidden_size, self.hidden_size])

        self.state_linearity.bias.data.fill_(0.0)
        # Initialize forget gate biases to 1.0 as per An Empirical
        # Exploration of Recurrent Network Architectures, (Jozefowicz, 2015).
        self.state_linearity.bias.data[self.hidden_size:2 * self.hidden_size].fill_(1.0)

    def forward(self,  # pylint: disable=arguments-differ
                inputs: PackedSequence,
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Parameters
        ----------
        inputs : PackedSequence, required.
            A tensor of shape (batch_size, num_timesteps, input_size)
            to apply the LSTM over.

        initial_state : Tuple[torch.Tensor, torch.Tensor], optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. Each tensor has shape (1, batch_size, output_dimension).

        Returns
        -------
        A PackedSequence containing a torch.FloatTensor of shape
        (batch_size, num_timesteps, output_dimension) representing
        the outputs of the LSTM per timestep and a tuple containing
        the LSTM state, with shape (1, batch_size, hidden_size) to
        match the Pytorch API.
        """
        if isinstance(inputs, PackedSequence):
            sequence_tensor, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
            batch_size, total_timesteps = sequence_tensor.shape[: 2]
        elif isinstance(inputs, torch.Tensor):
            sequence_tensor = inputs
            batch_size, total_timesteps = sequence_tensor.shape[: 2]
            batch_lengths = torch.ones(batch_size, dtype=torch.int64) * total_timesteps
        else:
            raise ValueError('inputs must be PackedSequence but got %s' % (type(inputs)))

        output_accumulator = sequence_tensor.new_zeros(batch_size, total_timesteps, self.hidden_size)
        if initial_state is None:
            full_batch_previous_memory = sequence_tensor.new_zeros(batch_size, self.hidden_size)
            full_batch_previous_state = sequence_tensor.new_zeros(batch_size, self.hidden_size)
        else:
            full_batch_previous_state = initial_state[0].squeeze(1).contiguous()
            full_batch_previous_memory = initial_state[1].squeeze(1).contiguous()

        current_length_index = batch_size - 1 if self.go_forward else 0
        if self.recurrent_dropout_probability > 0.0:
            dropout_mask = get_dropout_mask(self.recurrent_dropout_probability, full_batch_previous_memory)
        else:
            dropout_mask = None

        for timestep in range(total_timesteps):
            # The index depends on which end we start.
            index = timestep if self.go_forward else total_timesteps - timestep - 1

            # What we are doing here is finding the index into the batch dimension
            # which we need to use for this timestep, because the sequences have
            # variable length, so once the index is greater than the length of this
            # particular batch sequence, we no longer need to do the computation for
            # this sequence. The key thing to recognise here is that the batch inputs
            # must be _ordered_ by length from longest (first in batch) to shortest
            # (last) so initially, we are going forwards with every sequence and as we
            # pass the index at which the shortest elements of the batch finish,
            # we stop picking them up for the computation.
            if self.go_forward:
                while batch_lengths[current_length_index] <= index:
                    current_length_index -= 1
            # If we're going backwards, we are _picking up_ more indices.
            else:
                # First conditional: Are we already at the maximum number of elements in the batch?
                # Second conditional: Does the next shortest sequence beyond the current batch
                # index require computation use this timestep?
                while current_length_index < (len(batch_lengths) - 1) and \
                                batch_lengths[current_length_index + 1] > index:
                    current_length_index += 1

            # Actually get the slices of the batch which we need for the computation at this timestep.
            previous_memory = full_batch_previous_memory[0: current_length_index + 1].clone()
            previous_state = full_batch_previous_state[0: current_length_index + 1].clone()
            # Only do recurrent dropout if the dropout prob is > 0.0 and we are in training mode.
            if dropout_mask is not None and self.training:
                previous_state = previous_state * dropout_mask[0: current_length_index + 1]
            timestep_input = sequence_tensor[0: current_length_index + 1, index]

            # Do the projections for all the gates all at once.
            projected_input = self.input_linearity(timestep_input)
            projected_state = self.state_linearity(previous_state)
            # Main LSTM equations using relevant chunks of the big linear
            # projections of the hidden state and inputs.
            input_gate = torch.sigmoid(projected_input[:, 0 * self.hidden_size:1 * self.hidden_size] +
                                       projected_state[:, 0 * self.hidden_size:1 * self.hidden_size])
            forget_gate = torch.sigmoid(projected_input[:, 1 * self.hidden_size:2 * self.hidden_size] +
                                        projected_state[:, 1 * self.hidden_size:2 * self.hidden_size])
            memory_init = torch.tanh(projected_input[:, 2 * self.hidden_size:3 * self.hidden_size] +
                                     projected_state[:, 2 * self.hidden_size:3 * self.hidden_size])
            output_gate = torch.sigmoid(projected_input[:, 3 * self.hidden_size:4 * self.hidden_size] +
                                        projected_state[:, 3 * self.hidden_size:4 * self.hidden_size])

            memory = input_gate * memory_init + forget_gate * previous_memory
            timestep_output = output_gate * torch.tanh(memory)

            if self.use_highway:
                highway_gate = torch.sigmoid(projected_input[:, 4 * self.hidden_size:5 * self.hidden_size] +
                                             projected_state[:, 4 * self.hidden_size:5 * self.hidden_size])
                highway_input_projection = projected_input[:, 5 * self.hidden_size:6 * self.hidden_size]
                timestep_output = highway_gate * timestep_output + (1 - highway_gate) * highway_input_projection

            # We've been doing computation with less than the full batch, so here we create a new
            # variable for the the whole batch at this timestep and insert the result for the
            # relevant elements of the batch into it.
            full_batch_previous_memory = full_batch_previous_memory.clone()
            full_batch_previous_state = full_batch_previous_state.clone()
            full_batch_previous_memory[0:current_length_index + 1] = memory
            full_batch_previous_state[0:current_length_index + 1] = timestep_output
            output_accumulator[0:current_length_index + 1, index] = timestep_output

        output_accumulator = pack_padded_sequence(output_accumulator, batch_lengths, batch_first=True)
        # Mimic the pytorch API by returning state in the following shape:
        # (num_layers * num_directions, batch_size, hidden_size). As this
        # LSTM cannot be stacked, the first dimension here is just 1.
        final_state = (full_batch_previous_state.unsqueeze(1),
                       full_batch_previous_memory.unsqueeze(1))

        return output_accumulator, final_state

def block_orthogonal(tensor: torch.Tensor,
                     split_sizes: List[int],
                     gain: float = 1.0) -> None:
    """
    An initializer which allows initializing model parameters in "blocks". This is helpful
    in the case of recurrent models which use multiple gates applied to linear projections,
    which can be computed efficiently if they are concatenated together. However, they are
    separate parameters which should be initialized independently.

    Parameters
    ----------
    tensor : ``torch.Tensor``, required.
        A tensor to initialize.
    split_sizes : List[int], required.
        A list of length ``tensor.ndim()`` specifying the size of the
        blocks along that particular dimension. E.g. ``[10, 20]`` would
        result in the tensor being split into chunks of size 10 along the
        first dimension and 20 along the second.
    gain : float, optional (default = 1.0)
        The gain (scaling) applied to the orthogonal initialization.
    """
    data = tensor.data
    sizes = list(tensor.size())
    if any([a % b != 0 for a, b in zip(sizes, split_sizes)]):
        raise ValueError("tensor dimensions must be divisible by their respective "
                                 "split_sizes. Found size: {} and split_sizes: {}".format(sizes, split_sizes))
    indexes = [list(range(0, max_size, split))
               for max_size, split in zip(sizes, split_sizes)]
    # Iterate over all possible blocks within the tensor.
    for block_start_indices in itertools.product(*indexes):
        # A list of tuples containing the index to start at for this block
        # and the appropriate step size (i.e split_size[i] for dimension i).
        index_and_step_tuples = zip(block_start_indices, split_sizes)
        # This is a tuple of slices corresponding to:
        # tensor[index: index + step_size, ...]. This is
        # required because we could have an arbitrary number
        # of dimensions. The actual slices we need are the
        # start_index: start_index + step for each dimension in the tensor.
        block_slice = tuple([slice(start_index, start_index + step)
                             for start_index, step in index_and_step_tuples])
        data[block_slice] = torch.nn.init.orthogonal_(tensor[block_slice].contiguous(), gain=gain)
