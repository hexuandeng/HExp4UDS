import torch
from typing import Dict, List, Optional, Callable
import math
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple, Iterable, Union, cast, Dict, Deque, Sequence, Callable, Any
from heuds.constant import DEFAULT_PAD_IDX
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as tqdm_
from tqdm import trange as trange_
#import multiprocess
import multiprocessing

class trange(tqdm_):
    def __init__(self, *args, **kwargs):
        super().__init__(range(*args), **kwargs, mininterval=5)

class tqdm(tqdm_):
    def __init__(cls, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, mininterval=5)

def pad_sequence_to_length(
    sequence: Sequence,
    desired_length: List,
    default_value: Callable[[], Any] = lambda: DEFAULT_PAD_IDX,
    padding_on_right: bool = True,
    batch_padding: bool = True
) -> List:
    """
    Take a list of objects and pads it to the desired length, returning the padded list.  The
    original list is not modified.

    # Parameters

    sequence : `List`
        A list of objects to be padded.

    desired_length : `list`
        Maximum length of each sequence on each dimension. Longer sequences are truncated to 
        this length, and shorter ones are padded to it.

    default_value: `Callable`, optional (default=`lambda: 0`)
        Callable that outputs a default value (of any type) to use as padding values.  This is
        a lambda to avoid using the same object when the default value is more complex, like a
        list.

    padding_on_right : `bool`, optional (default=`True`)
        When we add padding tokens (or truncate the sequence), should we do it on the right or
        the left?

    # Returns

    padded_sequence : `List`
    """
    sequence = list(sequence)
    if batch_padding:
        final_list = []
        for subseq in sequence:
            final_list.append(pad_sequence_to_length(
                subseq, desired_length, default_value, padding_on_right, batch_padding=False))
        return final_list

    pad_size = desired_length[1:]
    desired_length = desired_length[0]
    # Truncates the sequence to the desired length.
    if padding_on_right:
        padded_sequence = sequence[:desired_length]
    else:
        padded_sequence = sequence[-desired_length:]

    # pad sub-sequence iterately if exists
    if len(pad_size):
        for i, subseq in enumerate(padded_sequence):
            padded_sequence[i] = pad_sequence_to_length(
                subseq, pad_size, default_value, padding_on_right, batch_padding)

    # Continues to pad with default_value() until we reach the desired length.
    values_to_pad = default_value()
    for size in pad_size[::-1]:
        values_to_pad = [values_to_pad] * size
    pad_length = desired_length - len(padded_sequence)
    values_to_pad = [values_to_pad] * pad_length
    if padding_on_right:
        padded_sequence = padded_sequence + values_to_pad
    else:
        padded_sequence = values_to_pad + padded_sequence

    return padded_sequence


def pad_cat(inputs: List[Tensor], device=None, dim=0, left_padding=False, need_unsqueeze=True, pad_index=DEFAULT_PAD_IDX):
    max_shape = None
    final = []
    for input in inputs:
        if max_shape is None:
            max_shape = list(input.shape)
        else:
            for i in range(len(max_shape)):
                max_shape[i] = max(max_shape[i], input.shape[i])

    for input in inputs:
        pad = []
        for i in reversed(range(len(max_shape))):
            if not left_padding:
                pad.append(0)
            pad.append(max_shape[i] - input.shape[i])
            if left_padding:
                pad.append(0)
        if device is not None:
            input = input.to(device)
        final.append(F.pad(input, pad, "constant", pad_index))

    if need_unsqueeze:
        return torch.cat([i.unsqueeze(dim) for i in final], dim=dim)
    else:
        return torch.cat(final, dim=dim)

def pad_cat_generation(inputs: Dict):
    tokens = pad_cat([i[0]['tokens'] for i in inputs], dim=0)
    positional_scores = pad_cat([i[0]['positional_scores'] for i in inputs], dim=0)

    keep_dict = {}
    example = inputs[0][0]['keep_dict']
    for k, v in example.items():
        if isinstance(v, list) or isinstance(v, tuple):
            tmp = []
            for id in range(len(v)):
                tmp.append(pad_cat([i[0]['keep_dict'][k][id] for i in inputs], dim=0, need_unsqueeze=False))
            keep_dict[k] = tmp
        else:
            keep_dict[k] = pad_cat([i[0]['keep_dict'][k] for i in inputs], dim=0, need_unsqueeze=False)
    return {
        'tokens': tokens,
        'positional_scores': positional_scores,
        'keep_dict': keep_dict
    }

def pad_to_tensor(input: Tensor, target: Tensor, left_padding=False):
    tgt_shape = list(target.shape)
    dim = len(input.shape)
    if dim < len(tgt_shape):
        tgt_shape = tgt_shape[: dim]
    while dim > len(tgt_shape):
        tgt_shape.append(1)
    pad = []
    for i in reversed(range(len(tgt_shape))):
        if not left_padding:
            pad.append(0)
        pad.append(max(tgt_shape[i] - input.shape[i], 0))
        if left_padding:
            pad.append(0)
    return F.pad(input.to(target.device), pad, "constant", DEFAULT_PAD_IDX)

def get_activation_fn(activation: str) -> Callable:
    """Returns the activation function corresponding to `activation`"""
    def relu_squared(x: torch.Tensor):
        return F.relu(x).pow(2)

    def gelu(x: torch.Tensor) -> torch.Tensor:
        return nn.functional.gelu(x.float()).type_as(x)

    def gelu_accurate(x):
        if not hasattr(gelu_accurate, "_a"):
            gelu_accurate._a = math.sqrt(2 / math.pi)
        return (
            0.5 * x * (1 + torch.tanh(gelu_accurate._a *
                       (x + 0.044715 * torch.pow(x, 3))))
        )

    activation = activation.lower()
    if activation == "relu":
        return F.relu
    elif activation == "elu":
        return F.relu
    elif activation == "relu_squared":
        return relu_squared
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "swish":
        return nn.SiLU
    else:
        raise RuntimeError(
            "--activation-fn {} not supported".format(activation))


def get_activation_nn(activation: str) -> Callable:
    """Returns the activation function corresponding to `activation`"""
    activation = activation.lower()
    if activation == "relu":
        return nn.ReLU
    elif activation == "elu":
        return nn.ELU
    elif activation == "swish":
        return nn.SiLU
    else:
        raise RuntimeError(
            "--activation-fn {} not supported".format(activation))

def get_optimizer(optimizer: str) -> Callable:
    """Returns the activation function corresponding to `activation`"""
    optimizer = optimizer.lower()
    if optimizer == "adam":
        return optim.Adam
    elif optimizer == "adamw":
        return optim.AdamW
    else:
        raise RuntimeError(
            "--activation-fn {} not supported".format(optimizer))

def combine_initial_dims(tensor: torch.Tensor, keep_back=1) -> torch.Tensor:
    """
    Given a (possibly higher order) tensor of ids with shape
    (d1, ..., dn, sequence_length)
    Return a view that's (d1 * ... * dn, sequence_length).
    If original tensor is 1-d or 2-d, return it as is.
    """
    if tensor.dim() <= keep_back:
        return tensor
    elif keep_back > 0:
        return tensor.view(-1, *tensor.size()[-keep_back:])
    elif keep_back == 0:
        return tensor.view(-1)


def uncombine_initial_dims(tensor: torch.Tensor, original_size: torch.Size, keep_back=1) -> torch.Tensor:
    """
    Given a tensor of embeddings with shape
    (d1 * ... * dn, sequence_length, embedding_dim)
    and the original shape
    (d1, ..., dn, sequence_length),
    return the reshaped tensor of embeddings with shape
    (d1, ..., dn, sequence_length, embedding_dim).
    If original size is 1-d or 2-d, return it as is.
    """
    if len(original_size) <= keep_back:
        return tensor
    elif keep_back > 0:
        return tensor.view(*original_size, *tensor.size()[-keep_back:])
    elif keep_back == 0:
        return tensor.view(*original_size)


def fill_with_neg_inf(t):
    return t.float().fill_(float("-inf")).type_as(t)

def process_multiprocessing(func, iterable, num_cores=8, order=True):
    try:
        total = len(iterable)
    except:
        total = None
        
    if num_cores == 1:
        result = []
        for i in tqdm(iterable, total=total):
            result.append(func(i))
        return result

    try:
        with multiprocessing.Pool(num_cores) as p:
            # multiprocessing is much faster, but don't support local function (lambda, fuc def in another fun)
            if order:
                result = list(tqdm(p.imap(func, iterable), total=total))
            else:
                result = list(tqdm(p.imap_unordered(func, iterable), total=total))
    except:
        p.close()
        # multiprocess support local function, but extremly slow, discard it
        result = []
        for i in tqdm(iterable, total=total):
            result.append(func(i))
        return result
    return result

def get_dropout_mask(dropout_probability: float, tensor_for_masking: torch.Tensor):
    """
    Computes and returns an element-wise dropout mask for a given tensor, where
    each element in the mask is dropped out with probability dropout_probability.
    Note that the mask is NOT applied to the tensor - the tensor is passed to retain
    the correct CUDA tensor type for the mask.

    Parameters
    ----------
    dropout_probability : float, required.
        Probability of dropping a dimension of the input.
    tensor_for_masking : torch.Tensor, required.


    Returns
    -------
    A torch.FloatTensor consisting of the binary mask scaled by 1/ (1 - dropout_probability).
    This scaling ensures expected values and variances of the output of applying this mask
     and the original tensor are the same.
    """
    binary_mask = (torch.rand(tensor_for_masking.size()) > dropout_probability).to(tensor_for_masking.device)
    # Scale mask by 1/keep_prob to preserve output statistics.
    dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
    return dropout_mask

def batch_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index.long())

def batch_mask_diagonal(input, mask_index=DEFAULT_PAD_IDX):
    bsz, len, len_ = input.shape
    assert len == len_
    mask = torch.eye(len).repeat(bsz, 1, 1).bool()
    input[mask] = mask_index
    return input

def safe_softmax(input, dim=-1):
    # Sometimes there may be a full list to be all -inf,
    # which need to be softmax, and thus return Nan.
    # This softmax solve this problem, with softmax([-inf, ..., -inf]) = [0, ..., 0]
    mask = input.isinf()
    dangerous = mask.all(dim=-1)
    output = input.clone()
    output[dangerous] = 0
    output = F.softmax(output, dim=dim)
    output = output.masked_fill(mask, 0)
    return output
