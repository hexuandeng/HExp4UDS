import random
import numpy as np
from tqdm import trange
from dataclasses import dataclass, field
from typing import List, Iterable, Optional, Tuple, Union
from heuds.base.base_batch import Batch
from heuds.base.base_config import BaseConfig


@dataclass
class IteratorConfig(BaseConfig):
    skip_invalid_size_inputs_valid_test: bool = field(
        default=False,
        metadata={
            "help": "ignore too long or too short lines in valid and test set"},
    )
    max_tokens: Optional[int] = field(
        default=None, metadata={"help": "maximum number of tokens in a batch"}
    )
    batch_size: Optional[int] = field(
        default=16,
        metadata={
            "help": "number of examples in a batch",
            "argparse_alias": "--max-sentences",
        },
    )
    dev_batch_size: Optional[int] = field(
        default=64,
        metadata={
            "help": "number of examples in a batch",
            "argparse_alias": "--max-sentences",
        },
    )
    skip_smaller_batches: bool = field(
        default=False, metadata={"help": "Whether batches smaller than batch_size will be discarded"}
    )
    sorting_keys: Tuple[str] = field(
        default=("src_tokens", "to_sem"), metadata={"help": "ngram_filter_sizes for char embedding"}
    ) # 先后顺序和他的方案相反


class BaseIterator():
    """
    An iterator which by default, pads batches with respect to the maximum input lengths `per
    batch`. Additionally, you can provide a list of field names and padding keys which the dataset
    will be sorted by before doing this batching, causing inputs with similar length to be batched
    together, making computation more efficient (as less time is wasted on padded elements of the
    batch).

    Parameters
    ----------
    sorting_keys : List[str] or str, optional, (default = None)
        Inputs are list of dicts. We sort them according to the length of Dict[sorting_keys].
        Elements closer to the start have higher priority. None means to skip sorting.
    """

    def __init__(self, cfg: BaseConfig, train=True) -> None:
        self.cfg = cfg
        self._sorting_keys = cfg.sorting_keys
        # self._sorting_keys = ["src_mask"]
        if isinstance(self._sorting_keys, str):
            self._sorting_keys = [self._sorting_keys]

        self.batch_size = self.cfg.batch_size if train else self.cfg.dev_batch_size
    
    def get_batch_num(self, instances):
        if self.cfg.skip_smaller_batches:
            return int(np.floor(len(instances) / float(self.batch_size)))
        return int(np.ceil(len(instances) / float(self.batch_size)))

    def __call__(self, instances: List[Iterable], shuffle: bool = True) -> Iterable[Batch]:
        instances = list(instances)
        if shuffle: # Select different batchs for every epoch
            random.shuffle(instances)

        # sort the instances by _sorting_keys
        if self._sorting_keys is not None:
            instances_with_lengths = []
            for key in self._sorting_keys:
                if key not in instances[0].keys():
                    self._sorting_keys.remove(key)
            for instance in instances:
                instance_with_lengths = ([Batch(instance[field_name]).get_padding_lengths()[0]
                                        for field_name in self._sorting_keys], instance)
                instances_with_lengths.append(instance_with_lengths)

            instances_with_lengths.sort(key=lambda x: x[0])
            instances = [instance_with_lengths[-1] for instance_with_lengths in instances_with_lengths]
        
        # select batches sequentially according to batch_size
        batches = []
        self.batch_num = batch_num = int(np.ceil(len(instances) / float(self.batch_size)))
        for i in trange(batch_num, position=0, mininterval=10):
            cur_batch_size = self.batch_size if i < batch_num - 1 else len(instances) - self.batch_size * i
            batch_instances = [instances[i * self.batch_size + b] for b in range(cur_batch_size)]
            if self.cfg.skip_smaller_batches and len(batch_instances) < self.batch_size:
                self.batch_num -= 1
                continue
            batches.append(Batch(batch_instances).as_tensor_dict())

        if shuffle: # Don't feed data as _sorting_keys defined.
            random.shuffle(batches)

        yield from batches
