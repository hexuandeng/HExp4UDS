import torch
from typing import List, Union, Dict
from heuds.utils import pad_sequence_to_length


class Batch():
    """
    A batch of Instances. In addition to containing the instances themselves,
    it contains helper functions for converting the data into tensors.
    """
    def __init__(self, instances):
        self.instances = instances

    def get_padding_lengths(self) -> Union[Dict[str, List[int]], List[int]]:
        """
        Gets the maximum padding lengths from all ``Instances`` in this batch.  Each ``Instance``
        may have multiple ``Fields`` (or simple list is also accepted). We find the max values for 
        each field_name, returning them in a dictionary.

        This can then be used to convert this batch into arrays of consistent length, or to set
        model parameters, etc.
        """
        def get_pad_len(instances):
            # Iteratively find max size of each dimension
            l = [len(instances)]
            if l[0] == 0:
                return l
            if not isinstance(instances[0], list):
                return l

            asw = get_pad_len(instances[0])
            for sub_list in instances:
                size = get_pad_len(sub_list)
                asw = [max(i, j) for i, j in zip(asw, size)]

            l.extend(asw)
            return l

        if len(self.instances) == 0:
            return [0]

        if isinstance(self.instances[0], dict):
            padding_lengths: Dict[str, list] = {}
        elif isinstance(self.instances[0], list):
            padding_lengths: list = None
        elif isinstance(self.instances, list):
            return [len(self.instances)]
        else:
            raise ValueError('Unsupported type for batch!')

        # find max pad_len for every field and dimension
        for instance in self.instances:
            if isinstance(instance, dict):
                for k, v in instance.items():
                    if v is None:
                        continue
                    v = list(v)
                    if k in padding_lengths and padding_lengths[k] is not None:
                        padding_lengths[k] = [max(i, j) for i, j in zip(padding_lengths[k], get_pad_len(v))]
                    else:
                        padding_lengths[k] = get_pad_len(v)
            elif isinstance(instance, list):
                instance = list(instance)
                if padding_lengths is None:
                    padding_lengths = get_pad_len(instance)
                else:
                    padding_lengths = [max(i, j) for i, j in zip(padding_lengths, get_pad_len(instance))]

        return padding_lengths

    def as_tensor_dict(self) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        This method converts this ``Batch`` into a set of pytorch Tensors that can be passed
        through a model.  In order for the tensors to be valid tensors, all ``Instances`` in this
        batch need to be padded to the same lengths wherever padding is necessary.

        Returns
        -------
        tensors : ``Union[Dict[str, torch.Tensor], torch.Tensor]``
            A dictionary of tensors, keyed by field name, suitable for passing as input to a model.
            Single tensor is also allowed.
        """
        if len(self.instances) == 0:
            return torch.Tensor([])

        def list2tensor(instances, padding_lengths):
            # padding and convert instances to torch.tensor
            pad_instances = pad_sequence_to_length(instances, padding_lengths)
            return torch.Tensor(pad_instances)

        # Combining all data in the batch to one or several single matrix.
        padding_lengths = self.get_padding_lengths()
        if isinstance(self.instances[0], dict):
            final_dict = {}
            for k in self.instances[0].keys():
                try:
                    final_dict[k] = list2tensor([data[k] for data in self.instances], padding_lengths[k])
                except:
                    final_dict[k] = [data[k] for data in self.instances]
            return final_dict
        elif isinstance(self.instances[0], list):
            return list2tensor(self.instances, padding_lengths)
        elif isinstance(self.instances, list):
            return torch.Tensor(self.instances)
        else:
            raise ValueError("An unsupported input type!")

    def __iter__(self):
        return iter(self.instances)
