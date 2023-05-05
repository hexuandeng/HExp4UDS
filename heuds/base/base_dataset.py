from torch.utils.data import Dataset
from loguru import logger
from heuds.utils import tqdm
from dataclasses import dataclass, field
from heuds.base.base_vocab import BaseVocab
from heuds.base.base_config import BaseConfig
import math
from heuds.utils import process_multiprocessing

@dataclass
class DatasetConfig(BaseConfig):
    min_occur_count: int = field(
        default=2, metadata={"help": "min occur time in vocab"}
    )
    max_vocab_size: int = field(
        default=None, metadata={"help": "max vocab size for all vocab build from datasets"}
    )


class BaseDataset(Dataset):
    """
    My base class of Datasets for Information Extraction.

    # Attributes

    split: `Union['train', 'dev', 'test']`

    datasets: `List[Dict[Any]]`
        A list of preprocessed data, but havent been tokenize. Can have str, int, matrix, and so on.

    vectors: `List[Dict[Matrix]]`
        A list of preprocessed data, coming from self.datasets by tokenizer or vocabulary.

    data2vec: `Dict[(Callable, str)]`
        For each field, callable process the datasets to vectors. It can be class, function, and so 
        on. str is the corrresponding suffix of the field name, while the prefix is its name in 
        datasets. Callable outputs Matrix or Dict(Matrix). If it is a dict, then str = None, and
        the corresponding key is its suffix.

    vocab: `Dict[(str, Callable)]`
        Project every field (str) in its output to its corresbonding vocab.
    """

    def __init__(self):
        super().__init__()
        self.split = ''
        self.vocab = {}
        self.datasets = None
        self.vectors = None
        self.data2vec = None

    def emb_input_to_vector(self):
        logger.info(f"Converting Corpus to Vector!")
        # multiprocessing much slower, don't know why
        process_multiprocessing(self.emb_input_to_vector_iterable, self.datasets, num_cores=1)
        
    def emb_input_to_vector_iterable(self, data):
        if data is None:
            self.vectors.append(None)
            return None
            
        vector = {}

        def add_val(key, value):
            if key in vector.keys():
                raise ValueError(
                    f"Dumplicate keys {key} for {self.split} dataset vector!")
            vector[key] = value

        for field, value in data.items():
            if field in self.data2vec:
                for emb, name in self.data2vec[field]:
                    new_value = emb(value)
                    if name is None:
                        assert isinstance(new_value, dict)
                        for k, v in new_value.items():
                            k = '_' + \
                                k if (len(k) != 0 and k[0] != '_') else k
                            add_val(field + k, v)
                    else:
                        name = '_' + \
                            name if (
                                len(name) != 0 and name[0] != '_') else name
                        add_val(field + name, new_value)
                        # One vocab can only give one output, not dict
                        if isinstance(emb, BaseVocab):
                            self.vocab[field + name] = emb

            else:
                add_val(field, value)
        self.vectors.append(vector)
    
        return None

    def __getitem__(self, index):
        return self.vectors[index]

    def __len__(self):
        return len(self.vectors)

    def __iter__(self):
        return iter(self.vectors)
