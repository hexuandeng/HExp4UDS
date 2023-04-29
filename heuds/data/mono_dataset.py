from collections import defaultdict
from decomp import UDSCorpus
from loguru import logger
from heuds.utils import tqdm
from dataclasses import dataclass, field
from transformers import AutoTokenizer
from heuds.data.base_vocab import BaseVocab
from heuds.data.base_dataset import BaseDataset, DatasetConfig
from heuds.data.uds_preprocess import DecompGraph
from heuds.constant import NODE_ONTOLOGY, EDGE_ONTOLOGY
import pickle
import os
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
import subprocess
import random

@dataclass
class MonoDatasetConfig(DatasetConfig):
    drop_syntax: bool = field(
        default=True, metadata={"help": "whether to drop syntax edge class information"}
    )
    order: str = field(
        default="inorder", metadata={"help": "which order to use for decoder side"}
    )
    vocab_dir: str = field(
        default='tokenizer.pickle', metadata={"help": "Glove dim"}
    )
    config: str = field(
        default='bert-base-cased', metadata={"help": "bert model config"}
    )
    mono_file: str = field(
        default='datasets/news.2021.en.shuffled.deduped', metadata={"help": "bert model config"}
    )

class MonoDataset(BaseDataset):
    def __init__(self, cfg):
        super().__init__()
        cfg.vocab_dir = os.path.join("buffer/", cfg.vocab_dir)
        self.cfg = cfg
        self.vocab = {}
        self.datasets = []
        self.vectors = []
        total = 1e5 * 1.2

        if not os.path.exists(f'{cfg.mono_file}.clean'):
            logger.info(f"Generating {cfg.mono_file}.clean!")
            out = subprocess.getoutput(f"wc -l {cfg.mono_file}")
            out = total * 2. / float(out.split()[0])
            datasets = []
            normalizer = normalizers.Sequence([NFD(), StripAccents()])
            pre_tokenizer = Whitespace()
            with open(cfg.mono_file, "r", encoding="utf-8") as f:
                for line in f:
                    if random.random() < out:
                        line = normalizer.normalize_str(line)
                        line = pre_tokenizer.pre_tokenize_str(line)
                        line = [i[0] for i in line]
                        if len(line) >= 5 and len(line) <= 64:
                            datasets.append(line)
            datasets = random.sample(datasets, int(total))
            with open(f'{cfg.mono_file}.clean', "w", encoding="utf-8") as f:
                for line in datasets:
                    line = ' '.join(line)
                    f.write(line + "\n")

        logger.info(f"Reading and Preprocessing {cfg.mono_file}.clean!")
        with open(f'{cfg.mono_file}.clean', "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().split()
                self.datasets.append({"src_tokens": line, "src_tokens_str": line, "src_mask": [1 for _ in line],
                                    "syn_head_indices": None, "syn_edge_mask": [[1 for _ in line] for _ in line]})

        self.bert_tokenizer = AutoTokenizer.from_pretrained(cfg.config)
        # self.data2vec = defaultdict(list)
        # self.build_uds_emb_map()
        assert os.path.exists(cfg.vocab_dir)
        logger.info(f'Loading Vocabularys From {cfg.vocab_dir}!')
        with open(cfg.vocab_dir, 'rb') as f:
            self.data2vec = pickle.load(f)
        self.sp_token = self.data2vec["sp_token"]
        self.emb_input_to_vector()
        self.post_process()
        del self.datasets

    def post_process(self):
        self.generate_vocab_size = len(self.sp_token)
        self.upos_vocab_size = self.data2vec["src_pos_tags"][0][0].vocab_size
        self.label_vocab_size = self.data2vec["sem_label"][0][0].vocab_size
        self.edge_vocab_size = self.data2vec["sem_edges"][0][0].vocab_size
        self.syn_vocab_size = self.data2vec["syn_head_tags"][0][0].vocab_size
        self.node_attr_num = len(NODE_ONTOLOGY)
        self.edge_attr_num = len(EDGE_ONTOLOGY)

    def build_uds_emb_map(self):
        self.data2vec['src_tokens'].append((self._bert_tokenize, None))
        self.data2vec = dict(self.data2vec)

    def _bert_tokenize(self, tokens):
        token_ids = self.bert_tokenizer(tokens, is_split_into_words=True)
        word_ids = token_ids.word_ids()
        gather_indexes = [[] for _ in tokens]
        for k, v in enumerate(word_ids):
            if v is not None:
                gather_indexes[v].append(k)
        return {"_bert": token_ids['input_ids'], "_bert_map": gather_indexes}

def RandDataset():
    total = 1e5 * 4
    with open("train_intermediate.pickle", 'rb') as f:
        datasets = pickle.load(f)
    words = []
    for i in datasets:
        words += i["src_tokens"]

    with open("datasets/random.clean", "w", encoding="utf-8") as f:
        for _ in range(int(total)):
            length = random.randint(4, 64)
            line = " ".join(random.sample(words, length))
            f.write(line + "\n")
