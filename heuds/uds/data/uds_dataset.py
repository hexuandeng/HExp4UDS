from collections import defaultdict
from decomp import UDSCorpus
from loguru import logger
from heuds.utils import tqdm
from dataclasses import dataclass, field
from transformers import AutoTokenizer
from heuds.base.base_vocab import BaseVocab
from heuds.base.base_dataset import BaseDataset, DatasetConfig
from heuds.uds.data.uds_preprocess import DecompGraph
from heuds.constant import NODE_ONTOLOGY, EDGE_ONTOLOGY
from heuds.utils import process_multiprocessing
from tqdm.contrib.concurrent import process_map
import pickle
import os
from heuds.constant import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN

@dataclass
class UDSDatasetConfig(DatasetConfig):
    drop_syntax: bool = field(
        default=True, metadata={"help": "whether to drop syntax edge class information"}
    )
    glove_dir: str = field(
        default='datasets/glove.840B.300d.txt', metadata={"help": "Glove save dir"}
    )
    glove_dim: int = field(
        default=300, metadata={"help": "Glove dim"}
    )
    vocab_dir: str = field(
        default='buffer/tokenizer.pickle', metadata={"help": "Glove dim"}
    )
    config: str = field(
        default='bert-base-cased', metadata={"help": "bert model config"}
    )

class UDSDataset(BaseDataset):
    def __init__(self, cfg, split, train_dataset=None):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.vocab = {}
        self.datasets = []
        self.vectors = []

        logger.info(f"Preprocessing UDSCorpus {split} to default format!")
        # r = process_map(self.proc, [j for i, j in uds.graphs.items()], max_workers=4, chunksize=1)
        if os.path.exists(f'buffer/{split}.pickle') and os.path.exists(cfg.vocab_dir):
            with open(f'buffer/{split}.pickle', 'rb') as f:
                self.datasets = pickle.load(f)
        else:
            logger.info(f"Reading UDSCorpus {split} from decomp library!")
            if os.path.exists(f'buffer/{split}_raw.pickle'):
                with open(f'buffer/{split}_raw.pickle', 'rb') as f:
                    uds = pickle.load(f)
            else:
                uds = UDSCorpus(split=split)
                with open(f'buffer/{split}_raw.pickle', 'wb') as f:
                    pickle.dump(uds, f)
            datasets = process_multiprocessing(self.proc, uds.graphs.items(), order=False)
            self.datasets = [i for i in datasets if i is not None]
            self.sp_token = list(DecompGraph.get_sp_token())
            with open(f'buffer/{split}.pickle', 'wb') as f:
                pickle.dump(self.datasets, f)
        self.datasets = [DecompGraph.post_process(i) for i in self.datasets]

        if split == 'train':
            self.bert_tokenizer = AutoTokenizer.from_pretrained(cfg.config)
            if os.path.exists(cfg.vocab_dir):
                logger.info(f'Loading Vocabularys From {cfg.vocab_dir}!')
                with open(cfg.vocab_dir, 'rb') as f:
                    self.data2vec = pickle.load(f)
            else:
                self.data2vec = defaultdict(list)
                self.build_uds_emb_map()
                logger.info(f'Saving Vocabularys To {cfg.vocab_dir}!')
                try:
                    with open(cfg.vocab_dir, 'wb') as f:
                        pickle.dump(self.data2vec, f)
                except Exception as e:
                    f.close()
                    os.remove(cfg.vocab_dir)
                    exit(0)
        else:
            assert train_dataset is not None
            self.bert_tokenizer = train_dataset.bert_tokenizer
            self.data2vec = train_dataset.data2vec

        self.sp_token = self.data2vec["sp_token"]
        self.emb_input_to_vector()
        self.post_process()

    def proc(self, graph):
        _, graph = graph
        d = DecompGraph(graph)
        return d.get_list_data()

    def post_process(self):
        self.generate_vocab_size = len(self.sp_token)
        self.upos_vocab_size = self.vocab["src_pos_tags"].vocab_size
        self.label_vocab_size = self.vocab["sem_label"].vocab_size
        self.edge_vocab_size = self.vocab["sem_edges"].vocab_size
        self.syn_vocab_size = self.vocab["syn_head_tags"].vocab_size
        self.node_attr_num = len(NODE_ONTOLOGY)
        self.edge_attr_num = len(EDGE_ONTOLOGY)

    def build_uds_emb_map(self):
        logger.info(f"Building Vocabulary!")
        self.sp_token = [DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN] + self.sp_token
        self.data2vec["sp_token"] = self.sp_token

        char_emb = BaseVocab.from_datasets(self.datasets, 'src_tokens', pre_tokenize=self._token2char, min_occur_count=self.cfg.min_occur_count,
                                           max_vocab_size=self.cfg.max_vocab_size, sp_token=self.sp_token)
        word_emb = BaseVocab.from_datasets(self.datasets, 'src_tokens', min_occur_count=self.cfg.min_occur_count,
                                           max_vocab_size=self.cfg.max_vocab_size, sp_token=self.sp_token)
        word_emb.load_pretrained_embs(self.cfg.glove_dir, self.cfg.glove_dim)
        self.data2vec['src_tokens'].append((char_emb, '_char'))
        self.data2vec['src_tokens'].append((word_emb, ''))

        pos_emb = BaseVocab.from_datasets(self.datasets, 'src_pos_tags', min_occur_count=self.cfg.min_occur_count,
                                          max_vocab_size=self.cfg.max_vocab_size)
        self.data2vec['src_pos_tags'].append((pos_emb, ''))

        syn_head_tag_emb = BaseVocab.from_datasets(self.datasets, 'syn_head_tags', min_occur_count=self.cfg.min_occur_count,
                                                   max_vocab_size=self.cfg.max_vocab_size)
        self.data2vec['syn_head_tags'].append((syn_head_tag_emb, ''))

        self.data2vec['src_map'].append((self._adjacency_to_mat, ''))
        # self.data2vec['tgt_map'].append((self._adjacency_to_mat, ''))
        self.data2vec['src_tokens'].append((self._bert_tokenize, None))
        self.data2vec['node_attr'].append((self._node_attr2vec, None))
        self.data2vec['edge_attr'].append((self._edge_attr2vec, None))

        sem_label_emb = BaseVocab(['arg', 'pred', 'predhead', 'argpred', 'predpredhead'])
        self.data2vec['sem_label'].append((sem_label_emb, ''))
        self.data2vec['to_sem_type'].append((sem_label_emb, ''))

        sem_edges_emb = BaseVocab.from_datasets(self.datasets, 'sem_edges', min_occur_count=self.cfg.min_occur_count,
                                           max_vocab_size=self.cfg.max_vocab_size)
        self.data2vec['sem_edges'].append((sem_edges_emb, ''))
        
        self.data2vec = dict(self.data2vec)

    def _bert_tokenize(self, tokens):
        token_ids = self.bert_tokenizer(tokens, is_split_into_words=True)
        word_ids = token_ids.word_ids()
        gather_indexes = [[] for _ in tokens]
        for k, v in enumerate(word_ids):
            if v is not None:
                gather_indexes[v].append(k)
        return {"_bert": token_ids['input_ids'], "_bert_map": gather_indexes}

    def _token2char(self, lst):
        asw = []
        for i in lst:
            if i in self.sp_token:
                asw.append([i])
                continue
            tmp = []
            for j in i:
                tmp.append(j)
            asw.append(tmp)
        return asw

    def _to_lower(self, lst):
        if isinstance(lst, list):
            return [self._to_lower(x) for x in lst]
        return lst.lower() if lst not in self.sp_token else lst

    def _dict2vec(self, keys, field, data):
        vec = {}
        if isinstance(data, dict):
            for j in field:
                tmp = []
                for key in keys:
                    if key in data.keys():
                        tmp.append(data[key][j])
                    else:
                        tmp.append(0)
                vec[j]= tmp
            return vec
        elif isinstance(data, list):
            for i in field:
                vec[i] = []
            for i in data:
                dct = self._dict2vec(keys, field, i)
                for j in field:
                    vec[j].append(dct[j])
            return vec
        else:
            raise NotImplementedError

    def _edge_attr2vec(self, data):
        return self._dict2vec(EDGE_ONTOLOGY, ['confidence', 'value'], data)

    def _node_attr2vec(self, data):
        return self._dict2vec(NODE_ONTOLOGY, ['confidence', 'value'], data)

    def _adjacency_to_mat(self, data):
        sze = len(data)
        mat = [[0 for _ in range(sze)] for _ in range(sze)]
        for i in data:
            mat[i[0]][i[1]] = 1
        return mat
