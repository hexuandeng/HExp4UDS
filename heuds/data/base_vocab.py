import numpy as np
from heuds.utils import tqdm
from collections import Counter
from loguru import logger
from heuds.constant import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN
from heuds.utils import process_multiprocessing

class BaseVocab(object):
    '''
    Build Vocab from file (now for glove only) and datasets (by counting), and add
    special tokens, while ensuring PAD_ID = 0.

    pre_tokenize is Callable func with only data as input, can do lowercase, 
    pre_tokenize, etc. Call BaseVocab directly to tokenize data, and nested list
    is supported.
    '''

    def __init__(self, vocab_list: list, sp_token: list = [], **kwargs):
        sp_token = sp_token.copy()
        self._padding_token = DEFAULT_PADDING_TOKEN
        self._oov_token = DEFAULT_OOV_TOKEN
        self.sp_token = [DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN]
        for i in self.sp_token:
            if i in sp_token:
                sp_token.remove(i)
        self.sp_token += sp_token

        for i in self.sp_token:
            if i in vocab_list:
                vocab_list.remove(i)
        self._id2word = self.sp_token + vocab_list
        def reverse(x): return dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)
        if len(self._word2id) != len(self._id2word):
            raise ValueError("serious bug: words dumplicated, please check!")

        self.unk = self._word2id[DEFAULT_OOV_TOKEN]
        self.pad = self._word2id[DEFAULT_PADDING_TOKEN]
        if DEFAULT_BOS_TOKEN in self._word2id:
            self.bos = self._word2id[DEFAULT_BOS_TOKEN]
        if DEFAULT_EOS_TOKEN in self._word2id:
            self.eos = self._word2id[DEFAULT_EOS_TOKEN]

        for key, value in kwargs.items():
            setattr(self, key, value)
        logger.info("Vocab info: #voc sizes %d" % (self.vocab_size))

        if hasattr(self, 'emb_matrix'):
            _, self.emb_size = self.emb_matrix.shape
            for _ in self.sp_token:
                to_add = np.array([[0] * self.emb_size])
                self.emb_matrix = np.insert(
                    self.emb_matrix, 0, values=to_add, axis=0)

    @classmethod
    def from_word_counter(cls, word_counter: Counter(), sp_token: list = [], min_occur_count=2, max_vocab_size=None, **kwargs):
        vocab_list = []
        for word, count in word_counter.most_common(max_vocab_size):
            if count >= min_occur_count:
                vocab_list.append(word)

        return cls(vocab_list=vocab_list, sp_token=sp_token, **kwargs)

    @classmethod
    def from_datasets(cls, inputs, key=None, pre_tokenize=None, sp_token: list = [], min_occur_count=0, max_vocab_size=None, **kwargs):
        '''
        inputs: all input datasets
        key: if input is a dict, we build vocab one at a time
        pre_tokenize: a func to tokenize the input to list of tokens
        sp_token: BOS, EOS, SEP, ect. PAD and UNK will be added automaticly
        min_occur_count: if a token occur less than min_occur_count times, we ignore it
        max_vocab_size: the maximum vocab size we allowed, and tokens that occured less are omitted
        '''
        word_counter = Counter()
        for input in inputs:
            if key is not None:
                if isinstance(key, list):
                    tmp = []
                    for i in key:
                        tmp.append(input[i])
                    input = tmp
                else:
                    input = input[key]

            # Now input should be list of tokens in dict;
            # Nested is allowed, and we flatten it afterwards.
            def flat(nested):
                res = []
                for i in nested:
                    if isinstance(i, list):
                        res.extend(flat(i))
                    else:
                        res.append(i)
                return res

            input = flat(input)
            if pre_tokenize is not None:
                input = pre_tokenize(input)
                input = flat(input)
            word_counter.update(input)

        if key is not None:
            logger.info(f"From field {key}: ", end='')
        logger.info("Loading vocab from datasets.")

        if pre_tokenize is not None:
            return cls.from_word_counter(word_counter=word_counter, sp_token=sp_token, min_occur_count=min_occur_count, max_vocab_size=max_vocab_size, pre_tokenize=pre_tokenize, **kwargs)
        else:
            return cls.from_word_counter(word_counter=word_counter, sp_token=sp_token, min_occur_count=min_occur_count, max_vocab_size=max_vocab_size, **kwargs)

    @classmethod
    def from_pretrained_embs(cls, filename, emb_size = None, sp_token: list = [], **kwargs):
        '''
        Loading from glove embedding file
        '''
        vocab_list = []
        vocab_emb = []
        
        with open(filename, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip().split()
                if emb_size is None:
                    emb_size = len(line) - 1
                vocab_list.append(' '.join(line[:-emb_size]))
                line = [float(i) if i != '.' else float(0)
                        for i in line[-emb_size:]]
                vocab_emb.append(line)
        logger.info(f"Loading vocab from embedding file {filename}.")
        return cls(vocab_list=vocab_list, sp_token=sp_token, emb_matrix=np.asarray(vocab_emb), **kwargs)

    def load_pretrained_embs(self, filename, emb_size):
        vocab_emb = {}
        embeddings = [None for _ in self._id2word]
        embeddings[self.unk] = np.zeros(emb_size, dtype='float64')
        count = 0.0

        with open(filename, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip().split()
                    
                vocab = ' '.join(line[:-emb_size])
                line = np.array([float(i) if i != '.' else 0.0
                        for i in line[-emb_size:]], dtype='float64')
                vocab_emb[vocab] = line
                embeddings[self.unk] += line
                count += 1
        embeddings[self.unk] /= count

        for id, word in enumerate(self._id2word):
            if word in self.sp_token:
                embeddings[id] = embeddings[self.unk]
                if id == self.pad:
                    embeddings[id] = np.zeros(emb_size, dtype='float64')
            elif word in vocab_emb:
                embeddings[id] = vocab_emb[word]
            elif word.lower() in vocab_emb:
                embeddings[id] = vocab_emb[word.lower()]
            else:
                embeddings[id] = embeddings[self.unk]

        self.emb_matrix = np.stack(embeddings, axis=0).astype(np.float32)
        self.emb_size = emb_size

    def word2id(self, xs):
        if isinstance(xs, list):
            return [self.word2id(x) for x in xs]
        return self._word2id.get(xs, self.unk)

    def id2word(self, xs):
        if isinstance(xs, list):
            return [self.id2word(x) for x in xs]
        return self._id2word[xs]

    @property
    def vocab_size(self):
        return len(self._id2word)

    def __len__(self):
        return len(self._id2word)

    def __call__(self, xs):
        if hasattr(self, 'pre_tokenize'):
            xs = self.pre_tokenize(xs)
        return self.word2id(xs)
