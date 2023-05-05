import torch
from torch import dropout, nn
from heuds.modules.embeddings.char_embedding import TokenCharactersEncoder
from heuds.modules.embeddings.word_embedding import WordEmbedding, WordEmbeddingConfig
from heuds.modules.embeddings.bert_word_embedding import BertWordEmbedding, BertWordEmbedConfig
from heuds.modules.seq2vec_encoders.cnn_seq2vec import CnnEncoder, CnnEncoderConfig
from heuds.models.transformer.transformer_encoder import TransformerEncoder
from heuds.base.base_config import BaseConfig
from dataclasses import dataclass, field, fields
from typing import List, Optional

@dataclass
class EncoderEmbeddingConfig(BaseConfig):
    char_emb: WordEmbeddingConfig = field(
        default=WordEmbeddingConfig(), metadata={"help": "char_emb config"}
    )
    word_emb: WordEmbeddingConfig = field(
        default=WordEmbeddingConfig(), metadata={"help": "word_emb config"}
    )
    char2word: CnnEncoderConfig = field(
        default=CnnEncoderConfig(), metadata={"help": "char2word config"}
    )
    bert_emb: BertWordEmbedConfig = field(
        default=BertWordEmbedConfig(), metadata={"help": "char2word config"}
    )

class EncoderEmbedding(nn.Module):
    def __init__(self, cfg: BertWordEmbedConfig, char_vocab, glove_vocab) -> None:
        super().__init__()
        self.cfg = cfg
        self.char_emb = WordEmbedding.from_vocab(cfg.char_emb, char_vocab)
        self.char_to_word = CnnEncoder(cfg.char2word, self.char_emb.output_dim)
        self.word_emb = WordEmbedding.from_vocab(cfg.word_emb, glove_vocab)
        if cfg.bert_emb:
            self.bert_emb = BertWordEmbedding(cfg.bert_emb)
        
        self.embed_dim = self.char_to_word.output_dim + self.word_emb.output_dim
        if cfg.bert_emb:
            self.embed_dim += self.bert_emb.output_dim

    @property
    def output_dim(self):
        return self.embed_dim

    def forward(self, words, chars, tokens_bert=None, bert_map=None):
        char_mask = (chars != 0).long()
        word_mask = (~words.eq(0)).long()
        emb_char = self.char_to_word(self.char_emb(chars), char_mask)
        emb_word = self.word_emb(words)
        encoder_inputs = torch.cat((emb_char, emb_word), 2)
        if hasattr(self, 'bert_emb'):
            emb_bert = self.bert_emb(tokens_bert, token_recovery_matrix=bert_map)
            encoder_inputs = torch.cat((encoder_inputs, emb_bert), 2)

        return {
            "token_embeddings": encoder_inputs, # B x T x C
            "mask": word_mask                   # B x T
        }
