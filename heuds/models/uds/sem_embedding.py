import torch
from torch import nn
from heuds.modules.embeddings.word_embedding import WordEmbedding, WordEmbeddingConfig
from torch.nn import Parameter
from heuds.utils import batch_index_select, batch_mask_diagonal, pad_to_tensor
from heuds.modules.attention.multihead_attention import MultiheadAttention

class SemEmbedding(nn.Module):
    def __init__(self, cfg, input_dim, word_dim, span_expression):
        super(SemEmbedding, self).__init__()
        self.input_dim = input_dim
        self.arg_embedding = WordEmbedding(cfg, word_dim)
        self.emb_proj = nn.Linear(input_dim + self.arg_embedding.output_dim, input_dim)
        self.bias_root = Parameter(torch.Tensor(1, 1, input_dim))
        nn.init.xavier_normal_(self.bias_root)
        if span_expression:
            self.encoder_attn = MultiheadAttention(
                self.input_dim,
                8,
                dropout=0.3,
                self_attention=False
            )
        
    def forward(self, encoder_outs, to_sem, to_sem_type, sem_syn=None):
        bsz, len_sem = to_sem.shape
        sem_vec = batch_index_select(encoder_outs, 1, to_sem)
        if sem_syn is not None:
            buf = []
            for i in range(len_sem):
                tmp = (sem_syn == i + 1)
                tmp[torch.arange(bsz).long(), to_sem[:, i].long()] = True
                buf.append(tmp.unsqueeze(1))
            buf = torch.cat(buf, dim=1)
            attn_mask = encoder_outs.new_zeros(buf.shape)
            attn_mask[~buf] = -torch.inf

            x, attn = self.encoder_attn(
                query=sem_vec.transpose(0, 1),
                key=encoder_outs.transpose(0, 1),
                value=encoder_outs.transpose(0, 1),
                need_weights=True,
                attn_mask=attn_mask
            )
            sem_vec = x.transpose(0, 1)

        type_vec = self.arg_embedding(to_sem_type)
        embedding = torch.cat([sem_vec, type_vec], dim=-1)
        embedding = self.emb_proj(embedding)

        return embedding, torch.cat([self.bias_root.repeat(bsz, 1, 1), embedding], dim=1)
