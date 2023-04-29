import torch
from torch import nn
from transformers import BertModel
from dataclasses import dataclass, field
from heuds.config.base_config import BaseConfig
from heuds.modules.embeddings.base_embedding import BaseEmbedding

@dataclass
class WordLevelBertConfig(BaseConfig):
    pooling: str = field(
        default='average', metadata={"help": "bert model config"}
    )
    config: str = field(
        default='bert-base-cased', metadata={"help": "bert model config"}
    )
    output_dim: int = field(
        default=-1, metadata={"help": "bert model config"}
    )
    layers: int = field(
        default=12, metadata={"help": "bert model config"}
    )

class WordLevelBert(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.pooling = cfg.pooling
        self.bert_model = BertModel.from_pretrained(cfg.config)
        self.output_dim = self.bert_model.config.hidden_size
        if cfg.output_dim > 0:
            self.project_out_dim = nn.Linear(self.output_dim, cfg.output_dim, bias=False)
            self.output_dim = cfg.output_dim

    def forward(self,
                input_ids: torch.LongTensor,
                token_recovery_matrix: torch.LongTensor = None,
                encoder_mask: torch.Tensor = None,
                token_type_ids: torch.Tensor = None,
                return_all_hiddens: bool = False) -> torch.Tensor:
        """
        :param input_ids: same as it in BertModel
        :param token_type_ids: same as it in BertModel
        :param attention_mask: same as it in BertModel
        :param output_all_encoded_layers: same as it in BertModel
        :param token_recovery_matrix: [batch_size, num_tokens, num_subwords]
        """
        # encoded_layers: [batch_size, num_subword_pieces, hidden_size]
        input_ids = input_ids.to(torch.long)
        bert_output = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, 
            encoder_attention_mask=encoder_mask, output_hidden_states=return_all_hiddens)
        
        hidden_state = bert_output['last_hidden_state']
        hidden_state = self.post_process(hidden_state, token_recovery_matrix)

        return {
            "encoder_out": hidden_state,  # B x T x C
            "encoder_embedding": bert_output['hidden_states'][0],  # B x T x C
            "encoder_states": bert_output['hidden_states'][1: ],  # List[T x B x C]
            "mask": encoder_mask
        }

    def post_process(self, hidden_state, token_recovery_matrix, *args):
        token_recovery_matrix = token_recovery_matrix.to(torch.long)
        if token_recovery_matrix is not None:
            if 'max' in self.pooling:
                hidden_state = self.max_pooling(hidden_state, token_recovery_matrix)
            else:
                hidden_state = self.average_pooling(hidden_state, token_recovery_matrix)
        if hasattr(self, 'project_out_dim'):
            hidden_state = self.project_out_dim(hidden_state)
            
        return hidden_state

    @staticmethod
    def average_pooling(encoded_layers: torch.FloatTensor,
                        token_subword_index: torch.LongTensor) -> torch.Tensor:
        batch_size, num_tokens, num_subwords = token_subword_index.size()
        batch_index = torch.arange(
            batch_size).view(-1, 1, 1).type_as(token_subword_index)
        token_index = torch.arange(num_tokens).view(
            1, -1, 1).type_as(token_subword_index)
        _, num_total_subwords, hidden_size = encoded_layers.size()
        expanded_encoded_layers = encoded_layers.unsqueeze(1).expand(
            batch_size, num_tokens, num_total_subwords, hidden_size)
        # [batch_size, num_tokens, num_subwords, hidden_size]
        token_reprs = expanded_encoded_layers[batch_index,
                                              token_index, token_subword_index]
        subword_pad_mask = token_subword_index.eq(0).unsqueeze(3).expand(
            batch_size, num_tokens, num_subwords, hidden_size)
        token_reprs.masked_fill_(subword_pad_mask, 0)
        # [batch_size, num_tokens, hidden_size]
        sum_token_reprs = torch.sum(token_reprs, dim=2)
        # [batch_size, num_tokens]
        num_valid_subwords = token_subword_index.ne(0).sum(dim=2)
        pad_mask = num_valid_subwords.eq(0).long()
        # Add ones to arrays where there is no valid subword.
        divisor = (num_valid_subwords +
                   pad_mask).unsqueeze(2).type_as(sum_token_reprs)
        # [batch_size, num_tokens, hidden_size]
        avg_token_reprs = sum_token_reprs / divisor
        return avg_token_reprs

    @staticmethod
    def max_pooling(encoded_layers: torch.FloatTensor,
                    token_subword_index: torch.LongTensor) -> torch.Tensor:
        batch_size, num_tokens, num_subwords = token_subword_index.size()
        batch_index = torch.arange(
            batch_size).view(-1, 1, 1).type_as(token_subword_index)
        token_index = torch.arange(num_tokens).view(
            1, -1, 1).type_as(token_subword_index)
        _, num_total_subwords, hidden_size = encoded_layers.size()
        expanded_encoded_layers = encoded_layers.unsqueeze(1).expand(
            batch_size, num_tokens, num_total_subwords, hidden_size)
        # [batch_size, num_tokens, num_subwords, hidden_size]
        token_reprs = expanded_encoded_layers[batch_index,
                                              token_index, token_subword_index]
        subword_pad_mask = token_subword_index.eq(0).unsqueeze(3).expand(
            batch_size, num_tokens, num_subwords, hidden_size)
        token_reprs.masked_fill_(subword_pad_mask, -float('inf'))
        # [batch_size, num_tokens, hidden_size]
        max_token_reprs, _ = torch.max(token_reprs, dim=2)
        # [batch_size, num_tokens]
        num_valid_subwords = token_subword_index.ne(0).sum(dim=2)
        pad_mask = num_valid_subwords.eq(0).unsqueeze(2).expand(
            batch_size, num_tokens, hidden_size)
        max_token_reprs.masked_fill(pad_mask, 0)
        return max_token_reprs
