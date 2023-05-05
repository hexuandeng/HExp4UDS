import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, field
from heuds.base.base_config import BaseConfig
from heuds.constant import register_model
from heuds.uds.syn_edge_parser import SynEdgeParser, SynEdgeParserConfig
from heuds.uds.sem_label_parser import NodeClassification, NodeClassificationConfig
from heuds.models.bert.bert_word_level import WordLevelBertConfig, WordLevelBert
from heuds.scoring.attachment_score import AttachmentScores

@dataclass
class SyntacticBERTConfig(BaseConfig):
    encoder: WordLevelBertConfig = field(
        default=WordLevelBertConfig(), metadata={"help": "encoder model config"}
    )
    syntax_upos: NodeClassificationConfig = field(
        default=NodeClassificationConfig(), metadata={"help": "SynEdgeParserConfig"}
    )
    syntax_edge: SynEdgeParserConfig = field(
        default=SynEdgeParserConfig(), metadata={"help": "SynEdgeParserConfig"}
    )
    conllu_file: str = field(
        default="datasets/tmp.conllu", metadata={"help": "how many subprocesses to use for data loading"}
    )

class SyntacticModel(nn.Module):
    def __init__(self, arch, cfg, dataset, *args, **kwargs) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_name = arch
        self.dataset = dataset

        # model initializing
        self.encoder = WordLevelBert(cfg.encoder)
        self.syntax_upos = NodeClassification(cfg.syntax_upos, self.encoder.output_dim, dataset.upos_vocab_size)
        self.syntax_edge = SynEdgeParser(
            cfg.syntax_edge, self.encoder.output_dim, self.encoder.output_dim, dataset.syn_vocab_size)
        self.syn_metric = AttachmentScores()

    def forward(self, input) -> Dict:
        if 'src_mask' in input.keys():
            word_mask = input['src_mask']
        else:
            word_mask = ~input['src_tokens'].eq(0)

        encoder_output = self.encoder(input['src_tokens_bert'], input['src_tokens_bert_map'], word_mask, return_all_hiddens=True)
        syntax_upos = self.syntax_upos(encoder_output["encoder_out"])
        syntax_edge = self.syntax_edge(encoder_output["encoder_out"], 
            encoder_output["encoder_out"], encoder_output["mask"], input["syn_head_indices"], input["syn_edge_mask"])

        return {
            "encoder_mask": encoder_output["mask"],
            "syntax_upos": syntax_upos,
            "syntax_edge": syntax_edge
        }

    def compute_loss(self, input):
        output = self.forward(input)
        loss = self.syntax_upos.loss(output['syntax_upos'], input['src_pos_tags'], output["encoder_mask"])
        loss += self.syntax_edge.loss(output["syntax_edge"], input['syn_head_indices'], 
                                            input['syn_head_tags'], output["encoder_mask"], self.syn_metric)

        return loss

    def get_conllu(self, input):
        output = self.forward(input)
        mask = output["encoder_mask"]
        syntax_upos = output["syntax_upos"]
        syntax_edge = output["syntax_edge"]

        pos_tag_vocab = self.dataset.data2vec['src_pos_tags'][0][0].id2word
        head_tag_vocab = self.dataset.data2vec['syn_head_tags'][0][0].id2word
        sentences = input['src_tokens_str']
        _, pos_tags = syntax_upos.max(dim=-1)

        shape = mask.size()[:-1] + torch.Size([1])
        key_mask = torch.cat([mask, mask.new_zeros(shape)], dim=-1).unsqueeze(1).bool()
        edge_head_ll = syntax_edge["edge_head_ll"].masked_fill(~key_mask, float('-inf'))
        _, edge_heads = edge_head_ll.max(dim=-1)
        edge_types = syntax_edge["edge_types"]

        conllu = ""
        for id, sentence in enumerate(sentences):
            for place, token in enumerate(sentence):
                pos = pos_tags[id][place].item()
                head = edge_heads[id][place].item()
                type = edge_types[id][place].item()
                if place == head:
                    head = -1
                conllu += "\t".join([str(place + 1), token, token, pos_tag_vocab(pos), pos_tag_vocab(pos), 
                                        "_", str(head + 1), head_tag_vocab(type), "_", "_"])
                conllu += "\n"
            conllu += "\n"

        with open(self.cfg.conllu_file, "a", encoding="utf-8") as f:
            f.write(conllu)

    def reset_metric(self):
        self.syntax_upos.reset()
        self.syntax_edge.reset()

    def get_metric(self, *args, **kwargs):
        metric = {  
            "Syntax_Upos": self.syntax_upos.get_metric(),
            "Syntax_Edge": self.syntax_edge.get_metric()
        }
        metric["Metric"] = metric["Syntax_Edge"]["UAS"] + metric["Syntax_Edge"]["LAS"]
        return metric  

register_model((SyntacticModel, SyntacticBERTConfig), 'Bert_Syntactic')
