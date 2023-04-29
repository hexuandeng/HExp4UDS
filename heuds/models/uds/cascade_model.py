import torch
from torch import nn
import torch.nn.functional as F
from collections import namedtuple
from loguru import logger
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, field
from heuds.constant import register_model
from heuds.data.base_batch import Batch
from heuds.config.base_config import BaseConfig
from heuds.models.transformer.transformer_config import EncDecBaseConfig
from heuds.models.transformer.transformer_encoder import TransformerEncoder
from heuds.models.rnn.argumented_stacked_bilstm import StackedBiLstmConfig, StackedBidirectionalLstm
from heuds.models.uds.sem_embedding import SemEmbedding
from heuds.models.bert.bert_word_level import WordLevelBertConfig, WordLevelBert
from heuds.models.uds.encoder_embedding import EncoderEmbeddingConfig, EncoderEmbedding
from heuds.models.uds.node_attr import NodeAttrParser, NodeAttrParserConfig
from heuds.models.uds.edge_attr import EdgeAttrParser, EdgeAttrParserConfig
from heuds.models.uds.syn_edge_parser import SynEdgeParser, SynEdgeParserConfig
from heuds.models.uds.sem_label_parser import NodeClassification, NodeClassificationConfig
from heuds.models.uds.sem_edge_parser import SemEdgeParser, SemEdgeParserConfig
from heuds.models.uds.sem_span_parser import SemSpanParser, SemSpanParserConfig
from heuds.modules.embeddings.word_embedding import WordEmbedding, WordEmbeddingConfig
from heuds.scoring.f1_score import F1Metrics
from heuds.scoring.attr_uds import DecompAttrMetrics
from heuds.scoring.attachment_score import AttachmentScores
from heuds.scoring.s_metric.s_metric import S
from heuds.utils import pad_cat

@dataclass
class CascadeLSTMConfig(BaseConfig):
    encoder_embed: EncoderEmbeddingConfig = field(
        default=EncoderEmbeddingConfig(), metadata={"help": "encoder model config"}
    )
    encoder: StackedBiLstmConfig = field(
        default=StackedBiLstmConfig(), metadata={"help": "encoder model config"}
    )
    syntax_upos: NodeClassificationConfig = field(
        default=NodeClassificationConfig(), metadata={"help": "SynEdgeParserConfig"}
    )
    syntax_edge: SynEdgeParserConfig = field(
        default=SynEdgeParserConfig(), metadata={"help": "SynEdgeParserConfig"}
    )
    semantic_embedding: WordEmbeddingConfig = field(
        default=WordEmbeddingConfig(), metadata={"help": "SynEdgeParserConfig"}
    )
    semantic_label: NodeClassificationConfig = field(
        default=NodeClassificationConfig(), metadata={"help": "how many subprocesses to use for data loading"}
    )
    semantic_edge: SemEdgeParserConfig = field(
        default=SemEdgeParserConfig(), metadata={"help": "how many subprocesses to use for data loading"}
    )
    semantic_span: SemSpanParserConfig = field(
        default=SemSpanParserConfig(), metadata={"help": "how many subprocesses to use for data loading"}
    )
    node_attr: NodeAttrParserConfig = field(
        default=NodeAttrParserConfig(), metadata={"help": "NodeAttrParserConfig"}
    )
    edge_attr: EdgeAttrParserConfig = field(
        default=EdgeAttrParserConfig(), metadata={"help": "EdgeAttrParserConfig"}
    )
    contact_ud: bool = field(
        default=False, metadata={"help": "SynEdgeParserConfig"}
    )
    layer_order: Tuple[int, ...] = field(
        default=(4, 4, 4, 4, 4, 4, 4), metadata={"help": "ngram_filter_sizes for char embedding"}
    )
    layer_in_use: Tuple[int, ...] = field(
        default=(1, 1, 1, 1, 1, 1, 1), metadata={"help": "ngram_filter_sizes for char embedding"}
    )
    span_expression: bool = field(
        default=False, metadata={"help": "edge_head_vector_dim for SynEdgeParser"}
    )

@dataclass
class CascadeTransformerConfig(CascadeLSTMConfig):
    encoder: EncDecBaseConfig = field(
        default=EncDecBaseConfig(), metadata={"help": "encoder model config"}
    )
    layer_order: Tuple[int, ...] = field(
        default=(6, 6, 6, 6, 6, 6, 6), metadata={"help": "ngram_filter_sizes for char embedding"}
    )

@dataclass
class CascadeBertConfig(CascadeLSTMConfig):
    encoder: WordLevelBertConfig = field(
        default=WordLevelBertConfig(), metadata={"help": "encoder model config"}
    )
    layer_order: Tuple[int, ...] = field(
        default=(12, 12, 12, 12, 12, 12, 12), metadata={"help": "ngram_filter_sizes for char embedding"}
    )

class CascadeModel(nn.Module):
    def __init__(self, arch, cfg, dataset) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.model_name = arch
        self.contact_ud = cfg.contact_ud
        self.sem_label_vocab = dataset.data2vec['sem_label'][0][0]

        self.model_init(cfg)
        self.model_list = [self._syntax_upos, self._syntax_edge, self._semantic_label, self._semantic_span, 
                           self._semantic_edge, self._node_attr, self._edge_attr]
        assert len(cfg.layer_order) == len(self.model_list)
        assert len(cfg.layer_in_use) == len(self.model_list)
        self.cfg.layer_order = list(cfg.layer_order)
        for i in range(len(cfg.layer_order)):
            self.cfg.layer_order[i] -= 1

    def model_init(self, cfg):
        for i, j in self.dataset.data2vec['src_tokens']:
            if j == '_char':
                char_vocab = i
            if j == '':
                glove_vocab = i

        # encoder initializing
        if "Transformer" in self.model_name:
            self.encoder_embed = EncoderEmbedding(
                cfg.encoder_embed, char_vocab, glove_vocab)
            self.encoder = TransformerEncoder(
                cfg.encoder, self.encoder_embed.output_dim)
        elif "LSTM" in self.model_name:
            self.encoder_embed = EncoderEmbedding(
                cfg.encoder_embed, char_vocab, glove_vocab)
            self.encoder = StackedBidirectionalLstm(
                cfg.encoder, self.encoder_embed.output_dim)
        elif "Bert" in self.model_name:
            self.encoder = WordLevelBert(cfg.encoder)
        else:
            raise NotImplementedError("Arch not Supported!")

        # ud syntactic tree part initializing
        if cfg.layer_in_use[0]:
            self.syntax_upos = NodeClassification(cfg.syntax_upos, self.encoder.output_dim, self.dataset.upos_vocab_size)
        if cfg.layer_in_use[1]:
            self.syntax_edge = SynEdgeParser(cfg.syntax_edge, self.encoder.output_dim, self.encoder.output_dim, self.dataset.syn_vocab_size)
            if self.contact_ud:
                self.ud_proj = nn.Linear(self.syntax_edge.output_dim + self.encoder.output_dim, self.encoder.output_dim)

        # uds semantic graph part initializing
        self.semantic_embedding = SemEmbedding(
            cfg.semantic_embedding, self.encoder.output_dim, self.dataset.label_vocab_size, cfg.span_expression)
        if cfg.layer_in_use[2]:
            self.semantic_label = NodeClassification(
                cfg.semantic_label, self.encoder.output_dim, self.dataset.label_vocab_size)
        if cfg.layer_in_use[3]:
            self.semantic_span = SemSpanParser(cfg.semantic_span, self.encoder.output_dim)
        if cfg.layer_in_use[4]:
            self.semantic_edge = SemEdgeParser(
                cfg.semantic_edge, self.encoder.output_dim, self.dataset.edge_vocab_size)
        self.s_score = F1Metrics()

        # uds sematic attribute part initializing
        if cfg.layer_in_use[5]:
            self.node_attr_decode = NodeAttrParser(cfg.node_attr, self.encoder.output_dim, self.dataset.node_attr_num)
        if cfg.layer_in_use[6]:
            self.edge_attr_model = EdgeAttrParser(cfg.edge_attr, self.encoder.output_dim, self.dataset.edge_attr_num)
        self.attr_metric = DecompAttrMetrics()

    def forward(self, input) -> Dict:
        encoder_output = self._encoder(input)
        encoder_state = encoder_output.copy()
        encoder_states = encoder_output["encoder_states"]
        output = [None for _ in self.model_list]
        encoder_out_buffer = [None for _ in range(self.cfg.encoder.layers)]

        for id, model in enumerate(self.model_list):
            if model is not None and self.cfg.layer_in_use[id] == 1:
                cnt = self.cfg.layer_order[id]
                # Transfer the encorder word embeddings into the syntactic node embeddings
                if encoder_out_buffer[cnt] is None:
                    if 'src_tokens_bert_map' in input:
                        encoder_out_buffer[cnt] = self.encoder.post_process(encoder_states[cnt], input['src_tokens_bert_map'])
                    else:
                        encoder_out_buffer[cnt] = self.encoder.post_process(encoder_states[cnt])
                encoder_state["encoder_out"] = encoder_out_buffer[cnt]

                if self.cfg.layer_in_use[1] and self.contact_ud and id >= 2:
                    encoder_state["encoder_out"] = torch.cat([encoder_state["encoder_out"], output[1]["edge_reps"]], dim=2)
                    encoder_state["encoder_out"] = self.ud_proj(encoder_state["encoder_out"])

                if id >= 3:
                    if self.cfg.span_expression and id >= 4:
                        sem_embedding, sem_embedding_root = self.semantic_embedding(
                            encoder_state["encoder_out"], input["to_sem"], input["to_sem_type"], input["sem_syn"])
                    else:
                        sem_embedding, sem_embedding_root = self.semantic_embedding(
                            encoder_state["encoder_out"], input["to_sem"], input["to_sem_type"])
                    encoder_state["sem_embedding"] = sem_embedding
                    encoder_state["sem_embedding_root"] = sem_embedding_root

                output[id] = model(input, encoder_state)

        [syntax_upos, syntax_edge, label, semantic_span, semantic_edge, node_attr, edge_attr] = output
        
        instance = None
        if not self.training and self.cfg.layer_in_use[2] and self.cfg.layer_in_use[3] and self.cfg.layer_in_use[4]:
            predict = label.max(dim=-1)[1]
            instance = self.batch_to_sem(predict, encoder_output["mask"])

            test_input = input.copy()
            test_input.update(instance)
            test_output = [None for _ in self.model_list]

            for id in [3, 4]:
                if self.model_list[id] is not None and self.cfg.layer_in_use[id] == 1:
                    encoder_state["encoder_out"] = encoder_out_buffer[cnt]
                    if self.cfg.layer_in_use[1] and self.contact_ud:
                        encoder_state["encoder_out"] = torch.cat([encoder_state["encoder_out"], output[1]["edge_reps"]], dim=2)
                        encoder_state["encoder_out"] = self.ud_proj(encoder_state["encoder_out"])

                    if self.cfg.span_expression and id == 4:
                        sem_embedding, sem_embedding_root = self.semantic_embedding(
                            encoder_state["encoder_out"], test_input["to_sem"], test_input["to_sem_type"], test_output[3].max(dim=-1).indices)
                    else:
                        sem_embedding, sem_embedding_root = self.semantic_embedding(
                            encoder_state["encoder_out"], test_input["to_sem"], test_input["to_sem_type"])
                    encoder_state["sem_embedding"] = sem_embedding
                    encoder_state["sem_embedding_root"] = sem_embedding_root

                    test_output[id] = self.model_list[id](test_input, encoder_state)

            semantic_span = test_output[3]
            semantic_edge = test_output[4]
            
        return {
            "encoder_mask": encoder_output["mask"],
            "label": label,
            "semantic_edge": semantic_edge,
            "semantic_span": semantic_span,
            "node_attr": node_attr,
            "edge_attr": edge_attr,
            "syntax_upos": syntax_upos,
            "syntax_edge": syntax_edge,
            "instance": instance
        }

    def distill(self, input) -> Dict:
        assert not self.training
        encoder_output = self._encoder(input)
        encoder_state = encoder_output.copy()
        encoder_states = encoder_output["encoder_states"]
        output = [None for _ in self.model_list]
        encoder_out_buffer = [None for _ in range(self.cfg.encoder.layers)]
        
        for id, model in enumerate(self.model_list[: 3]):
            if model is not None and self.cfg.layer_in_use[id] == 1:
                cnt = self.cfg.layer_order[id]
                # Transfer the encorder word embeddings into the syntactic node embeddings
                if encoder_out_buffer[cnt] is None:
                    if 'src_tokens_bert_map' in input:
                        encoder_out_buffer[cnt] = self.encoder.post_process(encoder_states[cnt], input['src_tokens_bert_map'])
                    else:
                        encoder_out_buffer[cnt] = self.encoder.post_process(encoder_states[cnt])
                encoder_state["encoder_out"] = encoder_out_buffer[cnt]

                if self.cfg.layer_in_use[1] and self.contact_ud and id >= 2:
                    encoder_state["encoder_out"] = torch.cat([encoder_state["encoder_out"], output[1]["edge_reps"]], dim=2)
                    encoder_state["encoder_out"] = self.ud_proj(encoder_state["encoder_out"])

                output[id] = model(input, encoder_state)

        [syntax_upos, syntax_edge, label, semantic_span, semantic_edge, node_attr, edge_attr] = output

        predict = label.max(dim=-1)[1]
        instance = self.batch_to_sem(predict, encoder_output["mask"])

        test_input = input.copy()
        test_input.update(instance)
        test_output = [None for _ in self.model_list]

        for id in [3, 4]:
            if self.model_list[id] is not None and self.cfg.layer_in_use[id] == 1:
                encoder_state["encoder_out"] = encoder_out_buffer[cnt]
                if self.cfg.layer_in_use[1] and self.contact_ud:
                    encoder_state["encoder_out"] = torch.cat([encoder_state["encoder_out"], output[1]["edge_reps"]], dim=2)
                    encoder_state["encoder_out"] = self.ud_proj(encoder_state["encoder_out"])

                if self.cfg.span_expression and id == 4:
                    sem_embedding, sem_embedding_root = self.semantic_embedding(
                        encoder_state["encoder_out"], test_input["to_sem"], test_input["to_sem_type"], test_output[3].max(dim=-1).indices)
                else:
                    sem_embedding, sem_embedding_root = self.semantic_embedding(
                        encoder_state["encoder_out"], test_input["to_sem"], test_input["to_sem_type"])
                encoder_state["sem_embedding"] = sem_embedding
                encoder_state["sem_embedding_root"] = sem_embedding_root

                test_output[id] = self.model_list[id](test_input, encoder_state)

        semantic_span = test_output[3]
        semantic_edge = test_output[4]

        input.update(instance)
        input["sem_label"] = label
        input["sem_edges"] = semantic_edge
        input["sem_syn"] = semantic_span
        input["src_pos_tags"] = syntax_upos
        input["syn_head_indices"] = syntax_edge["edge_head_ll"]
        input["syn_head_tags"] = syntax_edge["edge_type_ll"]

        return input

    def compute_loss(self, input):
        output = self.forward(input)
        loss = 0
        if self.cfg.layer_in_use[0]:
            loss += self.syntax_upos.loss(output['syntax_upos'], input['src_pos_tags'], output["encoder_mask"])
        if self.cfg.layer_in_use[1]:
            loss += self.syntax_edge.loss(output["syntax_edge"], input['syn_head_indices'], 
                        input['syn_head_tags'], output["encoder_mask"])
        if self.cfg.layer_in_use[2]:
            loss += self.semantic_label.loss(output["label"], 
                        input["sem_label"], output["encoder_mask"])
        if self.cfg.layer_in_use[3]:
            loss += self.semantic_span.loss(output["semantic_span"],
                        input["sem_syn"], input["syn_mask"])
        if self.cfg.layer_in_use[4]:
            loss += self.semantic_edge.loss(output["semantic_edge"],
                        input["sem_edges"], input["sem_mask"])
        if self.cfg.layer_in_use[5]:
            loss += self.node_attr_decode.loss(output["node_attr"], input['node_attr_value'],
                        input['node_attr_confidence'], input["sem_mask"], self.attr_metric)
        if self.cfg.layer_in_use[6]:
            loss += self.edge_attr_model.loss(output["edge_attr"], input['edge_attr_value'],
                        input['edge_attr_confidence'], input['sem_edges'].gt(1), self.attr_metric)
        
        if not self.training and self.cfg.layer_in_use[2] and self.cfg.layer_in_use[3] and self.cfg.layer_in_use[4]:
            instance = output["instance"]
            gold_instance_triple, gold_edge_triple = self.get_triple(
                input["to_sem"], input["sem_edges"], input["sem_syn"], input['src_tokens_str'], input["sem_mask"])
            pred_instance_triple, pred_edge_triple = self.get_triple(
                instance["to_sem"], output["semantic_edge"].max(dim=-1)[1], output["semantic_span"].max(dim=-1)[1], 
                input['src_tokens_str'], instance["sem_mask"])
            
            compute_args = {"seed": 0,
                    "iter_num": 1,
                    "compute_instance": True,
                    "compute_attribute": False,
                    "compute_relation": True,
                    "log_level": None,
                    "sanity_check": False,
                    "mode": "normal"
                    }

            ComputeTup = namedtuple("compute_args", sorted(compute_args))
            c_args = ComputeTup(**compute_args)

            bsz = output["encoder_mask"].shape[0]
            for batch in range(bsz):
                best_mapping, best_match_num, test_triple_num, gold_triple_num = S.get_best_match(
                    gold_instance_triple[batch], [], gold_edge_triple[batch],
                    pred_instance_triple[batch], [], pred_edge_triple[batch], c_args)
                self.s_score.update(best_match_num, test_triple_num, gold_triple_num)

        return loss

    def compute_soft_loss(self, input):
        # for those target is also a probability distribution
        output = self.forward(input)
        loss = 0
        if self.cfg.layer_in_use[0]:
            loss += self.syntax_upos.soft_loss(output['syntax_upos'], input['src_pos_tags'], output["encoder_mask"])
        if self.cfg.layer_in_use[1]:
            loss += self.syntax_edge.soft_loss(output["syntax_edge"], input['syn_head_indices'], 
                        input['syn_head_tags'], output["encoder_mask"])
        if self.cfg.layer_in_use[2]:
            loss += self.semantic_label.soft_loss(output["label"], 
                        input["sem_label"], output["encoder_mask"])
        if self.cfg.layer_in_use[3]:
            loss += self.semantic_span.soft_loss(output["semantic_span"],
                        input["sem_syn"], input["syn_mask"])
        if self.cfg.layer_in_use[4]:
            loss += self.semantic_edge.soft_loss(output["semantic_edge"],
                        input["sem_edges"], input["sem_mask"])
        if self.cfg.layer_in_use[5]:
            loss += self.node_attr_decode.soft_loss(output["node_attr"], input['node_attr_value'],
                        input['node_attr_confidence'], input["sem_mask"], self.attr_metric)
        if self.cfg.layer_in_use[6]:
            loss += self.edge_attr_model.soft_loss(output["edge_attr"], input['edge_attr_value'],
                        input['edge_attr_confidence'], input['sem_edges'].gt(1), self.attr_metric)

        return loss

    def _encoder(self, input):
        if 'Bert' in self.model_name:
            word_mask = ~input['src_tokens'].eq(0)
            encoder_output = self.encoder(input['src_tokens_bert'], input['src_tokens_bert_map'], word_mask, return_all_hiddens=True)
            return encoder_output

        enc_emb = self.encoder_embed(
            input['src_tokens'], input['src_tokens_char'], input['src_tokens_bert'], input['src_tokens_bert_map'])
        encoder_output = self.encoder(
            enc_emb["token_embeddings"], enc_emb["mask"], return_all_hiddens=True)
        return encoder_output

    def _syntax_upos(self, input, encoder_output):
        syntax_upos = self.syntax_upos(encoder_output["encoder_out"])
        return syntax_upos

    def _syntax_edge(self, input, encoder_output):
        syntax_edge = self.syntax_edge(
            encoder_output["encoder_out"], encoder_output["encoder_out"], encoder_output["mask"], input["syn_head_indices"], input["syn_edge_mask"])
        # concatenate in biaffine reps
        if self.contact_ud:
            encoder_output["encoder_out"] = torch.cat([encoder_output["encoder_out"], syntax_edge["edge_reps"]], dim=2)
            encoder_output["encoder_out"] = self.ud_proj(encoder_output["encoder_out"])
        return syntax_edge

    def _semantic_label(self, input, encoder_output):
        label = self.semantic_label(encoder_output["encoder_out"])
        return label
    
    def _semantic_span(self, input, encoder_output):
        semantic_span = self.semantic_span(
            encoder_output["encoder_out"], encoder_output["sem_embedding"], input["syn_mask"], input["sem_mask"])
        return semantic_span
    
    def _semantic_edge(self, input, encoder_output):
        semantic_edge = self.semantic_edge(encoder_output["sem_embedding_root"], input["sem_mask"])
        return semantic_edge

    def _node_attr(self, input, encoder_output):
        node_attr = self.node_attr_decode(encoder_output["sem_embedding"])
        return node_attr
    
    def _edge_attr(self, input, encoder_output):
        edge_attr = self.edge_attr_model(encoder_output["sem_embedding_root"], encoder_output["sem_embedding_root"], input['sem_edges'])
        return edge_attr

    def get_triple(self, to_sem, sem_edges, sem_syn, src_tokens_str, sem_mask):
        bsz = sem_mask.shape[0]

        gold_sem_node = to_sem.masked_fill(~sem_mask.bool(), -1).int().tolist()
        gold_instance_triple = [[('instance', id + 1, src_tokens_str[batch_id][i])
                                 for id, i in enumerate(batch) if i >= 0]
                                for batch_id, batch in enumerate(gold_sem_node)]
        gold_instance_triple = [[('instance', 0, 'root')] + i for i in gold_instance_triple]

        gold_sem_edge = sem_edges.ge(2).nonzero().tolist()
        gold_sem_edge_type = sem_edges.masked_select(sem_edges.ge(2)).int().tolist()
        gold_edge_triple = [[('type-' + str(type), head, tail) for type, (bid, head, tail) in zip(gold_sem_edge_type, gold_sem_edge) 
                            if bid == batch] for batch in range(bsz)]

        gold_syn_edge = sem_syn.ge(1).nonzero().tolist()
        gold_sem_edge_type = sem_syn.masked_select(sem_syn.ge(1)).int().tolist()
        now = 0
        for batch in range(bsz):
            cnt = gold_instance_triple[batch][-1][1]
            while now < len(gold_syn_edge) and gold_syn_edge[now][0] == batch:
                cnt += 1
                gold_instance_triple[batch].append(('instance', cnt, src_tokens_str[batch][gold_syn_edge[now][1]]))
                gold_edge_triple[batch].append(('type-syn', cnt, gold_sem_edge_type[now]))
                now += 1

        return gold_instance_triple, gold_edge_triple

    def batch_to_sem(self, predict, mask):
        bsz = predict.shape[0]
        predict = predict.masked_fill(~mask.bool(), 0)
        # Get every words that corresponding to one or two semantic nodes, with a triple [batch, sent_pos, type]
        one = predict.ge(2).nonzero()
        one_type = predict[predict.ge(2)].int()
        one_type[one_type.ge(5)] = one_type[one_type.ge(5)] - 3
        one = torch.cat([one, one_type.unsqueeze(1)], dim=-1)
        # Get every words that corresponding to two semantic nodes, with a triple [batch, sent_pos, type]
        two = predict.ge(5).nonzero()
        two_type = predict[predict.ge(5)].int() - 2
        two = torch.cat([two, two_type.unsqueeze(1)], dim=-1)
        # Here we get the triple of every semantic nodes
        triple = torch.cat([one, two], dim=0)
        # Sort the triple of every semantic nodes for easier comparison with gold ones afterwards, seems impossible for pytorch? Here comes an alternative solution.
        rank = triple[:, 0] * predict.shape[1] * 8 + triple[:, 1] * 8 + triple[:, 2]
        rank = torch.sort(rank).indices
        triple = triple[rank]
        triple = torch.cat([triple, triple.new_ones([triple.shape[0], 1])], dim=-1)
        # The words that corresponding to no semantic nodes are regarded as the syntactic nodes
        syn_mask = mask
        syn_mask[predict.ge(2)] = False

        to_sem = []
        to_sem_type = []
        sem_mask = []
        # Extract the information for every sentences in the batch
        for i in range(bsz):
            ind = triple[:, 0] == i
            to_sem.append(triple[:, 1][ind])
            to_sem_type.append(triple[:, 2][ind])
            sem_mask.append(triple[:, 3][ind])
        # Pad the information into a single matrix
        to_sem = pad_cat(to_sem)
        to_sem_type = pad_cat(to_sem_type)
        sem_mask = pad_cat(sem_mask)

        return {"to_sem": to_sem,
                "to_sem_type": to_sem_type,
                "sem_mask": sem_mask,
                "syn_mask": syn_mask}

    def get_metric(self, thresholds=None):
        UDS_in_use = self.cfg.layer_in_use[2] and self.cfg.layer_in_use[3] and self.cfg.layer_in_use[4]
        UD_in_use = self.cfg.layer_in_use[0] and self.cfg.layer_in_use[1]
        Attr_in_use = self.cfg.layer_in_use[5] and self.cfg.layer_in_use[6]
        metric = {
            "S_Score": self.s_score.get_metric() if UDS_in_use else 0,
            "Attr": self.attr_metric.get_metric(thresholds=thresholds) if Attr_in_use else 0,
            "Syntax_Upos": self.syntax_upos.get_metric() if self.cfg.layer_in_use[0] else 0,
            "Syntax_Edge": self.syntax_edge.get_metric() if self.cfg.layer_in_use[1] else 0,
            "Semantic_Label": self.semantic_label.get_metric() if self.cfg.layer_in_use[2] else 0,
            "Semantic_Span": self.semantic_span.get_metric() if self.cfg.layer_in_use[3] else 0, 
            "Semantic_Edge": self.semantic_edge.get_metric() if self.cfg.layer_in_use[4] else 0,
        }
        metric["Metric"] = 0
        if UDS_in_use:
            metric["Metric"] += metric["S_Score"]["f_score"]
        if Attr_in_use:
            metric["Metric"] += metric["Attr"]["pearson_r"]
        if not UDS_in_use and not Attr_in_use and UD_in_use:
            metric["Metric"] += metric["Syntax_Edge"]["UAS"]
        return metric 
     
    def reset_metric(self):
        if self.cfg.layer_in_use[0]:
            self.syntax_upos.reset()
        if self.cfg.layer_in_use[1]:
            self.syntax_edge.reset()
        if self.cfg.layer_in_use[2]:
            self.semantic_label.reset()
        if self.cfg.layer_in_use[3]:
            self.semantic_span.reset()
        if self.cfg.layer_in_use[4]:
            self.semantic_edge.reset()
        if self.cfg.layer_in_use[5] and self.cfg.layer_in_use[5]:
            self.attr_metric.reset()
        if self.cfg.layer_in_use[6] and self.cfg.layer_in_use[4] and self.cfg.layer_in_use[6]:
            self.s_score.reset()
    
register_model((CascadeModel, CascadeLSTMConfig), 'LSTM_UDS')
register_model((CascadeModel, CascadeTransformerConfig), 'Transformer_UDS')
register_model((CascadeModel, CascadeBertConfig), 'Bert_UDS')
