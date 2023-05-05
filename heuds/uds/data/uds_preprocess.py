import re
import json
import sys
import pdb
import logging
from collections import defaultdict, Counter, namedtuple
from typing import List, Dict, Hashable, TextIO, Optional, Union
from heuds.constant import *
import networkx as nx
import numpy as np
from decomp.semantics.uds.graph import UDSSentenceGraph
from loguru import logger

WORDSENSE_RE = re.compile(r'-\d\d$')
QUOTED_RE = re.compile(r'^".*"$')
NODE_ATTRIBUTES = [re.compile("wordsense.*"),
                   re.compile("genericity.*"),
                   re.compile("factuality.*"),
                   re.compile("time.*")]
EDGE_ATTRIBUTES = [re.compile("protoroles.*")]

def is_english_punct(c):
    return re.search(r'^[,.?!:;"\'-(){}\[\]]$', c)

def parse_attributes(attr_list: List, mask_list: List, ontology: List) -> Dict:
    """
    parses slices of predicted attribute/mask matrices into a dictionary
    with the ontology as keys and the attributes as values if the mask value is > 0.5. 
    Lists given per node (i.e. attr_list is the list of attribute values predicted for a single
    node) 

    Parameters
        attr_list: List
                a list of predicted attribute values where each 
                index corresponds to an ontology key 
        mask_list: List
                a list of predicted mask values 
                (does attribute apply to the node?)
        ontology: List
                the node/edge attribute ontology used 
    Returns:
        to_ret: Dict
            a dict with ontology labels as keys and attribute
            values as values, where k,v pairs only included if 
            mask value > 0.5 (i.e. attribute applies) 
    """
    if mask_list is None:
        assert (len(attr_list) == len(ontology))
        return {k: v for k, v in zip(ontology, attr_list)}

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    assert (len(attr_list) == len(ontology))
    assert (len(mask_list) == len(ontology))
    to_ret = {}

    for k, attr_v, mask_v in zip(ontology, attr_list, mask_list):
        mask_val = sigmoid(mask_v)
        if mask_val > 0.5:
            # upper and lower bound
            if attr_v > 0:
                attr_v = min(3, attr_v)
            if attr_v < 0:
                attr_v = max(-3, attr_v)
            to_ret[k] = attr_v
    return to_ret


class DecompGraph():
    def __init__(self, graph, keep_punct=False, drop_syntax=True, order="inorder", full_ud_parse=False):
        """
        :param graph: nx.Digraph
            the input decomp graph from UDSv1.0
        :param keep_punct: bool
            keep punctuation in the graph
        :param drop_syntax: bool
            flag to replace non-head syntactic relations with "nonhead" edge label
        :param order: 'sorted', 'inorder', or 'reversed'
            the linearization order. 'inorder' means that nodes are not sorted and 
            match the true word-order most closely. 'inorder' has highest performance
        """
        # remove non-semantics, non-syntax nodes
        self.graph = graph
        just_graph, self.sem_roots = self.remove_performative(graph)
        self.graph_size = len(just_graph.semantics_subgraph)

        self.name = graph.name
        self.ignore_list = []

        for node in self.graph.nodes:
            if "author" in node or \
               "addressee" in node or \
               "speaker" in node or \
                    node.endswith("arg-0") or \
                    node.endswith("-pred-root") or \
                    node.endswith("-root-0"):
                self.ignore_list.append(node)

        self.keep_punct = keep_punct
        self.drop_syntax = drop_syntax
        self.order = order
        self.full_ud_parse = full_ud_parse
        # We will set this with the gold graph and the predicted graph and use for eval
        self.semantic_graph = None

    def remove_performative(self, graph):
        """
        remove performative nodes, since in UDSv1.0 they are deterministically 
        added and have no attributes attached to them
        """
        to_remove = []
        sem_roots = []
        for node in graph.nodes:
            if "author" in node or "addressee" in node or node.endswith("arg-0") or node.endswith("pred-root"):
                if node.endswith("arg-0"):
                    sem_roots = [e[1] for e in graph.edges if e[0]
                                 == node and "root" not in e[1]]
                to_remove.append(node)
        for node in to_remove:
            graph.graph.remove_node(node)
        return graph, sem_roots

    def get_semantic_graph(self, semantic_only: bool = None):
        """
        Does DFS to linearize the decomp graph. 

        If semantics_only = False, then removing all syntax-syntax edges and propagating 
        syntax info to semantics nodes;

        Else, keeping all syntax-syntax edges and still propagating syntax info to semantics 
        nodes.

        Params
        ------
            drop_sytax: true if you want to replace all syntax-semantics edges that do not represent head relations with "nonhead"
            semantics_only: true if you only want to parse semantics nodes and drop syntax nodes from the output arboresence entirely 
        """
        sem_tag = []
        sem_syn = []
        sem_sem = []

        semantic_graph = nx.DiGraph()
        root_id = self.graph.rootid
        if semantic_only is None:
            semantic_only = "semantic_only"

        # check if the graph is valid
        if len(self.graph.semantics_subgraph.nodes) == 0 or self.graph_size == 0:
            if not self.full_ud_parse:
                #logger.info(f"skipping for not having semantics nodes")
                return None, None, None, None, None, None
            else:
                # FOR UD SOTA RESULT: assign a null semantics graph to the datapoint
                nx_g = self.graph.graph
                name = self.graph.name
                syn_node_name = list(
                    self.graph.syntax_subgraph.nodes.keys())[0]
                sem_node_name = re.sub(
                    "syntax", "semantics-arg", syn_node_name)

                attr_dict = {"domain": "semantics",
                             'frompredpatt': True, "type": "argument"}
                nx_g.add_node(sem_node_name, **attr_dict)
                edge_dict = {'domain': 'interface',
                             'type': 'head', 'id': syn_node_name}
                nx_g.add_edge(sem_node_name, syn_node_name, **edge_dict)
                # overwrite
                self.graph = UDSSentenceGraph(nx_g, name)
                self.remove_performative(self.graph)

        # adding this because you can visit a node as many times as it has incoming edges
        visitation_limit = {}
        synt = self.graph.syntax_subgraph

        # add node ids to synatx nodes
        attrs = {}
        for node in synt.nodes:
            try:
                form = synt.nodes[node]['form']
            except KeyError:
                form = ""
            try:
                upos = synt.nodes[node]['upos']
            except KeyError:
                upos = ""
            attrs[node] = {"form": form,
                           "upos": upos,
                           "id": node}

        self.graph.add_annotation(node_attrs=attrs, edge_attrs={})

        assert len(list(self.graph.nodes)) == len(set(self.graph.nodes))
        # deal with embedded preds
        for node_a in self.graph.nodes:
            if "syntax" in node_a:
                continue
            for node_b in self.graph.nodes:
                if "syntax" in node_b:
                    continue
                if node_a == node_b:
                    continue

                # node a is pred, node b is arg
                if node_b == re.sub("-pred-", "-arg-", node_a):
                    self.graph.nodes[node_b]['text'] = "SOMETHING"

        self.sem_visited = []
        spans = {}

        def children(node):
            return [e[1] for e in self.graph.graph.edges if e[0] == node]

        # semantic node dfs to deal with the removal of preformative
        # and reattach the nodes to a root node, and figure out
        # which semantic parent node is closest to a span, which deals
        # with overlapping spans
        def sem_depth_dfs(node, depth):
            if node not in self.sem_visited:
                try:
                    span = set([x for l in self.graph.span(
                        node, attrs=['id']).values() for x in l]) - set([root_id])
                except (ValueError, KeyError):
                    span = set([])
                spans[node] = (depth, span)
                self.sem_visited += [node]

                for c in children(node):
                    sem_depth_dfs(c, depth+1)

        for root in self.sem_roots:
            sem_depth_dfs(root, 1)

        # assign the syntactic span to the nearest dominating
        # syntax node
        for n1, (d1, span1) in spans.items():
            for n2, (d2, span2) in spans.items():
                if n1 == n2:
                    continue
                else:
                    if (d1 < d2 and len(span1 & span2) > 0):
                        no_intersect = span1 - span2
                        spans[n1] = (d1, no_intersect)

        syn_deps = {}
        added_synt = []

        # find the syntactic head, resorting to replacing "semantics"
        # with "syntax" where necessary (when error in UDSv1.0 graph) , rare
        for node in sorted(self.graph.semantics_subgraph.nodes):
            if node in self.ignore_list:
                continue
            try:
                syn_dep = self.graph.head(node, attrs=["form", "upos", 'id'])
            except (IndexError, KeyError) as e:
                # add syntactic dependency head manually
                synt_node = re.sub("semantics", "syntax", node)
                synt_node = re.sub("-arg", "", synt_node)
                synt_node = re.sub("-pred", "", synt_node)
                logger.info(f"tryint to get {node} head {synt_node}")
                num = int(synt_node.split("-")[2])
                synt_d = self.graph.nodes[synt_node]
                syn_dep = (num, [synt_d['form'], synt_d['upos'], synt_d['id']])

            syn_head_id = syn_dep[1][2]
            sem_tag.append((node, syn_head_id))
            try:
                semantic_graph.add_node(node, text=syn_dep[1][0], pos=syn_dep[1][1],
                                        **{k: v for k, v in self.graph.nodes[node].items()
                                           if k not in ['form', 'upos', 'id']})
            except TypeError:
                # already added SOMETHING text
                '''
                global counts
                counts += 1
                if counts % 10 == 0:
                    print(counts)
                '''
                semantic_graph.add_node(node, text=syn_dep[1][0], pos=syn_dep[1][1],
                                        **{k: v for k, v in self.graph.nodes[node].items()
                                           if k not in ['form', 'upos', 'id', 'text']})

            added_synt.append(syn_head_id)

            incoming = [se for se in self.graph.semantics_edges(
                node) if se[1] == node]

            visitation_limit[node] = len(incoming)
            syn_deps[node] = syn_dep

            # add the other syntactic children of the sem node
            if not semantic_only:
                # only add semantics children if we're not training strictly on semantics nodes
                try:
                    __, syn_children_ids = spans[node]
                    syn_children = {i: [self.graph.nodes[c]['form'], self.graph.nodes[c]['upos'],
                                        self.graph.nodes[c]['id']] for i, c in enumerate(syn_children_ids)}

                except KeyError:
                    # copula
                    assert ('semantics' in node and 'arg' in node)
                    syn_children = {}

                for (idx, (text, pos, syn_child)) in syn_children.items():
                    if syn_child not in added_synt:
                        semantic_graph.add_node(syn_child, text=text, pos=pos,
                                                **{k: v for k, v in self.graph.nodes[syn_child].items()
                                                   if k not in ['form', 'upos', 'id']})

                        if self.drop_syntax:
                            edge_label = 'nonhead'
                        else:
                            edge = (syn_head_id, syn_child)
                            try:
                                edge_label = self.graph.edges[edge]['type']
                            except KeyError:
                                # sometimes it's not in the graph
                                edge_label = "nonhead"
                        sem_syn.append((syn_child, node, edge_label))
                        semantic_graph.add_edge(
                            node, syn_child, semrel=edge_label)

                        visitation_limit[syn_child] = 1
                        added_synt.append(syn_child)

        if not semantic_only:
            ids_by_node = {}
            for node in sorted(self.graph.semantics_subgraph.nodes):
                if node in self.ignore_list:
                    continue
                syn_dep = syn_deps[node]
                syn_head_id = syn_dep[1][2]
                # now collect the span dominated by the syntactic head
                # ids keeps track locally of what is dominated by syn_head_id
                # added_synt keeps track globally of which syntax nodes have been added to graph so that we don't double-add
                # depth keeps track of how distant syn node is from root so that if it's headed by two we can assign to closer
                ids = []

                def syn_dfs(top_node, depth):
                    if top_node not in ids:
                        syn_children = [e[1] for e in self.graph.edges(
                            top_node) if "syntax" in e[1]]
                        for child in syn_children:
                            if not self.keep_punct:
                                if self.graph.nodes[child]['upos'].lower() not in ['punct']:
                                    syn_dfs(child, depth + 1)
                                    if child not in added_synt:
                                        ids.append((child, depth))
                            else:
                                syn_dfs(child, depth + 1)
                                if child not in added_synt:
                                    ids.append((child, depth))

                syn_dfs(syn_head_id, 0)
                # remove syntactic head so that it's not doubled
                ids = [x for x in ids if x not in added_synt]
                ids_by_node[node] = ids

            # postprocess ids to resolve cases where a syntactic node
            # is dominated by two different nodes and we have to pick the closer one
            for node_a, ids_and_depths_a in ids_by_node.items():
                for node_b, ids_and_depths_b in ids_by_node.items():
                    if node_a == node_b:
                        continue
                    for a_idx, (id_a, d_a) in enumerate(ids_and_depths_a):
                        for b_idx, (id_b, d_b) in enumerate(ids_and_depths_b):
                            if id_a == id_b:
                                # same syn node headed by two different nodes
                                # take less depth
                                if d_a < d_b:
                                    # delete b
                                    popped = ids_by_node[node_b].pop(b_idx)
                                else:
                                    # delete a
                                    popped = ids_by_node[node_a].pop(a_idx)

            # expand ids to include all syntactic children
            # add all syntax nodes under the semantics node up to leaves
            nodes_to_add = []
            already_dominated = []

            for node in sorted(self.graph.semantics_subgraph.nodes):
                if node in self.ignore_list:
                    continue
                ids = ids_by_node[node]
                for i, (syn_node_id, __) in enumerate(ids):
                    visitation_limit[syn_node_id] = 1
                    nodes_to_add.append((syn_node_id,
                                        self.graph.nodes[syn_node_id]['form'],
                                        self.graph.nodes[syn_node_id]['upos']))

                for (syn_node_id, text, pos) in nodes_to_add:
                    if syn_node_id not in added_synt:
                        semantic_graph.add_node(syn_node_id,
                                                text=text,
                                                pos=pos)

                        added_synt.append(syn_node_id)

                    if self.drop_syntax:
                        edge_label = 'nonhead'
                    else:
                        edge = (syn_head_id, syn_node_id)
                        try:
                            edge_label = self.graph.edges[edge]['type']
                        except KeyError:
                            # sometimes it's not in the graph
                            edge_label = "nonhead"

                    if syn_node_id not in already_dominated:
                        sem_syn.append((syn_node_id, node, edge_label))
                        semantic_graph.add_edge(
                            node, syn_node_id, semrel=edge_label)
                        already_dominated.append(syn_node_id)

        # copy semantics edges
        for e in self.graph.semantics_subgraph.edges:
            if e[0] in self.ignore_list or e[1] in self.ignore_list:
                continue

            e_val = self.graph.semantics_subgraph.edges[e]
            if e[0] != e[1]:
                e_val['semrel'] = e_val['type']
                sem_sem.append((e[1], e[0], e_val['type']))
                semantic_graph.add_edge(*e, **e_val)

        visited = defaultdict(int)
        node_list = []

        # get root, the only node that has nothing incoming
        all_sources = [e[0] for e in semantic_graph.edges]
        all_targets = [e[1] for e in semantic_graph.edges]
        potential_roots = [
            x for x in semantic_graph.nodes if x in all_sources and x not in all_targets]

        # add dummy root
        semantic_root = "dummy-semantics-root"
        visitation_limit[semantic_root] = 1
        semantic_graph.add_node(semantic_root, domain='semantics')
        for pot_root in potential_roots:
            # semantic_graph.add_edge(semantic_root, pot_root, semrel = "root")
            sem_sem.append((pot_root, semantic_root, "dependency"))
            semantic_graph.add_edge(
                semantic_root, pot_root, semrel="dependency")

        def dfs(node, relations, parent):
            if visited[node] <= visitation_limit[node]: # 此处应该没有‘=’
                node_list.append((node, relations, parent))
                # haven't visited, visit children
                visited[node] += 1
                if self.order not in ["sorted", "inorder", "reverse"]:
                    logger.warn(f"Invalid sorting order: {self.order}")
                    logger.warn(f"Reverting to 'inorder' ordering")
                    self.order = "inorder"

                if self.order == "sorted":
                    child_edges = sorted(
                        [e for e in semantic_graph.edges if e[0] == node])

                elif self.order == "inorder":
                    # inorder is the best sorting order and corresponds most closely
                    # to the order of the words in the text
                    child_edges = [
                        e for e in semantic_graph.edges if e[0] == node]
                    sem_edges = [
                        e for e in child_edges if "semantics" in e[0] and "semantics" in e[1]]
                    syn_edges = sorted([e for e in child_edges if "syntax" in e[0]
                                       or "syntax" in e[1]], key=lambda x: int(x[1].split("-")[-1]))

                    child_edges = sem_edges + syn_edges

                elif self.order == "reverse":
                    child_edges = sorted(
                        [e for e in semantic_graph.edges if e[0] == node], reverse=True)

                parent = node

                # linearize
                for child_e in child_edges:
                    relations = {k: v for k, v in semantic_graph.edges[child_e].items() if k not in [
                        "domain", "type", "frompredpatt"]}
                    child = child_e[1]
                    dfs(child, relations, parent)

        dfs(semantic_root,
            {'semrel': 'dependency'},
            semantic_root)

        # set arbor graph for eval later
        self.semantic_graph = semantic_graph

        return node_list, [semantic_root], semantic_graph, sem_tag, sem_sem, sem_syn

    def get_src_tokens(self):
        src_tokens = self.graph.sentence.split(" ")
        # get tags if read from UDlines
        pos_tags = []
        from_lines = False
        if "-root-0" in self.graph.nodes:
            try:
                pos_tags = self.graph.nodes['-root-0']['pos_tags']
            except KeyError:
                pos_tags = []
            from_lines = True
        else:
            for node in self.graph.syntax_subgraph:
                pos_tags.append(self.graph.nodes[node]['upos'])
        return src_tokens, pos_tags, from_lines

    def re_in(self, key, list_rerum):
        for regex in list_rerum:
            if regex.match(key):
                return True
        return False

    def serialize(self):
        return nx.adjacency_data(self.semantic_graph)

    def linearize_syntactic_graph(self):
        """ 
        do BFS on the syntax graph to get 
        a list of nodes, head indices, edge labels 
        doesn't add EOS or BOS tokens since those 
        change depending on concat/combo strategy 
        """
        node_list = []
        node_name_list = []
        head_inds = []
        head_labels = []
        pos_list = []

        syntax_graph = self.graph.syntax_subgraph
        if len(syntax_graph.nodes) == 0:
            # test-time, lenght is 0
            return node_list, node_name_list, head_inds, head_labels, pos_list

        possible_roots = set(syntax_graph.nodes.keys())
        for source_node, target_node in syntax_graph.edges:
            possible_roots -= set([target_node])
        try:
            assert (len(possible_roots) == 1)
        except AssertionError:
            return [], [], [], [], []
        root = list(possible_roots)[0]

        # do BFS
        edges = [(root, root)]
        used = [(root, root)]
        while len(edges) > 0:
            edge = edges.pop(0)
            head_node = edge[0]
            curr_node = edge[1]
            node_list.append(syntax_graph.nodes[curr_node]['form'])
            node_name_list.append(curr_node)
            head_inds.append(head_node)
            pos_list.append(syntax_graph.nodes[curr_node]['upos'])

            try:
                label = syntax_graph.edges[edge]['deprel']
            except KeyError:  # root
                label = "root"
            head_labels.append(label)
            for e in syntax_graph.edges:
                if e[0] == curr_node:
                    assert e not in used
                    edges.append(e)
                    used.append(e)

        return node_list, node_name_list, head_inds, head_labels, pos_list

    @staticmethod
    def get_sp_token(bos=True, eos=True):
        sp_token = set()
        sp_token.add(DEFAULT_ROOT_TOKEN)
        if bos:
            sp_token.add(DEFAULT_BOS_TOKEN)
        if eos:
            sp_token.add(DEFAULT_EOS_TOKEN)
        return sp_token
        
    def get_list_data(self, bos=True, eos=True, max_tgt_length=512):
        """
        convert a decomp graph into a shallow format where semantics nodes are labelled with their syntactic head
        and syntax nodes are simplified to have their semantic parent node as governor. All semantic edges are preserved
        except in cases of embedded predicates, where only the predicate node is preserved and all children of the 
        semantic-arg node supervening on the predicate node in the original graph are assigned to the predicate node,
        simplifying the graph and removing the argument node which does not have a unique span in the surface form. 

        After converting the graph, traverses the new graph (called an semantic_graph) and returns a list of nodes and relations.
        Using this list, builds all the data to be put into fields by ~/data/datset_readers/decomp_reader
        """
        self.sp_token = DecompGraph.get_sp_token()
        node_list, sem_roots, semantic_graph, sem_tag, sem_sem, sem_syn = self.get_semantic_graph()
        if node_list is None or sem_roots is None or semantic_graph is None:
            return None

        tgt_tokens = []
        tgt_node_name = []
        tgt_head_name = []
        tgt_head_tags = []
        tgt_node_attr = []
        tgt_edge_attr = []
        tgt_pos_tags = []

        def flatten_attrs(layered_dict):
            # flatten decomp new structure and get masks
            to_ret = {}
            for outer_key, inner_dict in layered_dict.items():
                try:
                    for inner_key, inner_vals in inner_dict.items():
                        new_key = f"{outer_key}-{inner_key}"
                        to_ret[new_key] = inner_vals
                except (KeyError, AttributeError) as e:
                    to_ret[outer_key] = {
                        "value": inner_dict, "confidence": 1.0}

            return to_ret

        node_name_list = list(semantic_graph.nodes)
        node_attr_list = []
        edge_name_list = list(semantic_graph.edges)
        edge_attr_list = []
        for node in node_name_list:
            attrs = {k: v for k, v in semantic_graph.nodes[node].items(
                ) if self.re_in(k, NODE_ATTRIBUTES)}
            node_attr_list.append(flatten_attrs(attrs))
        for edge in edge_name_list:
            attrs = {k: v for k, v in semantic_graph.edges[edge].items(
                )  if self.re_in(k, EDGE_ATTRIBUTES)}
            edge_attr_list.append(flatten_attrs(attrs))

        # add semantic subgraph
        for node, relation, parent in node_list:
            try:
                token = semantic_graph.nodes[node]['text']
                pos = semantic_graph.nodes[node]['pos']
                attrs = {k: v for k, v in semantic_graph.nodes[node].items(
                ) if self.re_in(k, NODE_ATTRIBUTES)}
            except KeyError:  # is root
                token = DEFAULT_ROOT_TOKEN
                pos = "ROOT"
                attrs = {}

            if 'semrel' in relation.keys() and relation['semrel'] not in ['dependency', 'head']:
                tgt_head_tags.append("EMPTY")
            else:
                tgt_head_tags.append(relation['semrel'])
            tgt_tokens.append(token)
            tgt_pos_tags.append(pos)
            tgt_node_name.append(node)
            tgt_head_name.append(parent)
            tgt_node_attr.append(flatten_attrs(attrs))
            edge_attrs = {k: v for k, v in relation.items() if k not in [
                'semrel', 'id']}
            tgt_edge_attr.append(flatten_attrs(edge_attrs))
        tgt_mask = [1 for _ in tgt_tokens]

        # add syntactic subgraph
        (syn_tokens, syn_node_name,
         syn_head_name, syn_head_tags, syn_pos_tags) = self.linearize_syntactic_graph()
        syn_mask = [1 for _ in syn_tokens]

        def reorder_syntax_for_encoder(tokens, inds, tags, mask, pos, nodes):
            """
            reorder tokens and relabel indices so that order corresponds to syntactic order 
            """
            # nodes has corrected ordering
            everything_zipped = zip(tokens, inds, tags, mask, pos, nodes)
            correct_order_zipped = sorted(
                everything_zipped, key=lambda x: int(x[-1].split("-")[-1]))
            if len(correct_order_zipped) == 0:
                return [], [], [], [], [], []
            new_tokens, new_inds, new_tags, new_mask, new_pos, new_nodes = [
                list(x) for x in zip(*correct_order_zipped)]
            ids = [int(x.split("-")[-1]) for x in new_nodes]
            id = 1
            for i in ids:
                assert i == id
                id += 1
            return new_tokens, new_inds, new_tags, new_mask, new_pos, new_nodes

        # no bos or eos for syntax, but it needs to re-ordered
        (syn_tokens, syn_head_name, syn_head_tags, syn_mask, syn_pos_tags, syn_node_name) = \
            reorder_syntax_for_encoder(
                syn_tokens, syn_head_name, syn_head_tags, syn_mask, syn_pos_tags, syn_node_name)

        # Source Tokens
        src_tokens = self.graph.sentence.split(" ")
        src_pos_tags = []
        src_node_name = []
        cur_id = 1
        syntax_list = sorted(list(self.graph.syntax_subgraph),
                             key=lambda x: int(x.split("-")[-1]))
        for node in syntax_list:
            assert int(node.split("-")[-1]) == cur_id
            assert src_tokens[cur_id-1] == self.graph.nodes[node]['form']
            cur_id += 1
            src_pos_tags.append(self.graph.nodes[node]['upos'])
            src_node_name.append(node)

        return {
            "src_tokens": src_tokens,
            "src_pos_tags": src_pos_tags,
            "tgt_tokens": tgt_tokens,
            "syn_head_tags": syn_head_tags,

            "src_node_name": src_node_name,
            "tgt_node_name": tgt_node_name,
            "tgt_head_name": tgt_head_name,
            "syn_node_name": syn_node_name,
            "syn_head_name": syn_head_name,

            "sem_tag": sem_tag,
            "sem_sem": sem_sem,
            "sem_syn": sem_syn,

            "node_name_list": node_name_list,
            "node_attr_list": node_attr_list,
            "edge_name_list": edge_name_list,
            "edge_attr_list": edge_attr_list
        }

    @staticmethod
    def post_process(dct):
        if dct is None:
            return None
        src_tokens = dct['src_tokens']
        src_pos_tags = dct['src_pos_tags']
        syn_head_tags = dct['syn_head_tags']
        src_node_name = dct['src_node_name']
        syn_node_name = dct['syn_node_name']
        syn_head_name = dct['syn_head_name']
        sem_tag = dct['sem_tag']
        sem_sem = dct['sem_sem']
        sem_syn_proc = dct['sem_syn']
        node_name_list = dct['node_name_list']
        node_attr_list = dct['node_attr_list']
        edge_name_list = dct['edge_name_list']
        edge_attr_list = dct['edge_attr_list']

        sem_label = [DEFAULT_OOV_TOKEN for _ in src_node_name]
        syn_mask = [i == DEFAULT_OOV_TOKEN for i in sem_label]
        to_place = {}
        for sem, syn in sem_tag:
            place = int(syn.split('-')[-1]) - 1
            label = sem.split('-')[-2]
            assert src_node_name[place] == syn
            if sem_label[place] != DEFAULT_OOV_TOKEN:
                if set([label, sem_label[place]]) == set(['arg', 'pred']):
                    sem_label[place] = 'argpred'
                elif set([label, sem_label[place]]) == set(['predhead', 'pred']):
                    sem_label[place] = 'predpredhead'
                else:
                    raise ValueError("Error in Dataset!")
            else:
                sem_label[place] = label
            to_place[sem] = place

        sem_nodes = sorted(sorted([i[0] for i in sem_tag]), key=lambda x: to_place[x])
        sem_nodes = ['dummy-semantics-root'] + sem_nodes
        def reverse(x): return dict(zip(x, range(len(x))))
        sem_map = reverse(sem_nodes)

        to_sem = []
        to_sem_type = []
        sem_mask = []
        sem_node_name = []
        node_attr = []
        cnt = 1
        def apd(id, pos, cnt):
            name = [i for i in node_name_list if i.endswith('-' + str(id + 1)) and '-' + pos + '-' in i]
            assert len(name) == 1
            sem_node_name.append(name[0])
            node_attr.append(node_attr_list[node_name_list.index(name[0])])
            to_sem.append(id)
            to_sem_type.append(pos)
            sem_mask.append(1)
            assert int(sem_nodes[cnt].split('-')[-1]) == id + 1
            assert pos in sem_nodes[cnt]
            return cnt + 1
        for id, i in enumerate(sem_label):
            if i == 'arg':
                cnt = apd(id, 'arg', cnt)
            elif i == 'pred':
                cnt = apd(id, 'pred', cnt)
            elif i == 'predhead':
                cnt = apd(id, 'arg', cnt)
            elif i == 'argpred':
                cnt = apd(id, 'arg', cnt)
                cnt = apd(id, 'pred', cnt)
            elif i == 'predpredhead':
                cnt = apd(id, 'pred', cnt)
                cnt = apd(id, 'predhead', cnt)
                edge = [(i, j, k) for (i, j, k) in sem_sem if sem_nodes[cnt-1] == i or sem_nodes[cnt - 1] == j]
                assert len(edge) == 1
                assert edge[0] == (sem_nodes[cnt-1], sem_nodes[cnt-2], 'head')

        sem_edges = [[DEFAULT_OOV_TOKEN for _ in sem_nodes] for _ in sem_nodes]
        edge_attr = [[{} for _ in sem_nodes] for _ in sem_nodes]
        for node, head, type in sem_sem:
            sem_edges[sem_map[node]][sem_map[head]] = type
            edge_attr[sem_map[node]][sem_map[head]] = edge_attr_list[edge_name_list.index((head, node))]

        sem_syn = [DEFAULT_PAD_IDX for _ in src_node_name]
        for syn, sem, type in sem_syn_proc:
            place = int(syn.split('-')[-1]) - 1
            label = sem_map[sem]
            assert src_node_name[place] == syn
            assert type == 'nonhead'
            sem_syn[place] = label

        src_map = []
        for cnt, v in enumerate(src_tokens):
            same = cnt
            for id, k in enumerate(src_tokens[: cnt]):
                if k == v:
                    same = id
                    break
            src_map.append((cnt, same))
        
        # root is identified by a self-loop, which is allowed
        syn_edge_mask = [[1 for _ in src_tokens] for _ in src_tokens]
        
        syn_head_indices = []
        for now, node in enumerate(syn_head_name):
            tgt = -1
            for id, src in enumerate(syn_node_name):
                if node == src:
                    tgt = id
            assert tgt != -1
            syn_head_indices.append(tgt)

        return {
            "src_tokens": src_tokens,
            "src_tokens_str": src_tokens,
            "src_pos_tags": src_pos_tags,

            "src_map": src_map,
            "syn_head_indices": syn_head_indices,
            "syn_head_tags": syn_head_tags,
            "syn_edge_mask": syn_edge_mask,

            "sem_label": sem_label,
            "sem_edges": sem_edges,
            "sem_syn": sem_syn,
            "to_sem": to_sem,
            "to_sem_type": to_sem_type,
            "sem_mask": sem_mask,
            "syn_mask": syn_mask,

            "node_attr": node_attr,
            "edge_attr": edge_attr
        }

    
    @staticmethod
    def build_syn_graph(nodes, edge_heads, edge_labels):
        """
        build the syntactic graph from a predicted set of nodes, 
        edge heads, and edge labels
        """
        try:
            graph = nx.DiGraph()
            for i, n in enumerate(nodes):
                attr = {"form": n}
                graph.add_node(i, **attr)

            for i, (head, label) in enumerate(zip(edge_heads, edge_labels)):
                if head == 0:
                    # root node not present, add self edge
                    edge = (i, i)
                else:
                    edge = (i, head-1)
                    if i not in graph.nodes or head-1 not in graph.nodes:
                        pdb.set_trace()
                attr = {"deprel": label}
                graph.add_edge(*edge, **attr)
            return graph

        except IndexError:
            return None

    @staticmethod
    def build_sem_graph(nodes, node_attr, corefs,
                        edge_heads, edge_labels, edge_attr):
        """
        build the semantic arbor graph from a predicted output
        """
        graph = nx.DiGraph()
        real_node_mapping = {}

        # Steps
        #########################################
        # 1. add all nodes and get node mapping #
        #########################################

        for i, (node, attr, coref) in enumerate(zip(nodes, node_attr, corefs)):
            if int(coref) != i:
                # node is a copy of a previous node, need to adjust heads, etc.
                real_node_mapping[i] = int(coref)
                # don't need to add the node
                continue

            else:
                real_node_mapping[i] = i

            node_id = f"predicted-{i}"
            attr['text'] = node
            # by default make everything semantic
            attr['type'] = 'semantics'
            graph.add_node(node_id, **attr)

        #############################################
        # 2. add semantic edges and syntactic edges #
        #############################################

        done_heads = []
        assert (len(edge_labels) == len(edge_heads) == len(edge_attr))
        for i, (label, head, attr) in enumerate(zip(edge_labels, edge_heads, edge_attr)):

            # add the edge between the original coreferrent node and new head, if they're repeated
            try:
                child_idx = real_node_mapping[i]
                head_idx = real_node_mapping[head]
            except KeyError:
                continue

            child = f"predicted-{child_idx}"
            parent = f"predicted-{head_idx}"

            if child == parent:
                # skip root-root edge
                continue

            if label != "EMPTY":
                # both parent and child are semantics nodes
                #logger.info(f"semrl is {attr['semrel']}")
                attr['semrel'] = label
                graph.nodes[parent]['type'] = 'semantics'
                graph.nodes[child]['type'] = 'semantics'
            else:
                attr = {'semrel': 'nonhead'}
                # don't set parent to syntax! that just makes all semantics nodes with syntactic children syntax nodes fool
                #graph.nodes[parent]['type'] = 'syntax'
                graph.nodes[child]['type'] = 'token'

            graph.add_edge(parent, child, **attr)

        ##########################################################
        # 3. clean up orphan nodes for later checking in scoring #
        ##########################################################

        for i, node in enumerate(graph.nodes):
            try:
                __ = graph.nodes[node]['type']
            except KeyError:
                graph.nodes[node]['type'] = "token"

        return graph
