from copy import deepcopy

from typing import Dict
import pandas as pd
from itertools import product
import networkx as nx

from lm_service.folding import fold_graph_top
from spacy.tokens import Doc
import logging

# import pygraphviz as pgv
from lm_service.graph import dep_tree_from_phrase

from dataclasses import dataclass, field
from typing import Optional, List, Set, Tuple


logger = logging.getLogger(__name__)

"""
    relation extraction module based on spacy.
    For each phrase:
    1. cast the dependency tree as a metagraph (unify noun chunk) 
    2. identify relation candidates in the metagraph based of tags
    3. for each relation candidate identify a pool of candidate sources and targets
    4. choose closest / best sources/ target candidates for each relation to form (relation, source, target) triples    

"""


class RelationHasNoTargetCandidatesError(Exception):
    pass


class ACandidate:
    @staticmethod
    def concretize(x, graph):
        # return lemma if not entity, otherwise return text
        return (
            graph.nodes[x]["lemma"]
            if not graph.nodes[x]["ent_iob"] not in (0, 2)
            else graph.nodes[x]["text"]
        )


class Token:
    def __init__(self, i, dep_, tag_, **kwargs):
        self.i: int = i
        self.dep_: str = dep_
        self.tag_: str = tag_


@dataclass
class Relation(ACandidate):
    @property
    def tokens(self):
        return [t.i for t in self.atokens]

    r0: Optional[int] = None
    passive: bool = False
    atokens: Optional[List[Token]] = field(default_factory=list)

    def project_to_text(self, graph):
        ll = [ACandidate.concretize(r, graph) for r in self.tokens]
        return ll

    def project_to_text_str(self, graph):
        ll = self.project_to_text(graph)
        txt = "".join([ll[0]] + [x.capitalize() for x in ll[1:]])
        return txt


@dataclass
class TripleCandidate:
    source: int
    relation: Relation
    target: int

    def project_to_text(self, graph):
        s = ACandidate.concretize(self.source, graph)
        t = ACandidate.concretize(self.target, graph)
        r = self.relation.project_to_text_str(graph)
        return s, r, t


class RelationPile:
    def __init__(self, relations: List[Relation]):
        self.relations: List[Relation] = relations
        self.tokens: Set[int] = set([x for r in self.relations for x in r.tokens])
        # self.map: pd.DataFrame = pd.DataFrame(
        #     [(r._id, x) for r in self.relations for x in r.tokens],
        #     columns=["rid", "token"],
        # )
        self.map: Dict[int, List[int]] = {
            r.r0: [x for x in r.tokens] for r in self.relations
        }

    def __repr__(self):
        return str(self.map)

    def __iter__(self):
        for r in self.relations:
            yield r


class SourcePile:
    def __init__(self, sources: List[int]):
        self.data: List[int] = sources


class TargetPile:
    def __init__(self, targets: List[int]):
        self.data: List[int] = targets


@dataclass
class CandidatePile:
    sources: SourcePile
    targets: TargetPile
    relations: RelationPile


class CorefGraph:
    def __init__(self, graph: nx.DiGraph, root: int, map_specific: Dict[int, int]):
        self.graph: nx.DiGraph = graph
        self.root: int = root
        self.map_specific: Dict[int:int] = map_specific


def render_coref_graph(rdoc: Doc, graph: nx.DiGraph, full=False):

    chains = rdoc._.coref_chains
    vs_coref = []
    es_coref = []

    mention_nodes = []
    chain_specific_mention = dict()
    concept_specific_blank = dict()

    coref_root = max([token.i for token in rdoc]) + 1
    jc = coref_root + 1

    vs_coref += [
        (
            coref_root,
            {"label": f"{coref_root}-*-coref-root", "tag_": "coref", "dep_": "root"},
        )
    ]

    for jchain, chain in enumerate(chains):
        coref_chain = jc
        chain_specific_mention[coref_chain] = chain[chain.most_specific_mention_index]
        vs_coref += [
            (
                coref_chain,
                {
                    "label": f"{coref_chain}-*-coref-chain",
                    "tag_": "coref",
                    "dep_": "chain",
                    "chain": jchain,
                },
            )
        ]
        es_coref.append((coref_root, coref_chain))
        jc += 1
        for kth, x in enumerate(chain.mentions):
            coref_blank = jc
            if kth == chain.most_specific_mention_index:
                concept_specific_blank[coref_chain] = coref_blank
            vs_coref += [
                (
                    coref_blank,
                    {
                        "label": f"{coref_blank}-*-coref-blank",
                        "tag_": "coref",
                        "dep_": "blank",
                        "chain": jchain,
                    },
                )
            ]
            es_coref.append((coref_chain, coref_blank))
            jc += 1
            mention_nodes.extend(x.token_indexes)
            for y in x.token_indexes:
                vs_coref += [(y, graph.nodes[y])]
                es_coref.append((coref_blank, y))

    chs = {
        j: [[graph.nodes[x]["lower"] for x in item] for item in k.mentions]
        for j, k in enumerate(chains)
    }
    logger.info(f"{chs}")
    chs = {
        j: [graph.nodes[x]["lower"] for x in k.mentions[k.most_specific_mention_index]]
        for j, k in enumerate(chains)
    }
    logger.info(f"specifics {chs}")

    if full:
        coref_graph = deepcopy(graph)
    else:
        coref_graph = nx.DiGraph()

    coref_graph.add_nodes_from(vs_coref)
    coref_graph.add_edges_from(es_coref)
    # cg = CorefGraph(coref_graph, coref_root, chain_specific_mention)
    return coref_graph, mention_nodes, concept_specific_blank


def render_mstar_graph(rdoc: Doc, graph: nx.DiGraph) -> nx.DiGraph:
    coref_graph, mention_nodes, concept_specific_blank = render_coref_graph(rdoc, graph)

    for m in mention_nodes:
        # find m_star
        blanks = list(coref_graph.predecessors(m))
        blank_metrics = []
        for b in blanks:
            # one concept per blank
            c0 = [concept for concept in coref_graph.predecessors(b)][0]
            best_blank_per_concept = concept_specific_blank[c0]
            specific_mentions = list(coref_graph.successors(best_blank_per_concept))
            blank_metrics += [
                (
                    best_blank_per_concept,
                    propotion_of_pronouns(coref_graph, specific_mentions),
                )
            ]
        blank_metrics = sorted(blank_metrics, key=lambda item: item[1])
        coref_graph.nodes[m]["m*"] = list(coref_graph.successors(blank_metrics[0][0]))

    return coref_graph


def propotion_of_pronouns(graph, mentions):
    return sum([graph.nodes[m]["tag_"].startswith("PRP") for m in mentions]) / len(
        mentions
    )


def find_relation_candidates(graph):
    r_candidates = [
        v
        for v in graph.nodes()
        if graph.nodes[v]["tag_"][:2] == "VB" and graph.nodes[v]["dep_"] != "aux"
    ]
    return r_candidates


def find_relation_candidates_new(graph: nx.DiGraph):
    r_candidates = []
    for v in graph.nodes():
        cand = Relation()
        if graph.nodes[v]["tag_"].startswith("VB"):
            # TODO check graph.nodes[v]["dep_"] != "aux"
            vtoken = Token(**graph.nodes[v])
            if vtoken.tag_ == "VBN" and vtoken.dep_ == "acl":
                cand.passive = True

            if len(list(graph.successors(v))) > 0:
                cand.atokens = [vtoken]
            for w in graph.successors(v):
                wtoken = Token(**graph.nodes[w])

                if wtoken.tag_.startswith("VB"):
                    if (
                        vtoken.tag_ == "VBN"
                        and
                        # VBN or VBZ
                        (
                            "VB" in wtoken.tag_
                            # auxpass or aux
                            and "aux" in wtoken.dep_
                        )
                    ):
                        cand.atokens = [wtoken] + cand.atokens
                        cand.passive = True
                    elif (
                        vtoken.tag_ == "VBZ"
                        and wtoken.tag_ == "VBN"
                        and wtoken.dep_ == "advcl"
                    ):
                        cand.atokens = cand.atokens + [wtoken]
                        cand.passive = True
                if (wtoken.tag_ == "IN" and wtoken.dep_ == "prep") or (
                    wtoken.tag_ == "IN" and wtoken.dep_ == "agent"
                ):
                    if any([t.tag_ == "IN" for t in cand.atokens]):
                        cand2 = deepcopy(cand)
                        for j, t in enumerate(cand2.atokens):
                            if t.tag_ == "IN":
                                cand2.atokens[j] = wtoken
                        r_candidates += [cand2]
                    else:
                        cand.atokens = cand.atokens + [wtoken]
        if cand.tokens:
            r_candidates += [cand]
    for j, r in enumerate(r_candidates):
        r.r0 = j

    rp = RelationPile(relations=r_candidates)
    return rp


def maybe_source(n) -> bool:
    return (("NN" in n["tag_"]) or (n["tag_"] == "PRP")) and (n["dep_"] != "pobj")


def maybe_target(n) -> bool:
    return ("NN" in n["tag_"]) or (n["dep_"] == "pobj") or (n["dep_"] == "ccomp")


def check_condition(graph, s, foo_condition) -> bool:
    logger.debug(f" {s} : {id(graph)} : {graph.nodes[s]}")
    flag = [foo_condition(graph.nodes[s])]
    if "leaf" in graph.nodes[s]:
        leaf = graph.nodes[s]["leaf"]
        flag += [foo_condition(prop) for n, prop in leaf.nodes if n != s]
    return any(flag)


def find_candidates(graph: nx.DiGraph):
    rp = find_relation_candidates_new(graph)

    source_candidates = [
        i for i in graph.nodes if check_condition(graph, i, maybe_source)
    ]
    target_candidates = [
        i for i in graph.nodes if check_condition(graph, i, maybe_target)
    ]

    logger.info(f" relations: {rp}")
    for r in rp.relations:
        logger.info(f" relations: {[graph.nodes[r0]['lower'] for r0 in r.tokens]}")
    logger.info(
        f" sources: {source_candidates} {[graph.nodes[r]['lower'] for r in source_candidates]}"
    )
    logger.info(
        f" targets: {target_candidates} {[graph.nodes[r]['lower'] for r in target_candidates]}"
    )

    return CandidatePile(
        relations=rp,
        sources=SourcePile(source_candidates),
        targets=TargetPile(target_candidates),
    )


def compute_distances(graph: nx.DiGraph, rp: RelationPile):

    undirected = graph.to_undirected()
    greverse = graph.reverse()
    nx.set_edge_attributes(greverse, values=-1, name="weight")
    gextra = graph.copy()
    nx.set_edge_attributes(gextra, values=1, name="weight")

    gextra.add_weighted_edges_from(
        [(u, v, greverse.edges[u, v]["weight"]) for u, v in greverse.edges],
        weight="weight",
    )

    # compute distances
    paths = {r: nx.shortest_path(gextra, r) for r in rp.tokens}
    path_weights = {
        r: {v: nx.path_weight(gextra, pp, "weight") for v, pp in batch.items()}
        for r, batch in paths.items()
    }

    distance_directed = {r: nx.shortest_path_length(graph, r) for r in rp.tokens}

    # dm = pd.DataFrame.from_dict(distance_directed).sort_index(axis=0)

    # distance_reverse = {r: nx.shortest_path_length(greverse, r) for r in rs}
    # rdm = pd.DataFrame.from_dict(distance_reverse).sort_index(axis=0)

    distance_undirected = {r: nx.shortest_path_length(undirected, r) for r in rp.tokens}

    # undirected graph distance matrix
    udm = pd.DataFrame.from_dict(distance_undirected).sort_index(axis=0)

    # weighted graph distance matrix
    wdm = pd.DataFrame.from_dict(path_weights).sort_index(axis=0)

    return undirected, distance_directed, udm, wdm


def parse_relations_basic(graph):
    """
    find triplets in a dep graph:
        a. find relation candidates
        b. find source candidates
        c. find target candidates

    :param graph: nx.Digraph

    :return:
    """

    triples = []

    cp = find_candidates(graph)

    # create relevant graphs for distance calculations : undirected, reversed ...

    undirected, distance_directed, udm, wdm = compute_distances(graph, cp.relations)

    t_cand = dict()
    s_cand = dict()

    target_candidates = cp.targets.data
    source_candidates = cp.sources.data

    # find targets per relation; targets are down the tree
    for r_parent, rels in cp.relations.map.items():
        dist_r_parent = []
        for r in rels:
            dist = distance_directed[r]
            # for r, dist in distance_directed.items():
            # find min distance to source candidate on the tree wrt relation r
            # target could be the same as r (if subgraph is hiding in r)
            dist_to_targets = [
                (r, k, dist[k])
                for k in target_candidates
                if k in dist and k not in rels
            ]
            dist_r_parent += dist_to_targets

        if dist_r_parent:
            min_dist = min([d for _, _, d in dist_r_parent])
            # find all such targets
            t_cand[r_parent] = set(
                [
                    target
                    for relation_part, target, d_rk in dist_r_parent
                    if d_rk == min_dist
                ]
            )

        if not dist_r_parent:
            t_cand[r_parent] = set()
            logger.error(f" relation {r_parent} has not target candidates")
            # raise RelationHasNoTargetCandidatesError(f" relation {r} has not target candidates")

    udm_source = list(set(source_candidates) & set(udm.index))
    wdm_source = list(set(source_candidates) & set(wdm.index))

    # find sources per relation; sources may be up the tree, using undirected graph
    # for each relation find source candidates
    # a. close to relation on the tree
    # b. negative cost preferred (close in reverse direction),
    # c. add penalty if dep is attr for given node
    for r_parent, rels in cp.relations.map.items():
        try:
            undirected_to_source = udm.loc[udm_source, rels].unstack()
        except ValueError:
            s_cand[r_parent] = []
            continue

        try:
            cost_to_source = wdm.loc[wdm_source, rels].unstack()
        except ValueError:
            s_cand[r_parent] = []
            continue

        decision = pd.concat(
            [undirected_to_source.rename("undirected"), cost_to_source.rename("cost")],
            axis=1,
        )

        # if candidate is "attr" or "dobj" add penalty (because they are likely to be targets
        decision["syn_penalty"] = pd.Series(
            decision.index.map(
                lambda x: int(graph.nodes[x[1]]["dep_"] in ["attr", "dobj"])
            ),
            index=decision.index,
        )

        decision["mcost"] = (
            decision["undirected"] + decision["cost"] + decision["syn_penalty"]
        )

        decision = decision.sort_values(
            ["mcost", "undirected", "cost", "syn_penalty"],
            ascending=[True, True, True, True],
        )
        top_row = decision.iloc[0]
        mask = (decision == top_row).all(axis=1)
        s_cand[r_parent] = decision[mask].index.get_level_values(1).tolist()

    for r in cp.relations:
        sources = s_cand[r.r0]
        targets = set(t_cand[r.r0]) - set(s_cand[r.r0])

        for s, t in product(sources, targets):
            path = nx.shortest_path(undirected, s, t)
            if any([token in path for token in r.tokens]) and s != t:
                triples += [TripleCandidate(s, r, t)]
                logger.info(
                    f" {graph.nodes[s]['lower']}, {r.project_to_text_str(graph)}, {graph.nodes[t]['lower']}"
                )
    return triples


def graph_to_relations(
    graph: nx.DiGraph, rules
) -> Tuple[nx.DiGraph, List[TripleCandidate], List[Tuple[str, str, str]], nx.DiGraph]:

    # fold graph : fold certain vertices, representing context into subgraphs
    folded_graph = fold_graph_top(graph, rules)

    logger.info(
        f"{[(n, folded_graph.nodes[n]['lower']) for n in sorted(folded_graph.nodes())]}"
    )

    # extract relation from the folded graph
    triples = parse_relations_basic(folded_graph)

    triples_projected = [tri.project_to_text(graph) for tri in triples]
    return graph, triples, triples_projected, folded_graph


def yield_star_nodes(graph, node_list):
    """
    yield most specific mentions for any mentions, given a coref graph
    :param graph:
    :param node_list:
    :return:
    """
    nlist = set()
    for n in node_list:
        if "m*" in graph.nodes[n] and n in graph.nodes[n]["m*"]:
            nlist |= {n}
        else:
            nlist |= yield_star_nodes(graph, graph.nodes[n]["m*"])
    return nlist


def expand_mstar(candidates, coref_graph):
    candidates_out = set()
    for c in candidates:
        if c in coref_graph.nodes():
            candidates_out |= yield_star_nodes(coref_graph, coref_graph.nodes[c]["m*"])
        else:
            candidates_out |= {c}
    return list(candidates_out)


def expand_candidate(candidate_token: int, metagraph, coref_graph):

    # t = st.tree
    # [(t.nodes[n]["lower"], t.nodes[n]["tag_"], t.nodes[n]["dep_"]) for n in t.nodes() if "lower" in t.nodes[n]]

    candidates = [candidate_token]
    if metagraph.nodes[candidate_token]["leaf"].is_compound():
        candidates = metagraph.nodes[candidate_token]["leaf"].compute_conj()
    candidates = expand_mstar(candidates, coref_graph)
    return candidates


def doc_to_chunks(rdoc):
    """

    :param rdoc:
    :return: (root, start, end) NB: last token is at end-1
    """
    acc = []
    for chunk in rdoc.noun_chunks:
        acc += [(chunk.root.i, chunk.start, chunk.end)]
    return acc


def parse_relations_advanced(
    phrase, nlp, rules
) -> Tuple[
    nx.DiGraph,
    nx.DiGraph,
    nx.DiGraph,
    List[TripleCandidate],
    List[Tuple[str, str, str]],
]:
    logging.info(f"{phrase}")

    rdoc, graph = dep_tree_from_phrase(nlp, phrase)

    # chunks = doc_to_chunks(rdoc)

    _, triples, triples_projected, metagraph = graph_to_relations(graph, rules)

    coref_graph = render_mstar_graph(rdoc, graph)

    triples_expanded = []

    for triple in triples:
        s_candidates = expand_candidate(
            triple.source, metagraph=metagraph, coref_graph=coref_graph
        )
        t_candidates = expand_candidate(
            triple.target, metagraph=metagraph, coref_graph=coref_graph
        )

        triples_expanded += [
            TripleCandidate(source=sp, relation=triple.relation, target=tp)
            for sp, tp in product(s_candidates, t_candidates)
        ]

    triples_proj = [tri.project_to_text(graph) for tri in triples_expanded]
    return graph, coref_graph, metagraph, triples_expanded, triples_proj
