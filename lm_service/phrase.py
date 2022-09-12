from __future__ import annotations

from itertools import product
from typing import List, Tuple

import networkx as nx

from lm_service.coref import (
    coref_candidates,
    graph_component_maps,
    render_coref_maps_wrapper,
)
from lm_service.graph import phrase_to_deptree, relabel_nodes_and_key
from lm_service.onto import Token, TripleCandidate, apply_map
from lm_service.relation import (
    form_triples,
    graph_to_candidate_pile,
    graph_to_maps,
    logger,
    realign_prepositions,
)


def phrase_to_triples(
    phrase, nlp, rules, filter_pronouns=True
) -> Tuple[List[TripleCandidate], List[Tuple[str, str, str]], nx.DiGraph]:

    logger.info(f"{phrase}")

    rdoc, graph = phrase_to_deptree(nlp, phrase)

    map_tree_subtree_index = graph_component_maps(graph)

    graph_relabeled = relabel_nodes_and_key(graph, map_tree_subtree_index, "s")

    # coref maps
    (
        map_subbable_to_chain,
        map_chain_to_most_specific,
    ) = render_coref_maps_wrapper(rdoc)

    (map_subbable_to_chain_str, map_chain_to_most_specific_str,) = apply_map(
        [map_subbable_to_chain, map_chain_to_most_specific],
        map_tree_subtree_index,
    )

    triples = graph_to_triples(
        graph_relabeled,
        map_subbable_to_chain_str,
        map_chain_to_most_specific_str,
        rules,
    )

    triples = [tri.normalize_relation() for tri in triples]
    if filter_pronouns:
        triples = [tri for tri in triples if not tri.has_pronouns()]
    triples_proj = [tri.project_to_text() for tri in triples]
    return triples, triples_proj, graph


def graph_to_triples(
    graph,
    map_subbable_to_chain_str,
    map_chain_to_most_specific_str,
    rules,
) -> list[TripleCandidate]:
    """
    find triplets in a dep graph:
        a. find relation candidates
        b. find source candidates
        c. find target candidates

    :param graph: nx.Digraph
    :param map_subbable_to_chain_str:
    :param map_chain_to_most_specific_str:
    :param rules:

    :return:
    """

    # derive
    # i) pile of source, relation, target
    # ii) depot of candidates (mixed)
    # iii) modified graph to use for distance computation
    pile, candidate_depot, mod_graph = graph_to_candidate_pile(graph, rules)

    (
        sources_per_relation,
        targets_per_relation,
        rel_sources_per_relation,
        rel_targets_per_relation,
        g_undirected,
    ) = graph_to_maps(mod_graph=mod_graph, pile=pile)

    itriples = form_triples(
        pile,
        sources_per_relation,
        targets_per_relation,
        rel_sources_per_relation,
        rel_targets_per_relation,
        g_undirected,
    )

    tokens = [
        Token(
            **graph.nodes[i],
            successors=graph.successors(i),
            predecessors=graph.predecessors(i),
        )
        for i in graph.nodes()
    ]

    token_dict = {t.s: t for t in tokens}

    ecl = candidate_depot.unfold_conjunction()

    # ncp : dict[TokenIndexT, list[Candidate]]
    # for each root -> a list of relevant candidates
    ncp = coref_candidates(
        ecl,
        map_subbable_to_chain_str,
        map_chain_to_most_specific_str,
        token_dict,
    )

    triples = []
    # expand using coref and conj maps
    for tri in itriples:
        s, r, t = tri
        for sprime, tprime in product(ncp[s], ncp[t]):
            relation = realign_prepositions(pile.relations[r], tprime, graph)
            # relation = pile.relations[r]
            triples += [
                TripleCandidate(
                    source=sprime, target=tprime, relation=relation
                )
            ]

    triples = sorted(
        triples,
        key=lambda x: (x.source.sroot, x.relation.sroot, x.target.sroot),
    )

    return triples
