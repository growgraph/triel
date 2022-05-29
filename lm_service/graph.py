import networkx as nx
from spacy import Language
from spacy.tokens import Doc


def dep_tree_from_phrase(nlp: Language, document: str) -> (nx.DiGraph, Doc):
    """
    given nlp and a phrase (string) - yield spacy doc and a digraph representing syn parsing
    :param nlp:
    :param document:
    :return:
    """
    graph = nx.DiGraph()

    rdoc = nlp(document)
    vs = [
        (
            token.i,
            {
                "lower": token.lower_,
                "dep": token.dep_,
                "tag": token.tag_,
                "lemma": token.lemma_,
                "ent_iob": token.ent_iob,
                "text": token.text,
                "label": f"{token.i}-{token.lower_}-{token.dep_}-{token.tag_}",
            },
        )
        for token in rdoc
    ]
    # root = [v[0] for v in vs if v[1]["dep"] == "ROOT"][0]
    # FYI https://spacy.io/docs/api/token
    # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    es = []
    for token in rdoc:
        for child in token.children:
            es.append((token.i, child.i))

    graph.add_nodes_from(vs)
    graph.add_edges_from(es)

    return rdoc, graph


def transform_advcl(nlp: Language, phrase):
    """
    it is assumed there are no fullstops
    :param nlp:
    :param phrase:
    :return:
    """
    # find vbz
    rdoc, graph = dep_tree_from_phrase(nlp, phrase)
    vbzs = [u for u in graph.nodes() if graph.nodes[u]["tag"] == "VBZ"]
    # for each vbz perform operation
    for root in vbzs:
        succs = list(graph.successors(root))
        while succs:
            s = succs.pop()
            if graph.nodes[s]["tag"] == "VBN" and graph.nodes[s]["dep"] == "advcl":
                subgraph = nx.ego_graph(graph, root, radius=50)
                subgraph.remove_edge(root, s)
                component = nx.ego_graph(subgraph, s, radius=50)
                main_component = nx.ego_graph(subgraph, root, radius=50)
                ixs = sorted(subgraph.nodes)
                map_component_ix = dict(
                    zip(sorted(component.nodes), ixs[-len(component.nodes) :])
                )
                map_main_component_ix = dict(
                    zip(sorted(main_component.nodes), ixs[: len(main_component.nodes)])
                )
                map_component_ix.update(map_main_component_ix)
                graph = nx.relabel_nodes(graph, map_component_ix)
                succs = [map_component_ix[x] for x in succs]
            elif graph.nodes[s]["dep"] == "punct":
                graph.remove_edge(root, s)
                graph.remove_node(s)
    phrase_rep = [graph.nodes[i]["text"] for i in sorted(graph.nodes)]
    phrase_rep[0] = phrase_rep[0].capitalize()
    transformed_phrase = " ".join(phrase_rep)
    return transformed_phrase
