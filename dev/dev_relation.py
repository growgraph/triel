import yaml
import pkgutil
import logging
import sys
import spacy
import coreferee
import hashlib
from itertools import product
from lm_service.relation import graph_to_relations, dep_tree_from_phrase
from lm_service.relation import render_coref_graph, render_mstar_graph

logger = logging.getLogger(__name__)


def yield_star_nodes(graph, node_list):
    """
    yield most specific mentions for any mentions, given a coref graph
    :param graph:
    :param node_list:
    :return:
    """
    nlist = []
    for n in node_list:
        if "m*" in graph.nodes[n] and n in graph.nodes[n]["m*"]:
            nlist += [n]
        else:
            nlist += yield_star_nodes(graph, graph.nodes[n]["m*"])
    return nlist


def main(phrase, nlp):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound.yaml")
    add_dict_rules = yaml.load(fp, Loader=yaml.FullLoader)

    chash = hashlib.sha256(phrase.encode("utf-8")).hexdigest()

    rdoc, graph = dep_tree_from_phrase(nlp, phrase)

    _, relations, rproj, mg = graph_to_relations(graph, add_dict_rules)

    cg = render_mstar_graph(rdoc, graph)

    relations_transformed = []

    for s, r, t in relations:
        s_candidates = [s]
        t_candidates = [t]
        if s in cg.nodes():
            s_candidates = yield_star_nodes(cg, cg.nodes[s]["m*"])
        if t in cg.nodes():
            t_candidates = yield_star_nodes(cg, cg.nodes[t]["m*"])

        relations_transformed += [
            (sp, r, tp) for sp, tp in product(s_candidates, t_candidates)
        ]

    def project(x):
        return graph.nodes[x]["lemma"]

    relations_proj = [[project(u) for u in item] for item in relations_transformed]

    logger.info(relations_transformed)
    logger.info(relations_proj)


if __name__ == "__main__":
    # nlp = spacy.load("en_core_web_sm")
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")

    # phrase = (
    #     "CHEOPS (CHaracterising ExOPlanets Satellite) is a European space telescope "
    #     "to determine the size of known extrasolar planets, which will allow the estimation "
    #     "of their mass, density, composition and their formation"
    # )
    # main(phrase, nlp)

    phrase2 = (
        "Although he was very busy with his work, Peter had had enough of it. "
        "He and his wife decided they needed a holiday. "
        "They travelled to Spain because they loved the country very much."
    )
    main(phrase2, nlp)
