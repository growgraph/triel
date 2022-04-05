import yaml
import pkgutil
import logging
import sys
import spacy
import coreferee
import hashlib
from itertools import product
from lm_service.relation import graph_to_relations, dep_tree_from_phrase
from lm_service.relation import (
    render_coref_graph,
    render_mstar_graph,
    yield_star_nodes,
    expand_mstar,
    expand_candidate,
)

logger = logging.getLogger(__name__)


def main(phrase, nlp):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound.yaml")
    add_dict_rules = yaml.load(fp, Loader=yaml.FullLoader)

    chash = hashlib.sha256(phrase.encode("utf-8")).hexdigest()

    rdoc, graph = dep_tree_from_phrase(nlp, phrase)

    _, relations, rproj, metagraph = graph_to_relations(graph, add_dict_rules)

    coref_graph = render_mstar_graph(rdoc, graph)

    relations_transformed = []

    for s, r, t in relations:
        s_candidates = expand_candidate(s, metagraph=metagraph, coref_graph=coref_graph)
        t_candidates = expand_candidate(t, metagraph=metagraph, coref_graph=coref_graph)

        relations_transformed += [
            (sp, r, tp) for sp, tp in product(s_candidates, t_candidates)
        ]

    def project(x):
        return graph.nodes[x]["lemma"]

    relations_proj = [[project(u) for u in item] for item in relations_transformed]

    logger.info(relations_transformed)
    logger.info(relations_proj)


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")

    phrase = (
        "CHEOPS (CHaracterising ExOPlanets Satellite) is a European space telescope "
        "to determine the size of known extrasolar planets, "
        "which will allow the estimation of their mass, density, composition and their formation. "
        "Launched on 18 December 2019, it is the first Small-class "
        "mission in ESA's Cosmic Vision science programme."
    )
    main(phrase, nlp)

    # phrase2 = (
    #     "Although he was very busy with his work, Peter had had enough of it. "
    #     "He and his wife decided they needed a holiday. "
    #     "They travelled to Spain because they loved the country very much."
    # )
    # main(phrase2, nlp)
