import yaml
import pkgutil
import logging
import sys
import spacy
import coreferee
import hashlib
from networkx.drawing.nx_agraph import to_agraph
from lm_service.relation import phrase_to_relations, dep_tree_from_phrase
from lm_service.relation import render_coref_graph

logger = logging.getLogger(__name__)


def main(phrase, nlp):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound.yaml")
    add_dict_rules = yaml.load(fp, Loader=yaml.FullLoader)

    chash = hashlib.sha256(phrase.encode("utf-8")).hexdigest()
    rdoc, graph = dep_tree_from_phrase(nlp, phrase)

    metagraph, r, rproj = phrase_to_relations(graph, add_dict_rules)

    coref_graph = render_coref_graph(rdoc, graph)

    coref_root = [n for n in coref_graph.nodes if coref_graph.nodes[n]["tag"] == "coref_root"][0]
    # for coref_class in coref_graph.neighbors(coref_root):
    #
    # rdoc._.coref_chains

    # simplified
    # chain_refs
    # for item in ch:

    # dot = to_agraph(mg)
    # dot.layout("dot")
    # dot.draw(path=f"./figs/{chash[:6]}.png", format="png", prog="dot")
    # dot.draw(path=f"./figs/{chash[:6]}.pdf", format="pdf", prog="dot")


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
