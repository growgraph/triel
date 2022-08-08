import hashlib
import os
import pkgutil
from pathlib import Path

import coreferee
import spacy
import yaml

from lm_service.coref import render_coref_graph
from lm_service.folding import fold_graph_top
from lm_service.graph import phrase_to_deptree
from lm_service.util import plot_graph


def main(phrase, nlp):
    path = Path(__file__).parent
    fig_path = os.path.join(path, "figs")

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)

    chash = hashlib.sha256(phrase.encode("utf-8")).hexdigest()
    rdoc, nx_graph = phrase_to_deptree(nlp, phrase)

    plot_graph(nx_graph, fig_path, f"{chash[:6]}")

    gmetagraph = fold_graph_top(nx_graph, rules)

    plot_graph(gmetagraph, fig_path, f"{chash[:6]}_folded")

    coref_graph = render_coref_graph(rdoc, nx_graph)

    plot_graph(coref_graph, fig_path, f"{chash[:6]}_coref")


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")

    phrases = [
        "CHEOPS (CHaracterising ExOPlanets Satellite) is a European space"
        " telescope to determine the size of known extrasolar planets, which"
        " will allow the estimation of their mass, density, composition and"
        " their formation. Launched on 18 December 2019, it is the first"
        " Small-class mission in ESA's Cosmic Vision science programme.",
        "CHEOPS (CHaracterising ExOPlanets Satellite) is a European space"
        " telescope to determine the size of known extrasolar planets,"
        " which will allow the estimation of their mass, density,"
        " composition and their formation.",
        "Although he was very busy with his work, Peter had had enough of it. "
        "He and his wife decided they needed a holiday. "
        "They travelled to Spain because they loved the country very much.",
        "Peter would have caught the fish with a fishing rod, if not the"
        " darkness. ",
        "The medium was affected by the near-field radiation",
        "He treated her unfairly.",
    ]

    for phrase in phrases:
        main(phrase, nlp)
