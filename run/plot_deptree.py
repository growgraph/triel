import hashlib
import os
import pathlib
from pathlib import Path

import coreferee
import spacy

from lm_service.coref import render_coref_graph
from lm_service.preprocessing import normalize_input_text, transform_advcl
from lm_service.relation import text_to_compound_index_graph
from lm_service.util import plot_graph


def main(phrase, nlp):

    figs_folder = "./figs"
    current_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), figs_folder
    )
    pathlib.Path(current_path).mkdir(parents=True, exist_ok=True)

    path = Path(__file__).parent
    fig_path = os.path.join(path, "figs")

    chash = hashlib.sha256(phrase.encode("utf-8")).hexdigest()
    (
        nx_graph,
        rdoc,
        map_tree_subtree_index,
    ) = text_to_compound_index_graph(nlp, phrase, 0)

    plot_graph(nx_graph, fig_path, f"{chash[:6]}")

    coref_graph = render_coref_graph(rdoc)

    plot_graph(coref_graph, fig_path, f"{chash[:6]}_coref")


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")

    phrases = [
        # "CHEOPS (CHaracterising ExOPlanets Satellite) is a European space"
        # " telescope to determine the size of known extrasolar planets,"
        # " which will allow the estimation of their mass, density,"
        # " composition and their formation.",
        # "The medium was affected by the near-field radiation",
        "Although he was very busy with his work, Peter Brown had had enough"
        " of it. He and his wife decided they needed a holiday. They travelled"
        " to Spain because they loved the country very much.",
        # "Peter would have caught the fish with a fishing rod, if not the"
        # " darkness. ",
        # "He treated her unfairly.",
        "CHEOPS (CHaracterising ExOPlanets Satellite) is a European space"
        " telescope to determine the size of known extrasolar planets, which"
        " will allow the estimation of their mass, density, composition and"
        " their formation. Launched on 18 December 2019, it is the first"
        " Small-class mission in ESA's Cosmic Vision science programme.",
        # "Part of the GTO programme is to find transits of known exoplanets "
        # "that were confirmed by other techniques, such as radial-velocity, "
        # "but not by the transit-method. Another part of the GTO programme "
        # "includes exploration of multi-systems "
        # "and search of additional planets in those systems, "
        # "for example using the transit-timing-variation (TTV) method.",
        "Thousands of exoplanets have been discovered by the end of the 2010s;"
        " some have minimum mass measurements from the radial velocity method"
        " while others that are seen to transit their parent stars have"
        " measures of their physical size.",
        "CHEOPS measures photometric signals with a precision limited by"
        " stellar photon noise of 150 ppm/min for a 9th magnitude star. This"
        " corresponds to the transit of an Earth-sized planet orbiting a star"
        " of 0.9 R☉ in 60 days detected with a S/Ntransit >10 (100 ppm transit"
        " depth).",
    ]

    phrases = [
        " ".join(normalize_input_text(p, terminal_full_stop=True))
        for p in phrases
    ]
    phrases = [transform_advcl(nlp, p) for p in phrases]

    for phrase in phrases:
        main(phrase, nlp)
