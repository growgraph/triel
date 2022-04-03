import os
from pathlib import Path
import spacy
import pkgutil
import yaml
import coreferee
import hashlib
from networkx.drawing.nx_agraph import to_agraph
from lm_service.relation import (
    dep_tree_from_phrase,
    render_coref_graph,
    render_coref_graph_reduced,
    phrase_to_relations,
)


def main(phrase, nlp):
    path = Path(__file__).parent
    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound.yaml")
    add_dict_rules = yaml.load(fp, Loader=yaml.FullLoader)

    fig_path = os.path.join(path, "figs")
    chash = hashlib.sha256(phrase.encode("utf-8")).hexdigest()
    rdoc, nx_graph = dep_tree_from_phrase(nlp, phrase)

    dot = to_agraph(nx_graph)
    dot.layout("dot")
    dot.draw(path=os.path.join(fig_path, f"{chash[:6]}.png"), format="png", prog="dot")
    dot.draw(path=os.path.join(fig_path, f"{chash[:6]}.pdf"), format="pdf", prog="dot")

    _, relations, rproj, folded = phrase_to_relations(nx_graph, add_dict_rules)
    dot = to_agraph(folded)
    dot.layout("dot")
    dot.draw(
        path=os.path.join(fig_path, f"{chash[:6]}_folded.png"), format="png", prog="dot"
    )
    dot.draw(
        path=os.path.join(fig_path, f"{chash[:6]}_folded.pdf"), format="pdf", prog="dot"
    )

    coref_graph = render_coref_graph(rdoc, nx_graph)
    dot = to_agraph(coref_graph.graph)
    dot.layout("dot")
    dot.draw(
        path=os.path.join(fig_path, f"{chash[:6]}_coref.png"), format="png", prog="dot"
    )
    dot.draw(
        path=os.path.join(fig_path, f"{chash[:6]}_coref.pdf"), format="pdf", prog="dot"
    )

    coref_graph = render_coref_graph_reduced(rdoc, nx_graph)
    dot = to_agraph(coref_graph)
    dot.layout("dot")
    dot.draw(
        path=os.path.join(fig_path, f"{chash[:6]}_coref_reduced.png"),
        format="png",
        prog="dot",
    )
    dot.draw(
        path=os.path.join(fig_path, f"{chash[:6]}_coref_reduced.pdf"),
        format="pdf",
        prog="dot",
    )


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    # nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")

    phrase = (
        "CHEOPS (CHaracterising ExOPlanets Satellite) is a European space telescope "
        "to determine the size of known extrasolar planets, which will allow the estimation "
        "of their mass, density, composition and their formation"
    )
    main(phrase, nlp)

    phrase2 = (
        "Although he was very busy with his work, Peter had had enough of it. "
        "He and his wife decided they needed a holiday. "
        "They travelled to Spain because they loved the country very much."
    )
    main(phrase2, nlp)
