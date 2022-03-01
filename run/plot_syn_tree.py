import spacy
import coreferee
import hashlib
from networkx.drawing.nx_agraph import to_agraph
from lm_service.relation import dep_tree_from_phrase, add_coref


def main(phrase, nlp):
    chash = hashlib.sha256(phrase.encode("utf-8")).hexdigest()
    rdoc, nx_graph = dep_tree_from_phrase(nlp, phrase)
    coref_graph = add_coref(rdoc)

    dot = to_agraph(coref_graph)
    dot.layout("dot")
    dot.draw(path=f"./figs/{chash[:6]}.png", format="png", prog="dot")
    dot.draw(path=f"./figs/{chash[:6]}.pdf", format="pdf", prog="dot")


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
