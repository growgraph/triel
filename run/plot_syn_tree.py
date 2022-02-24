import spacy
import networkx
import hashlib
from networkx.drawing.nx_agraph import to_agraph


def main(phrase, nlp):
    # Load spacy's dependency tree into a networkx graph
    edges = []
    chash = hashlib.sha256(phrase.encode("utf-8")).hexdigest()
    for token in nlp(phrase):
        # FYI https://spacy.io/docs/api/token
        # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        for child in token.children:
            edges.append(
                (
                    f"{token.lower_}-{token.i}-{token.dep_}-{token.tag_}",
                    f"{child.lower_}-{child.i}-{child.dep_}-{child.tag_}",
                )
            )

    graph = networkx.DiGraph(edges)
    dot = to_agraph(graph)
    dot.layout("dot")
    dot.draw(path=f"./figs/{chash[:6]}.png", format="png", prog="dot")
    dot.draw(path=f"./figs/{chash[:6]}.pdf", format="pdf", prog="dot")


if __name__ == "__main__":
    phrase = (
        "CHEOPS (CHaracterising ExOPlanets Satellite) is a European space telescope "
        "to determine the size of known extrasolar planets, which will allow the estimation "
        "of their mass, density, composition and their formation"
    )
    phrase2 = (
        "Although he was very busy with his work, Peter had had enough of it. "
        "He and his wife decided they needed a holiday. "
        "They travelled to Spain because they loved the country very much."
    )
    nlp = spacy.load("en_core_web_sm")
    main(phrase, nlp)
    main(phrase2, nlp)
