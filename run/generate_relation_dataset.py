import spacy
import pandas as pd
from spacy import displacy
import yaml
from lm_service.relation import graph_to_relations


def main():
    with open("./lm_service/config/prune_noun_compound.yaml") as file:
        add_dict_rules = yaml.load(file, Loader=yaml.FullLoader)

    with open("./test/data/cheops.txt", "r") as f:
        text = f.read()

    nlp = spacy.load("en_core_web_sm")

    phrases = text.split(".")

    acc = []
    # for j, document in enumerate(phrases[:2]):
    #     mg, r, rproj = phrase_to_relations(document, add_dict_rules)
    #     for iis, triplet in zip(r, rproj):
    #         acc += [[document] + list(iis) + triplet]
    # df = pd.DataFrame(acc, columns=["phrase", "iR", "iS", "iO", "R", "S", "O"])


if __name__ == "__main__":
    main()
