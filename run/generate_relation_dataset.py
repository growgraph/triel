import spacy
import coreferee
import pandas as pd
from spacy import displacy
import networkx as nx
import pkgutil
import os
from pathlib import Path
import yaml
from lm_service.relation import parse_relations_advanced
from lm_service.util import plot_graph


def main(nlp):
    path = Path(__file__).parent
    fig_path = os.path.join(path, "figs")

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)

    with open(os.path.join(path, "..", "./test/data/cheops.txt"), "r") as f:
        text = f.read()

    phrases = text.split(".")

    acc = []
    window_size = 2

    # for i in range(len(phrases) - window_size + 1):
    for i in range(4):
        fragment = ".".join(phrases[i : i + window_size])
        (
            graph,
            coref_graph,
            metagraph,
            relations_transformed,
            relations_proj,
        ) = parse_relations_advanced(fragment, nlp, rules)
        acc += [(i, fragment, relations_transformed, relations_proj)]

        plot_flag = False
        for text_triplet in relations_proj:
            s, r, t = text_triplet
            if s == r or r == t:
                plot_flag = True
        if plot_flag:
            plot_graph(metagraph, fig_path, f"fragment_mg_{i}")
            plot_graph(coref_graph, fig_path, f"fragment_coref_{i}")

    sources = [s for _, _, _, relations_proj in acc for s, _, _ in relations_proj]
    relations = [r for _, _, _, relations_proj in acc for _, r, _ in relations_proj]
    targets = [t for _, _, _, relations_proj in acc for _, _, t in relations_proj]

    tokens = sorted(set(sources) | set(targets) | set(relations))
    token_map = {t: ii for ii, t in enumerate(tokens)}
    g = nx.DiGraph()

    for i, _, _, text_relations in acc:
        for text_triplet in text_relations:
            s, r, t = text_triplet
            triplet = [token_map[x] for x in text_triplet]
            si, ri, ti = triplet
            g.add_node(si, label=s)
            g.add_node(ti, label=t)
            g.add_edge(si, ti, label=r)
    plot_graph(g, os.path.join(path, "figs"), f"doc")

    dacc = []
    for i, phrase, relations, text_relations in acc:

        for relation, text_relation in zip(text_relations, relations):
            dacc += [[i] + list(relation) + list(text_relation) + [phrase]]
    df = pd.DataFrame(dacc, columns=["ip", "is", "ir", "it", "s", "r", "t", "phrase"])
    df.to_csv("~/tmp/relations.csv")


if __name__ == "__main__":

    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")
    main(nlp)
