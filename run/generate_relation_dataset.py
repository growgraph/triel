import spacy
import coreferee
import pandas as pd
import sys
import networkx as nx
import pkgutil
import os
import yaml
import logging
import argparse
from lm_service.relation import parse_relations_advanced
from lm_service.util import plot_graph, plot_leaves
from lm_service.preprocessing import normalize_input_text
from lm_service.graph import transform_advcl


def main(nlp, text, fig_path, head=None, window_size=2, plot=True):
    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)

    # phrases = normalize_input_text(text)

    phrases = normalize_input_text(text, terminal_full_stop=False)
    phrases = [transform_advcl(nlp, p) for p in phrases]

    acc = []

    nmax = len(phrases) - window_size + 1
    if head is not None:
        nmax = min([nmax, head])
    for i in range(nmax):
        fragment = ". ".join(phrases[i : i + window_size])
        (
            graph,
            coref_graph,
            metagraph,
            triples_expanded,
            triples_proj,
        ) = parse_relations_advanced(fragment, nlp, rules)
        acc += [(i, fragment, triples_expanded, triples_proj)]
        if plot:
            plot_graph(graph, fig_path, f"fragment_{i}_full")
            plot_graph(metagraph, fig_path, f"fragment_{i}_mg")
            plot_graph(coref_graph, fig_path, f"fragment_{i}_coref")
            plot_leaves(metagraph, fig_path, f"fragment_{i}")

    sources = [s for _, _, _, triples_proj in acc for s, _, _ in triples_proj]
    relations = [r for _, _, _, triples_proj in acc for _, r, _ in triples_proj]
    targets = [t for _, _, _, triples_proj in acc for _, _, t in triples_proj]

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
    plot_graph(g, fig_path, f"doc")

    dacc = []
    for i, phrase, triples, text_triples in acc:
        for triple, text_relation in zip(triples, text_triples):
            dacc += [[i] + [triple.source, triple.relation.tokens, triple.target] + list(text_relation) + [phrase]]
    df = pd.DataFrame(dacc, columns=["ip", "is", "ir", "it", "s", "r", "t", "phrase"])
    df.to_csv(os.path.join(fig_path, "relations.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filemode="w",
        stream=sys.stdout,
    )

    parser.add_argument(
        "--head", nargs="?", type=int, help="number of phrases to parse"
    )
    parser.add_argument("--outpath", type=str, help="output folder path")
    parser.add_argument("--input-txt", type=str, help="input text file path")
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()

    with open(args.input_txt, "r") as f:
        text = f.read()

    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")
    main(nlp, text, fig_path=os.path.expanduser(args.outpath), head=args.head, plot=args.plot)
