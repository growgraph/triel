import argparse
import logging
import os
import pkgutil
import sys

import coreferee
import networkx as nx
import pandas as pd
import spacy
import yaml

from lm_service.text import cast_simplified_triples_table, text_to_triples
from lm_service.util import plot_graph


def main(nlp, text, fig_path, head=None, window_size=2, plot_path=None):
    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound_v2.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)

    triples, map_mu_index_triple = text_to_triples(
        text,
        nlp,
        rules,
        window_size=window_size,
        head=head,
        return_phrases=True,
        plot_path=plot_path,
    )

    triples_text = cast_simplified_triples_table(triples, map_mu_index_triple)

    sources = [s for s, _, _ in triples_text]
    relations = [r for _, r, _ in triples_text]
    targets = [t for _, _, t in triples_text]

    tokens = sorted(set(set(sources) | set(targets) | set(relations)))
    token_map = {t: ii for ii, t in enumerate(tokens)}
    g = nx.DiGraph()

    for text_triplet in triples_text:
        s, r, t = text_triplet
        triplet = [token_map[x] for x in text_triplet]
        si, ri, ti = triplet
        g.add_node(si, label=s)
        g.add_node(ti, label=t)
        g.add_edge(si, ti, label=r)
    if plot_path:
        plot_graph(g, fig_path, f"doc", prog="sfdp")

    df_acc = []
    for tri, tri_txt in zip(triples, triples_text):
        s, r, t = tri
        ix = r.phrase
        df_acc += [
            [
                ix,
                tuple(s.stokens),
                tuple(r.stokens),
                tuple(t.stokens),
                *tri_txt,
                # phrases[ix],
            ]
        ]
    df = pd.DataFrame(
        df_acc,
        columns=[
            "phrase_ix",
            "source_ix",
            "relation_ix",
            "target_ix",
            "relation_txt",
            "relation_txt",
            "target_txt",
            # "phrase",
        ],
    ).sort_values(["phrase_ix", "relation_ix", "source_ix", "target_ix"])
    df.to_csv(os.path.join(fig_path, "relations.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    logging.basicConfig(
        format=(
            "%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.ERROR,
        # filemode="w",
        stream=sys.stdout,
    )

    parser.add_argument(
        "--head", nargs="?", type=int, help="number of phrases to parse"
    )
    parser.add_argument("--outpath", type=str, help="output folder path")
    parser.add_argument("--input-txt", type=str, help="input text file path")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if args.plot:
        plot_path = os.path.expanduser(args.outpath)
    else:
        plot_path = None

    with open(args.input_txt, "r") as f:
        text = f.read()

    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")
    main(
        nlp,
        text,
        fig_path=os.path.expanduser(args.outpath),
        head=args.head,
        plot_path=plot_path,
    )
