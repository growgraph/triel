from __future__ import annotations

import logging
import re

import networkx as nx
from pylatexenc.latex2text import LatexNodes2Text
from spacy import Language
from unidecode import unidecode

from lm_service.graph import get_subtree_wrapper, phrase_to_deptree

logger = logging.getLogger(__name__)


def split_tokens_into_phrases(tokens_list, terminal_puncts=None):
    if terminal_puncts is None:
        terminal_puncts = {".", "!", "?"}
    phrases = []
    cur_phrase = [tokens_list[0]]

    # split only if a white space follows and the following letter is capital
    for token0, token1, token2 in zip(
        tokens_list[:-2], tokens_list[1:-1], tokens_list[2:]
    ):
        cur_phrase.append(token1)
        if (
            # not token0[-1].isupper() and
            token1 in terminal_puncts
            and token2[0].isupper()
            # and not token2[0].islower()
        ):
            phrases.append(cur_phrase)
            cur_phrase = []
    cur_phrase.append(tokens_list[-1])
    phrases.append(cur_phrase)
    return phrases


def normalize_input_text(text, terminal_full_stop=True):
    """

    :param text:
    :param terminal_full_stop: add terminal full stop to each phrase, or not
    :return:
    """

    # condense white spaces
    text = re.sub(r"\s+", " ", text)

    # split on .!? if followed by a capital and not preceded by a capital
    pat = r"(?<=[^A-Z][.!?])\s*(?=[A-Z])"
    phrases_ = re.split(pat, text)
    # trim initial/terminal whitespaces
    trim_whitespace = re.compile(r"^[\s+]+|[\s+]+$")
    phrases_ = [trim_whitespace.sub("", p) for p in phrases_]

    phrases = []
    for p in phrases_:
        try:
            text = LatexNodes2Text().latex_to_text(p)
        except:
            logger.error(f" LatexNodes2Text could not process : {p}")
            text = p

        # cast possible diacritics to ascii
        try:
            text = unidecode(text)
        except:
            logger.error(f" unidecode failed on: {text}")

        # deal with double backslash
        try:
            text = bytes(text, "utf-8").decode("unicode_escape")
        except:
            # TODO breaking example :
            #  text =  'The program is freely available at \\url{http://graphics.med.yale.edu/cgi-bin/lib_comp.pl}.'
            # in \\url is interpreted as the beginning of escape sequence
            logger.error(
                f"unicode decoding failed; more latex in text? {text}"
            )

        phrases += [text]

    return phrases


def pivot_around_advcl(nlp: Language, phrase) -> list[str]:
    """
        coreference works better after pivot_around_advcl is applied
        idea take advcl component and move complement to the end of the subtree with advcl root
    :param nlp:
    :param phrase:
    :return:
    """

    rdoc, graph = phrase_to_deptree(nlp, phrase)

    advcls = [
        u
        for u in graph.nodes()
        if graph.nodes[u]["tag_"] == "VBN"
        and graph.nodes[u]["dep_"] == "advcl"
    ]

    for advcl in advcls:
        root = next(iter(graph.predecessors(advcl)))

        succs = sorted(graph.successors(root))
        advcl_index = succs.index(advcl)

        if advcl_index < len(succs) - 1:
            next_ = succs[advcl_index + 1]

            if (
                graph.nodes[next_]["dep_"] == "punct"
                and graph.nodes[next_]["tag_"] == ","
            ):
                if len(list(graph.successors(next_))) != 0:
                    raise Exception(f" `,` punct has successors")
                graph.remove_node(next_)

        # refresh succs just in case: advcl has the same index
        succs = sorted(graph.successors(root))
        jstar = None
        for j in range(advcl_index, len(succs)):
            if graph.nodes[succs[j]]["tag_"] == "NN":
                jstar = j
                break

        if jstar is not None:
            next_ = succs[jstar]

            adv_nodes = sorted(get_subtree_wrapper(graph, advcl))
            next_nodes = sorted(get_subtree_wrapper(graph, next_))
            root_nodes = sorted(get_subtree_wrapper(graph, root))

            if (
                list(range(adv_nodes[0], adv_nodes[0] + len(adv_nodes)))
                != adv_nodes
            ):
                raise Exception(f" subtree nodes are not a sequence {phrase}")
            if (
                list(range(next_nodes[0], next_nodes[0] + len(next_nodes)))
                != next_nodes
            ):
                raise Exception(" subtree nodes are not a sequence")

            i0 = root_nodes.index(adv_nodes[0])
            ilast = root_nodes.index(next_nodes[-1])
            index_remap = root_nodes[i0 : ilast + 1]
            full_index_shifted = (
                index_remap[len(adv_nodes) :] + index_remap[: len(adv_nodes)]
            )
            mapping = dict(zip(full_index_shifted, index_remap))
            if i0 == 0:
                s = graph.nodes[adv_nodes[0]]["text"]
                graph.nodes[adv_nodes[0]]["text"] = s[0].lower() + s[1:]

            graph = nx.relabel_nodes(graph, mapping)

    tphrases = []
    sgraphs = sorted(
        nx.weakly_connected_components(graph), key=lambda x: min(x)
    )
    for sg in sgraphs:
        phrase_rep = [graph.nodes[i]["text"] for i in sorted(sg)]
        phrase_rep[0] = phrase_rep[0][0].capitalize() + phrase_rep[0][1:]
        tphrases += [" ".join(phrase_rep)]
    return tphrases
