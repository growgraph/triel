from __future__ import annotations

from itertools import product

from lm_service.graph import transform_advcl
from lm_service.onto import TripleCandidate
from lm_service.piles import ExtCandidateList
from lm_service.preprocessing import normalize_input_text
from lm_service.relation import (
    phrases_to_basis_triples,
    text_to_coref_sourcetarget,
)


def text_to_triples(
    text, nlp, rules, window_size, head=None, return_phrases=False
) -> list[TripleCandidate] | tuple[list[TripleCandidate], list[str]]:
    phrases_original = normalize_input_text(text, terminal_full_stop=True)
    if head is not None:
        phrases_original = phrases_original[:head]
    phrases = [transform_advcl(nlp, p) for p in phrases_original]

    striples, candidate_depot, relations = phrases_to_basis_triples(
        nlp, rules, phrases
    )

    ecl = candidate_depot.unfold_conjunction()

    global_ecl = ExtCandidateList()
    window_size = 2
    nmax = len(phrases) - window_size + 1
    for i in range(nmax):
        fragment = " ".join(phrases[i : i + window_size])
        ecl.set_filter(lambda x: i <= x[0] < i + window_size)
        ncp = text_to_coref_sourcetarget(
            nlp, fragment, ecl, initial_phrase_index=i
        )

        for key, candidate_list in ncp.items():
            for c in candidate_list:
                global_ecl.append(key, c)

    global_ecl.filter_out_pronouns()
    global_triples = []

    # expand using coref and conj maps
    for tri in striples:
        s, r, t = tri
        for sprime, tprime in product(global_ecl[s], global_ecl[t]):
            global_triples += [
                TripleCandidate(
                    source=sprime, target=tprime, relation=relations[r]
                )
            ]

    global_triples = [tri.normalize_relation() for tri in global_triples]
    global_triples = sorted(
        global_triples,
        key=lambda x: (x.source.sroot, x.relation.sroot, x.target.sroot),
    )
    if return_phrases:
        return global_triples, phrases_original
    else:
        return global_triples
