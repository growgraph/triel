from __future__ import annotations

import logging
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, Field
from spacy import Language
from spacy.tokens import Doc, Span

logger = logging.getLogger(__name__)


class CorefAdapterError(Exception):
    """Base exception for coref adapter failures."""


class CorefSetupError(CorefAdapterError):
    """Raised when backend setup fails."""


class CorefInferenceError(CorefAdapterError):
    """Raised when backend inference/parsing fails."""


class CorefBackend(StrEnum):
    COREFEREE = "coreferee"
    SPACY_NATIVE = "spacy_native"
    FASTCOREF = "fastcoref"
    NONE = "none"


class CorefMention(BaseModel):
    token_indexes: tuple[int, ...]
    most_specific: bool = False


class CorefChain(BaseModel):
    chain_index: int
    mentions: list[CorefMention]


class CorefResolution(BaseModel):
    chains: list[CorefChain] = Field(default_factory=list)


class CorefResolver(Protocol):
    backend: CorefBackend

    def is_ready(self, nlp: Language | None = None) -> bool: ...

    def resolve_doc(self, rdoc: Doc) -> CorefResolution: ...


def _span_to_token_indexes(span: Span) -> tuple[int, ...]:
    return tuple(range(span.start, span.end))


class NoneCorefResolver:
    backend = CorefBackend.NONE

    def is_ready(self, nlp: Language | None = None) -> bool:
        _ = nlp
        return True

    def resolve_doc(self, rdoc: Doc) -> CorefResolution:
        _ = rdoc
        return CorefResolution()


class CorefereeResolver:
    backend = CorefBackend.COREFEREE

    def is_ready(self, nlp: Language | None = None) -> bool:
        if nlp is None:
            return True
        return "coreferee" in nlp.pipe_names

    def resolve_doc(self, rdoc: Doc) -> CorefResolution:
        try:
            if not hasattr(rdoc._, "coref_chains"):
                return CorefResolution()
            chains = rdoc._.coref_chains if rdoc._.coref_chains is not None else []
            out: list[CorefChain] = []
            for jchain, chain in enumerate(chains):
                mentions = [
                    CorefMention(
                        token_indexes=tuple(x.token_indexes),
                        most_specific=(kth == chain.most_specific_mention_index),
                    )
                    for kth, x in enumerate(chain.mentions)
                ]
                out.append(CorefChain(chain_index=jchain, mentions=mentions))
            return CorefResolution(chains=out)
        except Exception as e:
            raise CorefInferenceError(f"Coreferee resolution failed: {e}") from e


class SpacyNativeCorefResolver:
    backend = CorefBackend.SPACY_NATIVE

    def is_ready(self, nlp: Language | None = None) -> bool:
        if nlp is None:
            return True
        return "coref" in nlp.pipe_names or "coreference_resolver" in nlp.pipe_names

    def resolve_doc(self, rdoc: Doc) -> CorefResolution:
        try:
            chain_spans = sorted(
                (
                    (k, v)
                    for k, v in rdoc.spans.items()
                    if k.startswith("coref_clusters")
                ),
                key=lambda item: item[0],
            )
            chains: list[CorefChain] = []
            for jchain, (_, spangroup) in enumerate(chain_spans):
                mentions: list[CorefMention] = []
                spans = list(spangroup)
                for kth, span in enumerate(spans):
                    tokens = _span_to_token_indexes(span)
                    if not tokens:
                        continue
                    # Use longest mention as most specific approximation.
                    mentions.append(
                        CorefMention(token_indexes=tokens, most_specific=(kth == 0))
                    )
                if mentions:
                    longest_ix = max(
                        range(len(mentions)),
                        key=lambda i: len(mentions[i].token_indexes),
                    )
                    mentions = [
                        CorefMention(
                            token_indexes=m.token_indexes,
                            most_specific=(i == longest_ix),
                        )
                        for i, m in enumerate(mentions)
                    ]
                    chains.append(CorefChain(chain_index=jchain, mentions=mentions))
            return CorefResolution(chains=chains)
        except Exception as e:
            raise CorefInferenceError(
                f"spaCy native coref resolution failed: {e}"
            ) from e


class FastCorefResolver:
    backend = CorefBackend.FASTCOREF

    def is_ready(self, nlp: Language | None = None) -> bool:
        if nlp is None:
            return True
        return "fastcoref" in nlp.pipe_names

    def resolve_doc(self, rdoc: Doc) -> CorefResolution:
        try:
            if not hasattr(rdoc._, "coref_clusters"):
                return CorefResolution()
            chains: list[CorefChain] = []
            raw_clusters = (
                rdoc._.coref_clusters if rdoc._.coref_clusters is not None else []
            )
            for jchain, cluster in enumerate(raw_clusters):
                mentions: list[CorefMention] = []
                for mention in cluster:
                    if not isinstance(mention, tuple) or len(mention) != 2:
                        continue
                    char_a, char_b = mention
                    span = rdoc.char_span(char_a, char_b, alignment_mode="expand")
                    if span is None:
                        continue
                    mentions.append(
                        CorefMention(
                            token_indexes=_span_to_token_indexes(span),
                            most_specific=False,
                        )
                    )
                if mentions:
                    longest_ix = max(
                        range(len(mentions)),
                        key=lambda i: len(mentions[i].token_indexes),
                    )
                    mentions = [
                        CorefMention(
                            token_indexes=m.token_indexes,
                            most_specific=(i == longest_ix),
                        )
                        for i, m in enumerate(mentions)
                    ]
                    chains.append(CorefChain(chain_index=jchain, mentions=mentions))
            return CorefResolution(chains=chains)
        except Exception as e:
            raise CorefInferenceError(f"fastcoref resolution failed: {e}") from e


def get_coref_resolver(backend: str | CorefBackend | None = None) -> CorefResolver:
    selected = CorefBackend(backend or CorefBackend.COREFEREE)
    if selected == CorefBackend.COREFEREE:
        return CorefereeResolver()
    if selected == CorefBackend.SPACY_NATIVE:
        return SpacyNativeCorefResolver()
    if selected == CorefBackend.FASTCOREF:
        return FastCorefResolver()
    return NoneCorefResolver()


def configure_nlp_coref_backend(
    nlp: Language, backend: str | CorefBackend | None = None
) -> Language:
    selected = CorefBackend(backend or CorefBackend.COREFEREE)
    if selected == CorefBackend.COREFEREE:
        if "coreferee" not in nlp.pipe_names:
            try:
                nlp.add_pipe("coreferee")
            except Exception as e:
                raise CorefSetupError(
                    f"Failed to configure coref backend '{selected.value}': {e}"
                ) from e
        return nlp

    if selected == CorefBackend.FASTCOREF:
        if "fastcoref" not in nlp.pipe_names:
            try:
                nlp.add_pipe(
                    "fastcoref",
                    config={
                        "model_architecture": "LingMessCoref",
                        "model_path": "biu-nlp/lingmess-coref",
                    },
                )
            except Exception as e:
                raise CorefSetupError(
                    f"Failed to configure coref backend '{selected.value}': {e}"
                ) from e
        return nlp

    if selected == CorefBackend.SPACY_NATIVE:
        # Native spaCy coref models can provide the component already.
        # Try adding only if factory is available.
        if (
            "coref" not in nlp.pipe_names
            and "coreference_resolver" not in nlp.pipe_names
        ):
            try:
                nlp.add_pipe("coref")
            except Exception as e:
                raise CorefSetupError(
                    f"Failed to configure coref backend '{selected.value}': {e}"
                ) from e
        return nlp

    return nlp


def get_ready_coref_runtime(
    nlp: Language,
    backend: str | CorefBackend | None = None,
    *,
    fallback_to_none: bool = False,
) -> tuple[Language, CorefResolver]:
    selected = CorefBackend(backend or CorefBackend.COREFEREE)
    try:
        nlp_ready = configure_nlp_coref_backend(nlp, selected)
        resolver = get_coref_resolver(selected)
        if not resolver.is_ready(nlp_ready):
            raise CorefSetupError(
                f"Coref backend '{selected.value}' is not ready after setup."
            )
        return nlp_ready, resolver
    except CorefAdapterError:
        if fallback_to_none:
            logger.warning(
                "Falling back to 'none' coref backend after setup failure "
                f"for backend='{selected.value}'."
            )
            return nlp, get_coref_resolver(CorefBackend.NONE)
        raise
