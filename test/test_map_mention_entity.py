from lm_service.linking.onto import Entity, interval_overlap_metric
from lm_service.linking.util import (
    link_candidate_entity,
    link_unlinked_entities,
    process_entities,
    process_entity_cluster,
    render_entity_clusters,
)
from lm_service.piles import ExtCandidateList


def test_overlap():
    d = interval_overlap_metric((0, 2), (1, 3))
    assert d == 0.5
    d = interval_overlap_metric((0, 2), (5, 6))
    assert d == 0
    d = interval_overlap_metric((5, 6), (0, 2))
    assert d == 0
    d = interval_overlap_metric((1, 2), (-5, 4))
    assert d == 1


def test_render_entity_clusters(entities):
    clusters = render_entity_clusters(entities)
    lens = [len(c) for c in clusters]
    assert sum(lens) == 122
    assert len(clusters) == 91
    assert lens[:10] == [1, 1, 2, 1, 2, 1, 3, 2, 1, 1]
    assert lens[-5:] == [2, 1, 1, 1, 1]


def test_process_entity_cluster(entity_cluster, score_mapper_trivial):
    e, edges = process_entity_cluster(entity_cluster, score_mapper_trivial)
    assert len(edges) == 2
    assert e.hash == "BERN_V2.NCBIGene.925"
    assert all([d == 1.0 for _, _, d in edges])


def test_process_entities(entities, score_mapper_trivial):
    pri_entites, edges = process_entities(entities, score_mapper_trivial)
    normalized_entities = set([Entity.from_local_entity(e) for e in pri_entites])
    assert len(edges) == 31
    assert len(normalized_entities) == 56
    assert len(pri_entites) == 91


def test_mapping(
    ecl: ExtCandidateList,
    entities_local,
    phrase_mapper,
    score_mapper_trivial,
    muindex_candidate,
):
    map_candidate2entity, edges = link_candidate_entity(
        phrase_mapper=phrase_mapper,
        muindex_candidate=muindex_candidate,
        entities_local=entities_local,
        score_mapper=score_mapper_trivial,
        overlap_thr=0.8,
    )

    assert len(map_candidate2entity) == 153
    assert len(edges) == 78

    normalized_entities = set([Entity.from_local_entity(e) for e in entities_local])

    map_eindex_entity: dict[str, Entity] = {e.hash: e for e in normalized_entities}

    map_eindex_entity, map_c2e = link_unlinked_entities(
        map_candidate2entity, muindex_candidate
    )
