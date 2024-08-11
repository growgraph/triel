from lm_service.linking.string import render_gap_mappers, render_index_mapper


def test_render(strings):
    s_a, s_b = strings
    mapper_a, mapper_b = render_gap_mappers(s_a, s_b)
    assert mapper_b == [(2, 3), (4, 2)]
    ioi = list(range(len(s_b)))
    map_io = render_index_mapper(ioi, mapper_b)
    assert all(s_a[v] == s_b[k] for k, v in map_io.items())


def test_render_trivial(strings):
    s_a = ""
    s_b = "No ascii letters. Please enter your text in English."
    mapper_a, mapper_b = render_gap_mappers(s_a, s_b)
    assert mapper_a == []
    assert mapper_b == []
