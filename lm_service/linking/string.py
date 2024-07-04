def render_gap_mappers(s1, s2):
    """
    given strings s1 and s2,
    return two lists of tuples: [(pos, gap)]

    where the first contains index `pos` where an insertion of size `gap` happens in s2 wrt to s1
    and vice versa
    :param s1:
    :param s2:
    :return:
    """
    max_len = max([len(s1), len(s2)])
    p1, p2 = 0, 0

    ix1 = []
    ix2 = []

    while p1 < max_len and p2 < max_len:
        if s1[p1] != s2[p2]:
            for k in range(1, 5):
                if s1[p1 + k] == s2[p2]:
                    ix2 += [(p2, k)]
                    p1 += k
                    break
                if s1[p1] == s2[p2 + k]:
                    ix1 += [(p1, k)]
                    p2 += k
                    break

        p1 += 1
        p2 += 1
    return ix1, ix2


def render_index_mapper(index_oi, gap_mapper):
    """
    for a set of indexes index_oi return
    :param index_oi:
    :param gap_mapper:
    :return:
    """
    index_mapper = dict()
    mapper2_acc = [(0, 0)]
    for pos, gap_size in gap_mapper:
        mapper2_acc += [(pos, gap_size + mapper2_acc[-1][-1])]

    pnt = 0
    for index in index_oi:
        while pnt < len(mapper2_acc) and index >= mapper2_acc[pnt][0]:
            pnt += 1
        ans = index + mapper2_acc[pnt - 1][1]
        index_mapper[index] = ans
    return index_mapper
