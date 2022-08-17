from typing import Optional

import networkx as nx

from lm_service.folding import get_flag


class Leaf:
    def __init__(self, root, root_props):
        self.root = root
        self.tree: nx.DiGraph = nx.DiGraph()
        self.tree.add_node(root, **root_props)
        self.conj = []

    def __len__(self):
        return 0 if self.tree is None else self.tree.number_of_nodes()

    def is_compound(self):
        return self.tree.number_of_nodes() > 1

    def add_node(self, v, **vprops):
        self.tree.add_node(v, **vprops)

    def add_edge(self, u, v):
        self.tree.add_edge(u, v)

    @property
    def nodes(self):
        return self.tree.nodes(data=True)

    def compute_conj(self):
        dists = nx.shortest_path_length(self.tree, self.root)
        self.conj = [self.root]
        conj_candidates = [
            (i, dists[i])
            for i, data in self.tree.nodes(data=True)
            if data["dep_"] == "conj"
        ]
        step = 1
        while True:
            conj_candidates_at_level = [
                n for n, dist in conj_candidates if dist == step
            ]
            step += 1
            if conj_candidates_at_level:
                self.conj += conj_candidates_at_level
            else:
                break
        return self.conj


def fold_graph_top(
    nx_graph: nx.DiGraph,
    rules,
) -> nx.DiGraph:
    gmetagraph = nx.DiGraph()

    roots = [n for n in nx_graph.nodes() if nx_graph.in_degree(n) == 0]

    for root in roots:
        metagraph = nx.DiGraph()
        gmetagraph.update(
            fold_graph(nx_graph, metagraph, None, root, None, rules)
        )

    return gmetagraph


def fold_graph(
    graph: nx.DiGraph,
    metagraph: nx.DiGraph,
    u: Optional[int],
    v: int,
    local_root: Optional[int],
    rules,
) -> nx.DiGraph:

    vprops = graph.nodes[v]
    vflag = get_flag(vprops, rules)

    if vflag and local_root is not None and u is not None:
        subgraph = metagraph.nodes[local_root]["leaf"]
        subgraph.add_node(v, **vprops)
        subgraph.add_edge(u, v)
    else:
        metagraph.add_node(v, **vprops)
        metagraph.nodes[v]["leaf"] = Leaf(v, vprops)
        if local_root is not None:
            metagraph.add_edge(local_root, v)
        local_root = v

    for w in graph.successors(v):
        metagraph = fold_graph(graph, metagraph, v, w, local_root, rules)
    return metagraph
