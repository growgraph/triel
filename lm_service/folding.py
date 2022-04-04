import networkx as nx
from typing import Optional


class Leaf:
    def __init__(self, root):
        self.root = root
        self.tree: Optional[nx.DiGraph] = None

    def __len__(self):
        return 0 if self.tree is None else self.tree.number_of_nodes()

    def add_node(self, v, **vprops):
        if self.tree is None:
            self.tree = nx.DiGraph()
        self.tree.add_node(v, **vprops)

    def add_edge(self, u, v):
        if self.tree is None:
            self.tree = nx.DiGraph()
        self.tree.add_edge(u, v)

    @property
    def nodes(self):
        if self.tree is None:
            return []
        else:
            return self.tree.nodes(data=True)


def fold_graph_top(
    nx_graph: nx.DiGraph,
    rules,
) -> nx.DiGraph:
    gmetagraph = nx.DiGraph()

    roots = [n for n in nx_graph.nodes() if nx_graph.in_degree(n) == 0]

    for root in roots:
        metagraph = nx.DiGraph()
        gmetagraph.update(fold_graph(nx_graph, metagraph, None, root, None, rules))

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
        metagraph.nodes[v]["leaf"] = Leaf(v)
        if local_root is not None:
            metagraph.add_edge(local_root, v)
        local_root = v

    for w in graph.successors(v):
        metagraph = fold_graph(graph, metagraph, v, w, local_root, rules)
    return metagraph


def get_flag(props, rules):
    conclusion = []
    for r in rules:
        flag = []
        for subrule in r:
            if "how" not in subrule:
                flag.append(props[subrule["key"]] == subrule["value"])
            elif subrule["how"] == "contains":
                flag.append(subrule["value"] in props[subrule["key"]])
        conclusion += [all(flag)]
    return any(conclusion)
