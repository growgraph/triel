import networkx as nx
from typing import Optional


class Leaf:
    def __init__(self, root):
        self.root = None
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
