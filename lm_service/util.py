import os
from timeit import default_timer

from networkx.drawing.nx_agraph import to_agraph


def plot_graph(graph, path, name, plot_png=False, plot_pdf=True, prog="dot"):
    """

    :param graph:
    :param path:
    :param name:
    :param plot_png:
    :param plot_pdf:
    :param prog: prog=[‘neato’|’dot’|’twopi’|’circo’|’fdp’|’nop’]
    :return:
    """

    dot = to_agraph(graph)
    dot.layout("dot")
    if plot_png:
        dot.draw(path=os.path.join(path, f"{name}.png"), format="png", prog=prog)
    if plot_pdf:
        dot.draw(path=os.path.join(path, f"{name}.pdf"), format="pdf", prog=prog)


def plot_leaves(metagraph, path, root_fname):
    graphs = [
        metagraph.nodes[n]["leaf"]
        for n in metagraph.nodes()
        if metagraph.nodes[n]["leaf"].is_compound()
    ]
    for j, mg in enumerate(graphs):
        plot_graph(mg.tree, path, f"{root_fname}_leaf_{j}")


class Timer:
    def __init__(self):
        self.timer = default_timer

    def __enter__(self):
        self.start = self.timer()
        return self

    def __exit__(self, *args):
        end = self.timer()
        self.elapsed = end - self.start
