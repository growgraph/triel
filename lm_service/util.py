import os

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
    # for n in graph.nodes:
    #     label = graph.nodes[n]["label"] if "label" in graph.nodes[n] else ""
    #     graph.nodes[n]["label"] = f"{n}#{label}"
    dot = to_agraph(graph)
    dot.layout("dot")
    if plot_png:
        dot.draw(
            path=os.path.join(path, f"{name}.png"), format="png", prog=prog
        )
    if plot_pdf:
        dot.draw(
            path=os.path.join(path, f"{name}.pdf"), format="pdf", prog=prog
        )


def plot_leaves(metagraph, path, root_fname):
    graphs = [
        metagraph.nodes[n]["leaf"]
        for n in metagraph.nodes()
        if metagraph.nodes[n]["leaf"].is_compound()
    ]
    for j, mg in enumerate(graphs):
        plot_graph(mg.tree, path, f"{root_fname}_leaf_{j}")
