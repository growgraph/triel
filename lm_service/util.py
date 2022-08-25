import os

from networkx.drawing.nx_agraph import to_agraph


def plot_graph(graph, path, name, plot_png=False, plot_pdf=True):
    dot = to_agraph(graph)
    dot.layout("dot")
    if plot_png:
        dot.draw(
            path=os.path.join(path, f"{name}.png"), format="png", prog="dot"
        )
    if plot_pdf:
        dot.draw(
            path=os.path.join(path, f"{name}.pdf"), format="pdf", prog="dot"
        )


def plot_leaves(metagraph, path, root_fname):
    graphs = [
        metagraph.nodes[n]["leaf"]
        for n in metagraph.nodes()
        if metagraph.nodes[n]["leaf"].is_compound()
    ]
    for j, mg in enumerate(graphs):
        plot_graph(mg.tree, path, f"{root_fname}_leaf_{j}")


def to_string(obj):
    if isinstance(obj, dict):
        return {str(k): to_string(item) for k, item in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_string(item) for item in obj]
    else:
        return str(obj)


def to_string_keys(obj):
    if isinstance(obj, dict):
        return {to_string_keys(k): item for k, item in obj.items()}
    if isinstance(obj, (list, tuple)):
        return tuple(to_string(item) for item in obj)
    else:
        return str(obj)
