import os
from networkx.drawing.nx_agraph import to_agraph
import requests


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

