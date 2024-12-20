# Local number of directed triangles
# Copied directly from NetworkX source code for nx.clustering as counting directed triangles is not implemented directly in networkx:
# https://networkx.org/documentation/stable/_modules/networkx/algorithms/cluster.html#clustering

from itertools import chain

''' Function for listing directed triangles

Parameters:
G - network object

Returns:
List of directed triangles

'''
def triangles(G):
    to_return = []
    nodes = list(G.nodes())
    td_iter = _directed_triangles_and_degree_iter(G, nodes)
    for v, dt, db, t in td_iter:
        to_return.append(t)
    assert len(G.nodes()) == len(to_return), "wrong length"
    return(to_return)

def _directed_triangles_and_degree_iter(G, nodes=None):
    """Return an iterator of
    (node, total_degree, reciprocal_degree, directed_triangles).

    Used for directed clustering.
    Note that unlike `_triangles_and_degree_iter()`, this function counts
    directed triangles so does not count triangles twice.

    """
    nodes_nbrs = ((n, G._pred[n], G._succ[n]) for n in G.nbunch_iter(nodes))

    for i, preds, succs in nodes_nbrs:
        ipreds = set(preds) - {i}
        isuccs = set(succs) - {i}

        directed_triangles = 0
        for j in chain(ipreds, isuccs):
            jpreds = set(G._pred[j]) - {j}
            jsuccs = set(G._succ[j]) - {j}
            directed_triangles += sum(
                1
                for k in chain(
                    (ipreds & jpreds),
                    (ipreds & jsuccs),
                    (isuccs & jpreds),
                    (isuccs & jsuccs),
                )
            )
        dtotal = len(ipreds) + len(isuccs)
        dbidirectional = len(ipreds & isuccs)
        yield (i, dtotal, dbidirectional, directed_triangles)