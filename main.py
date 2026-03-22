# We cannot, for obvious reasons, represent Conway's infinite grid in a matrix.
# We may represent is as an ever-expanding graph; each node corresponds to a
# coordinate pair $(i, j)$, and X and Y are connected if their (supremum) distance is at most 1.
# The rules are simple:
# 1. A node with less than 2 neighbors is removed.
# 2. A node with two or three neighbors survives.
# 3. A node with more than three neighbors die.
# 4. A node is added if it has exactly three neighborhoods in the graph.
# To track (4), we also create a _border_ graph that maintains the potential new nodes to the original graph.
# The rules for expanding this border graph are similarly simple.
# 1. A node is added to the border graph
#    if (i) its supremum distance to the original graph is at most 1 and (ii) the node is not in the border graph.
# 2. A node is removed from the border graph if it
#    is either (i) added to the border graph or (ii) its distance to the original graph is at least 2.
# I will call the original and border graphs OG and BG, respectively.
# The implementation will be GPU friendly (i.e., operations will be executed in matrix form)
# and in torch (JAX is not suitable for this).
# This will also be a bare bones implementation.

from dataclasses import dataclass, replace

import torch


@dataclass
class OG:
    nodes: torch.Tensor  # (N, 2)


@dataclass
class BG:
    nodes: torch.Tensor  # (N, 2)


def count_neighbors(new_nodes: torch.Tensor, curr_nodes: torch.Tensor):
    # This function should not be Jitted, as the shape of nodes' will change.
    distance = torch.abs(
        new_nodes[:, None, ...] - curr_nodes[None, ...]
    )  # (K, N, 2)
    distance = distance.max(dim=2)  # (K, N)
    distance = distance.values
    return distance.sum(dim=1)  # (K,)


def get_border(og: OG):
    e1 = torch.tensor([1, 0])
    e2 = torch.tensor([0, 1])
    ed = e1 + e2

    candidates = torch.vstack(
        [
            og.nodes + e1[None, ...],
            og.nodes + e2[None, ...],
            og.nodes + ed[None, ...],
            og.nodes - e1[None, ...],
            og.nodes - e2[None, ...],
            og.nodes - ed[None, ...],
        ]
    )  # (6N, 2)

    candidates_in_graph = torch.all(
        candidates[:, None, ...] == og.nodes[None, ...], dim=2
    ).any(dim=1)  # (6N,)

    return candidates[~candidates_in_graph]


def evolve(og: OG, bg: BG):
    # This creates the next generation of Conway's game of life.

    # First: check which nodes should be added
    n_neigh_bg = count_neighbors(bg.nodes, og.nodes)

    # Fourth rule: If n_neigh == 3, becomes alive.
    nodes_born = bg.nodes[n_neigh_bg == 3]

    # First and third rules: check which nodes will die
    n_neigh_og = count_neighbors(og.nodes, og.nodes)
    should_die = (n_neigh_og < 2) | (n_neigh_og > 3)

    # Third rule: the other nodes survive
    remaining_nodes = torch.vstack([og.nodes[~should_die], nodes_born])

    # Update the bg
    og = replace(og, nodes=remaining_nodes)
    bg = replace(bg, nodes=get_border(og))

    return og, bg


def run(seed: torch.Tensor, iterations: int):
    og = OG(nodes=seed)
    bg = BG(nodes=get_border(og))

    for _ in range(iterations):
        og, bg = evolve(og, bg)

    print(og, bg)


if __name__ == "__main__":
    seed = torch.tensor([[0, 0], [0, 1], [2, 0]])
    run(seed, iterations=512)
