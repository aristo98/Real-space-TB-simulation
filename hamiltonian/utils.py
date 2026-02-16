from math import isclose, sqrt
from collections import deque, defaultdict
from numpy import array
import numpy as np


def nearest_k_per_site(positions: array, k: int = 2) -> dict:
    """
    Find k nearest neighbors of each site (OBC assumed).

    :param int k: Number of nearest neighbors
    :return: Dictionary containing indices of k-NNs of each site and their corresponding distances
    :rtype: dict
    """

    N = positions.shape[0]
    # pairwise difference (N, N, d)
    diff = positions[:, None, :] - positions[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)  # squared distances
    # exclude self
    np.fill_diagonal(d2, np.inf)

    # full ordering per row (ascending distances)
    full_order = np.argsort(d2, axis=1)  # shape (N, N)
    full_dists_sorted = np.sqrt(np.take_along_axis(d2, full_order, axis=1))

    # k nearest
    k = min(k, N - 1)
    nearest_idx = full_order[:, :k]
    nearest_dists = full_dists_sorted[:, :k]

    return {
        "nearest_idx": nearest_idx,
        "nearest_dists": nearest_dists,
    }


def find_discrete_combinations_optimized(
    vectors: list | tuple, d: float, max_coeff_per_dim: int = 10
) -> list:
    """
    Find integer combinations of vectors that yield a point at distance d from origin.

    :param list | array | tuple vectors: Arrays of Bravais lattice vectors + basis vectors
    :param float d: Target distance
    :param int max_coeff_per_dim: Maximum value of coefficients for each vector
    :return: List of coefficients and the resulting vectors
    :rtype: list
    """
    vectors = [np.array(v, dtype=float) for v in vectors]
    k = len(vectors)

    # Precompute squared norms and dot products
    norms_sq = [np.dot(v, v) for v in vectors]

    solutions = []

    # Use depth-first search with pruning
    def dfs(current_coeffs, current_vec, current_norm_sq, idx):
        if idx == k:
            # Check if we found a solution
            if isclose(sqrt(current_norm_sq), d):
                solutions.append((tuple(current_coeffs), array(current_vec)))
            return

        # Estimate bounds for remaining coefficients
        # Simple bound: |coeff| ≤ ceil((d^2 - current_norm_sq) / min_norm_sq)
        min_remaining_norm_sq = min(norms_sq[idx:]) if idx < k else 0

        # Calculate maximum possible coefficient magnitude
        if min_remaining_norm_sq > 0:
            max_coeff = int(np.ceil((d**2 - current_norm_sq) / min_remaining_norm_sq))
            max_coeff = min(max_coeff, max_coeff_per_dim)
        else:
            max_coeff = max_coeff_per_dim

        coeff_range = range(-max_coeff, max_coeff + 1)

        for coeff in coeff_range:
            new_vec = current_vec + coeff * vectors[idx]
            new_norm_sq = np.dot(new_vec, new_vec)

            # Prune if norm already exceeds d^2 (for positive coeffs)
            if new_norm_sq > d**2 * 1.1:  # 10% tolerance
                continue

            dfs(current_coeffs + [coeff], new_vec, new_norm_sq, idx + 1)

    dfs([], np.zeros_like(vectors[0]), 0.0, 0)
    return solutions


def canonical_cycle_tuple(nodes):
    """
    Find cycles by scanning edges: remove edge (u,v) and find shortest u->v path; path + edge -> cycle
    canonicalize and deduplicate
    """
    # nodes: list without repeated start/end, e.g. [n0,n1,...,n_{m-1}]
    m = len(nodes)
    if m == 0:
        return ()
    # produce all rotations and reversed rotations; pick lexicographically smallest tuple
    seqs = []
    for k in range(m):
        seqs.append(tuple(nodes[k:] + nodes[:k]))
    rev = list(reversed(nodes))
    for k in range(m):
        seqs.append(tuple(rev[k:] + rev[:k]))
    return min(seqs)


def bfs_shortest_path_excluding_edge(u, v, adjacency, max_len=None, max_paths=1000):
    """
    Return a list of all shortest simple paths from u to v *excluding* the direct edge (u,v).
    Each path is a list of nodes [u, ..., v].

    - adjacency: dict(node)->iterable_of_neighbors (your adjacency dict-of-sets)
    - max_len: optional maximum allowed path length (number of nodes in path). If provided,
               any shortest path longer than max_len is discarded (and an empty list returned).
    - max_paths: safety cap on how many paths to return (prevent combinatorial explosion).
    """
    # Standard BFS to get distances and parents for shortest paths
    q = deque([u])
    dist = {u: 0}
    parents = defaultdict(list)  # node -> list of predecessors on shortest paths
    found_dist = None

    while q:
        cur = q.popleft()

        # If we've already found v at distance found_dist, don't explore nodes at greater distance
        if found_dist is not None and dist[cur] > found_dist:
            break

        for nb in adjacency[cur]:
            # forbid the direct edge in either direction
            if (cur == u and nb == v) or (cur == v and nb == u):
                continue

            if nb not in dist:
                dist[nb] = dist[cur] + 1
                parents[nb].append(cur)
                q.append(nb)
            else:
                # another shortest predecessor
                if dist[nb] == dist[cur] + 1:
                    parents[nb].append(cur)

        if cur == v:
            found_dist = dist[cur]
            # if shortest distance is already longer than allowed max_len (count nodes)
            if (max_len is not None) and (found_dist + 1 > max_len):
                return []

    # no path found
    if v not in dist:
        return []

    # enforce max_len (path length = dist[v] + 1)
    path_len = dist[v] + 1
    if (max_len is not None) and (path_len > max_len):
        return []

    # backtrack all shortest paths from v to u using parents
    results = []

    def backtrack(node, acc):
        # acc currently holds nodes after `node` on the path (from node->...->v)
        if len(results) >= max_paths:
            return  # safety cap
        if node == u:
            results.append([u] + list(reversed(acc)))
            return
        for p in parents[node]:
            backtrack(p, acc + [node])

    backtrack(v, [])
    return results


# def bfs_shortest_path_excluding_edge(u, v, adjacency, max_len):
#     """
#     BFS shortest-path helper (returns path list [u,...,v]) with edge (u,v) removed temporarily
#     """
#     # adjacency is dict of sets; make local shallow copy of neighbors for traversal but do not mutate original
#     # We'll treat the edge (u,v) as forbidden.
#     q = deque([u])
#     prev = {u: None}
#     while q:
#         cur = q.popleft()
#         # early pruning
#         if (
#             prev[cur] is not None
#             and (max_len is not None)
#             and (len(prev) > max_len + 1)
#         ):
#             # rough bound; not precise but prevents infinite loops; actual checking done when reconstructing
#             pass
#         for nb in adjacency[cur]:
#             if (cur == u and nb == v) or (cur == v and nb == u):
#                 # skip the forbidden direct edge
#                 continue
#             if nb not in prev:
#                 prev[nb] = cur
#                 if nb == v:
#                     # reconstruct path
#                     path = [v]
#                     while prev[path[-1]] is not None:
#                         path.append(prev[path[-1]])
#                     path.reverse()
#                     return path  # [u,...,v]
#                 q.append(nb)
#     return None
