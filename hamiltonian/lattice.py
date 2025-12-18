from itertools import product
import numpy as np
from numpy import array
from scipy.spatial import cKDTree
from utils import bfs_shortest_path_excluding_edge, canonical_cycle_tuple


class Lattice:
    """
    Prepare a crystal lattice for the construction of a tight-binding-Hamiltonian

    Features:
    - Dimension of the crystal D=1/2/3
    - Bravais lattice vectors
    - Basis vectors in each unit cell
    - Size customization
    - Enumeration of each atomic site
    - Finding a pair of atomic sites with distance d
    - Enlist all plaquetes with a side length equals d
    """

    def __init__(
        self,
        D: int,
        Rs: array,
        taus: array,
        size: tuple,
        periodic=True,
        enumerate_site: bool = True,
    ):
        """
        Initialize the crystal

        :param int D: Dimension of the crystal (D=1/2/3)
        :param array Rs: Bravais lattice vectors stored in (m,n)-array.

                        - (1 <= m <= 3, 1 <= n <= 3)
                        - Row vector is the lattice vector
        :param array taus: Vectors for basis points stored in (m,n)-array.

                        - (n <= 3)
                        - Row vector is the basis vector
        :param tuple size: Size of the crystals stored in tuples of length 1/2/3. (Nx,) in case of 1D
        :param bool periodic: Impose PBC? Default is True
        :param bool enumerate_site: Directly enumerate_site? Default is True
        """
        assert D == Rs.shape[0] and D == Rs.shape[1], "Dimension mismatch Rs!"
        assert D == taus.shape[1], "Dimension mismatch taus!"
        assert D == len(size), "Dimension mismatch crystal sizes!"
        self._D = D
        self._Rs = Rs
        self._taus = taus
        self._size = size
        self._periodic = periodic
        self._idx_lst: np.ndarray = np.array([], dtype=int)
        self._R_site_lst: np.ndarray = np.empty((0, D))
        self._R_lattice_lst: np.ndarray = np.empty((0, D))
        self._n_lst: np.ndarray = np.empty((0, D), dtype=int)
        self._tau_lst: np.ndarray = np.empty((0, D))
        self._ntau_lst: np.ndarray = np.empty((0, len(taus)), dtype=int)

        if enumerate_site:
            self.enumerate_site()

    @classmethod
    def two_d_honeycomb_lattice(
        cls,
        Nx: int,
        Ny: int,
        a: float = 1.0,
        periodic: bool = True,
        enumerate_site: bool = True,
    ):
        """
        Instantiate a 2D honeycomb graphene lattice with the lattice constant a
        and with the dimension (Nx,Ny)

        :param int Nx: Span of the first lattice vector
        :param int Ny: Span of the second lattice vector
        :param float a: Lattice constant
        """
        D = 2
        Rs = a * array([[1.0, 0.0], [1 / 2, np.sqrt(3) / 2]])
        taus = a * array([[0.0, 0.0], [1 / 2, np.sqrt(3) / 6]])
        size = (Nx, Ny)
        return cls(D, Rs, taus, size, periodic, enumerate_site)

    @classmethod
    def two_d_square_lattice(
        cls,
        Nx: int,
        Ny: int,
        a: float = 1.0,
        b: float = 1.0,
        periodic: bool = True,
        enumerate_site: bool = True,
    ):
        """
        Instantiate a 2D square lattice with the lattice constant (a,b)
        andwith the dimension (Nx,Ny). Each unit cell contains only one atom

        :param int Nx: Span of the first lattice vector
        :param int Ny: Span of the second lattice vector
        :param float a: Lattice constant of the first lattice vector
        :param float b: Lattice constant of the second lattice vector
        """
        D = 2
        Rs = array([[a * 1.0, 0.0], [0.0, b * 1.0]])
        taus = array([[0.0, 0.0]])
        size = (Nx, Ny)
        return cls(D, Rs, taus, size, periodic, enumerate_site)

    @classmethod
    def three_d_square_lattice(
        cls,
        Nx: int,
        Ny: int,
        Nz: int,
        a: float = 1.0,
        b: float = 1.0,
        c: float = 1.0,
        periodic: bool = True,
        enumerate_site: bool = True,
    ):
        """
        Instantiate a 3D square lattice with the lattice constant (a,b,c)
        and with the dimension (Nx,Ny,Nz). Each unit cell contains only one atom

        :param int Nx: Span of the first lattice vector
        :param int Ny: Span of the second lattice vector
        :param int Nz: Span of the third lattice vector
        :param float a: Lattice constant of the first lattice vector
        :param float b: Lattice constant of the second lattice vector
        :param float c: Lattice constant of the third lattice vector
        """
        D = 3
        Rs = array([[a * 1.0, 0.0, 0.0], [0.0, b * 1.0, 0.0], [0.0, 0.0, c * 1.0]])
        taus = array([[0.0, 0.0, 0.0]])
        size = (Nx, Ny, Nz)
        return cls(D, Rs, taus, size, periodic, enumerate_site)

    def enumerate_site(self) -> None:
        """
        Enumerate each site. Order of scan:
        z -> enumerate each atom inside the unit cell
        -> y -> enumerate each atom inside the unit cell
        -> x -> enumerate each atom inside the unit cell
        """
        # 1. Generate Lattice Indices Grid
        # np.indices returns shape (D, Nz, Ny, Nx)
        grids = np.indices(self._size[::-1])
        # Reshape to (D, N_cells) and Transpose to (N_cells, D)
        # This gives us the list of [nz, ny, nx] for every unit cell
        n_vecs = grids.reshape(self._D, -1).T
        N_cells = n_vecs.shape[0]
        # 2. Calculate Lattice Points (R_latt = n . Rs)
        # If n_vecs is (n3, n2, n1), we align it to allow broadcasting.
        R_lattice_all = n_vecs @ self._Rs
        # 3. Handle Basis Vectors (Broadcasting)
        # taus shape: (N_basis, D)
        N_taus = len(self._taus)
        # R_lattice_all[:, None, :] shape: (N_cells, 1, D)
        # taus[None, :, :]          shape: (1, N_basis, D)
        # Result                    shape: (N_cells, N_basis, D)
        R_site_all = R_lattice_all[:, None, :] + self._taus[None, :, :]
        # 4. Flatten and Store
        # We flatten the first two dimensions to get a simple list of all sites
        N_total = N_cells * N_taus
        self._R_site_lst = R_site_all.reshape(N_total, self._D)
        self._idx_lst = np.arange(N_total)
        # Expand metadata to match the flattened structure
        # Repeat lattice vector for each basis atom in the cell
        self._R_lattice_lst = np.repeat(R_lattice_all, N_taus, axis=0)
        # Repeat lattice indices (n vectors)
        self._n_lst = np.repeat(n_vecs, N_taus, axis=0)
        # Tile basis vectors (repeat the basis set for every cell)
        self._tau_lst = np.tile(self._taus, (N_cells, 1))
        # Generate one-hot encoding for basis indices
        # e.g., if N_basis=2: [1,0], [0,1], [1,0], [0,1]...
        tau_identity = np.eye(N_taus, dtype=int)
        self._ntau_lst = np.tile(tau_identity, (N_cells, 1))

    def customize_crystal(
        self, n_lst: list | np.ndarray, ntau_lst: list | np.ndarray
    ) -> None:
        """
        Create a customized crystal that is not necessarily the sum over all Nx,Ny,Nz and basis vectors
        Make sure that each element of ntau_lst reads (0,0,...,1,...,0). Calling this method could overwrite
        previously generated attributes.

        :param n_lst: List/Array of lattice indices (e.g., [[0,0], [0,1], ...])
        :param ntau_lst: List/Array of basis selectors (e.g., [[1,0], [0,1], ...])
        """
        n_arr = np.asarray(n_lst)
        ntau_arr = np.asarray(ntau_lst)
        assert len(n_arr) == len(
            ntau_arr
        ), "Length mismatch between n_lst and ntau_lst!"
        assert n_arr.shape[1] == self._D, "Dimension mismatch in n_lst!"

        # 1. Update Storage Attributes
        # We overwrite them completely (no appending to old data)
        self._n_lst = n_arr
        self._ntau_lst = ntau_arr
        self._idx_lst = np.arange(len(n_arr))

        # 2. Vectorized Calculation of Positions
        # R_lattice = n_vecs @ Rs
        R_lattice_vals = n_arr @ self._Rs
        self._R_lattice_lst = R_lattice_vals

        # Tau = ntau_vecs @ taus
        # This extracts the correct basis vector using the one-hot encoding (or weights) in ntau
        tau_vals = ntau_arr @ self._taus
        self._tau_lst = tau_vals

        # R_site = R_lattice + Tau
        self._R_site_lst = R_lattice_vals + tau_vals

    def find_pairs_at_distance(
        self, dist_target: float | int, tol: float = 1e-4
    ) -> list:
        """
        Find all unordered pairs of sites whose (minimum-image if periodic) distance is within [target-tol, target+tol].

        :param float | int dist_target: Target distance
        :param float tol: Numerical tolerance for distance-matching
        :param bool periodic: Apply periodic boundary conditions across the supercell
        :return: List containing all site pairs with the distance d=dist
        :rtype: array (list of tuples)
        """
        r_min = dist_target - tol
        r_max = dist_target + tol

        # Ensure positions are float
        positions = np.asarray(self._R_site_lst, dtype=np.float64)
        N = positions.shape[0]

        # 1. Build Tree once
        tree = cKDTree(positions)

        # ---------------------------------------------------------
        # PART A: Internal Pairs (Inside the central box)
        # ---------------------------------------------------------
        # query_pairs is highly optimized for self-search (returns set of i<j)
        # It finds everything < r_max. We filter for r_min later.
        raw_pairs = tree.query_pairs(r=r_max, output_type="set")

        # Convert to array for filtering
        if raw_pairs:
            pairs_array = np.array(list(raw_pairs))  # Shape (M, 2)

            # Calculate exact distances to filter lower bound
            diffs = positions[pairs_array[:, 0]] - positions[pairs_array[:, 1]]
            dists = np.linalg.norm(diffs, axis=1)

            # Keep only those >= r_min
            mask = dists >= r_min
            pairs_internal = pairs_array[mask]
        else:
            pairs_internal = np.empty((0, 2), dtype=int)

        # ---------------------------------------------------------
        # PART B: Periodic Pairs (Crossing boundaries)
        # ---------------------------------------------------------
        periodic_pairs_list = []

        if self._periodic:
            # Calculate box vectors (works for triclinic too if _Rs is rotation)
            box_vecs = (self._Rs.T * np.array(self._size)).T

            # Generate minimal shifts (26 images for 3D)
            # We skip (0,0,0) because that is covered by Part A
            ranges = [range(-1, 2) for _ in range(self._D)]
            shift_indices = np.array(list(product(*ranges)))
            shift_indices = shift_indices[np.any(shift_indices != 0, axis=1)]

            # Convert to physical vectors
            shift_vecs = shift_indices @ box_vecs

            for shift in shift_vecs:
                shifted_positions = positions + shift

                # 1. Radius Query
                # Find neighbors within r_max. No 'k' limit implies no missed neighbors.
                # Returns a list of lists: idx_list[i] contains neighbors of atom i
                idx_list = tree.query_ball_point(shifted_positions, r=r_max)

                # 2. Vectorized Flattening (The Optimization)
                # Count neighbors per atom to prepare for flattening
                counts = [len(x) for x in idx_list]
                total_found = sum(counts)

                if total_found == 0:
                    continue

                # 'rows' = indices of the shifted particles (i)
                # 'cols' = indices of the tree particles (j)
                rows = np.repeat(np.arange(N), counts)
                cols = np.concatenate(idx_list).astype(int)

                # 3. Filter by Distance (Vectorized)
                # We must re-calculate distance because query_ball_point is approximate (ball)
                # and we need to check the r_min lower bound.
                # Vector math: pos[i]_shifted - pos[j]_original
                diffs = shifted_positions[rows] - positions[cols]
                dists = np.linalg.norm(diffs, axis=1)

                # Check strict shell [r_min, r_max]
                # Note: query_ball_point guarantees dist <= r_max, so we primarily check r_min
                mask = dists >= r_min

                if np.any(mask):
                    valid_rows = rows[mask]
                    valid_cols = cols[mask]

                    # Stack them. Note: We might find (i, j) here and (j, i) in the
                    # opposite shift direction. We let `np.unique` handle this at the end.
                    new_pairs = np.column_stack((valid_rows, valid_cols))
                    periodic_pairs_list.append(new_pairs)

        # ---------------------------------------------------------
        # PART C: Merge and Cleanup
        # ---------------------------------------------------------
        all_pairs_list = [pairs_internal] + periodic_pairs_list
        if not all_pairs_list:
            return np.array([], dtype=int), np.array([], dtype=int)

        all_pairs = np.vstack(all_pairs_list)

        if all_pairs.size == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        # Sort each pair so (i, j) is always (min, max) to detect (i, j) vs (j, i) duplicates
        all_pairs = np.sort(all_pairs, axis=1)

        # Remove duplicates
        unique_pairs = np.unique(all_pairs, axis=0)

        # Return as two arrays (source, target)
        return unique_pairs[:, 0], unique_pairs[:, 1]

    def find_plaquettes(
        self,
        bond_lengths: list | None = None,
        tol: float = 1e-6,
        max_cycle_length: int = 8,
    ):
        """
        Identify and enumerate all elementary plaquettes (chordless cycles) in the lattice.

        This method builds a connectivity graph based on the provided bond length(s) and
        performs a depth-first search to find closed loops. It explicitly filters for
        "chordless" cycles to ensure only the smallest interaction faces are returned
        (e.g., preventing a rhombus formed by two triangles from being counted as a plaquette).

        If bond_lengths == None -> identify the elementary plaquette with the smallest distance

        :param bond_lengths: Target bond length(s) that define a valid connection between sites.
        :type bond_lengths: list[float] | None
        :param float tol: Numerical tolerance for matching the given bond length.
        :param bool periodic: If True, Periodic Boundary Conditions (PBC) are applied.
                              The algorithm will correctly identify cycles crossing boundaries
                              and unwrap coordinates for geometric calculations.
        :param int max_cycle_length: Maximum number of sides (sites) in a cycle.
                                     Limiting this (e.g., to 6 or 8) prevents combinatorial
                                     blow-up in complex lattices.

        :return: A list of plaquettes, where each plaquette is a dictionary containing:

                 * **'id'** (*int*): Enumeration index of the plaquette.
                 * **'nodes'** (*tuple[int]*): Ordered sequence of site indices forming the boundary.
                   Canonicalized to start with the smallest index.
                 * **'length'** (*int*): Number of sites/edges in the plaquette.
                 * **'center'** (*np.ndarray*): Geometric center of the plaquette (shape ``(D,)``).
                   Calculated using unwrapped coordinates for periodic systems.
                 * **'normal'** (*np.ndarray*): Unit normal vector (shape ``(3,)``).
                   For 2D lattices (D=2), this defaults to ``[0, 0, 1]``.
        :rtype: list[dict]
        """
        positions = np.asarray(self._R_site_lst, dtype=float)
        N = positions.shape[0]
        if N == 0:
            return []

        # -------------------------
        # determine bond_lengths if not provided (take distinct distances >0 sorted, choose smallest)
        # -------------------------
        bond_lengths_use = []
        if bond_lengths is None:
            # Check just the first atom (or first few) to find nearest neighbors
            # efficient for large N
            sample_indices = [0]

            if not self._periodic:
                tree = cKDTree(positions)
                # query 10 nearest neighbors for the sample
                dists, _ = tree.query(positions[sample_indices], k=10)
                # Flatten and sort valid distances > tol
                candidates = np.unique(dists)
                candidates = candidates[candidates > tol]
            else:
                # Periodic: Calculate distances from atom 0 to all others (1xN, not NxN)
                # Re-use logic similar to find_pairs but just for row 0
                diffs = positions[None, :, :] - positions[sample_indices, None, :]

                # Manual Periodic Min-Image for 1 row
                super_Rs = (self._Rs.T * np.array(self._size)).T
                try:
                    inv_super_Rs = np.linalg.inv(super_Rs)
                except np.linalg.LinAlgError:
                    inv_super_Rs = np.eye(self._D)

                coeffs = diffs @ inv_super_Rs
                coeffs_mic = coeffs - np.round(coeffs)
                dr_mic = coeffs_mic @ super_Rs

                # Check neighbors for skew
                min_dists_sq = np.sum(dr_mic**2, axis=-1)
                aux_shifts_int = np.array(list(product([-1, 0, 1], repeat=self._D)))
                aux_shifts_int = aux_shifts_int[np.any(aux_shifts_int != 0, axis=1)]
                if aux_shifts_int.size > 0:
                    aux_vecs = aux_shifts_int @ super_Rs
                    for shift in aux_vecs:
                        d_cand = np.sum((dr_mic + shift) ** 2, axis=-1)
                        min_dists_sq = np.minimum(min_dists_sq, d_cand)

                dists = np.sqrt(min_dists_sq)
                candidates = np.unique(dists)
                candidates = candidates[candidates > tol]

            if candidates.size > 0:
                # Take the smallest non-zero distance found
                bond_lengths_use = [candidates[0]]
            else:
                return []
        else:
            bond_lengths_use = list(bond_lengths)

        # -------------------------
        # 2. Build Adjacency Graph (Optimized)
        # -------------------------
        adj = {i: set() for i in range(N)}

        for b in bond_lengths_use:
            # Call the optimized finder (returns tuple of arrays)
            rows, cols = self.find_pairs_at_distance(b, tol=tol)

            # fast iteration
            for i, j in zip(rows, cols):
                adj[i].add(j)
                adj[j].add(i)

        # Pre-compute edge set for O(1) chord checks
        edge_set = set()
        for i in adj:
            for j in adj[i]:
                if i < j:
                    edge_set.add(frozenset((i, j)))

        # -------------------------
        # 3. Find Cycles (BFS/DFS)
        # -------------------------
        cycle_set = set()
        # We iterate edges to seed the search, reducing search space compared to all-node DFS
        sorted_edges = sorted(list(edge_set))  # Deterministic order

        for u, v in sorted_edges:
            # Find shortest path from u to v excluding the direct edge (u,v)
            path = bfs_shortest_path_excluding_edge(u, v, adj, max_cycle_length - 1)

            if path:
                # Cycle found: u -> ... -> v -> u
                # Path includes u and v. Cycle length is len(path).
                if len(path) < 3:
                    continue

                # Chordless check:
                # A cycle is chordless if the only edges between nodes in the cycle
                # are the ones forming the cycle itself.
                cycle_nodes = path  # List of nodes
                n_c = len(cycle_nodes)
                is_chordless = True

                # Check all non-adjacent pairs in the cycle
                # In the path [n0, n1, ... nk], n0 and nk are connected by the edge we excluded.
                # So we check indices i, j where distance is not 1 and not (0, last).

                # Quick optimization: mapping node -> index in cycle
                node_to_idx = {node: k for k, node in enumerate(cycle_nodes)}

                for k, node in enumerate(cycle_nodes):
                    # check neighbors of this node in the full graph
                    for neighbor in adj[node]:
                        # if neighbor is in cycle
                        if neighbor in node_to_idx:
                            idx_neighbor = node_to_idx[neighbor]
                            # verify topological distance
                            diff = abs(k - idx_neighbor)
                            # Adjacent in cycle means diff is 1 or (n_c - 1)
                            if diff > 1 and diff < n_c - 1:
                                is_chordless = False
                                break
                    if not is_chordless:
                        break

                if is_chordless:
                    cycle_set.add(canonical_cycle_tuple(list(path)))

        cycles = sorted(list(cycle_set))

        # -------------------------
        # 4. Compute Geometry (Unwrapping & Normals)
        # -------------------------
        # Pre-calculate matrix for unwrapping if periodic
        if self._periodic:
            super_Rs = (self._Rs.T * np.array(self._size)).T
            try:
                inv_super_Rs = np.linalg.inv(super_Rs)
            except np.linalg.LinAlgError:
                inv_super_Rs = np.eye(self._D)
                super_Rs = np.eye(self._D)

        plaquettes = []
        for pid, cyc_tuple in enumerate(cycles):
            cyc_nodes = list(cyc_tuple)

            # --- Unwrapping Logic (Local) ---
            # We reconstruct the polygon by walking the cycle
            poly_coords = [positions[cyc_nodes[0]]]

            for k in range(1, len(cyc_nodes)):
                prev_pos = poly_coords[-1]
                curr_pos = positions[cyc_nodes[k]]
                diff = curr_pos - prev_pos

                if self._periodic:
                    # Minimum Image Convention
                    coeffs = diff @ inv_super_Rs
                    coeffs_mic = coeffs - np.round(coeffs)
                    diff = coeffs_mic @ super_Rs

                poly_coords.append(prev_pos + diff)

            poly_coords = np.array(poly_coords)

            # Center
            center = poly_coords.mean(axis=0)

            # Normal (Newell's method / Cross product)
            normal = np.array([0.0, 0.0, 1.0])  # Default

            if self._D == 3:
                # Calculate normal via cross products of edges relative to center
                # Robust for non-planar 3D polygons
                n_acc = np.zeros(3)
                n_nodes = len(poly_coords)
                for k, p_curr in enumerate(poly_coords):
                    p1 = p_curr - center
                    # We still use index k for the neighbor, but now we use p_curr directly
                    p2 = poly_coords[(k + 1) % n_nodes] - center
                    n_acc += np.cross(p1, p2)

                norm_mag = np.linalg.norm(n_acc)
                if norm_mag > 1e-8:
                    normal = n_acc / norm_mag

            elif self._D == 2:
                # 2D Signed Area for orientation
                xs = poly_coords[:, 0]
                ys = poly_coords[:, 1]
                area = np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))
                if abs(area) > 1e-12:
                    normal = np.array([0.0, 0.0, 1.0 if area > 0 else -1.0])

            plaquettes.append(
                {
                    "id": pid,
                    "nodes": cyc_nodes,
                    "length": len(cyc_nodes),
                    "center": center,
                    "normal": normal,
                }
            )

        return plaquettes
