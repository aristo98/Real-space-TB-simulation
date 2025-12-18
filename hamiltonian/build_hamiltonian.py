import warnings
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import expm_multiply
from lattice import Lattice


class TightBindingModel:
    """
    A class to define and build Tight-Binding Hamiltonians on a Lattice.

    Features:
    - distinct separation of geometry (Lattice) and physics (Model).
    - caching of neighbor pairs (geometry) for fast rebuilding when parameters change.
    - fluent interface (method chaining).
    """

    def __init__(self, lattice: Lattice, orbitals: list | int = 1) -> None:
        """
        Initialize a (multi-orbital) Tight-Binding Model.

        :param Lattice lattice: Instance of the Lattice class.
        :param list | int orbitals: Either an integer (number of orbitals) or a list of names
                         (e.g., ['s', 'px', 'py', 'pz']).
        """
        self.lattice = lattice
        self._N_sites = len(lattice._idx_lst)

        # Handle orbital definition
        if isinstance(orbitals, int):
            self.n_orbitals = orbitals
            self.orb_names = [str(i) for i in range(orbitals)]
        else:
            self.n_orbitals = len(orbitals)
            self.orb_names = orbitals

        # Total matrix size
        self._dim = self._N_sites * self.n_orbitals

        # Storage
        self._hopping_terms = []
        self._onsite_couplings = []  # Stores off-diagonal on-site terms
        self._onsite_terms = []  # For local couplings (SOC, crystal field)
        self._onsite_energies = np.zeros(self._dim, dtype=complex)

        # Geometric Cache
        self._pair_cache = {}

    def set_onsite_energies(self, energies: float | list | np.ndarray = 0.0) -> None:
        r"""
        Set on-site energies. Supports both uniform (translational invariant)
        and site-specific (disordered/potential profile) configurations.

        Equation: :math:`\hat{c}^{\dag}_{R\alpha}\hat{c}_{R\alpha}`

        :param float | list | np.ndarray energies:

            - If float: Set uniform energies for all sites and all orbitals

            - If shape is ``(n_orbitals,)``: Sets uniform energy for each orbital type across all sites.

              Example: ``[E_s, E_p]`` -> Every site has E_s and E_p.

            - If shape is ``(N_sites, n_orbitals)``: Sets specific energy for every orbital on every site.

              Example: ``[[E_s0, E_p0], [E_s1, E_p1], ...]``.

            - If shape is ``(N_total_dim,)``: Direct assignment to the diagonal.
        """
        # Case 1: Uniform scalar
        if (
            isinstance(energies, float)
            or isinstance(energies, int)
            or isinstance(energies, complex)
        ):
            self._onsite_energies = energies * np.ones(self._dim, dtype=complex)
        else:
            energies = np.asarray(energies, dtype=complex)

            # Case 2: Uniform per orbital type (Vector of length n_orbitals)
            if energies.ndim == 1 and len(energies) == self.n_orbitals:
                # Tile: Repeat [E_s, E_p] for N_sites
                # Result: [E_s, E_p, E_s, E_p, ...]
                self._onsite_energies = np.tile(energies, self._N_sites)

            # Case 3: Site-Specific (Matrix of N_sites x n_orbitals)
            elif energies.ndim == 2 and energies.shape == (
                self._N_sites,
                self.n_orbitals,
            ):
                # Flatten row by row to match the Hamiltonian basis order:
                # (Site 0 orb 0, Site 0 orb 1, Site 1 orb 0, ...)
                self._onsite_energies = energies.flatten()

            # Case 4: Flat array of full dimension (N_sites * n_orbitals)
            elif energies.ndim == 1 and len(energies) == self._dim:
                self._onsite_energies = energies

            else:
                raise ValueError(
                    f"Shape mismatch! System has {self._N_sites} sites and {self.n_orbitals} orbitals.\n"
                    f"Input shape {energies.shape} must be ({self.n_orbitals},), "
                    f"({self._N_sites}, {self.n_orbitals}), or ({self._dim},)."
                )

    def set_onsite_coupling(
        self, t: complex | np.ndarray, orb_i: int, orb_j: int
    ) -> None:
        r"""
        Set an on-site off-diagonal coupling between DIFFERENT orbitals i and j.
        Use set_onsite_energies instead for two identical orbitals i and j.

        Equation: :math:`\hat{c}^{\dag}_{R\alpha}\hat{c}_{R\beta}`

        :param complex | np.ndarray t: Coupling strength.

                  - Scalar: Uniform across all sites.
                  - Array (N_sites,): Different value for each site.
        :param int orb_i: Row orbital index.
        :param int orb_j: Col orbital index.
        """
        assert orb_i != orb_j, "Orbitals have to be different!"
        # Validate if array
        t = np.asarray(t) if not np.isscalar(t) else t
        if t.ndim > 0 and len(t) != self._N_sites:
            raise ValueError(
                f"Coupling array length {len(t)} does not match N_sites {self._N_sites}"
            )

        self._onsite_couplings.append({"t": t, "orb_i": orb_i, "orb_j": orb_j})
        return self

    def set_hopping(
        self, t: complex, d: float, orb_i: int, orb_j: int, tol: float = 1e-4
    ) -> None:
        r"""
        Set a hopping term (tunneling energy) between two orbitals on DIFFERENT atomic sites.
        Two orbitals may be both equal or different to/from each other.

        Rule: "For every pair of atomic sites separated
        by distance 'd', add a hopping 't' from orbital 'orb_i' on the first site to
        orbital 'orb_j' on the second site."

        Equation: :math:`\hat{c}^{\dag}_{R\alpha}\hat{c}_{R'\alpha}`

        :param complex t:
            The hopping amplitude (energy in eV).

            - Real values represent standard kinetic hopping.
            - Complex values often represent Peierls phases (magnetic fields) or SOC.

        :param float d:
            The bond length (distance between atomic centers) where this hopping is active.
            Example: For Nearest Neighbor in Graphene, d ≈ 1.42 Angstrom.

        :param int orb_i:
            The index of the source orbital on the starting site.
            (e.g., if orbitals=['s', 'px', 'py'], then 0='s', 1='px').

        :param int orb_j:
            The index of the destination orbital on the neighbor site.

            - If orb_i == orb_j, it's a "diagonal" hopping (e.g., s-s or px-px).
            - If orb_i != orb_j, it's an orbital mixing hopping (e.g., s-px).

        :param float tol:
            Numerical tolerance for identifying the distance 'd'.
            Defaults to 1e-4. Increase if lattice positions are slightly noisy.
        """
        assert d > tol, "Atomic sites have to be different!"
        self._hopping_terms.append(
            {"t": t, "d": d, "orb_i": orb_i, "orb_j": orb_j, "tol": tol}
        )

    def _get_pairs(self, d: float, tol: float = 1e-4):
        """
        Retrieve neighbor pairs from cache or calculate them via Lattice.
        Rounding d ensures tolerance consistency for dictionary keys.
        """
        # Create a unique key for distance (rounded to handle float tolerance)
        key = round(d, int(-np.log10(tol)))

        if key not in self._pair_cache:

            self._pair_cache[key] = self.lattice.find_pairs_at_distance(d, tol=tol)

        return self._pair_cache[key]

    def build(self) -> csr_matrix:
        """
        Constructs the sparse Hamiltonian from scratch.

        Suitable for static case or single-shot calculations

        :return: Sparse matrix of H
        :rtype: csr_matrix
        """
        row_list = []
        col_list = []
        data_list = []

        # 1. Diagonal Energies
        diag_idx = np.arange(self._dim)
        row_list.append(diag_idx)
        col_list.append(diag_idx)
        data_list.append(self._onsite_energies)

        # 2. On-site Couplings (Off-Diagonal, e.g., SOC)
        sites = np.arange(self._N_sites)
        for term in self._onsite_couplings:
            oi, oj = term["orb_i"], term["orb_j"]
            val = term["t"]

            # Map to global indices: site_k * n_orb + orb_idx
            g_rows = sites * self.n_orbitals + oi
            g_cols = sites * self.n_orbitals + oj

            # Handle scalar vs array values
            if np.isscalar(val):
                values = np.full(self._N_sites, val, dtype=complex)
            else:
                values = val.astype(complex)

            # Add H_ij and H_ji (Hermitian Conjugate)
            row_list.extend([g_rows, g_cols])
            col_list.extend([g_cols, g_rows])
            data_list.extend([values, values.conj()])

        # 3. Inter-site Hoppings
        for term in self._hopping_terms:
            d, tol = term["d"], term["tol"]
            oi, oj = term["orb_i"], term["orb_j"]
            val = term["t"]

            # Retrieve geometry
            s_rows, s_cols = self._get_pairs(d, tol)
            if len(s_rows) == 0:
                continue

            # Global Indices
            g_rows = s_rows * self.n_orbitals + oi
            g_cols = s_cols * self.n_orbitals + oj

            values = np.full(len(g_rows), val, dtype=complex)

            # Add Forward and Backward
            row_list.extend([g_rows, g_cols])
            col_list.extend([g_cols, g_rows])
            data_list.extend([values, values.conj()])

        # 4. Final Construction
        # Concatenate is very fast in NumPy
        all_rows = np.concatenate(row_list)
        all_cols = np.concatenate(col_list)
        all_data = np.concatenate(data_list)

        # COO sums duplicates automatically
        H = coo_matrix((all_data, (all_rows, all_cols)), shape=(self._dim, self._dim))
        return H.tocsr()

    def solve_full(self):
        """
        Calculate ALL eigenenergies and eigenstates. Inadvisable for large systems

        :param sort_by: 'energy' (ascending) or None.
        :return: (eigenvalues, eigenvectors)

                - eigenvalues: (N,) array
                - eigenvectors: (N, N) matrix where column v[:, i] is the ith eigenvector
        :rtype: tuple
        """
        H_mat = self.build().toarray()  # Convert sparse to dense
        # eigh is optimized for Hermitian matrices
        evals, evecs = np.linalg.eigh(H_mat)
        return evals, evecs

    def solve_sparse(self, k: int = 10, sigma: float = None, which: str = "SA"):
        """
        Calculate a SUBSET of eigenenergies using a sparse iterative solver (ARPACK).
        Recommended for large systems.

        :param int k: Number of eigenvalues/vectors to compute.
        :param float sigma: (Shift-Invert mode) If provided, find eigenvalues near this energy value.
                            If None, it finds the extremes of the spectrum.
        :param str which: Which eigenvalues to find (if sigma is None).
                          'LM': Largest Magnitude (default)
                          'SA': Smallest Algebraic (lowest energy)
                          'LA': Largest Algebraic (highest energy)
        :return: (eigenvalues, eigenvectors)
        """
        H_sparse = self.build()

        # ARPACK solver
        # sigma != None triggers "shift-invert" mode which is efficient for finding specific energies
        evals, evecs = eigsh(H_sparse, k=k, sigma=sigma, which=which)

        # eigsh does not guarantee sorted order
        idx = np.argsort(evals)
        return evals[idx], evecs[:, idx]

    def simulate_static_schroedinger(
        self, psi0: np.ndarray, t_span: tuple, num_steps: int, hbar: float = 1.0
    ):
        """
        Simulate the time evolution of a state psi0 under the STATIC Hamiltonian H.
        Solves: i * hbar * d/dt |psi> = H |psi>

        :param psi0: Initial state vector (shape: N_dim,)
        :param t_span: Tuple (t_start, t_end)
        :param num_steps: Number of time steps to retrieve
        :param hbar: Planck constant.

                    - If energy is in eV and time in fs, hbar ~ 0.6582.
                    - If units are dimensionless (t=1), use hbar=1.0 <-- default
        :return: (t_eval, psi_t)
                 t_eval: Array of time points
                 psi_t: Array of states at each time point (shape: num_steps, N_dim)
        """
        H = self.build()
        factor = -1j / hbar
        t_start, t_stop = t_span

        # Estimate memory usage for output
        # Complex128 = 16 bytes. Size = num_steps * N_dim * 16
        mem_bytes = num_steps * self._dim * 16
        if mem_bytes > 1e9:  # Warning if > 1 GB
            warnings.warn(
                f"Output array will consume {mem_bytes/1e9:.2f} GB of RAM. "
                "Consider reducing num_steps or N_dim."
            )

        # Evaluation time
        t_eval = np.linspace(t_start, t_stop, num_steps)
        A = H * factor  # -i/hbar H

        psi_t = expm_multiply(A, psi0, start=t_start, stop=t_stop, num=num_steps)

        return t_eval, psi_t
