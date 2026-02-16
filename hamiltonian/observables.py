import numpy as np
from lattice import Lattice


# class Vorticity:
#    r"""Measure current vortices around the smallest possible plaquettes
#    :math:`\hat{\omega}_{P_{i}}=\sum_{(\mathbf{R},\mathbf{R}')\in\partial P_{i}}\hat{J}_{\mathbf{R}\rightarrow\mathbf{R}'}`
#    """

#    def __init__(
#        self,
#        lattice: Lattice,
#        bond_lengths: list | None = None,
#    ):
#        """Find all plaquettes, whose vertices have given bond_lengths, from the given lattice.
#        ATTENTION: Periodicity of the lattice is already stored in the object Lattice"""
