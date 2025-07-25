"""
Topological molecular descriptors for ChemiAccel.

This module implements comprehensive topological descriptors including:
- Wiener, Zagreb, and Randic connectivity indices
- Topological charge and electrotopological state indices
- Kappa shape indices and molecular path counts
- Autocorrelation descriptors for spatial property correlations
- Graph-based CUDA kernels for topological calculations
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

try:
    from rdkit import Chem
    from rdkit.Chem import (
        AllChem,
        Crippen,
        Descriptors,
        Lipinski,
        rdMolDescriptors,
        rdPartialCharges,
    )

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from ..core.exceptions import ChemiAccelError
from ..core.molecular import MolecularData
from ..utils.logging import get_logger
from .graph import MolecularGraphBuilder

logger = get_logger()


class TopologicalDescriptorCalculator:
    """Calculate comprehensive topological molecular descriptors."""

    def __init__(self, use_gpu: bool = True):
        """Initialize topological descriptor calculator.

        Args:
            use_gpu: Whether to use GPU acceleration when available
        """
        if not RDKIT_AVAILABLE:
            raise ChemiAccelError(
                "RDKit is required for topological descriptor calculations"
            )

        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.graph_builder = MolecularGraphBuilder()

        # Initialize descriptor calculation functions
        self.descriptor_functions = {
            # Connectivity indices
            "wiener_index": self._calculate_wiener_index,
            "modified_wiener_index": self._calculate_modified_wiener_index,
            "zagreb_index_1": self._calculate_zagreb_index_1,
            "zagreb_index_2": self._calculate_zagreb_index_2,
            "modified_zagreb_index": self._calculate_modified_zagreb_index,
            "randic_index": self._calculate_randic_index,
            "modified_randic_index": self._calculate_modified_randic_index,
            "balaban_j": self._calculate_balaban_j,
            "harary_index": self._calculate_harary_index,
            # Topological charge indices
            "topological_charge_1": self._calculate_topological_charge_1,
            "topological_charge_2": self._calculate_topological_charge_2,
            "topological_charge_3": self._calculate_topological_charge_3,
            "topological_charge_4": self._calculate_topological_charge_4,
            "topological_charge_5": self._calculate_topological_charge_5,
            "topological_charge_6": self._calculate_topological_charge_6,
            "topological_charge_7": self._calculate_topological_charge_7,
            "topological_charge_8": self._calculate_topological_charge_8,
            "topological_charge_9": self._calculate_topological_charge_9,
            "topological_charge_10": self._calculate_topological_charge_10,
            # Electrotopological state indices
            "estate_vsa_1": self._calculate_estate_vsa_1,
            "estate_vsa_2": self._calculate_estate_vsa_2,
            "estate_vsa_3": self._calculate_estate_vsa_3,
            "estate_vsa_4": self._calculate_estate_vsa_4,
            "estate_vsa_5": self._calculate_estate_vsa_5,
            "estate_vsa_6": self._calculate_estate_vsa_6,
            "estate_vsa_7": self._calculate_estate_vsa_7,
            "estate_vsa_8": self._calculate_estate_vsa_8,
            "estate_vsa_9": self._calculate_estate_vsa_9,
            "estate_vsa_10": self._calculate_estate_vsa_10,
            "estate_vsa_11": self._calculate_estate_vsa_11,
            # Kappa shape indices
            "kappa_1": self._calculate_kappa_1,
            "kappa_2": self._calculate_kappa_2,
            "kappa_3": self._calculate_kappa_3,
            "kappa_alpha_1": self._calculate_kappa_alpha_1,
            "kappa_alpha_2": self._calculate_kappa_alpha_2,
            "kappa_alpha_3": self._calculate_kappa_alpha_3,
            "flexibility_index": self._calculate_flexibility_index,
            # Path counts
            "path_count_1": self._calculate_path_count_1,
            "path_count_2": self._calculate_path_count_2,
            "path_count_3": self._calculate_path_count_3,
            "path_count_4": self._calculate_path_count_4,
            "path_count_5": self._calculate_path_count_5,
            "path_count_6": self._calculate_path_count_6,
            "cluster_count_1": self._calculate_cluster_count_1,
            "cluster_count_2": self._calculate_cluster_count_2,
            "cluster_count_3": self._calculate_cluster_count_3,
            "cluster_count_4": self._calculate_cluster_count_4,
            "cluster_count_5": self._calculate_cluster_count_5,
            "cluster_count_6": self._calculate_cluster_count_6,
            # Autocorrelation descriptors
            "moran_autocorr_mass": self._calculate_moran_autocorr_mass,
            "moran_autocorr_volume": self._calculate_moran_autocorr_volume,
            "moran_autocorr_electronegativity": self._calculate_moran_autocorr_electronegativity,
            "moran_autocorr_polarizability": self._calculate_moran_autocorr_polarizability,
            "geary_autocorr_mass": self._calculate_geary_autocorr_mass,
            "geary_autocorr_volume": self._calculate_geary_autocorr_volume,
            "geary_autocorr_electronegativity": self._calculate_geary_autocorr_electronegativity,
            "geary_autocorr_polarizability": self._calculate_geary_autocorr_polarizability,
            # Additional topological indices
            "schultz_index": self._calculate_schultz_index,
            "gutman_index": self._calculate_gutman_index,
            "xu_index": self._calculate_xu_index,
            "eccentric_connectivity_index": self._calculate_eccentric_connectivity_index,
            "total_adjacency_index": self._calculate_total_adjacency_index,
            "symmetric_division_index": self._calculate_symmetric_division_index,
            "harmonic_topological_index": self._calculate_harmonic_topological_index,
            "geometric_topological_index": self._calculate_geometric_topological_index,
            "arithmetic_topological_index": self._calculate_arithmetic_topological_index,
        }

        # Atomic properties for autocorrelation descriptors
        self.atomic_properties = {
            "mass": self._get_atomic_masses(),
            "volume": self._get_atomic_volumes(),
            "electronegativity": self._get_electronegativities(),
            "polarizability": self._get_polarizabilities(),
        }

        logger.info(
            f"TopologicalDescriptorCalculator initialized with {len(self.descriptor_functions)} descriptors (GPU: {self.use_gpu})"
        )

    def _get_atomic_masses(self) -> Dict[int, float]:
        """Get atomic masses for common elements."""
        return {
            1: 1.008,
            6: 12.011,
            7: 14.007,
            8: 15.999,
            9: 18.998,
            15: 30.974,
            16: 32.065,
            17: 35.453,
            35: 79.904,
            53: 126.904,
        }

    def _get_atomic_volumes(self) -> Dict[int, float]:
        """Get atomic volumes for common elements."""
        return {
            1: 7.24,
            6: 20.58,
            7: 15.60,
            8: 14.71,
            9: 13.31,
            15: 24.43,
            16: 24.43,
            17: 22.45,
            35: 26.52,
            53: 32.52,
        }

    def _get_electronegativities(self) -> Dict[int, float]:
        """Get Pauling electronegativities for common elements."""
        return {
            1: 2.20,
            6: 2.55,
            7: 3.04,
            8: 3.44,
            9: 3.98,
            15: 2.19,
            16: 2.58,
            17: 3.16,
            35: 2.96,
            53: 2.66,
        }

    def _get_polarizabilities(self) -> Dict[int, float]:
        """Get atomic polarizabilities for common elements."""
        return {
            1: 0.667,
            6: 1.76,
            7: 1.10,
            8: 0.802,
            9: 0.557,
            15: 3.63,
            16: 2.90,
            17: 2.18,
            35: 3.05,
            53: 5.35,
        }

    def _get_distance_matrix(self, adj_matrix: np.ndarray) -> np.ndarray:
        """Calculate shortest path distance matrix."""
        if self.use_gpu:
            return self._get_distance_matrix_gpu(adj_matrix)
        else:
            return self._get_distance_matrix_cpu(adj_matrix)

    def _get_distance_matrix_cpu(self, adj_matrix: np.ndarray) -> np.ndarray:
        """Calculate distance matrix using CPU."""
        # Convert to sparse matrix for efficiency
        sparse_adj = csr_matrix(adj_matrix)
        dist_matrix = shortest_path(sparse_adj, directed=False, unweighted=True)

        # Replace infinite distances with a large number
        dist_matrix[np.isinf(dist_matrix)] = 999

        return dist_matrix

    def _get_distance_matrix_gpu(self, adj_matrix: np.ndarray) -> np.ndarray:
        """Calculate distance matrix using GPU (CuPy)."""
        try:
            # Transfer to GPU
            adj_gpu = cp.asarray(adj_matrix)
            n = adj_gpu.shape[0]

            # Initialize distance matrix
            dist_gpu = cp.full((n, n), cp.inf, dtype=cp.float32)

            # Set distances for adjacent atoms
            dist_gpu[adj_gpu == 1] = 1.0
            cp.fill_diagonal(dist_gpu, 0.0)

            # Floyd-Warshall algorithm on GPU
            for k in range(n):
                dist_gpu = cp.minimum(
                    dist_gpu, dist_gpu[:, k : k + 1] + dist_gpu[k : k + 1, :]
                )

            # Transfer back to CPU
            dist_matrix = cp.asnumpy(dist_gpu)
            dist_matrix[np.isinf(dist_matrix)] = 999

            return dist_matrix

        except Exception as e:
            logger.warning(
                f"GPU distance matrix calculation failed: {e}, falling back to CPU"
            )
            return self._get_distance_matrix_cpu(adj_matrix)

    # Connectivity indices
    def _calculate_wiener_index(self, graph_data: Dict[str, Any]) -> float:
        """Calculate Wiener index (sum of shortest path distances)."""
        adj_matrix = graph_data["adjacency_matrix"]
        dist_matrix = self._get_distance_matrix(adj_matrix)

        # Sum upper triangle to avoid double counting
        n = dist_matrix.shape[0]
        wiener_index = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if dist_matrix[i, j] < 999:
                    wiener_index += dist_matrix[i, j]

        return wiener_index

    def _calculate_modified_wiener_index(self, graph_data: Dict[str, Any]) -> float:
        """Calculate modified Wiener index."""
        adj_matrix = graph_data["adjacency_matrix"]
        dist_matrix = self._get_distance_matrix(adj_matrix)
        degrees = np.sum(adj_matrix, axis=1)

        modified_wiener = 0.0
        n = dist_matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if dist_matrix[i, j] < 999:
                    modified_wiener += dist_matrix[i, j] * degrees[i] * degrees[j]

        return modified_wiener

    def _calculate_zagreb_index_1(self, graph_data: Dict[str, Any]) -> float:
        """Calculate first Zagreb index (sum of squared degrees)."""
        adj_matrix = graph_data["adjacency_matrix"]
        degrees = np.sum(adj_matrix, axis=1)
        return float(np.sum(degrees**2))

    def _calculate_zagreb_index_2(self, graph_data: Dict[str, Any]) -> float:
        """Calculate second Zagreb index (sum of products of adjacent vertex degrees)."""
        adj_matrix = graph_data["adjacency_matrix"]
        degrees = np.sum(adj_matrix, axis=1)

        zagreb_2 = 0.0
        n = adj_matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j] == 1:
                    zagreb_2 += degrees[i] * degrees[j]

        return zagreb_2

    def _calculate_modified_zagreb_index(self, graph_data: Dict[str, Any]) -> float:
        """Calculate modified Zagreb index."""
        adj_matrix = graph_data["adjacency_matrix"]
        degrees = np.sum(adj_matrix, axis=1)

        modified_zagreb = 0.0
        n = adj_matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j] == 1:
                    if degrees[i] > 0 and degrees[j] > 0:
                        modified_zagreb += 1.0 / (degrees[i] * degrees[j])

        return modified_zagreb

    def _calculate_randic_index(self, graph_data: Dict[str, Any]) -> float:
        """Calculate Randic connectivity index."""
        adj_matrix = graph_data["adjacency_matrix"]
        degrees = np.sum(adj_matrix, axis=1)

        randic_index = 0.0
        n = adj_matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j] == 1:
                    if degrees[i] > 0 and degrees[j] > 0:
                        randic_index += 1.0 / np.sqrt(degrees[i] * degrees[j])

        return randic_index

    def _calculate_modified_randic_index(self, graph_data: Dict[str, Any]) -> float:
        """Calculate modified Randic index."""
        adj_matrix = graph_data["adjacency_matrix"]
        degrees = np.sum(adj_matrix, axis=1)

        modified_randic = 0.0
        n = adj_matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j] == 1:
                    if degrees[i] > 1 and degrees[j] > 1:
                        modified_randic += 1.0 / np.sqrt(
                            (degrees[i] - 1) * (degrees[j] - 1)
                        )

        return modified_randic

    def _calculate_balaban_j(self, graph_data: Dict[str, Any]) -> float:
        """Calculate Balaban J index."""
        num_atoms = graph_data["num_atoms"]
        num_bonds = graph_data["num_bonds"]

        if num_bonds == 0 or num_atoms <= 2:
            return 0.0

        adj_matrix = graph_data["adjacency_matrix"]
        dist_matrix = self._get_distance_matrix(adj_matrix)
        degrees = np.sum(adj_matrix, axis=1)

        # Calculate cyclomatic number
        cyclomatic_number = max(0, num_bonds - num_atoms + 1)

        if cyclomatic_number == 0:
            return 0.0

        # Calculate sum of reciprocals of geometric means of distance sums
        sum_reciprocals = 0.0
        n = adj_matrix.shape[0]

        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j] == 1:
                    # Distance sums for vertices i and j
                    dist_sum_i = np.sum(dist_matrix[i, :])
                    dist_sum_j = np.sum(dist_matrix[j, :])

                    if dist_sum_i > 0 and dist_sum_j > 0:
                        sum_reciprocals += 1.0 / np.sqrt(dist_sum_i * dist_sum_j)

        balaban_j = (num_bonds / (cyclomatic_number + 1)) * sum_reciprocals
        return balaban_j

    def _calculate_harary_index(self, graph_data: Dict[str, Any]) -> float:
        """Calculate Harary index (sum of reciprocals of distances)."""
        adj_matrix = graph_data["adjacency_matrix"]
        dist_matrix = self._get_distance_matrix(adj_matrix)

        harary_index = 0.0
        n = dist_matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if dist_matrix[i, j] > 0 and dist_matrix[i, j] < 999:
                    harary_index += 1.0 / dist_matrix[i, j]

        return harary_index

    # Topological charge indices
    def _calculate_topological_charge_1(self, graph_data: Dict[str, Any]) -> float:
        """Calculate topological charge index 1."""
        adj_matrix = graph_data["adjacency_matrix"]
        degrees = np.sum(adj_matrix, axis=1)

        tc1 = 0.0
        n = adj_matrix.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j and adj_matrix[i, j] == 1:
                    tc1 += (degrees[i] - degrees[j]) ** 2

        return tc1 / 2.0  # Divide by 2 to avoid double counting

    def _calculate_topological_charge_2(self, graph_data: Dict[str, Any]) -> float:
        """Calculate topological charge index 2."""
        adj_matrix = graph_data["adjacency_matrix"]
        degrees = np.sum(adj_matrix, axis=1)

        tc2 = 0.0
        n = adj_matrix.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Calculate distance-2 connectivity
                    dist_2_connected = False
                    for k in range(n):
                        if (
                            adj_matrix[i, k] == 1
                            and adj_matrix[k, j] == 1
                            and i != k
                            and k != j
                        ):
                            dist_2_connected = True
                            break

                    if dist_2_connected:
                        tc2 += (degrees[i] - degrees[j]) ** 2

        return tc2 / 2.0

    def _calculate_topological_charge_3(self, graph_data: Dict[str, Any]) -> float:
        """Calculate topological charge index 3."""
        adj_matrix = graph_data["adjacency_matrix"]
        degrees = np.sum(adj_matrix, axis=1)
        dist_matrix = self._get_distance_matrix(adj_matrix)

        tc3 = 0.0
        n = adj_matrix.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j and dist_matrix[i, j] == 3:
                    tc3 += (degrees[i] - degrees[j]) ** 2

        return tc3 / 2.0

    def _calculate_topological_charge_4(self, graph_data: Dict[str, Any]) -> float:
        """Calculate topological charge index 4."""
        adj_matrix = graph_data["adjacency_matrix"]
        degrees = np.sum(adj_matrix, axis=1)
        dist_matrix = self._get_distance_matrix(adj_matrix)

        tc4 = 0.0
        n = adj_matrix.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j and dist_matrix[i, j] == 4:
                    tc4 += (degrees[i] - degrees[j]) ** 2

        return tc4 / 2.0

    def _calculate_topological_charge_5(self, graph_data: Dict[str, Any]) -> float:
        """Calculate topological charge index 5."""
        adj_matrix = graph_data["adjacency_matrix"]
        degrees = np.sum(adj_matrix, axis=1)
        dist_matrix = self._get_distance_matrix(adj_matrix)

        tc5 = 0.0
        n = adj_matrix.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j and dist_matrix[i, j] == 5:
                    tc5 += (degrees[i] - degrees[j]) ** 2

        return tc5 / 2.0

    def _calculate_topological_charge_6(self, graph_data: Dict[str, Any]) -> float:
        """Calculate topological charge index 6."""
        adj_matrix = graph_data["adjacency_matrix"]
        degrees = np.sum(adj_matrix, axis=1)
        dist_matrix = self._get_distance_matrix(adj_matrix)

        tc6 = 0.0
        n = adj_matrix.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j and dist_matrix[i, j] == 6:
                    tc6 += (degrees[i] - degrees[j]) ** 2

        return tc6 / 2.0

    def _calculate_topological_charge_7(self, graph_data: Dict[str, Any]) -> float:
        """Calculate topological charge index 7."""
        adj_matrix = graph_data["adjacency_matrix"]
        degrees = np.sum(adj_matrix, axis=1)
        dist_matrix = self._get_distance_matrix(adj_matrix)

        tc7 = 0.0
        n = adj_matrix.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j and dist_matrix[i, j] == 7:
                    tc7 += (degrees[i] - degrees[j]) ** 2

        return tc7 / 2.0

    def _calculate_topological_charge_8(self, graph_data: Dict[str, Any]) -> float:
        """Calculate topological charge index 8."""
        adj_matrix = graph_data["adjacency_matrix"]
        degrees = np.sum(adj_matrix, axis=1)
        dist_matrix = self._get_distance_matrix(adj_matrix)

        tc8 = 0.0
        n = adj_matrix.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j and dist_matrix[i, j] == 8:
                    tc8 += (degrees[i] - degrees[j]) ** 2

        return tc8 / 2.0

    def _calculate_topological_charge_9(self, graph_data: Dict[str, Any]) -> float:
        """Calculate topological charge index 9."""
        adj_matrix = graph_data["adjacency_matrix"]
        degrees = np.sum(adj_matrix, axis=1)
        dist_matrix = self._get_distance_matrix(adj_matrix)

        tc9 = 0.0
        n = adj_matrix.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j and dist_matrix[i, j] == 9:
                    tc9 += (degrees[i] - degrees[j]) ** 2

        return tc9 / 2.0

    def _calculate_topological_charge_10(self, graph_data: Dict[str, Any]) -> float:
        """Calculate topological charge index 10."""
        adj_matrix = graph_data["adjacency_matrix"]
        degrees = np.sum(adj_matrix, axis=1)
        dist_matrix = self._get_distance_matrix(adj_matrix)

        tc10 = 0.0
        n = adj_matrix.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j and dist_matrix[i, j] == 10:
                    tc10 += (degrees[i] - degrees[j]) ** 2

        return tc10 / 2.0

    # Electrotopological state indices
    def _get_estate_values(self, mol_data: MolecularData) -> np.ndarray:
        """Calculate E-state values for all atoms."""
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            raise ChemiAccelError(
                f"No RDKit mol object found for molecule {mol_data.mol_id}"
            )

        try:
            # Try to calculate E-state values using RDKit
            if hasattr(rdMolDescriptors, "EState"):
                estate_values = rdMolDescriptors.EState.EStateIndices(rdkit_mol)
                return np.array(estate_values)
            else:
                # Fallback: use a simplified E-state calculation
                return self._calculate_simple_estate_values(rdkit_mol)
        except Exception as e:
            logger.debug(f"E-state calculation failed, using fallback: {e}")
            return self._calculate_simple_estate_values(rdkit_mol)

    def _calculate_simple_estate_values(self, mol: Chem.Mol) -> np.ndarray:
        """Calculate simplified E-state values as fallback."""
        num_atoms = mol.GetNumAtoms()
        estate_values = np.zeros(num_atoms)

        for i, atom in enumerate(mol.GetAtoms()):
            # Simplified E-state calculation based on atom properties
            atomic_num = atom.GetAtomicNum()
            degree = atom.GetDegree()
            formal_charge = atom.GetFormalCharge()

            # Basic E-state approximation
            estate_value = atomic_num / 10.0 + degree * 0.5 + formal_charge
            estate_values[i] = estate_value

        return estate_values

    def _calculate_estate_vsa_1(
        self, graph_data: Dict[str, Any], mol_data: MolecularData
    ) -> float:
        """Calculate E-state VSA descriptor 1 (E-state < -0.390)."""
        estate_values = self._get_estate_values(mol_data)
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")

        vsa_sum = 0.0
        for i, estate in enumerate(estate_values):
            if estate < -0.390:
                try:
                    atom = rdkit_mol.GetAtomWithIdx(i)
                    vsa_sum += Descriptors.LabuteASA([atom])
                except (AttributeError, IndexError, RuntimeError) as e:
                    vsa_sum += 0.0

        return vsa_sum

    def _calculate_estate_vsa_2(
        self, graph_data: Dict[str, Any], mol_data: MolecularData
    ) -> float:
        """Calculate E-state VSA descriptor 2 (-0.390 <= E-state < 0.290)."""
        estate_values = self._get_estate_values(mol_data)
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")

        vsa_sum = 0.0
        for i, estate in enumerate(estate_values):
            if -0.390 <= estate < 0.290:
                try:
                    atom = rdkit_mol.GetAtomWithIdx(i)
                    vsa_sum += Descriptors.LabuteASA([atom])
                except (AttributeError, IndexError, RuntimeError) as e:
                    vsa_sum += 0.0

        return vsa_sum

    def _calculate_estate_vsa_3(
        self, graph_data: Dict[str, Any], mol_data: MolecularData
    ) -> float:
        """Calculate E-state VSA descriptor 3 (0.290 <= E-state < 0.717)."""
        estate_values = self._get_estate_values(mol_data)
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")

        vsa_sum = 0.0
        for i, estate in enumerate(estate_values):
            if 0.290 <= estate < 0.717:
                try:
                    atom = rdkit_mol.GetAtomWithIdx(i)
                    vsa_sum += Descriptors.LabuteASA([atom])
                except (AttributeError, IndexError, RuntimeError) as e:
                    vsa_sum += 0.0

        return vsa_sum

    def _calculate_estate_vsa_4(
        self, graph_data: Dict[str, Any], mol_data: MolecularData
    ) -> float:
        """Calculate E-state VSA descriptor 4 (0.717 <= E-state < 1.165)."""
        estate_values = self._get_estate_values(mol_data)
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")

        vsa_sum = 0.0
        for i, estate in enumerate(estate_values):
            if 0.717 <= estate < 1.165:
                try:
                    atom = rdkit_mol.GetAtomWithIdx(i)
                    vsa_sum += Descriptors.LabuteASA([atom])
                except (AttributeError, IndexError, RuntimeError) as e:
                    vsa_sum += 0.0

        return vsa_sum

    def _calculate_estate_vsa_5(
        self, graph_data: Dict[str, Any], mol_data: MolecularData
    ) -> float:
        """Calculate E-state VSA descriptor 5 (1.165 <= E-state < 1.540)."""
        estate_values = self._get_estate_values(mol_data)
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")

        vsa_sum = 0.0
        for i, estate in enumerate(estate_values):
            if 1.165 <= estate < 1.540:
                try:
                    atom = rdkit_mol.GetAtomWithIdx(i)
                    vsa_sum += Descriptors.LabuteASA([atom])
                except (AttributeError, IndexError, RuntimeError) as e:
                    vsa_sum += 0.0

        return vsa_sum

    def _calculate_estate_vsa_6(
        self, graph_data: Dict[str, Any], mol_data: MolecularData
    ) -> float:
        """Calculate E-state VSA descriptor 6 (1.540 <= E-state < 1.807)."""
        estate_values = self._get_estate_values(mol_data)
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")

        vsa_sum = 0.0
        for i, estate in enumerate(estate_values):
            if 1.540 <= estate < 1.807:
                try:
                    atom = rdkit_mol.GetAtomWithIdx(i)
                    vsa_sum += Descriptors.LabuteASA([atom])
                except (AttributeError, IndexError, RuntimeError) as e:
                    vsa_sum += 0.0

        return vsa_sum

    def _calculate_estate_vsa_7(
        self, graph_data: Dict[str, Any], mol_data: MolecularData
    ) -> float:
        """Calculate E-state VSA descriptor 7 (1.807 <= E-state < 2.050)."""
        estate_values = self._get_estate_values(mol_data)
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")

        vsa_sum = 0.0
        for i, estate in enumerate(estate_values):
            if 1.807 <= estate < 2.050:
                try:
                    atom = rdkit_mol.GetAtomWithIdx(i)
                    vsa_sum += Descriptors.LabuteASA([atom])
                except (AttributeError, IndexError, RuntimeError) as e:
                    vsa_sum += 0.0

        return vsa_sum

    def _calculate_estate_vsa_8(
        self, graph_data: Dict[str, Any], mol_data: MolecularData
    ) -> float:
        """Calculate E-state VSA descriptor 8 (2.050 <= E-state < 2.618)."""
        estate_values = self._get_estate_values(mol_data)
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")

        vsa_sum = 0.0
        for i, estate in enumerate(estate_values):
            if 2.050 <= estate < 2.618:
                try:
                    atom = rdkit_mol.GetAtomWithIdx(i)
                    vsa_sum += Descriptors.LabuteASA([atom])
                except (AttributeError, IndexError, RuntimeError) as e:
                    vsa_sum += 0.0

        return vsa_sum

    def _calculate_estate_vsa_9(
        self, graph_data: Dict[str, Any], mol_data: MolecularData
    ) -> float:
        """Calculate E-state VSA descriptor 9 (2.618 <= E-state < 3.177)."""
        estate_values = self._get_estate_values(mol_data)
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")

        vsa_sum = 0.0
        for i, estate in enumerate(estate_values):
            if 2.618 <= estate < 3.177:
                try:
                    atom = rdkit_mol.GetAtomWithIdx(i)
                    vsa_sum += Descriptors.LabuteASA([atom])
                except (AttributeError, IndexError, RuntimeError) as e:
                    vsa_sum += 0.0

        return vsa_sum

    def _calculate_estate_vsa_10(
        self, graph_data: Dict[str, Any], mol_data: MolecularData
    ) -> float:
        """Calculate E-state VSA descriptor 10 (3.177 <= E-state < 4.000)."""
        estate_values = self._get_estate_values(mol_data)
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")

        vsa_sum = 0.0
        for i, estate in enumerate(estate_values):
            if 3.177 <= estate < 4.000:
                try:
                    atom = rdkit_mol.GetAtomWithIdx(i)
                    vsa_sum += Descriptors.LabuteASA([atom])
                except (AttributeError, IndexError, RuntimeError) as e:
                    vsa_sum += 0.0

        return vsa_sum

    def _calculate_estate_vsa_11(
        self, graph_data: Dict[str, Any], mol_data: MolecularData
    ) -> float:
        """Calculate E-state VSA descriptor 11 (E-state >= 4.000)."""
        estate_values = self._get_estate_values(mol_data)
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")

        vsa_sum = 0.0
        for i, estate in enumerate(estate_values):
            if estate >= 4.000:
                try:
                    atom = rdkit_mol.GetAtomWithIdx(i)
                    vsa_sum += Descriptors.LabuteASA([atom])
                except (AttributeError, IndexError, RuntimeError) as e:
                    vsa_sum += 0.0

        return vsa_sum

    # Kappa shape indices
    def _calculate_kappa_1(self, graph_data: Dict[str, Any]) -> float:
        """Calculate Kappa 1 shape index."""
        num_atoms = graph_data["num_atoms"]
        if num_atoms <= 2:
            return 0.0

        adj_matrix = graph_data["adjacency_matrix"]

        # Count paths of length 1 (bonds)
        path_count_1 = np.sum(adj_matrix) // 2

        if path_count_1 == 0:
            return 0.0

        # Kappa 1 = (n-1)(n-2)^2 / P1^2
        # where n is number of atoms and P1 is number of paths of length 1
        kappa_1 = ((num_atoms - 1) * (num_atoms - 2) ** 2) / (path_count_1**2)

        return kappa_1

    def _calculate_kappa_2(self, graph_data: Dict[str, Any]) -> float:
        """Calculate Kappa 2 shape index."""
        num_atoms = graph_data["num_atoms"]
        if num_atoms <= 3:
            return 0.0

        adj_matrix = graph_data["adjacency_matrix"]

        # Count paths of length 2
        adj_squared = np.dot(adj_matrix, adj_matrix)
        path_count_2 = (np.sum(adj_squared) - np.trace(adj_squared)) // 2

        if path_count_2 == 0:
            return 0.0

        # Kappa 2 = (n-1)(n-3)^2 / P2^2
        kappa_2 = ((num_atoms - 1) * (num_atoms - 3) ** 2) / (path_count_2**2)

        return kappa_2

    def _calculate_kappa_3(self, graph_data: Dict[str, Any]) -> float:
        """Calculate Kappa 3 shape index."""
        num_atoms = graph_data["num_atoms"]
        if num_atoms <= 4:
            return 0.0

        adj_matrix = graph_data["adjacency_matrix"]

        # Count paths of length 3
        adj_cubed = np.dot(np.dot(adj_matrix, adj_matrix), adj_matrix)
        path_count_3 = 0

        # Count 3-paths (avoiding cycles)
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    path_count_3 += adj_cubed[i, j]

        path_count_3 //= 2  # Avoid double counting

        if path_count_3 == 0:
            return 0.0

        # Kappa 3 calculation for odd/even number of atoms
        if num_atoms % 2 == 1:
            kappa_3 = ((num_atoms - 1) * (num_atoms - 4) ** 2) / (path_count_3**2)
        else:
            kappa_3 = ((num_atoms - 3) * (num_atoms - 4) ** 2) / (path_count_3**2)

        return kappa_3

    def _calculate_kappa_alpha_1(
        self, graph_data: Dict[str, Any], mol_data: MolecularData
    ) -> float:
        """Calculate Kappa alpha 1 (considering heteroatoms)."""
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            return self._calculate_kappa_1(graph_data)

        # Count non-carbon atoms
        alpha_count = 0
        for atom in rdkit_mol.GetAtoms():
            if atom.GetAtomicNum() != 6:  # Not carbon
                alpha_count += 1

        kappa_1 = self._calculate_kappa_1(graph_data)

        # Adjust for heteroatoms
        if alpha_count > 0:
            kappa_alpha_1 = kappa_1 * (alpha_count + 1) / graph_data["num_atoms"]
        else:
            kappa_alpha_1 = kappa_1

        return kappa_alpha_1

    def _calculate_kappa_alpha_2(
        self, graph_data: Dict[str, Any], mol_data: MolecularData
    ) -> float:
        """Calculate Kappa alpha 2 (considering heteroatoms)."""
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            return self._calculate_kappa_2(graph_data)

        # Count non-carbon atoms
        alpha_count = 0
        for atom in rdkit_mol.GetAtoms():
            if atom.GetAtomicNum() != 6:  # Not carbon
                alpha_count += 1

        kappa_2 = self._calculate_kappa_2(graph_data)

        # Adjust for heteroatoms
        if alpha_count > 0:
            kappa_alpha_2 = kappa_2 * (alpha_count + 1) / graph_data["num_atoms"]
        else:
            kappa_alpha_2 = kappa_2

        return kappa_alpha_2

    def _calculate_kappa_alpha_3(
        self, graph_data: Dict[str, Any], mol_data: MolecularData
    ) -> float:
        """Calculate Kappa alpha 3 (considering heteroatoms)."""
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            return self._calculate_kappa_3(graph_data)

        # Count non-carbon atoms
        alpha_count = 0
        for atom in rdkit_mol.GetAtoms():
            if atom.GetAtomicNum() != 6:  # Not carbon
                alpha_count += 1

        kappa_3 = self._calculate_kappa_3(graph_data)

        # Adjust for heteroatoms
        if alpha_count > 0:
            kappa_alpha_3 = kappa_3 * (alpha_count + 1) / graph_data["num_atoms"]
        else:
            kappa_alpha_3 = kappa_3

        return kappa_alpha_3

    def _calculate_flexibility_index(
        self, graph_data: Dict[str, Any], mol_data: MolecularData
    ) -> float:
        """Calculate molecular flexibility index."""
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            return 0.0

        # Count rotatable bonds
        rotatable_bonds = 0
        for bond in rdkit_mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.SINGLE:
                begin_atom = bond.GetBeginAtom()
                end_atom = bond.GetEndAtom()

                # Check if bond is rotatable (not in ring, not terminal)
                if (
                    not bond.IsInRing()
                    and begin_atom.GetDegree() > 1
                    and end_atom.GetDegree() > 1
                ):
                    rotatable_bonds += 1

        # Flexibility index = rotatable bonds / total bonds
        total_bonds = graph_data["num_bonds"]
        if total_bonds > 0:
            flexibility_index = rotatable_bonds / total_bonds
        else:
            flexibility_index = 0.0

        return flexibility_index

    # Path counts
    def _calculate_path_count_1(self, graph_data: Dict[str, Any]) -> float:
        """Calculate number of paths of length 1."""
        adj_matrix = graph_data["adjacency_matrix"]
        return float(np.sum(adj_matrix) // 2)

    def _calculate_path_count_2(self, graph_data: Dict[str, Any]) -> float:
        """Calculate number of paths of length 2."""
        adj_matrix = graph_data["adjacency_matrix"]
        adj_squared = np.dot(adj_matrix, adj_matrix)
        return float((np.sum(adj_squared) - np.trace(adj_squared)) // 2)

    def _calculate_path_count_3(self, graph_data: Dict[str, Any]) -> float:
        """Calculate number of paths of length 3."""
        adj_matrix = graph_data["adjacency_matrix"]
        adj_cubed = np.dot(np.dot(adj_matrix, adj_matrix), adj_matrix)

        path_count = 0
        n = adj_matrix.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j:
                    path_count += adj_cubed[i, j]

        return float(path_count // 2)

    def _calculate_path_count_4(self, graph_data: Dict[str, Any]) -> float:
        """Calculate number of paths of length 4."""
        adj_matrix = graph_data["adjacency_matrix"]
        adj_4 = np.linalg.matrix_power(adj_matrix, 4)

        path_count = 0
        n = adj_matrix.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j:
                    path_count += adj_4[i, j]

        return float(path_count // 2)

    def _calculate_path_count_5(self, graph_data: Dict[str, Any]) -> float:
        """Calculate number of paths of length 5."""
        adj_matrix = graph_data["adjacency_matrix"]
        adj_5 = np.linalg.matrix_power(adj_matrix, 5)

        path_count = 0
        n = adj_matrix.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j:
                    path_count += adj_5[i, j]

        return float(path_count // 2)

    def _calculate_path_count_6(self, graph_data: Dict[str, Any]) -> float:
        """Calculate number of paths of length 6."""
        adj_matrix = graph_data["adjacency_matrix"]
        adj_6 = np.linalg.matrix_power(adj_matrix, 6)

        path_count = 0
        n = adj_matrix.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j:
                    path_count += adj_6[i, j]

        return float(path_count // 2)

    # Cluster counts (subgraph counts)
    def _calculate_cluster_count_1(self, graph_data: Dict[str, Any]) -> float:
        """Calculate number of 1-clusters (single atoms)."""
        return float(graph_data["num_atoms"])

    def _calculate_cluster_count_2(self, graph_data: Dict[str, Any]) -> float:
        """Calculate number of 2-clusters (bonds)."""
        return float(graph_data["num_bonds"])

    def _calculate_cluster_count_3(self, graph_data: Dict[str, Any]) -> float:
        """Calculate number of 3-clusters (triangles)."""
        adj_matrix = graph_data["adjacency_matrix"]
        adj_cubed = np.dot(np.dot(adj_matrix, adj_matrix), adj_matrix)
        return float(np.trace(adj_cubed) // 6)  # Each triangle counted 6 times

    def _calculate_cluster_count_4(self, graph_data: Dict[str, Any]) -> float:
        """Calculate number of 4-clusters (4-cycles and 4-paths)."""
        adj_matrix = graph_data["adjacency_matrix"]
        adj_4 = np.linalg.matrix_power(adj_matrix, 4)

        # Count 4-cycles
        cycles_4 = np.trace(adj_4) // 8  # Each 4-cycle counted 8 times

        return float(cycles_4)

    def _calculate_cluster_count_5(self, graph_data: Dict[str, Any]) -> float:
        """Calculate number of 5-clusters."""
        adj_matrix = graph_data["adjacency_matrix"]
        adj_5 = np.linalg.matrix_power(adj_matrix, 5)

        # Count 5-cycles
        cycles_5 = np.trace(adj_5) // 10  # Each 5-cycle counted 10 times

        return float(cycles_5)

    def _calculate_cluster_count_6(self, graph_data: Dict[str, Any]) -> float:
        """Calculate number of 6-clusters."""
        adj_matrix = graph_data["adjacency_matrix"]
        adj_6 = np.linalg.matrix_power(adj_matrix, 6)

        # Count 6-cycles
        cycles_6 = np.trace(adj_6) // 12  # Each 6-cycle counted 12 times

        return float(cycles_6)

    # Autocorrelation descriptors
    def _calculate_moran_autocorr_mass(
        self, graph_data: Dict[str, Any], mol_data: MolecularData
    ) -> float:
        """Calculate Moran autocorrelation for atomic mass."""
        return self._calculate_moran_autocorr(graph_data, mol_data, "mass")

    def _calculate_moran_autocorr_volume(
        self, graph_data: Dict[str, Any], mol_data: MolecularData
    ) -> float:
        """Calculate Moran autocorrelation for atomic volume."""
        return self._calculate_moran_autocorr(graph_data, mol_data, "volume")

    def _calculate_moran_autocorr_electronegativity(
        self, graph_data: Dict[str, Any], mol_data: MolecularData
    ) -> float:
        """Calculate Moran autocorrelation for electronegativity."""
        return self._calculate_moran_autocorr(graph_data, mol_data, "electronegativity")

    def _calculate_moran_autocorr_polarizability(
        self, graph_data: Dict[str, Any], mol_data: MolecularData
    ) -> float:
        """Calculate Moran autocorrelation for polarizability."""
        return self._calculate_moran_autocorr(graph_data, mol_data, "polarizability")

    def _calculate_moran_autocorr(
        self, graph_data: Dict[str, Any], mol_data: MolecularData, property_name: str
    ) -> float:
        """Calculate Moran autocorrelation for a given atomic property."""
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            return 0.0

        adj_matrix = graph_data["adjacency_matrix"]
        n = adj_matrix.shape[0]

        if n <= 1:
            return 0.0

        # Get atomic properties
        properties = []
        property_dict = self.atomic_properties[property_name]

        for atom in rdkit_mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            prop_value = property_dict.get(atomic_num, 0.0)
            properties.append(prop_value)

        properties = np.array(properties)

        if np.std(properties) == 0:
            return 0.0

        # Calculate Moran's I
        mean_prop = np.mean(properties)

        numerator = 0.0
        denominator = 0.0
        weight_sum = 0.0

        for i in range(n):
            for j in range(n):
                if i != j and adj_matrix[i, j] == 1:
                    weight = 1.0  # Binary adjacency weight
                    numerator += (
                        weight
                        * (properties[i] - mean_prop)
                        * (properties[j] - mean_prop)
                    )
                    weight_sum += weight

        for i in range(n):
            denominator += (properties[i] - mean_prop) ** 2

        if weight_sum > 0 and denominator > 0:
            moran_i = (n / weight_sum) * (numerator / denominator)
        else:
            moran_i = 0.0

        return moran_i

    def _calculate_geary_autocorr_mass(
        self, graph_data: Dict[str, Any], mol_data: MolecularData
    ) -> float:
        """Calculate Geary autocorrelation for atomic mass."""
        return self._calculate_geary_autocorr(graph_data, mol_data, "mass")

    def _calculate_geary_autocorr_volume(
        self, graph_data: Dict[str, Any], mol_data: MolecularData
    ) -> float:
        """Calculate Geary autocorrelation for atomic volume."""
        return self._calculate_geary_autocorr(graph_data, mol_data, "volume")

    def _calculate_geary_autocorr_electronegativity(
        self, graph_data: Dict[str, Any], mol_data: MolecularData
    ) -> float:
        """Calculate Geary autocorrelation for electronegativity."""
        return self._calculate_geary_autocorr(graph_data, mol_data, "electronegativity")

    def _calculate_geary_autocorr_polarizability(
        self, graph_data: Dict[str, Any], mol_data: MolecularData
    ) -> float:
        """Calculate Geary autocorrelation for polarizability."""
        return self._calculate_geary_autocorr(graph_data, mol_data, "polarizability")

    def _calculate_geary_autocorr(
        self, graph_data: Dict[str, Any], mol_data: MolecularData, property_name: str
    ) -> float:
        """Calculate Geary autocorrelation for a given atomic property."""
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            return 0.0

        adj_matrix = graph_data["adjacency_matrix"]
        n = adj_matrix.shape[0]

        if n <= 1:
            return 0.0

        # Get atomic properties
        properties = []
        property_dict = self.atomic_properties[property_name]

        for atom in rdkit_mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            prop_value = property_dict.get(atomic_num, 0.0)
            properties.append(prop_value)

        properties = np.array(properties)

        if np.std(properties) == 0:
            return 0.0

        # Calculate Geary's C
        mean_prop = np.mean(properties)

        numerator = 0.0
        denominator = 0.0
        weight_sum = 0.0

        for i in range(n):
            for j in range(n):
                if i != j and adj_matrix[i, j] == 1:
                    weight = 1.0  # Binary adjacency weight
                    numerator += weight * (properties[i] - properties[j]) ** 2
                    weight_sum += weight

        for i in range(n):
            denominator += (properties[i] - mean_prop) ** 2

        if weight_sum > 0 and denominator > 0:
            geary_c = ((n - 1) / (2 * weight_sum)) * (numerator / denominator)
        else:
            geary_c = 0.0

        return geary_c

    # Additional topological indices
    def _calculate_schultz_index(self, graph_data: Dict[str, Any]) -> float:
        """Calculate Schultz molecular topological index."""
        adj_matrix = graph_data["adjacency_matrix"]
        dist_matrix = self._get_distance_matrix(adj_matrix)
        degrees = np.sum(adj_matrix, axis=1)

        schultz_index = 0.0
        n = adj_matrix.shape[0]

        for i in range(n):
            for j in range(n):
                if i != j and dist_matrix[i, j] < 999:
                    schultz_index += (degrees[i] + degrees[j]) * dist_matrix[i, j]

        return schultz_index / 2.0  # Avoid double counting

    def _calculate_gutman_index(self, graph_data: Dict[str, Any]) -> float:
        """Calculate Gutman topological index."""
        adj_matrix = graph_data["adjacency_matrix"]
        degrees = np.sum(adj_matrix, axis=1)

        gutman_index = 0.0
        n = adj_matrix.shape[0]

        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j] == 1:
                    gutman_index += degrees[i] * degrees[j]

        return gutman_index

    def _calculate_xu_index(self, graph_data: Dict[str, Any]) -> float:
        """Calculate Xu index."""
        adj_matrix = graph_data["adjacency_matrix"]
        degrees = np.sum(adj_matrix, axis=1)

        xu_index = 0.0
        n = adj_matrix.shape[0]

        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j] == 1:
                    if degrees[i] > 0 and degrees[j] > 0:
                        xu_index += np.sqrt(degrees[i] * degrees[j])

        return xu_index

    def _calculate_eccentric_connectivity_index(
        self, graph_data: Dict[str, Any]
    ) -> float:
        """Calculate eccentric connectivity index."""
        adj_matrix = graph_data["adjacency_matrix"]
        dist_matrix = self._get_distance_matrix(adj_matrix)
        degrees = np.sum(adj_matrix, axis=1)

        eccentric_index = 0.0
        n = adj_matrix.shape[0]

        for i in range(n):
            # Find eccentricity (maximum distance from vertex i)
            eccentricity = 0.0
            for j in range(n):
                if dist_matrix[i, j] < 999:
                    eccentricity = max(eccentricity, dist_matrix[i, j])

            eccentric_index += degrees[i] * eccentricity

        return eccentric_index

    def _calculate_total_adjacency_index(self, graph_data: Dict[str, Any]) -> float:
        """Calculate total adjacency index."""
        adj_matrix = graph_data["adjacency_matrix"]
        degrees = np.sum(adj_matrix, axis=1)

        # Sum of all vertex degrees
        return float(np.sum(degrees))

    def _calculate_symmetric_division_index(self, graph_data: Dict[str, Any]) -> float:
        """Calculate symmetric division index."""
        adj_matrix = graph_data["adjacency_matrix"]
        degrees = np.sum(adj_matrix, axis=1)

        sdd_index = 0.0
        n = adj_matrix.shape[0]

        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j] == 1:
                    if degrees[i] > 0 and degrees[j] > 0:
                        min_deg = min(degrees[i], degrees[j])
                        max_deg = max(degrees[i], degrees[j])
                        sdd_index += min_deg / max_deg + max_deg / min_deg

        return sdd_index

    def _calculate_harmonic_topological_index(
        self, graph_data: Dict[str, Any]
    ) -> float:
        """Calculate harmonic topological index."""
        adj_matrix = graph_data["adjacency_matrix"]
        degrees = np.sum(adj_matrix, axis=1)

        harmonic_index = 0.0
        n = adj_matrix.shape[0]

        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j] == 1:
                    if degrees[i] > 0 and degrees[j] > 0:
                        harmonic_index += 2.0 / (degrees[i] + degrees[j])

        return harmonic_index

    def _calculate_geometric_topological_index(
        self, graph_data: Dict[str, Any]
    ) -> float:
        """Calculate geometric topological index."""
        adj_matrix = graph_data["adjacency_matrix"]
        degrees = np.sum(adj_matrix, axis=1)

        geometric_index = 0.0
        n = adj_matrix.shape[0]

        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j] == 1:
                    if degrees[i] > 0 and degrees[j] > 0:
                        geometric_index += (
                            2.0
                            * np.sqrt(degrees[i] * degrees[j])
                            / (degrees[i] + degrees[j])
                        )

        return geometric_index

    def _calculate_arithmetic_topological_index(
        self, graph_data: Dict[str, Any]
    ) -> float:
        """Calculate arithmetic topological index."""
        adj_matrix = graph_data["adjacency_matrix"]
        degrees = np.sum(adj_matrix, axis=1)

        arithmetic_index = 0.0
        n = adj_matrix.shape[0]

        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j] == 1:
                    if degrees[i] > 0 and degrees[j] > 0:
                        arithmetic_index += (degrees[i] + degrees[j]) / 2.0

        return arithmetic_index

    def calculate_topological_descriptors(
        self, mol_data: MolecularData
    ) -> Dict[str, float]:
        """Calculate all topological descriptors for a molecule.

        Args:
            mol_data: MolecularData with RDKit mol object

        Returns:
            Dictionary of topological descriptor values
        """
        try:
            # Build molecular graph
            graph_data = self.graph_builder.build_molecular_graph(mol_data)

            # Calculate descriptors
            descriptors = {}
            for desc_name, desc_func in self.descriptor_functions.items():
                try:
                    # Check if descriptor function needs mol_data
                    import inspect

                    sig = inspect.signature(desc_func)
                    if "mol_data" in sig.parameters:
                        value = desc_func(graph_data, mol_data)
                    else:
                        value = desc_func(graph_data)

                    descriptors[desc_name] = float(value)
                except Exception as e:
                    logger.warning(
                        f"Failed to calculate {desc_name} for molecule {mol_data.mol_id}: {e}"
                    )
                    descriptors[desc_name] = np.nan

            return descriptors

        except Exception as e:
            logger.error(
                f"Failed to calculate topological descriptors for molecule {mol_data.mol_id}: {e}"
            )
            return dict.fromkeys(self.descriptor_functions.keys(), np.nan)

    def get_available_descriptors(self) -> List[str]:
        """Get list of available topological descriptor names."""
        return list(self.descriptor_functions.keys())

    def get_descriptor_info(self) -> Dict[str, Dict[str, str]]:
        """Get information about available descriptors."""
        descriptor_info = {
            # Connectivity indices
            "wiener_index": {
                "category": "connectivity",
                "description": "Sum of shortest path distances between all pairs of vertices",
            },
            "modified_wiener_index": {
                "category": "connectivity",
                "description": "Wiener index weighted by vertex degrees",
            },
            "zagreb_index_1": {
                "category": "connectivity",
                "description": "Sum of squared vertex degrees",
            },
            "zagreb_index_2": {
                "category": "connectivity",
                "description": "Sum of products of adjacent vertex degrees",
            },
            "randic_index": {
                "category": "connectivity",
                "description": "Sum of reciprocals of square roots of adjacent vertex degree products",
            },
            "balaban_j": {
                "category": "connectivity",
                "description": "Balaban J index based on distance sums",
            },
            "harary_index": {
                "category": "connectivity",
                "description": "Sum of reciprocals of distances between all vertex pairs",
            },
            # Topological charge indices
            "topological_charge_1": {
                "category": "topological_charge",
                "description": "Topological charge index for distance 1",
            },
            # Kappa shape indices
            "kappa_1": {
                "category": "shape",
                "description": "Kappa 1 shape index based on path counts",
            },
            "kappa_2": {
                "category": "shape",
                "description": "Kappa 2 shape index based on path counts",
            },
            "kappa_3": {
                "category": "shape",
                "description": "Kappa 3 shape index based on path counts",
            },
            # Path counts
            "path_count_1": {
                "category": "path",
                "description": "Number of paths of length 1",
            },
            "path_count_2": {
                "category": "path",
                "description": "Number of paths of length 2",
            },
            # Autocorrelation
            "moran_autocorr_mass": {
                "category": "autocorrelation",
                "description": "Moran autocorrelation for atomic mass",
            },
            "geary_autocorr_mass": {
                "category": "autocorrelation",
                "description": "Geary autocorrelation for atomic mass",
            },
        }

        return descriptor_info


# CUDA kernel implementations (placeholder for future GPU acceleration)
class TopologicalCUDAKernels:
    """CUDA kernels for topological descriptor calculations."""

    def __init__(self):
        """Initialize CUDA kernels."""
        self.kernels_available = CUPY_AVAILABLE

        if self.kernels_available:
            logger.info("CUDA kernels available for topological descriptors")
        else:
            logger.warning(
                "CUDA kernels not available - falling back to CPU calculations"
            )

    def calculate_wiener_index_batch(
        self, adjacency_matrices: List[np.ndarray]
    ) -> List[float]:
        """Calculate Wiener index for a batch of molecules using GPU."""
        if not self.kernels_available:
            raise ChemiAccelError("CUDA kernels not available")

        # Placeholder for CUDA implementation
        # This would implement Floyd-Warshall on GPU for batch processing
        results = []
        for adj_matrix in adjacency_matrices:
            # Transfer to GPU and calculate
            adj_gpu = cp.asarray(adj_matrix)
            # ... CUDA kernel implementation ...
            results.append(0.0)  # Placeholder

        return results

    def calculate_zagreb_indices_batch(
        self, adjacency_matrices: List[np.ndarray]
    ) -> List[Tuple[float, float]]:
        """Calculate Zagreb indices for a batch of molecules using GPU."""
        if not self.kernels_available:
            raise ChemiAccelError("CUDA kernels not available")

        # Placeholder for CUDA implementation
        results = []
        for adj_matrix in adjacency_matrices:
            results.append((0.0, 0.0))  # Placeholder

        return results


# Convenience functions
def calculate_topological_descriptors(
    molecules: List[MolecularData], use_gpu: bool = True
) -> pd.DataFrame:
    """Calculate topological descriptors for a list of molecules.

    Args:
        molecules: List of MolecularData objects
        use_gpu: Whether to use GPU acceleration when available

    Returns:
        DataFrame with topological descriptors
    """
    if not molecules:
        return pd.DataFrame()

    calculator = TopologicalDescriptorCalculator(use_gpu=use_gpu)
    results = []

    for mol_data in molecules:
        descriptors = calculator.calculate_topological_descriptors(mol_data)
        descriptors["mol_id"] = mol_data.mol_id
        descriptors["smiles"] = mol_data.smiles
        results.append(descriptors)

    df = pd.DataFrame(results)

    # Reorder columns
    id_cols = ["mol_id", "smiles"]
    desc_cols = [col for col in df.columns if col not in id_cols]
    df = df[id_cols + desc_cols]

    logger.info(f"Calculated topological descriptors for {len(df)} molecules")

    return df


def get_topological_descriptor_info() -> Dict[str, Dict[str, str]]:
    """Get information about available topological descriptors.

    Returns:
        Dictionary with descriptor information
    """
    calculator = TopologicalDescriptorCalculator()
    return calculator.get_descriptor_info()
