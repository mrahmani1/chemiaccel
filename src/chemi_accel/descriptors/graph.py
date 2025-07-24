"""
Molecular graph representations and graph-based descriptors.

This module provides functionality to create molecular graph representations
including adjacency matrices, atom/bond feature matrices, and graph-based
molecular descriptors.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors, AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from ..core.molecular import MolecularData
from ..core.exceptions import ChemiAccelError, DescriptorCalculationError
from ..utils.logging import get_logger

logger = get_logger()


class MolecularGraphBuilder:
    """Build molecular graph representations from RDKit molecules."""
    
    def __init__(self, include_hydrogens: bool = False):
        """Initialize molecular graph builder.
        
        Args:
            include_hydrogens: Whether to include hydrogen atoms in graphs
        """
        if not RDKIT_AVAILABLE:
            raise ChemiAccelError("RDKit is required for molecular graph operations")
        
        self.include_hydrogens = include_hydrogens
        
        # Define atom feature encodings
        self.atom_features = {
            "atomic_number": list(range(1, 119)),  # H to Og
            "degree": list(range(0, 11)),  # 0-10 connections
            "formal_charge": list(range(-5, 6)),  # -5 to +5
            "hybridization": [
                Chem.HybridizationType.SP,
                Chem.HybridizationType.SP2,
                Chem.HybridizationType.SP3,
                Chem.HybridizationType.SP3D,
                Chem.HybridizationType.SP3D2,
                Chem.HybridizationType.UNSPECIFIED
            ],
            "num_hydrogens": list(range(0, 5)),  # 0-4 hydrogens
            "aromaticity": [True, False]
        }
        
        # Define bond feature encodings
        self.bond_features = {
            "bond_type": [
                Chem.BondType.SINGLE,
                Chem.BondType.DOUBLE,
                Chem.BondType.TRIPLE,
                Chem.BondType.AROMATIC
            ],
            "conjugated": [True, False],
            "in_ring": [True, False],
            "stereo": [
                Chem.BondStereo.STEREONONE,
                Chem.BondStereo.STEREOANY,
                Chem.BondStereo.STEREOZ,
                Chem.BondStereo.STEREOE
            ]
        }
        
        logger.debug(f"MolecularGraphBuilder initialized (include_hydrogens={include_hydrogens})")
    
    def _one_hot_encode(self, value: Any, possible_values: List[Any]) -> List[int]:
        """Create one-hot encoding for a value."""
        encoding = [0] * len(possible_values)
        try:
            idx = possible_values.index(value)
            encoding[idx] = 1
        except ValueError:
            # Unknown value - leave as all zeros
            pass
        return encoding
    
    def get_atom_features(self, atom: Chem.Atom) -> np.ndarray:
        """Extract feature vector for an atom.
        
        Args:
            atom: RDKit Atom object
            
        Returns:
            Numpy array of atom features
        """
        features = []
        
        # Atomic number (one-hot)
        features.extend(self._one_hot_encode(atom.GetAtomicNum(), self.atom_features["atomic_number"]))
        
        # Degree (one-hot)
        features.extend(self._one_hot_encode(atom.GetDegree(), self.atom_features["degree"]))
        
        # Formal charge (one-hot)
        features.extend(self._one_hot_encode(atom.GetFormalCharge(), self.atom_features["formal_charge"]))
        
        # Hybridization (one-hot)
        features.extend(self._one_hot_encode(atom.GetHybridization(), self.atom_features["hybridization"]))
        
        # Number of hydrogens (one-hot)
        features.extend(self._one_hot_encode(atom.GetTotalNumHs(), self.atom_features["num_hydrogens"]))
        
        # Aromaticity (one-hot)
        features.extend(self._one_hot_encode(atom.GetIsAromatic(), self.atom_features["aromaticity"]))
        
        # Additional scalar features
        features.append(float(atom.GetMass()))
        features.append(float(atom.GetTotalValence()))
        features.append(float(atom.IsInRing()))
        
        return np.array(features, dtype=np.float32)
    
    def get_bond_features(self, bond: Chem.Bond) -> np.ndarray:
        """Extract feature vector for a bond.
        
        Args:
            bond: RDKit Bond object
            
        Returns:
            Numpy array of bond features
        """
        features = []
        
        # Bond type (one-hot)
        features.extend(self._one_hot_encode(bond.GetBondType(), self.bond_features["bond_type"]))
        
        # Conjugated (one-hot)
        features.extend(self._one_hot_encode(bond.GetIsConjugated(), self.bond_features["conjugated"]))
        
        # In ring (one-hot)
        features.extend(self._one_hot_encode(bond.IsInRing(), self.bond_features["in_ring"]))
        
        # Stereo (one-hot)
        features.extend(self._one_hot_encode(bond.GetStereo(), self.bond_features["stereo"]))
        
        # Additional scalar features
        features.append(float(bond.GetBondTypeAsDouble()))
        
        return np.array(features, dtype=np.float32)
    
    def build_adjacency_matrix(self, mol: Chem.Mol) -> np.ndarray:
        """Build adjacency matrix for molecule.
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            Adjacency matrix as numpy array
        """
        num_atoms = mol.GetNumAtoms()
        adj_matrix = np.zeros((num_atoms, num_atoms), dtype=np.int32)
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
        
        return adj_matrix
    
    def build_bond_adjacency_matrix(self, mol: Chem.Mol) -> np.ndarray:
        """Build bond adjacency matrix (bonds connected through atoms).
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            Bond adjacency matrix as numpy array
        """
        num_bonds = mol.GetNumBonds()
        if num_bonds == 0:
            return np.zeros((0, 0), dtype=np.int32)
        
        bond_adj = np.zeros((num_bonds, num_bonds), dtype=np.int32)
        
        # Create mapping from atom pairs to bond indices
        bond_map = {}
        for i, bond in enumerate(mol.GetBonds()):
            atom_pair = tuple(sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]))
            bond_map[atom_pair] = i
        
        # Connect bonds that share an atom
        for i, bond1 in enumerate(mol.GetBonds()):
            atoms1 = {bond1.GetBeginAtomIdx(), bond1.GetEndAtomIdx()}
            
            for j, bond2 in enumerate(mol.GetBonds()):
                if i != j:
                    atoms2 = {bond2.GetBeginAtomIdx(), bond2.GetEndAtomIdx()}
                    if atoms1 & atoms2:  # Bonds share an atom
                        bond_adj[i, j] = 1
        
        return bond_adj
    
    def build_molecular_graph(self, mol_data: MolecularData) -> Dict[str, Any]:
        """Build complete molecular graph representation.
        
        Args:
            mol_data: MolecularData with RDKit mol object
            
        Returns:
            Dictionary containing graph components
        """
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            raise ChemiAccelError(f"No RDKit mol object found for molecule {mol_data.mol_id}")
        
        # Optionally add hydrogens
        if self.include_hydrogens:
            mol = Chem.AddHs(rdkit_mol)
        else:
            mol = rdkit_mol
        
        # Build atom feature matrix
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self.get_atom_features(atom))
        atom_feature_matrix = np.array(atom_features) if atom_features else np.zeros((0, 0))
        
        # Build bond feature matrix
        bond_features = []
        for bond in mol.GetBonds():
            bond_features.append(self.get_bond_features(bond))
        bond_feature_matrix = np.array(bond_features) if bond_features else np.zeros((0, 0))
        
        # Build adjacency matrices
        adjacency_matrix = self.build_adjacency_matrix(mol)
        bond_adjacency_matrix = self.build_bond_adjacency_matrix(mol)
        
        # Create sparse representations for efficiency
        adjacency_sparse = csr_matrix(adjacency_matrix)
        bond_adjacency_sparse = csr_matrix(bond_adjacency_matrix)
        
        graph_data = {
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds(),
            "atom_features": atom_feature_matrix,
            "bond_features": bond_feature_matrix,
            "adjacency_matrix": adjacency_matrix,
            "adjacency_sparse": adjacency_sparse,
            "bond_adjacency_matrix": bond_adjacency_matrix,
            "bond_adjacency_sparse": bond_adjacency_sparse,
            "atom_feature_names": self._get_atom_feature_names(),
            "bond_feature_names": self._get_bond_feature_names()
        }
        
        return graph_data
    
    def _get_atom_feature_names(self) -> List[str]:
        """Get names of atom features."""
        names = []
        
        # One-hot encoded features
        for atomic_num in self.atom_features["atomic_number"]:
            names.append(f"atomic_num_{atomic_num}")
        for degree in self.atom_features["degree"]:
            names.append(f"degree_{degree}")
        for charge in self.atom_features["formal_charge"]:
            names.append(f"formal_charge_{charge}")
        for hyb in self.atom_features["hybridization"]:
            names.append(f"hybridization_{hyb}")
        for num_h in self.atom_features["num_hydrogens"]:
            names.append(f"num_hydrogens_{num_h}")
        for aromatic in self.atom_features["aromaticity"]:
            names.append(f"aromatic_{aromatic}")
        
        # Scalar features
        names.extend(["mass", "total_valence", "in_ring"])
        
        return names
    
    def _get_bond_feature_names(self) -> List[str]:
        """Get names of bond features."""
        names = []
        
        # One-hot encoded features
        for bond_type in self.bond_features["bond_type"]:
            names.append(f"bond_type_{bond_type}")
        for conj in self.bond_features["conjugated"]:
            names.append(f"conjugated_{conj}")
        for in_ring in self.bond_features["in_ring"]:
            names.append(f"in_ring_{in_ring}")
        for stereo in self.bond_features["stereo"]:
            names.append(f"stereo_{stereo}")
        
        # Scalar features
        names.append("bond_order")
        
        return names
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get dimensions of feature vectors.
        
        Returns:
            Dictionary with feature dimensions
        """
        # Calculate atom feature dimension
        atom_dim = (
            len(self.atom_features["atomic_number"]) +
            len(self.atom_features["degree"]) +
            len(self.atom_features["formal_charge"]) +
            len(self.atom_features["hybridization"]) +
            len(self.atom_features["num_hydrogens"]) +
            len(self.atom_features["aromaticity"]) +
            3  # mass, total_valence, in_ring
        )
        
        # Calculate bond feature dimension
        bond_dim = (
            len(self.bond_features["bond_type"]) +
            len(self.bond_features["conjugated"]) +
            len(self.bond_features["in_ring"]) +
            len(self.bond_features["stereo"]) +
            1  # bond_order
        )
        
        return {
            "atom_features": atom_dim,
            "bond_features": bond_dim
        }


class GraphDescriptorCalculator:
    """Calculate graph-based molecular descriptors."""
    
    def __init__(self):
        """Initialize graph descriptor calculator."""
        if not RDKIT_AVAILABLE:
            raise ChemiAccelError("RDKit is required for graph descriptor calculations")
        
        self.graph_builder = MolecularGraphBuilder()
        
        # Define graph-based descriptors
        self.descriptor_functions = {
            "wiener_index": self._calculate_wiener_index,
            "zagreb_index_1": self._calculate_zagreb_index_1,
            "zagreb_index_2": self._calculate_zagreb_index_2,
            "randic_index": self._calculate_randic_index,
            "balaban_j": self._calculate_balaban_j,
            "graph_density": self._calculate_graph_density,
            "average_degree": self._calculate_average_degree,
            "max_degree": self._calculate_max_degree,
            "num_cycles": self._calculate_num_cycles,
            "cyclomatic_number": self._calculate_cyclomatic_number
        }
        
        logger.debug(f"GraphDescriptorCalculator initialized with {len(self.descriptor_functions)} descriptors")
    
    def _calculate_wiener_index(self, graph_data: Dict[str, Any]) -> float:
        """Calculate Wiener index (sum of shortest path distances)."""
        adj_matrix = graph_data["adjacency_matrix"]
        num_atoms = graph_data["num_atoms"]
        
        if num_atoms <= 1:
            return 0.0
        
        # Floyd-Warshall algorithm for all-pairs shortest paths
        dist_matrix = adj_matrix.astype(float)
        dist_matrix[dist_matrix == 0] = np.inf
        np.fill_diagonal(dist_matrix, 0)
        
        for k in range(num_atoms):
            for i in range(num_atoms):
                for j in range(num_atoms):
                    if dist_matrix[i, k] + dist_matrix[k, j] < dist_matrix[i, j]:
                        dist_matrix[i, j] = dist_matrix[i, k] + dist_matrix[k, j]
        
        # Sum upper triangle (avoid double counting)
        wiener_index = 0.0
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                if np.isfinite(dist_matrix[i, j]):
                    wiener_index += dist_matrix[i, j]
        
        return wiener_index
    
    def _calculate_zagreb_index_1(self, graph_data: Dict[str, Any]) -> float:
        """Calculate first Zagreb index (sum of squared degrees)."""
        adj_matrix = graph_data["adjacency_matrix"]
        degrees = np.sum(adj_matrix, axis=1)
        return float(np.sum(degrees ** 2))
    
    def _calculate_zagreb_index_2(self, graph_data: Dict[str, Any]) -> float:
        """Calculate second Zagreb index (sum of products of adjacent vertex degrees)."""
        adj_matrix = graph_data["adjacency_matrix"]
        degrees = np.sum(adj_matrix, axis=1)
        
        zagreb_2 = 0.0
        num_atoms = graph_data["num_atoms"]
        
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                if adj_matrix[i, j] == 1:  # Adjacent vertices
                    zagreb_2 += degrees[i] * degrees[j]
        
        return zagreb_2
    
    def _calculate_randic_index(self, graph_data: Dict[str, Any]) -> float:
        """Calculate Randic connectivity index."""
        adj_matrix = graph_data["adjacency_matrix"]
        degrees = np.sum(adj_matrix, axis=1)
        
        randic_index = 0.0
        num_atoms = graph_data["num_atoms"]
        
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                if adj_matrix[i, j] == 1:  # Adjacent vertices
                    if degrees[i] > 0 and degrees[j] > 0:
                        randic_index += 1.0 / np.sqrt(degrees[i] * degrees[j])
        
        return randic_index
    
    def _calculate_balaban_j(self, graph_data: Dict[str, Any]) -> float:
        """Calculate Balaban J index."""
        num_atoms = graph_data["num_atoms"]
        num_bonds = graph_data["num_bonds"]
        
        if num_bonds == 0 or num_atoms <= 2:
            return 0.0
        
        # This is a simplified version - full implementation requires
        # distance matrix and more complex calculations
        cyclomatic_number = self._calculate_cyclomatic_number(graph_data)
        if cyclomatic_number == 0:
            return 0.0
        
        # Placeholder calculation
        return float(num_bonds) / (cyclomatic_number + 1)
    
    def _calculate_graph_density(self, graph_data: Dict[str, Any]) -> float:
        """Calculate graph density (ratio of edges to possible edges)."""
        num_atoms = graph_data["num_atoms"]
        num_bonds = graph_data["num_bonds"]
        
        if num_atoms <= 1:
            return 0.0
        
        max_possible_bonds = num_atoms * (num_atoms - 1) / 2
        return num_bonds / max_possible_bonds
    
    def _calculate_average_degree(self, graph_data: Dict[str, Any]) -> float:
        """Calculate average vertex degree."""
        adj_matrix = graph_data["adjacency_matrix"]
        degrees = np.sum(adj_matrix, axis=1)
        return float(np.mean(degrees))
    
    def _calculate_max_degree(self, graph_data: Dict[str, Any]) -> float:
        """Calculate maximum vertex degree."""
        adj_matrix = graph_data["adjacency_matrix"]
        degrees = np.sum(adj_matrix, axis=1)
        return float(np.max(degrees)) if len(degrees) > 0 else 0.0
    
    def _calculate_num_cycles(self, graph_data: Dict[str, Any]) -> float:
        """Calculate number of cycles (rings)."""
        # This is approximated by the cyclomatic number
        return self._calculate_cyclomatic_number(graph_data)
    
    def _calculate_cyclomatic_number(self, graph_data: Dict[str, Any]) -> float:
        """Calculate cyclomatic number (edges - vertices + connected components)."""
        num_atoms = graph_data["num_atoms"]
        num_bonds = graph_data["num_bonds"]
        
        # For connected molecules, number of connected components = 1
        # Cyclomatic number = edges - vertices + 1
        return max(0.0, float(num_bonds - num_atoms + 1))
    
    def calculate_graph_descriptors(self, mol_data: MolecularData) -> Dict[str, float]:
        """Calculate all graph-based descriptors for a molecule.
        
        Args:
            mol_data: MolecularData with RDKit mol object
            
        Returns:
            Dictionary of graph descriptor values
        """
        try:
            # Build molecular graph
            graph_data = self.graph_builder.build_molecular_graph(mol_data)
            
            # Calculate descriptors
            descriptors = {}
            for desc_name, desc_func in self.descriptor_functions.items():
                try:
                    value = desc_func(graph_data)
                    descriptors[desc_name] = float(value)
                except Exception as e:
                    logger.warning(f"Failed to calculate {desc_name} for molecule {mol_data.mol_id}: {e}")
                    descriptors[desc_name] = np.nan
            
            return descriptors
            
        except Exception as e:
            logger.error(f"Failed to calculate graph descriptors for molecule {mol_data.mol_id}: {e}")
            return {desc_name: np.nan for desc_name in self.descriptor_functions.keys()}
    
    def get_available_descriptors(self) -> List[str]:
        """Get list of available graph descriptor names."""
        return list(self.descriptor_functions.keys())


def build_molecular_graphs(molecules: List[MolecularData], 
                          include_hydrogens: bool = False) -> List[Dict[str, Any]]:
    """Build molecular graphs for a list of molecules.
    
    Args:
        molecules: List of MolecularData objects
        include_hydrogens: Whether to include hydrogen atoms
        
    Returns:
        List of molecular graph dictionaries
    """
    builder = MolecularGraphBuilder(include_hydrogens=include_hydrogens)
    graphs = []
    
    for mol_data in molecules:
        try:
            graph = builder.build_molecular_graph(mol_data)
            graphs.append(graph)
        except Exception as e:
            logger.error(f"Failed to build graph for molecule {mol_data.mol_id}: {e}")
            graphs.append({})
    
    return graphs


def calculate_graph_descriptors(molecules: List[MolecularData]) -> pd.DataFrame:
    """Calculate graph-based descriptors for a list of molecules.
    
    Args:
        molecules: List of MolecularData objects
        
    Returns:
        DataFrame with graph descriptors
    """
    calculator = GraphDescriptorCalculator()
    results = []
    
    for mol_data in molecules:
        descriptors = calculator.calculate_graph_descriptors(mol_data)
        descriptors["mol_id"] = mol_data.mol_id
        descriptors["smiles"] = mol_data.smiles
        results.append(descriptors)
    
    df = pd.DataFrame(results)
    
    # Reorder columns
    id_cols = ["mol_id", "smiles"]
    desc_cols = [col for col in df.columns if col not in id_cols]
    df = df[id_cols + desc_cols]
    
    return df