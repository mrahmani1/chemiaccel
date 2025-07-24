"""
Constitutional molecular descriptors implementation.

This module provides comprehensive constitutional descriptors including atom and bond
counting variants, molecular complexity metrics, Bertz complexity, synthetic
accessibility scoring, and information content descriptors with entropy calculations.

Constitutional descriptors are fundamental molecular properties that describe the
basic composition and structure of molecules without considering 3D geometry.
They include atom counts, bond counts, ring counts, and various complexity measures.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
import math
from collections import Counter, defaultdict

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski
    from rdkit.Chem import rdchem, AllChem, rdMolChemicalFeatures
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem.Crippen import MolLogP, MolMR
    from rdkit.Chem.Descriptors import BertzCT
    from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    import cupy as cp
    from cupy import RawKernel
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

from ..core.molecular import MolecularData
from ..core.exceptions import ChemiAccelError, DescriptorCalculationError
from ..utils.logging import get_logger

logger = get_logger()


# CUDA kernel for constitutional descriptors
CONSTITUTIONAL_DESCRIPTORS_KERNEL = r'''
extern "C" __global__
void calculate_constitutional_descriptors(
    const int* atomic_numbers,
    const float* atom_weights,
    const int* formal_charges,
    const int* hybridizations,
    const bool* is_aromatic,
    const int* bond_types,
    const bool* bond_in_ring,
    const int* molecule_boundaries,
    const int* molecule_bond_boundaries,
    float* molecular_weights,
    int* atom_counts,
    int* heavy_atom_counts,
    int* carbon_counts,
    int* nitrogen_counts,
    int* oxygen_counts,
    int* sulfur_counts,
    int* phosphorus_counts,
    int* halogen_counts,
    int* aromatic_atom_counts,
    int* sp3_carbon_counts,
    int* sp2_carbon_counts,
    int* sp_carbon_counts,
    int* single_bond_counts,
    int* double_bond_counts,
    int* triple_bond_counts,
    int* aromatic_bond_counts,
    int* ring_bond_counts,
    float* complexity_indices,
    int num_molecules
) {
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (mol_idx >= num_molecules) return;
    
    int start_atom = (mol_idx == 0) ? 0 : molecule_boundaries[mol_idx - 1];
    int end_atom = molecule_boundaries[mol_idx];
    int start_bond = (mol_idx == 0) ? 0 : molecule_bond_boundaries[mol_idx - 1];
    int end_bond = molecule_bond_boundaries[mol_idx];
    
    // Initialize counters
    float mol_weight = 0.0f;
    int total_atoms = end_atom - start_atom;
    int heavy_atoms = 0;
    int carbons = 0, nitrogens = 0, oxygens = 0, sulfurs = 0, phosphorus = 0;
    int halogens = 0, aromatic_atoms = 0;
    int sp3_carbons = 0, sp2_carbons = 0, sp_carbons = 0;
    
    // Count atoms by type and properties
    for (int i = start_atom; i < end_atom; i++) {
        int atomic_num = atomic_numbers[i];
        mol_weight += atom_weights[i];
        
        if (atomic_num != 1) heavy_atoms++;  // Non-hydrogen
        
        // Count by element
        switch (atomic_num) {
            case 6: carbons++; break;           // Carbon
            case 7: nitrogens++; break;         // Nitrogen
            case 8: oxygens++; break;           // Oxygen
            case 16: sulfurs++; break;          // Sulfur
            case 15: phosphorus++; break;       // Phosphorus
            case 9: case 17: case 35: case 53:  // F, Cl, Br, I
                halogens++; break;
        }
        
        if (is_aromatic[i]) aromatic_atoms++;
        
        // Count carbon hybridization
        if (atomic_num == 6) {
            switch (hybridizations[i]) {
                case 3: sp3_carbons++; break;   // sp3
                case 2: sp2_carbons++; break;   // sp2
                case 1: sp_carbons++; break;    // sp
            }
        }
    }
    
    // Count bonds by type
    int single_bonds = 0, double_bonds = 0, triple_bonds = 0;
    int aromatic_bonds = 0, ring_bonds = 0;
    
    for (int i = start_bond; i < end_bond; i++) {
        switch (bond_types[i]) {
            case 1: single_bonds++; break;
            case 2: double_bonds++; break;
            case 3: triple_bonds++; break;
            case 12: aromatic_bonds++; break;  // Aromatic bond type
        }
        
        if (bond_in_ring[i]) ring_bonds++;
    }
    
    // Calculate simple complexity index
    float complexity = 0.0f;
    if (total_atoms > 0) {
        complexity = (float)(heavy_atoms + single_bonds + 2*double_bonds + 3*triple_bonds) / total_atoms;
    }
    
    // Store results
    molecular_weights[mol_idx] = mol_weight;
    atom_counts[mol_idx] = total_atoms;
    heavy_atom_counts[mol_idx] = heavy_atoms;
    carbon_counts[mol_idx] = carbons;
    nitrogen_counts[mol_idx] = nitrogens;
    oxygen_counts[mol_idx] = oxygens;
    sulfur_counts[mol_idx] = sulfurs;
    phosphorus_counts[mol_idx] = phosphorus;
    halogen_counts[mol_idx] = halogens;
    aromatic_atom_counts[mol_idx] = aromatic_atoms;
    sp3_carbon_counts[mol_idx] = sp3_carbons;
    sp2_carbon_counts[mol_idx] = sp2_carbons;
    sp_carbon_counts[mol_idx] = sp_carbons;
    single_bond_counts[mol_idx] = single_bonds;
    double_bond_counts[mol_idx] = double_bonds;
    triple_bond_counts[mol_idx] = triple_bonds;
    aromatic_bond_counts[mol_idx] = aromatic_bonds;
    ring_bond_counts[mol_idx] = ring_bonds;
    complexity_indices[mol_idx] = complexity;
}
'''


class ConstitutionalDescriptorCalculator:
    """Calculator for constitutional molecular descriptors.
    
    This class provides comprehensive constitutional descriptors including:
    - Basic atom and bond counts
    - Element-specific counts
    - Hybridization-specific counts
    - Constitutional indices
    - Molecular complexity metrics
    - Information content descriptors
    - Synthetic accessibility scoring
    """
    
    def __init__(self, use_cuda: bool = True):
        """Initialize constitutional descriptor calculator.
        
        Args:
            use_cuda: Whether to use CUDA acceleration when available
        """
        if not RDKIT_AVAILABLE:
            raise ChemiAccelError("RDKit is required for constitutional descriptor calculations")
        
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        
        if self.use_cuda:
            self._compile_cuda_kernels()
        
        # Define all constitutional descriptors
        self.descriptor_registry = self._build_descriptor_registry()
        
        logger.info(f"ConstitutionalDescriptorCalculator initialized with {len(self.descriptor_registry)} descriptors")
    
    def _compile_cuda_kernels(self):
        """Compile CUDA kernels for constitutional descriptors."""
        if not CUDA_AVAILABLE:
            return
        
        try:
            self.cuda_kernel = RawKernel(CONSTITUTIONAL_DESCRIPTORS_KERNEL, 'calculate_constitutional_descriptors')
            logger.info("CUDA kernels compiled successfully for constitutional descriptors")
        except Exception as e:
            logger.warning(f"Failed to compile CUDA kernels: {e}")
            self.use_cuda = False
    
    def _build_descriptor_registry(self) -> Dict[str, Dict[str, Any]]:
        """Build comprehensive registry of constitutional descriptors."""
        registry = {}
        
        # Basic molecular properties
        basic_descriptors = {
            "molecular_weight": {
                "function": Descriptors.MolWt,
                "description": "Molecular weight in Daltons",
                "category": "basic"
            },
            "exact_molecular_weight": {
                "function": Descriptors.ExactMolWt,
                "description": "Exact molecular weight using isotopic masses",
                "category": "basic"
            },
            "heavy_atom_molecular_weight": {
                "function": Descriptors.HeavyAtomMolWt,
                "description": "Molecular weight excluding hydrogens",
                "category": "basic"
            }
        }
        registry.update(basic_descriptors)
        
        # Atom counts - basic
        atom_count_descriptors = {
            "num_atoms": {
                "function": lambda mol: mol.GetNumAtoms(),
                "description": "Total number of atoms",
                "category": "atom_counts"
            },
            "num_heavy_atoms": {
                "function": Descriptors.HeavyAtomCount,
                "description": "Number of heavy (non-hydrogen) atoms",
                "category": "atom_counts"
            },
            "num_heteroatoms": {
                "function": Descriptors.NumHeteroatoms,
                "description": "Number of heteroatoms",
                "category": "atom_counts"
            }
        }
        registry.update(atom_count_descriptors)
        
        # Element-specific atom counts
        element_counts = {
            "num_carbons": {
                "function": lambda mol: self._count_atoms_by_element(mol, 6),
                "description": "Number of carbon atoms",
                "category": "element_counts"
            },
            "num_nitrogens": {
                "function": lambda mol: self._count_atoms_by_element(mol, 7),
                "description": "Number of nitrogen atoms",
                "category": "element_counts"
            },
            "num_oxygens": {
                "function": lambda mol: self._count_atoms_by_element(mol, 8),
                "description": "Number of oxygen atoms",
                "category": "element_counts"
            },
            "num_sulfurs": {
                "function": lambda mol: self._count_atoms_by_element(mol, 16),
                "description": "Number of sulfur atoms",
                "category": "element_counts"
            },
            "num_phosphorus": {
                "function": lambda mol: self._count_atoms_by_element(mol, 15),
                "description": "Number of phosphorus atoms",
                "category": "element_counts"
            },
            "num_halogens": {
                "function": lambda mol: self._count_halogens(mol),
                "description": "Number of halogen atoms (F, Cl, Br, I)",
                "category": "element_counts"
            },
            "num_fluorines": {
                "function": lambda mol: self._count_atoms_by_element(mol, 9),
                "description": "Number of fluorine atoms",
                "category": "element_counts"
            },
            "num_chlorines": {
                "function": lambda mol: self._count_atoms_by_element(mol, 17),
                "description": "Number of chlorine atoms",
                "category": "element_counts"
            },
            "num_bromines": {
                "function": lambda mol: self._count_atoms_by_element(mol, 35),
                "description": "Number of bromine atoms",
                "category": "element_counts"
            },
            "num_iodines": {
                "function": lambda mol: self._count_atoms_by_element(mol, 53),
                "description": "Number of iodine atoms",
                "category": "element_counts"
            }
        }
        registry.update(element_counts)
        
        # Hybridization-specific counts
        hybridization_counts = {
            "num_sp3_carbons": {
                "function": lambda mol: self._count_sp3_carbons(mol),
                "description": "Number of sp3 hybridized carbons",
                "category": "hybridization"
            },
            "num_sp2_carbons": {
                "function": lambda mol: self._count_sp2_carbons(mol),
                "description": "Number of sp2 hybridized carbons",
                "category": "hybridization"
            },
            "num_sp_carbons": {
                "function": lambda mol: self._count_sp_carbons(mol),
                "description": "Number of sp hybridized carbons",
                "category": "hybridization"
            },
            "fraction_csp3": {
                "function": Descriptors.FractionCSP3,
                "description": "Fraction of sp3 carbons",
                "category": "hybridization"
            }
        }
        registry.update(hybridization_counts)
        
        # Bond counts
        bond_count_descriptors = {
            "num_bonds": {
                "function": lambda mol: mol.GetNumBonds(),
                "description": "Total number of bonds",
                "category": "bond_counts"
            },
            "num_single_bonds": {
                "function": lambda mol: self._count_bonds_by_type(mol, Chem.BondType.SINGLE),
                "description": "Number of single bonds",
                "category": "bond_counts"
            },
            "num_double_bonds": {
                "function": lambda mol: self._count_bonds_by_type(mol, Chem.BondType.DOUBLE),
                "description": "Number of double bonds",
                "category": "bond_counts"
            },
            "num_triple_bonds": {
                "function": lambda mol: self._count_bonds_by_type(mol, Chem.BondType.TRIPLE),
                "description": "Number of triple bonds",
                "category": "bond_counts"
            },
            "num_aromatic_bonds": {
                "function": lambda mol: self._count_bonds_by_type(mol, Chem.BondType.AROMATIC),
                "description": "Number of aromatic bonds",
                "category": "bond_counts"
            }
        }
        registry.update(bond_count_descriptors)
        
        # Ring-related descriptors
        ring_descriptors = {
            "ring_count": {
                "function": Descriptors.RingCount,
                "description": "Total number of rings",
                "category": "rings"
            },
            "num_aromatic_rings": {
                "function": Descriptors.NumAromaticRings,
                "description": "Number of aromatic rings",
                "category": "rings"
            },
            "num_saturated_rings": {
                "function": Descriptors.NumSaturatedRings,
                "description": "Number of saturated rings",
                "category": "rings"
            },
            "num_aliphatic_rings": {
                "function": Descriptors.NumAliphaticRings,
                "description": "Number of aliphatic rings",
                "category": "rings"
            },
            "largest_ring_size": {
                "function": self._calculate_largest_ring_size,
                "description": "Size of the largest ring",
                "category": "rings"
            },
            "smallest_ring_size": {
                "function": self._calculate_smallest_ring_size,
                "description": "Size of the smallest ring",
                "category": "rings"
            }
        }
        registry.update(ring_descriptors)
        
        # Complexity and information content descriptors
        complexity_descriptors = {
            "bertz_complexity": {
                "function": BertzCT,
                "description": "Bertz complexity index",
                "category": "complexity"
            },
            "molecular_complexity": {
                "function": self._calculate_molecular_complexity,
                "description": "Molecular complexity based on atom and bond diversity",
                "category": "complexity"
            },
            "information_content": {
                "function": self._calculate_information_content,
                "description": "Information content based on atom type diversity",
                "category": "complexity"
            },
            "bond_information_content": {
                "function": self._calculate_bond_information_content,
                "description": "Information content based on bond type diversity",
                "category": "complexity"
            },
            "structural_information_content": {
                "function": self._calculate_structural_information_content,
                "description": "Combined structural information content",
                "category": "complexity"
            }
        }
        registry.update(complexity_descriptors)
        
        # Constitutional indices
        constitutional_indices = {
            "constitutional_index_1": {
                "function": self._calculate_constitutional_index_1,
                "description": "Constitutional index based on heavy atoms",
                "category": "indices"
            },
            "constitutional_index_2": {
                "function": self._calculate_constitutional_index_2,
                "description": "Constitutional index based on bonds",
                "category": "indices"
            },
            "constitutional_index_3": {
                "function": self._calculate_constitutional_index_3,
                "description": "Constitutional index based on rings",
                "category": "indices"
            },
            "constitutional_index_4": {
                "function": self._calculate_constitutional_index_4,
                "description": "Constitutional index based on heteroatoms",
                "category": "indices"
            },
            "constitutional_index_5": {
                "function": self._calculate_constitutional_index_5,
                "description": "Constitutional index based on multiple bonds",
                "category": "indices"
            }
        }
        registry.update(constitutional_indices)
        
        # Synthetic accessibility and drug-likeness
        accessibility_descriptors = {
            "synthetic_accessibility": {
                "function": self._calculate_synthetic_accessibility,
                "description": "Synthetic accessibility score (1=easy, 10=difficult)",
                "category": "accessibility"
            },
            "natural_product_likeness": {
                "function": self._calculate_natural_product_likeness,
                "description": "Natural product-likeness score",
                "category": "accessibility"
            }
        }
        registry.update(accessibility_descriptors)
        
        # Additional constitutional descriptors
        additional_descriptors = {
            "num_radical_electrons": {
                "function": Descriptors.NumRadicalElectrons,
                "description": "Number of radical electrons",
                "category": "electronic"
            },
            "formal_charge": {
                "function": Chem.rdmolops.GetFormalCharge,
                "description": "Total formal charge",
                "category": "electronic"
            },
            "num_valence_electrons": {
                "function": Descriptors.NumValenceElectrons,
                "description": "Number of valence electrons",
                "category": "electronic"
            },
            "molecular_formula": {
                "function": lambda mol: Chem.rdMolDescriptors.CalcMolFormula(mol),
                "description": "Molecular formula",
                "category": "basic"
            }
        }
        registry.update(additional_descriptors)
        
        return registry
    
    def _count_atoms_by_element(self, mol: Chem.Mol, atomic_number: int) -> int:
        """Count atoms of a specific element."""
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == atomic_number:
                count += 1
        return count
    
    def _count_halogens(self, mol: Chem.Mol) -> int:
        """Count halogen atoms (F, Cl, Br, I)."""
        halogen_numbers = {9, 17, 35, 53}  # F, Cl, Br, I
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in halogen_numbers:
                count += 1
        return count
    
    def _count_sp3_carbons(self, mol: Chem.Mol) -> int:
        """Count sp3 hybridized carbon atoms."""
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6 and atom.GetHybridization() == Chem.HybridizationType.SP3:
                count += 1
        return count
    
    def _count_sp2_carbons(self, mol: Chem.Mol) -> int:
        """Count sp2 hybridized carbon atoms."""
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6 and atom.GetHybridization() == Chem.HybridizationType.SP2:
                count += 1
        return count
    
    def _count_sp_carbons(self, mol: Chem.Mol) -> int:
        """Count sp hybridized carbon atoms."""
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6 and atom.GetHybridization() == Chem.HybridizationType.SP:
                count += 1
        return count
    
    def _count_bonds_by_type(self, mol: Chem.Mol, bond_type: Chem.BondType) -> int:
        """Count bonds of a specific type."""
        count = 0
        for bond in mol.GetBonds():
            if bond.GetBondType() == bond_type:
                count += 1
        return count
    
    def _calculate_largest_ring_size(self, mol: Chem.Mol) -> int:
        """Calculate the size of the largest ring."""
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        if not atom_rings:
            return 0
        return max(len(ring) for ring in atom_rings)
    
    def _calculate_smallest_ring_size(self, mol: Chem.Mol) -> int:
        """Calculate the size of the smallest ring."""
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        if not atom_rings:
            return 0
        return min(len(ring) for ring in atom_rings)
    
    def _calculate_molecular_complexity(self, mol: Chem.Mol) -> float:
        """Calculate molecular complexity based on atom and bond diversity."""
        # Count different atom types
        atom_types = Counter()
        for atom in mol.GetAtoms():
            atom_types[atom.GetAtomicNum()] += 1
        
        # Count different bond types
        bond_types = Counter()
        for bond in mol.GetBonds():
            bond_types[bond.GetBondType()] += 1
        
        # Calculate complexity as weighted sum
        atom_diversity = len(atom_types)
        bond_diversity = len(bond_types)
        total_atoms = mol.GetNumAtoms()
        total_bonds = mol.GetNumBonds()
        
        if total_atoms == 0:
            return 0.0
        
        complexity = (atom_diversity + bond_diversity) * math.log(total_atoms + 1)
        return complexity
    
    def _calculate_information_content(self, mol: Chem.Mol) -> float:
        """Calculate information content based on atom type diversity."""
        atom_types = Counter()
        for atom in mol.GetAtoms():
            atom_types[atom.GetAtomicNum()] += 1
        
        total_atoms = sum(atom_types.values())
        if total_atoms <= 1:
            return 0.0
        
        # Calculate Shannon entropy
        entropy = 0.0
        for count in atom_types.values():
            if count > 0:
                p = count / total_atoms
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _calculate_bond_information_content(self, mol: Chem.Mol) -> float:
        """Calculate information content based on bond type diversity."""
        bond_types = Counter()
        for bond in mol.GetBonds():
            bond_types[bond.GetBondType()] += 1
        
        total_bonds = sum(bond_types.values())
        if total_bonds <= 1:
            return 0.0
        
        # Calculate Shannon entropy
        entropy = 0.0
        for count in bond_types.values():
            if count > 0:
                p = count / total_bonds
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _calculate_structural_information_content(self, mol: Chem.Mol) -> float:
        """Calculate combined structural information content."""
        atom_ic = self._calculate_information_content(mol)
        bond_ic = self._calculate_bond_information_content(mol)
        return atom_ic + bond_ic
    
    def _calculate_constitutional_index_1(self, mol: Chem.Mol) -> float:
        """Constitutional index based on heavy atoms."""
        heavy_atoms = Descriptors.HeavyAtomCount(mol)
        total_atoms = mol.GetNumAtoms()
        if total_atoms == 0:
            return 0.0
        return heavy_atoms / total_atoms
    
    def _calculate_constitutional_index_2(self, mol: Chem.Mol) -> float:
        """Constitutional index based on bonds."""
        total_bonds = mol.GetNumBonds()
        heavy_atoms = Descriptors.HeavyAtomCount(mol)
        if heavy_atoms <= 1:
            return 0.0
        return total_bonds / heavy_atoms
    
    def _calculate_constitutional_index_3(self, mol: Chem.Mol) -> float:
        """Constitutional index based on rings."""
        ring_count = Descriptors.RingCount(mol)
        heavy_atoms = Descriptors.HeavyAtomCount(mol)
        if heavy_atoms == 0:
            return 0.0
        return ring_count / heavy_atoms
    
    def _calculate_constitutional_index_4(self, mol: Chem.Mol) -> float:
        """Constitutional index based on heteroatoms."""
        heteroatoms = Descriptors.NumHeteroatoms(mol)
        heavy_atoms = Descriptors.HeavyAtomCount(mol)
        if heavy_atoms == 0:
            return 0.0
        return heteroatoms / heavy_atoms
    
    def _calculate_constitutional_index_5(self, mol: Chem.Mol) -> float:
        """Constitutional index based on multiple bonds."""
        multiple_bonds = 0
        for bond in mol.GetBonds():
            if bond.GetBondType() in [Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]:
                multiple_bonds += 1
        
        total_bonds = mol.GetNumBonds()
        if total_bonds == 0:
            return 0.0
        return multiple_bonds / total_bonds
    
    def _calculate_synthetic_accessibility(self, mol: Chem.Mol) -> float:
        """Calculate synthetic accessibility score (simplified version)."""
        # This is a simplified version - in practice would use trained models
        # Based on molecular complexity, ring strain, and functional group complexity
        
        # Basic complexity factors
        heavy_atoms = Descriptors.HeavyAtomCount(mol)
        ring_count = Descriptors.RingCount(mol)
        rotatable_bonds = CalcNumRotatableBonds(mol)
        
        # Penalize complex ring systems
        ring_penalty = 0
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if len(ring) < 5 or len(ring) > 7:  # Strained rings
                ring_penalty += 2
            if len(ring) > 8:  # Large rings
                ring_penalty += 3
        
        # Penalize high molecular weight
        mw_penalty = max(0, (Descriptors.MolWt(mol) - 500) / 100)
        
        # Base score
        base_score = 1.0
        
        # Add penalties
        complexity_score = base_score + (heavy_atoms / 10) + (ring_penalty / 2) + (mw_penalty / 2)
        
        # Normalize to 1-10 scale
        sa_score = min(10.0, max(1.0, complexity_score))
        
        return sa_score
    
    def _calculate_natural_product_likeness(self, mol: Chem.Mol) -> float:
        """Calculate natural product-likeness score (simplified version)."""
        # This is a simplified version - in practice would use trained models
        
        # Factors that increase NP-likeness
        np_score = 0.0
        
        # Presence of oxygen (common in natural products)
        oxygens = self._count_atoms_by_element(mol, 8)
        np_score += min(oxygens * 0.2, 1.0)
        
        # Moderate molecular weight
        mw = Descriptors.MolWt(mol)
        if 200 <= mw <= 600:
            np_score += 0.5
        
        # Presence of rings (but not too many)
        rings = Descriptors.RingCount(mol)
        if 1 <= rings <= 4:
            np_score += 0.3
        
        # Moderate complexity
        complexity = BertzCT(mol)
        if 50 <= complexity <= 200:
            np_score += 0.2
        
        # Normalize to 0-1 scale
        return min(1.0, np_score)
    
    def get_available_descriptors(self) -> List[str]:
        """Get list of all available constitutional descriptor names."""
        return list(self.descriptor_registry.keys())
    
    def get_descriptors_by_category(self, category: str) -> List[str]:
        """Get descriptors by category."""
        return [
            name for name, info in self.descriptor_registry.items()
            if info["category"] == category
        ]
    
    def get_descriptor_categories(self) -> List[str]:
        """Get list of all descriptor categories."""
        categories = set()
        for info in self.descriptor_registry.values():
            categories.add(info["category"])
        return sorted(list(categories))
    
    def calculate_single_descriptor(self, 
                                  mol_data: MolecularData, 
                                  descriptor_name: str) -> Union[float, int, str]:
        """Calculate a single constitutional descriptor for a molecule."""
        if descriptor_name not in self.descriptor_registry:
            raise DescriptorCalculationError(
                descriptor=descriptor_name,
                molecule_id=mol_data.mol_id,
                reason="Unknown constitutional descriptor"
            )
        
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            raise DescriptorCalculationError(
                descriptor=descriptor_name,
                molecule_id=mol_data.mol_id,
                reason="No RDKit mol object found in molecular data"
            )
        
        try:
            desc_func = self.descriptor_registry[descriptor_name]["function"]
            value = desc_func(rdkit_mol)
            
            # Convert to appropriate type
            if isinstance(value, str):
                return value
            elif isinstance(value, (int, np.integer)):
                return int(value)
            else:
                return float(value) if value is not None else 0.0
                
        except Exception as e:
            raise DescriptorCalculationError(
                descriptor=descriptor_name,
                molecule_id=mol_data.mol_id,
                reason=f"Failed to calculate {descriptor_name}: {e}"
            )
    
    def calculate_descriptors(self, 
                            mol_data: MolecularData,
                            descriptor_names: Optional[List[str]] = None) -> Dict[str, Union[float, int, str]]:
        """Calculate multiple constitutional descriptors for a single molecule."""
        if descriptor_names is None:
            descriptor_names = self.get_available_descriptors()
        
        descriptors = {}
        errors = []
        
        for desc_name in descriptor_names:
            try:
                value = self.calculate_single_descriptor(mol_data, desc_name)
                descriptors[desc_name] = value
            except DescriptorCalculationError as e:
                logger.warning(f"Constitutional descriptor calculation failed: {e}")
                descriptors[desc_name] = np.nan
                errors.append(str(e))
        
        if errors:
            descriptors["_calculation_errors"] = errors
        
        return descriptors
    
    def calculate_descriptors_batch(self, 
                                  molecules: List[MolecularData],
                                  descriptor_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Calculate constitutional descriptors for a batch of molecules."""
        if not molecules:
            return pd.DataFrame()
        
        if descriptor_names is None:
            descriptor_names = self.get_available_descriptors()
        
        # Use CUDA acceleration if available and requested
        if self.use_cuda and len(molecules) > 100:
            try:
                return self._calculate_descriptors_cuda(molecules, descriptor_names)
            except Exception as e:
                logger.warning(f"CUDA calculation failed, falling back to CPU: {e}")
        
        # CPU calculation
        results = []
        for mol_data in molecules:
            try:
                descriptors = self.calculate_descriptors(mol_data, descriptor_names)
                
                result_row = {
                    "mol_id": mol_data.mol_id,
                    "smiles": mol_data.smiles,
                    **descriptors
                }
                results.append(result_row)
                
            except Exception as e:
                logger.error(f"Failed to calculate constitutional descriptors for molecule {mol_data.mol_id}: {e}")
                result_row = {
                    "mol_id": mol_data.mol_id,
                    "smiles": mol_data.smiles,
                    **{name: np.nan for name in descriptor_names}
                }
                results.append(result_row)
        
        df = pd.DataFrame(results)
        
        # Reorder columns
        id_cols = ["mol_id", "smiles"]
        desc_cols = [col for col in df.columns if col not in id_cols and not col.startswith("_")]
        error_cols = [col for col in df.columns if col.startswith("_")]
        
        df = df[id_cols + desc_cols + error_cols]
        
        logger.info(f"Calculated {len(descriptor_names)} constitutional descriptors for {len(df)} molecules")
        return df
    
    def _calculate_descriptors_cuda(self, 
                                  molecules: List[MolecularData],
                                  descriptor_names: List[str]) -> pd.DataFrame:
        """Calculate constitutional descriptors using CUDA acceleration."""
        # This is a placeholder for CUDA implementation
        # In practice, would need to convert molecular data to GPU arrays
        # and launch CUDA kernels
        
        logger.info("CUDA acceleration for constitutional descriptors not yet fully implemented")
        raise NotImplementedError("CUDA acceleration will be implemented in future versions")
    
    def get_descriptor_statistics(self, 
                                df: pd.DataFrame, 
                                descriptor_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Calculate statistics for constitutional descriptors in a DataFrame."""
        if descriptor_names is None:
            descriptor_names = [
                col for col in df.columns 
                if col not in ["mol_id", "smiles", "molecular_formula"] and 
                   df[col].dtype in [np.float64, np.int64, np.float32, np.int32]
            ]
        
        stats = df[descriptor_names].describe()
        
        # Add additional statistics
        stats.loc["missing"] = df[descriptor_names].isna().sum()
        stats.loc["missing_pct"] = (df[descriptor_names].isna().sum() / len(df)) * 100
        
        return stats


def calculate_constitutional_descriptors(molecules: List[MolecularData],
                                       descriptor_set: str = "basic") -> pd.DataFrame:
    """Convenience function to calculate constitutional descriptors.
    
    Args:
        molecules: List of MolecularData objects
        descriptor_set: Descriptor set to calculate ("basic", "all", or category name)
        
    Returns:
        DataFrame with constitutional descriptors
    """
    calculator = ConstitutionalDescriptorCalculator()
    
    if descriptor_set == "basic":
        descriptors = [
            "molecular_weight", "num_atoms", "num_heavy_atoms", "num_carbons",
            "num_nitrogens", "num_oxygens", "num_bonds", "ring_count",
            "bertz_complexity", "information_content"
        ]
    elif descriptor_set == "all":
        descriptors = calculator.get_available_descriptors()
    else:
        # Try to get descriptors by category
        descriptors = calculator.get_descriptors_by_category(descriptor_set)
        if not descriptors:
            raise ValueError(f"Unknown descriptor set: {descriptor_set}")
    
    return calculator.calculate_descriptors_batch(molecules, descriptors)


def get_constitutional_descriptor_info() -> Dict[str, Dict[str, Any]]:
    """Get information about all available constitutional descriptors."""
    calculator = ConstitutionalDescriptorCalculator()
    
    info = {}
    for name, desc_info in calculator.descriptor_registry.items():
        info[name] = {
            "description": desc_info["description"],
            "category": desc_info["category"]
        }
    
    return info