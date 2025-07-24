"""
Repeating unit identification and polymer fragment analysis module.

This module provides functionality for identifying repeating units in polymers,
analyzing polymer fragments, and handling branched polymer topologies.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass
from enum import Enum
import logging
from collections import Counter, defaultdict

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors, Descriptors, AllChem
    from rdkit.Chem.rdMolAlign import AlignMol
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available. Some repeating unit analysis features will be limited.")

from ..core.exceptions import ChemiAccelError, InvalidMoleculeError
from ..utils.rdkit_integration import parse_smiles_safe


class PolymerTopology(Enum):
    """Types of polymer topologies."""
    LINEAR = "linear"
    BRANCHED = "branched"
    CROSSLINKED = "crosslinked"
    CYCLIC = "cyclic"
    STAR = "star"
    COMB = "comb"
    DENDRITIC = "dendritic"


class RepeatingUnitType(Enum):
    """Types of repeating units."""
    MONOMER = "monomer"
    DIMER = "dimer"
    TRIMER = "trimer"
    OLIGOMER = "oligomer"
    IRREGULAR = "irregular"


@dataclass
class RepeatingUnit:
    """Represents a repeating unit in a polymer."""
    smiles: str
    frequency: int
    molecular_weight: float
    unit_type: RepeatingUnitType
    connectivity_points: List[int]
    stereochemistry: Optional[str] = None


@dataclass
class PolymerFragment:
    """Represents a fragment of a polymer structure."""
    fragment_smiles: str
    fragment_type: str
    molecular_weight: float
    atom_indices: List[int]
    connectivity: Dict[str, List[int]]
    properties: Dict[str, float]


@dataclass
class BranchPoint:
    """Represents a branching point in a polymer."""
    atom_index: int
    branch_degree: int
    branch_types: List[str]
    branch_lengths: List[int]


@dataclass
class PolymerStructureAnalysis:
    """Complete structural analysis of a polymer."""
    topology: PolymerTopology
    repeating_units: List[RepeatingUnit]
    fragments: List[PolymerFragment]
    branch_points: List[BranchPoint]
    molecular_weight: float
    degree_of_polymerization: int
    structural_regularity: float


class RepeatingUnitIdentifier:
    """Identifies repeating units in polymer structures."""
    
    def __init__(self):
        """Initialize the repeating unit identifier."""
        if not RDKIT_AVAILABLE:
            raise ChemiAccelError("RDKit is required for repeating unit identification")
        
        # Define common polymer patterns
        self.common_patterns = {
            'polyethylene': '[CH2][CH2]',
            'polypropylene': '[CH2][CH]([CH3])',
            'polystyrene': '[CH2][CH]([c])',
            'polyvinyl_chloride': '[CH2][CH]([Cl])',
            'polyethylene_terephthalate': '[CH2][CH2][O][C](=O)[c][c][C](=O)[O]',
            'nylon_6': '[NH][CH2][CH2][CH2][CH2][CH2][C](=O)',
            'polyurethane': '[NH][C](=O)[O]',
            'polyester': '[C](=O)[O]',
            'polyether': '[O][CH2][CH2]',
            'polyamide': '[NH][C](=O)'
        }
    
    def identify_repeating_units(self, polymer_smiles: str) -> List[RepeatingUnit]:
        """
        Identify repeating units in a polymer structure.
        
        Args:
            polymer_smiles: SMILES of the polymer
            
        Returns:
            List of identified repeating units
        """
        mol = parse_smiles_safe(polymer_smiles)
        if mol is None:
            raise InvalidMoleculeError(polymer_smiles, "Invalid polymer SMILES")
        
        # Try pattern-based identification first
        pattern_units = self._identify_by_patterns(mol)
        
        # Try structural decomposition
        structural_units = self._identify_by_structure(mol)
        
        # Combine and rank results
        all_units = pattern_units + structural_units
        
        # Remove duplicates and rank by frequency
        unique_units = self._deduplicate_units(all_units)
        
        return sorted(unique_units, key=lambda x: x.frequency, reverse=True)
    
    def _identify_by_patterns(self, mol: Chem.Mol) -> List[RepeatingUnit]:
        """Identify repeating units using known patterns."""
        identified_units = []
        
        for pattern_name, pattern_smarts in self.common_patterns.items():
            try:
                pattern_mol = Chem.MolFromSmarts(pattern_smarts)
                if pattern_mol is None:
                    continue
                
                matches = mol.GetSubstructMatches(pattern_mol)
                if matches:
                    # Calculate properties
                    pattern_mol_h = Chem.AddHs(pattern_mol)
                    mw = Descriptors.MolWt(pattern_mol_h)
                    
                    # Determine unit type based on size
                    num_heavy_atoms = pattern_mol.GetNumHeavyAtoms()
                    if num_heavy_atoms <= 5:
                        unit_type = RepeatingUnitType.MONOMER
                    elif num_heavy_atoms <= 10:
                        unit_type = RepeatingUnitType.DIMER
                    elif num_heavy_atoms <= 15:
                        unit_type = RepeatingUnitType.TRIMER
                    else:
                        unit_type = RepeatingUnitType.OLIGOMER
                    
                    # Get connectivity points (simplified)
                    connectivity_points = self._find_connectivity_points(pattern_mol)
                    
                    unit = RepeatingUnit(
                        smiles=Chem.MolToSmiles(pattern_mol),
                        frequency=len(matches),
                        molecular_weight=mw,
                        unit_type=unit_type,
                        connectivity_points=connectivity_points
                    )
                    identified_units.append(unit)
                    
            except Exception as e:
                logging.warning(f"Error matching pattern {pattern_name}: {e}")
                continue
        
        return identified_units
    
    def _identify_by_structure(self, mol: Chem.Mol) -> List[RepeatingUnit]:
        """Identify repeating units by structural analysis."""
        identified_units = []
        
        try:
            # Find potential repeating substructures
            substructures = self._find_repeating_substructures(mol)
            
            for substruct_smiles, frequency in substructures.items():
                if frequency < 2:  # Must repeat at least twice
                    continue
                
                substruct_mol = parse_smiles_safe(substruct_smiles)
                if substruct_mol is None:
                    continue
                
                # Calculate properties
                mw = Descriptors.MolWt(substruct_mol)
                
                # Determine unit type
                num_heavy_atoms = substruct_mol.GetNumHeavyAtoms()
                if num_heavy_atoms <= 3:
                    unit_type = RepeatingUnitType.MONOMER
                elif num_heavy_atoms <= 8:
                    unit_type = RepeatingUnitType.DIMER
                elif num_heavy_atoms <= 15:
                    unit_type = RepeatingUnitType.TRIMER
                else:
                    unit_type = RepeatingUnitType.OLIGOMER
                
                connectivity_points = self._find_connectivity_points(substruct_mol)
                
                unit = RepeatingUnit(
                    smiles=substruct_smiles,
                    frequency=frequency,
                    molecular_weight=mw,
                    unit_type=unit_type,
                    connectivity_points=connectivity_points
                )
                identified_units.append(unit)
                
        except Exception as e:
            logging.warning(f"Error in structural analysis: {e}")
        
        return identified_units
    
    def _find_repeating_substructures(self, mol: Chem.Mol) -> Dict[str, int]:
        """Find repeating substructures in the molecule."""
        substructures = defaultdict(int)
        
        # Generate fragments of different sizes
        for fragment_size in range(2, min(10, mol.GetNumAtoms() // 2)):
            fragments = self._generate_fragments(mol, fragment_size)
            
            for fragment_smiles in fragments:
                substructures[fragment_smiles] += 1
        
        # Filter out non-repeating fragments
        repeating_substructures = {
            smiles: count for smiles, count in substructures.items() 
            if count >= 2
        }
        
        return repeating_substructures
    
    def _generate_fragments(self, mol: Chem.Mol, fragment_size: int) -> List[str]:
        """Generate molecular fragments of specified size."""
        fragments = []
        
        try:
            # Use RDKit's fragment generation
            from rdkit.Chem import BRICS
            
            # Generate BRICS fragments
            brics_fragments = BRICS.BRICSDecompose(mol)
            
            for fragment_smiles in brics_fragments:
                fragment_mol = parse_smiles_safe(fragment_smiles)
                if fragment_mol and fragment_mol.GetNumHeavyAtoms() == fragment_size:
                    fragments.append(fragment_smiles)
            
        except Exception as e:
            logging.warning(f"BRICS fragmentation failed: {e}")
            
            # Fallback: simple path-based fragmentation
            fragments = self._generate_path_fragments(mol, fragment_size)
        
        return fragments
    
    def _generate_path_fragments(self, mol: Chem.Mol, fragment_size: int) -> List[str]:
        """Generate fragments based on molecular paths."""
        fragments = []
        
        # Find all paths of specified length
        for start_atom in range(mol.GetNumAtoms()):
            paths = self._find_paths_from_atom(mol, start_atom, fragment_size)
            
            for path in paths:
                try:
                    # Create fragment from path
                    fragment_mol = self._create_fragment_from_path(mol, path)
                    if fragment_mol:
                        fragment_smiles = Chem.MolToSmiles(fragment_mol)
                        fragments.append(fragment_smiles)
                except:
                    continue
        
        return fragments
    
    def _find_paths_from_atom(self, mol: Chem.Mol, start_atom: int, length: int) -> List[List[int]]:
        """Find all paths of specified length starting from an atom."""
        paths = []
        
        def dfs(current_path, visited):
            if len(current_path) == length:
                paths.append(current_path[:])
                return
            
            current_atom = current_path[-1]
            atom = mol.GetAtomWithIdx(current_atom)
            
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    current_path.append(neighbor_idx)
                    dfs(current_path, visited)
                    current_path.pop()
                    visited.remove(neighbor_idx)
        
        visited = {start_atom}
        dfs([start_atom], visited)
        
        return paths
    
    def _create_fragment_from_path(self, mol: Chem.Mol, path: List[int]) -> Optional[Chem.Mol]:
        """Create a molecular fragment from an atom path."""
        try:
            # Create a new molecule with only the atoms in the path
            fragment_mol = Chem.RWMol()
            atom_map = {}
            
            # Add atoms
            for i, atom_idx in enumerate(path):
                atom = mol.GetAtomWithIdx(atom_idx)
                new_atom = Chem.Atom(atom.GetSymbol())
                new_atom.SetFormalCharge(atom.GetFormalCharge())
                fragment_mol.AddAtom(new_atom)
                atom_map[atom_idx] = i
            
            # Add bonds
            for i in range(len(path) - 1):
                atom1_idx = path[i]
                atom2_idx = path[i + 1]
                
                bond = mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
                if bond:
                    fragment_mol.AddBond(
                        atom_map[atom1_idx],
                        atom_map[atom2_idx],
                        bond.GetBondType()
                    )
            
            # Sanitize and return
            Chem.SanitizeMol(fragment_mol)
            return fragment_mol.GetMol()
            
        except Exception as e:
            logging.warning(f"Error creating fragment: {e}")
            return None
    
    def _find_connectivity_points(self, mol: Chem.Mol) -> List[int]:
        """Find potential connectivity points in a repeating unit."""
        connectivity_points = []
        
        for atom in mol.GetAtoms():
            # Look for atoms that could connect to other units
            if (atom.GetDegree() < atom.GetTotalValence() or
                atom.GetSymbol() in ['C', 'N', 'O'] and atom.GetDegree() <= 2):
                connectivity_points.append(atom.GetIdx())
        
        return connectivity_points
    
    def _deduplicate_units(self, units: List[RepeatingUnit]) -> List[RepeatingUnit]:
        """Remove duplicate repeating units."""
        unique_units = {}
        
        for unit in units:
            # Use canonical SMILES as key
            try:
                mol = parse_smiles_safe(unit.smiles)
                if mol:
                    canonical_smiles = Chem.MolToSmiles(mol)
                    
                    if canonical_smiles in unique_units:
                        # Combine frequencies
                        unique_units[canonical_smiles].frequency += unit.frequency
                    else:
                        unit.smiles = canonical_smiles
                        unique_units[canonical_smiles] = unit
            except:
                continue
        
        return list(unique_units.values())


class PolymerFragmentAnalyzer:
    """Analyzes polymer fragments and their properties."""
    
    def __init__(self):
        """Initialize the polymer fragment analyzer."""
        if not RDKIT_AVAILABLE:
            raise ChemiAccelError("RDKit is required for polymer fragment analysis")
    
    def analyze_fragments(self, polymer_smiles: str) -> List[PolymerFragment]:
        """
        Analyze polymer fragments.
        
        Args:
            polymer_smiles: SMILES of the polymer
            
        Returns:
            List of PolymerFragment objects
        """
        mol = parse_smiles_safe(polymer_smiles)
        if mol is None:
            raise InvalidMoleculeError(polymer_smiles, "Invalid polymer SMILES")
        
        fragments = []
        
        # Identify different types of fragments
        backbone_fragments = self._identify_backbone_fragments(mol)
        side_chain_fragments = self._identify_side_chain_fragments(mol)
        end_group_fragments = self._identify_end_group_fragments(mol)
        
        fragments.extend(backbone_fragments)
        fragments.extend(side_chain_fragments)
        fragments.extend(end_group_fragments)
        
        return fragments
    
    def _identify_backbone_fragments(self, mol: Chem.Mol) -> List[PolymerFragment]:
        """Identify backbone fragments."""
        fragments = []
        
        try:
            # Find the main chain (longest path)
            main_chain = self._find_main_chain(mol)
            
            if main_chain:
                # Create backbone fragment
                backbone_mol = self._create_fragment_from_atoms(mol, main_chain)
                if backbone_mol:
                    fragment = PolymerFragment(
                        fragment_smiles=Chem.MolToSmiles(backbone_mol),
                        fragment_type="backbone",
                        molecular_weight=Descriptors.MolWt(backbone_mol),
                        atom_indices=main_chain,
                        connectivity=self._analyze_connectivity(mol, main_chain),
                        properties=self._calculate_fragment_properties(backbone_mol)
                    )
                    fragments.append(fragment)
                    
        except Exception as e:
            logging.warning(f"Error identifying backbone fragments: {e}")
        
        return fragments
    
    def _identify_side_chain_fragments(self, mol: Chem.Mol) -> List[PolymerFragment]:
        """Identify side chain fragments."""
        fragments = []
        
        try:
            main_chain = self._find_main_chain(mol)
            main_chain_set = set(main_chain) if main_chain else set()
            
            # Find side chains
            for atom_idx in main_chain:
                atom = mol.GetAtomWithIdx(atom_idx)
                
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    
                    if neighbor_idx not in main_chain_set:
                        # This is a side chain
                        side_chain = self._trace_side_chain(mol, neighbor_idx, main_chain_set)
                        
                        if len(side_chain) > 1:  # Only consider non-trivial side chains
                            side_chain_mol = self._create_fragment_from_atoms(mol, side_chain)
                            if side_chain_mol:
                                fragment = PolymerFragment(
                                    fragment_smiles=Chem.MolToSmiles(side_chain_mol),
                                    fragment_type="side_chain",
                                    molecular_weight=Descriptors.MolWt(side_chain_mol),
                                    atom_indices=side_chain,
                                    connectivity=self._analyze_connectivity(mol, side_chain),
                                    properties=self._calculate_fragment_properties(side_chain_mol)
                                )
                                fragments.append(fragment)
                                
        except Exception as e:
            logging.warning(f"Error identifying side chain fragments: {e}")
        
        return fragments
    
    def _identify_end_group_fragments(self, mol: Chem.Mol) -> List[PolymerFragment]:
        """Identify end group fragments."""
        fragments = []
        
        try:
            # Find terminal atoms
            terminal_atoms = []
            for atom in mol.GetAtoms():
                if atom.GetDegree() == 1:
                    terminal_atoms.append(atom.GetIdx())
            
            # Create end group fragments
            for terminal_atom in terminal_atoms:
                end_group = self._trace_end_group(mol, terminal_atom)
                
                if len(end_group) > 1:
                    end_group_mol = self._create_fragment_from_atoms(mol, end_group)
                    if end_group_mol:
                        fragment = PolymerFragment(
                            fragment_smiles=Chem.MolToSmiles(end_group_mol),
                            fragment_type="end_group",
                            molecular_weight=Descriptors.MolWt(end_group_mol),
                            atom_indices=end_group,
                            connectivity=self._analyze_connectivity(mol, end_group),
                            properties=self._calculate_fragment_properties(end_group_mol)
                        )
                        fragments.append(fragment)
                        
        except Exception as e:
            logging.warning(f"Error identifying end group fragments: {e}")
        
        return fragments
    
    def _find_main_chain(self, mol: Chem.Mol) -> List[int]:
        """Find the main polymer chain (longest path)."""
        longest_path = []
        
        # Try all possible starting points
        for start_atom in range(mol.GetNumAtoms()):
            path = self._find_longest_path_from_atom(mol, start_atom)
            if len(path) > len(longest_path):
                longest_path = path
        
        return longest_path
    
    def _find_longest_path_from_atom(self, mol: Chem.Mol, start_atom: int) -> List[int]:
        """Find the longest path starting from a specific atom."""
        longest_path = []
        
        def dfs(current_path, visited):
            nonlocal longest_path
            
            if len(current_path) > len(longest_path):
                longest_path = current_path[:]
            
            current_atom = current_path[-1]
            atom = mol.GetAtomWithIdx(current_atom)
            
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    current_path.append(neighbor_idx)
                    dfs(current_path, visited)
                    current_path.pop()
                    visited.remove(neighbor_idx)
        
        visited = {start_atom}
        dfs([start_atom], visited)
        
        return longest_path
    
    def _trace_side_chain(self, mol: Chem.Mol, start_atom: int, main_chain: Set[int]) -> List[int]:
        """Trace a side chain from a starting atom."""
        side_chain = []
        visited = set()
        
        def dfs(atom_idx):
            if atom_idx in visited or atom_idx in main_chain:
                return
            
            visited.add(atom_idx)
            side_chain.append(atom_idx)
            
            atom = mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                dfs(neighbor_idx)
        
        dfs(start_atom)
        return side_chain
    
    def _trace_end_group(self, mol: Chem.Mol, terminal_atom: int) -> List[int]:
        """Trace an end group from a terminal atom."""
        end_group = [terminal_atom]
        
        # Follow the chain for a few atoms to define the end group
        current_atom = terminal_atom
        visited = {terminal_atom}
        
        for _ in range(3):  # Limit end group size
            atom = mol.GetAtomWithIdx(current_atom)
            next_atom = None
            
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx not in visited:
                    next_atom = neighbor_idx
                    break
            
            if next_atom is None:
                break
            
            end_group.append(next_atom)
            visited.add(next_atom)
            current_atom = next_atom
        
        return end_group
    
    def _create_fragment_from_atoms(self, mol: Chem.Mol, atom_indices: List[int]) -> Optional[Chem.Mol]:
        """Create a molecular fragment from atom indices."""
        try:
            fragment_mol = Chem.RWMol()
            atom_map = {}
            
            # Add atoms
            for i, atom_idx in enumerate(atom_indices):
                atom = mol.GetAtomWithIdx(atom_idx)
                new_atom = Chem.Atom(atom.GetSymbol())
                new_atom.SetFormalCharge(atom.GetFormalCharge())
                fragment_mol.AddAtom(new_atom)
                atom_map[atom_idx] = i
            
            # Add bonds
            for i, atom1_idx in enumerate(atom_indices):
                for j, atom2_idx in enumerate(atom_indices[i+1:], i+1):
                    bond = mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
                    if bond:
                        fragment_mol.AddBond(
                            atom_map[atom1_idx],
                            atom_map[atom2_idx],
                            bond.GetBondType()
                        )
            
            # Sanitize
            Chem.SanitizeMol(fragment_mol)
            return fragment_mol.GetMol()
            
        except Exception as e:
            logging.warning(f"Error creating fragment: {e}")
            return None
    
    def _analyze_connectivity(self, mol: Chem.Mol, atom_indices: List[int]) -> Dict[str, List[int]]:
        """Analyze connectivity of a fragment."""
        connectivity = {
            'internal_bonds': [],
            'external_connections': []
        }
        
        atom_set = set(atom_indices)
        
        for atom_idx in atom_indices:
            atom = mol.GetAtomWithIdx(atom_idx)
            
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                
                if neighbor_idx in atom_set:
                    # Internal bond
                    bond_idx = mol.GetBondBetweenAtoms(atom_idx, neighbor_idx).GetIdx()
                    if bond_idx not in connectivity['internal_bonds']:
                        connectivity['internal_bonds'].append(bond_idx)
                else:
                    # External connection
                    connectivity['external_connections'].append((atom_idx, neighbor_idx))
        
        return connectivity
    
    def _calculate_fragment_properties(self, fragment_mol: Chem.Mol) -> Dict[str, float]:
        """Calculate properties of a fragment."""
        properties = {}
        
        try:
            properties['molecular_weight'] = Descriptors.MolWt(fragment_mol)
            properties['logp'] = Descriptors.MolLogP(fragment_mol)
            properties['tpsa'] = Descriptors.TPSA(fragment_mol)
            properties['num_heavy_atoms'] = fragment_mol.GetNumHeavyAtoms()
            properties['num_rotatable_bonds'] = rdMolDescriptors.CalcNumRotatableBonds(fragment_mol)
            properties['num_aromatic_rings'] = rdMolDescriptors.CalcNumAromaticRings(fragment_mol)
            
        except Exception as e:
            logging.warning(f"Error calculating fragment properties: {e}")
            properties = {
                'molecular_weight': 0.0,
                'logp': 0.0,
                'tpsa': 0.0,
                'num_heavy_atoms': 0,
                'num_rotatable_bonds': 0,
                'num_aromatic_rings': 0
            }
        
        return properties


class BranchedPolymerHandler:
    """Handles analysis of branched polymer structures."""
    
    def __init__(self):
        """Initialize the branched polymer handler."""
        if not RDKIT_AVAILABLE:
            raise ChemiAccelError("RDKit is required for branched polymer analysis")
    
    def analyze_branching(self, polymer_smiles: str) -> PolymerStructureAnalysis:
        """
        Analyze branching in a polymer structure.
        
        Args:
            polymer_smiles: SMILES of the polymer
            
        Returns:
            PolymerStructureAnalysis object
        """
        mol = parse_smiles_safe(polymer_smiles)
        if mol is None:
            raise InvalidMoleculeError(polymer_smiles, "Invalid polymer SMILES")
        
        # Determine topology
        topology = self._determine_topology(mol)
        
        # Identify repeating units
        unit_identifier = RepeatingUnitIdentifier()
        repeating_units = unit_identifier.identify_repeating_units(polymer_smiles)
        
        # Analyze fragments
        fragment_analyzer = PolymerFragmentAnalyzer()
        fragments = fragment_analyzer.analyze_fragments(polymer_smiles)
        
        # Find branch points
        branch_points = self._find_branch_points(mol)
        
        # Calculate properties
        molecular_weight = Descriptors.MolWt(mol)
        degree_of_polymerization = self._estimate_degree_of_polymerization(mol, repeating_units)
        structural_regularity = self._calculate_structural_regularity(mol, repeating_units, branch_points)
        
        return PolymerStructureAnalysis(
            topology=topology,
            repeating_units=repeating_units,
            fragments=fragments,
            branch_points=branch_points,
            molecular_weight=molecular_weight,
            degree_of_polymerization=degree_of_polymerization,
            structural_regularity=structural_regularity
        )
    
    def _determine_topology(self, mol: Chem.Mol) -> PolymerTopology:
        """Determine the topology of the polymer."""
        # Count branch points
        branch_count = 0
        high_degree_atoms = 0
        
        for atom in mol.GetAtoms():
            degree = atom.GetDegree()
            if degree > 2:
                high_degree_atoms += 1
                if degree > 3:
                    branch_count += 1
        
        # Simple topology classification
        if branch_count == 0:
            if high_degree_atoms == 0:
                return PolymerTopology.LINEAR
            else:
                # Check for cycles
                if mol.GetRingInfo().NumRings() > 0:
                    return PolymerTopology.CYCLIC
                else:
                    return PolymerTopology.LINEAR
        elif branch_count < 3:
            return PolymerTopology.BRANCHED
        elif self._is_star_topology(mol):
            return PolymerTopology.STAR
        elif self._is_comb_topology(mol):
            return PolymerTopology.COMB
        else:
            return PolymerTopology.DENDRITIC
    
    def _is_star_topology(self, mol: Chem.Mol) -> bool:
        """Check if the polymer has a star topology."""
        # Look for a central atom with multiple branches
        for atom in mol.GetAtoms():
            if atom.GetDegree() >= 3:
                # Check if this could be a star center
                branch_lengths = []
                for neighbor in atom.GetNeighbors():
                    branch_length = self._calculate_branch_length(mol, neighbor.GetIdx(), atom.GetIdx())
                    branch_lengths.append(branch_length)
                
                # Star topology has multiple branches of similar length from one center
                if len(branch_lengths) >= 3 and max(branch_lengths) > 3:
                    return True
        
        return False
    
    def _is_comb_topology(self, mol: Chem.Mol) -> bool:
        """Check if the polymer has a comb topology."""
        # Look for a main chain with regular side branches
        main_chain = self._find_main_chain(mol)
        
        if len(main_chain) < 5:
            return False
        
        # Count side branches along the main chain
        side_branches = 0
        for atom_idx in main_chain:
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetDegree() > 2:
                side_branches += 1
        
        # Comb topology has regular branching along the main chain
        return side_branches >= 3 and side_branches / len(main_chain) > 0.2
    
    def _find_main_chain(self, mol: Chem.Mol) -> List[int]:
        """Find the main polymer chain."""
        # Use the same method as in PolymerFragmentAnalyzer
        fragment_analyzer = PolymerFragmentAnalyzer()
        return fragment_analyzer._find_main_chain(mol)
    
    def _find_branch_points(self, mol: Chem.Mol) -> List[BranchPoint]:
        """Find branching points in the polymer."""
        branch_points = []
        
        for atom in mol.GetAtoms():
            if atom.GetDegree() > 2:
                # This is a potential branch point
                branches = []
                branch_lengths = []
                
                for neighbor in atom.GetNeighbors():
                    branch_length = self._calculate_branch_length(mol, neighbor.GetIdx(), atom.GetIdx())
                    branch_type = self._classify_branch_type(mol, neighbor.GetIdx(), atom.GetIdx())
                    
                    branches.append(branch_type)
                    branch_lengths.append(branch_length)
                
                branch_point = BranchPoint(
                    atom_index=atom.GetIdx(),
                    branch_degree=atom.GetDegree(),
                    branch_types=branches,
                    branch_lengths=branch_lengths
                )
                branch_points.append(branch_point)
        
        return branch_points
    
    def _calculate_branch_length(self, mol: Chem.Mol, start_atom: int, exclude_atom: int) -> int:
        """Calculate the length of a branch."""
        visited = {exclude_atom}
        length = 0
        current_atoms = [start_atom]
        
        while current_atoms:
            next_atoms = []
            length += 1
            
            for atom_idx in current_atoms:
                if atom_idx in visited:
                    continue
                
                visited.add(atom_idx)
                atom = mol.GetAtomWithIdx(atom_idx)
                
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx not in visited:
                        next_atoms.append(neighbor_idx)
            
            current_atoms = next_atoms
            
            if length > 20:  # Prevent infinite loops
                break
        
        return length
    
    def _classify_branch_type(self, mol: Chem.Mol, start_atom: int, exclude_atom: int) -> str:
        """Classify the type of branch."""
        # Simple classification based on branch composition
        visited = {exclude_atom}
        branch_atoms = []
        
        def collect_branch_atoms(atom_idx):
            if atom_idx in visited:
                return
            
            visited.add(atom_idx)
            branch_atoms.append(atom_idx)
            
            atom = mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                collect_branch_atoms(neighbor_idx)
        
        collect_branch_atoms(start_atom)
        
        if not branch_atoms:
            return "terminal"
        
        # Analyze branch composition
        has_aromatic = any(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in branch_atoms)
        has_heteroatom = any(mol.GetAtomWithIdx(idx).GetSymbol() != 'C' for idx in branch_atoms)
        
        if has_aromatic:
            return "aromatic"
        elif has_heteroatom:
            return "heteroatom"
        elif len(branch_atoms) == 1:
            return "methyl"
        elif len(branch_atoms) <= 3:
            return "short_alkyl"
        else:
            return "long_alkyl"
    
    def _estimate_degree_of_polymerization(self, mol: Chem.Mol, repeating_units: List[RepeatingUnit]) -> int:
        """Estimate the degree of polymerization."""
        if not repeating_units:
            return 1
        
        # Use the most frequent repeating unit
        main_unit = max(repeating_units, key=lambda x: x.frequency)
        
        # Estimate based on molecular weight ratio
        polymer_mw = Descriptors.MolWt(mol)
        unit_mw = main_unit.molecular_weight
        
        if unit_mw > 0:
            return max(1, int(polymer_mw / unit_mw))
        else:
            return 1
    
    def _calculate_structural_regularity(self, mol: Chem.Mol, repeating_units: List[RepeatingUnit], 
                                       branch_points: List[BranchPoint]) -> float:
        """Calculate structural regularity score."""
        regularity_score = 1.0
        
        # Penalize for irregular branching
        if branch_points:
            branch_degrees = [bp.branch_degree for bp in branch_points]
            degree_variance = np.var(branch_degrees) if len(branch_degrees) > 1 else 0
            regularity_score -= min(0.3, degree_variance / 10.0)
        
        # Reward for regular repeating units
        if repeating_units:
            total_frequency = sum(unit.frequency for unit in repeating_units)
            max_frequency = max(unit.frequency for unit in repeating_units)
            
            if total_frequency > 0:
                regularity_bonus = max_frequency / total_frequency
                regularity_score = min(1.0, regularity_score + regularity_bonus * 0.2)
        
        return max(0.0, min(1.0, regularity_score))