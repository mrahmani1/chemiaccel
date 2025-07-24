"""
Polymer-specific descriptor calculations.

This module provides specialized descriptors for polymer chains including
persistence length, radius of gyration, glass transition temperature predictors,
and mechanical/processing property descriptors.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import math

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
    from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule, MMFFOptimizeMolecule
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available. Some polymer descriptor features will be limited.")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logging.warning("CuPy not available. GPU acceleration will be disabled.")

from ..core.exceptions import ChemiAccelError, InvalidMoleculeError
from ..utils.rdkit_integration import parse_smiles_safe


@dataclass
class PolymerDescriptors:
    """Container for polymer-specific descriptors."""
    # Chain flexibility descriptors
    persistence_length: Optional[float] = None
    radius_of_gyration: Optional[float] = None
    end_to_end_distance: Optional[float] = None
    characteristic_ratio: Optional[float] = None
    kuhn_length: Optional[float] = None
    
    # Thermal property predictors
    glass_transition_temperature: Optional[float] = None
    melting_temperature: Optional[float] = None
    thermal_decomposition_temperature: Optional[float] = None
    
    # Mechanical property descriptors
    elastic_modulus_predictor: Optional[float] = None
    brittleness_index: Optional[float] = None
    toughness_index: Optional[float] = None
    hardness_predictor: Optional[float] = None
    
    # Processing property descriptors
    viscosity_predictor: Optional[float] = None
    solubility_parameter: Optional[float] = None
    melt_flow_index_predictor: Optional[float] = None
    processability_index: Optional[float] = None
    
    # Additional polymer properties
    crystallinity_index: Optional[float] = None
    chain_entanglement_density: Optional[float] = None
    free_volume_fraction: Optional[float] = None


class PolymerChainFlexibilityCalculator:
    """Calculator for polymer chain flexibility descriptors."""
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize the flexibility calculator.
        
        Args:
            use_gpu: Whether to use GPU acceleration when available
        """
        if not RDKIT_AVAILABLE:
            raise ChemiAccelError("RDKit is required for polymer flexibility calculations")
        
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        
        # Physical constants
        self.BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
        self.AVOGADRO_NUMBER = 6.02214076e23
        self.GAS_CONSTANT = 8.314  # J/(mol·K)
    
    def calculate_persistence_length(self, polymer_smiles: str, temperature: float = 298.15) -> float:
        """
        Calculate persistence length of a polymer chain.
        
        Args:
            polymer_smiles: SMILES of the polymer chain
            temperature: Temperature in Kelvin
            
        Returns:
            Persistence length in Angstroms
        """
        mol = parse_smiles_safe(polymer_smiles)
        if mol is None:
            raise InvalidMoleculeError(polymer_smiles, "Invalid polymer SMILES")
        
        # Generate conformers for analysis
        mol = Chem.AddHs(mol)
        conformer_ids = self._generate_conformers(mol, num_conformers=20)
        
        if not conformer_ids:
            return 0.0
        
        # Calculate bond correlation function
        bond_correlations = []
        
        for conf_id in conformer_ids:
            try:
                correlation = self._calculate_bond_correlation_function(mol, conf_id)
                bond_correlations.append(correlation)
            except Exception as e:
                logging.warning(f"Error calculating bond correlation for conformer {conf_id}: {e}")
                continue
        
        if not bond_correlations:
            return 0.0
        
        # Average over conformers
        avg_correlation = np.mean(bond_correlations, axis=0)
        
        # Fit exponential decay to get persistence length
        persistence_length = self._fit_persistence_length(avg_correlation)
        
        return persistence_length
    
    def calculate_radius_of_gyration(self, polymer_smiles: str) -> float:
        """
        Calculate radius of gyration for a polymer chain.
        
        Args:
            polymer_smiles: SMILES of the polymer chain
            
        Returns:
            Radius of gyration in Angstroms
        """
        mol = parse_smiles_safe(polymer_smiles)
        if mol is None:
            raise InvalidMoleculeError(polymer_smiles, "Invalid polymer SMILES")
        
        # Generate 3D structure
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        
        conf = mol.GetConformer()
        
        # Get atomic positions
        positions = []
        masses = []
        
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            positions.append([pos.x, pos.y, pos.z])
            
            atom = mol.GetAtomWithIdx(i)
            masses.append(atom.GetMass())
        
        positions = np.array(positions)
        masses = np.array(masses)
        
        if self.use_gpu:
            return self._calculate_rg_gpu(positions, masses)
        else:
            return self._calculate_rg_cpu(positions, masses)
    
    def calculate_end_to_end_distance(self, polymer_smiles: str) -> float:
        """
        Calculate end-to-end distance for a polymer chain.
        
        Args:
            polymer_smiles: SMILES of the polymer chain
            
        Returns:
            End-to-end distance in Angstroms
        """
        mol = parse_smiles_safe(polymer_smiles)
        if mol is None:
            raise InvalidMoleculeError(polymer_smiles, "Invalid polymer SMILES")
        
        # Generate 3D structure
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        
        # Find terminal atoms
        terminal_atoms = []
        for atom in mol.GetAtoms():
            if atom.GetDegree() == 1:
                terminal_atoms.append(atom.GetIdx())
        
        if len(terminal_atoms) < 2:
            return 0.0
        
        # Calculate distance between terminal atoms
        conf = mol.GetConformer()
        pos1 = conf.GetAtomPosition(terminal_atoms[0])
        pos2 = conf.GetAtomPosition(terminal_atoms[-1])
        
        distance = math.sqrt(
            (pos1.x - pos2.x)**2 + 
            (pos1.y - pos2.y)**2 + 
            (pos1.z - pos2.z)**2
        )
        
        return distance    

    def calculate_characteristic_ratio(self, polymer_smiles: str) -> float:
        """
        Calculate characteristic ratio (Cn) for a polymer chain.
        
        Args:
            polymer_smiles: SMILES of the polymer chain
            
        Returns:
            Characteristic ratio (dimensionless)
        """
        mol = parse_smiles_safe(polymer_smiles)
        if mol is None:
            raise InvalidMoleculeError(polymer_smiles, "Invalid polymer SMILES")
        
        # Calculate mean-square end-to-end distance
        end_to_end_distances = []
        
        # Generate multiple conformers
        mol = Chem.AddHs(mol)
        conformer_ids = self._generate_conformers(mol, num_conformers=50)
        
        for conf_id in conformer_ids:
            try:
                distance = self._calculate_end_to_end_distance_conformer(mol, conf_id)
                end_to_end_distances.append(distance)
            except:
                continue
        
        if not end_to_end_distances:
            return 1.0
        
        mean_square_distance = np.mean([d**2 for d in end_to_end_distances])
        
        # Calculate number of bonds and average bond length
        num_bonds = mol.GetNumBonds()
        avg_bond_length = self._calculate_average_bond_length(mol)
        
        if num_bonds == 0 or avg_bond_length == 0:
            return 1.0
        
        # Characteristic ratio = <r²> / (n * l²)
        characteristic_ratio = mean_square_distance / (num_bonds * avg_bond_length**2)
        
        return characteristic_ratio
    
    def _generate_conformers(self, mol: Chem.Mol, num_conformers: int = 20) -> List[int]:
        """Generate conformers for the molecule."""
        try:
            conf_ids = AllChem.EmbedMultipleConfs(
                mol, 
                numConfs=num_conformers,
                randomSeed=42,
                useExpTorsionAnglePrefs=True
            )
            
            # Optimize conformers
            for conf_id in conf_ids:
                try:
                    AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
                except:
                    continue
            
            return list(conf_ids)
            
        except Exception as e:
            logging.warning(f"Conformer generation failed: {e}")
            return [0] if mol.GetNumConformers() > 0 else []
    
    def _calculate_bond_correlation_function(self, mol: Chem.Mol, conf_id: int) -> np.ndarray:
        """Calculate bond-bond correlation function."""
        conf = mol.GetConformer(conf_id)
        
        # Get bond vectors
        bond_vectors = []
        for bond in mol.GetBonds():
            atom1_idx = bond.GetBeginAtomIdx()
            atom2_idx = bond.GetEndAtomIdx()
            
            pos1 = conf.GetAtomPosition(atom1_idx)
            pos2 = conf.GetAtomPosition(atom2_idx)
            
            vector = np.array([pos2.x - pos1.x, pos2.y - pos1.y, pos2.z - pos1.z])
            bond_vectors.append(vector / np.linalg.norm(vector))  # Normalize
        
        bond_vectors = np.array(bond_vectors)
        
        # Calculate correlation function
        num_bonds = len(bond_vectors)
        correlations = []
        
        for i in range(num_bonds):
            correlation_sum = 0
            count = 0
            
            for j in range(i, min(i + 20, num_bonds)):  # Limit range for efficiency
                if i != j:
                    correlation = np.dot(bond_vectors[i], bond_vectors[j])
                    correlation_sum += correlation
                    count += 1
            
            if count > 0:
                correlations.append(correlation_sum / count)
            else:
                correlations.append(0.0)
        
        return np.array(correlations)
    
    def _fit_persistence_length(self, correlation_function: np.ndarray) -> float:
        """Fit persistence length from correlation function."""
        if len(correlation_function) < 3:
            return 1.0
        
        # Simple exponential fit: C(n) = exp(-n*l/lp)
        # where l is bond length and lp is persistence length
        
        x = np.arange(len(correlation_function))
        y = np.maximum(correlation_function, 1e-10)  # Avoid log(0)
        
        try:
            # Linear fit to log(correlation)
            log_y = np.log(y)
            coeffs = np.polyfit(x, log_y, 1)
            
            # Persistence length (in units of bond length)
            if coeffs[0] < 0:
                persistence_length = -1.0 / coeffs[0]
            else:
                persistence_length = 1.0
            
            # Convert to Angstroms (assuming average bond length ~1.5 Å)
            return persistence_length * 1.5
            
        except Exception as e:
            logging.warning(f"Error fitting persistence length: {e}")
            return 1.0
    
    def _calculate_rg_cpu(self, positions: np.ndarray, masses: np.ndarray) -> float:
        """Calculate radius of gyration on CPU."""
        # Calculate center of mass
        total_mass = np.sum(masses)
        center_of_mass = np.sum(positions * masses[:, np.newaxis], axis=0) / total_mass
        
        # Calculate radius of gyration
        rg_squared = np.sum(masses * np.sum((positions - center_of_mass)**2, axis=1)) / total_mass
        
        return np.sqrt(rg_squared)
    
    def _calculate_rg_gpu(self, positions: np.ndarray, masses: np.ndarray) -> float:
        """Calculate radius of gyration on GPU."""
        if not CUPY_AVAILABLE:
            return self._calculate_rg_cpu(positions, masses)
        
        # Transfer to GPU
        positions_gpu = cp.asarray(positions)
        masses_gpu = cp.asarray(masses)
        
        # Calculate center of mass
        total_mass = cp.sum(masses_gpu)
        center_of_mass = cp.sum(positions_gpu * masses_gpu[:, cp.newaxis], axis=0) / total_mass
        
        # Calculate radius of gyration
        rg_squared = cp.sum(masses_gpu * cp.sum((positions_gpu - center_of_mass)**2, axis=1)) / total_mass
        
        return float(cp.sqrt(rg_squared))
    
    def _calculate_end_to_end_distance_conformer(self, mol: Chem.Mol, conf_id: int) -> float:
        """Calculate end-to-end distance for a specific conformer."""
        conf = mol.GetConformer(conf_id)
        
        # Find terminal atoms
        terminal_atoms = []
        for atom in mol.GetAtoms():
            if atom.GetDegree() == 1:
                terminal_atoms.append(atom.GetIdx())
        
        if len(terminal_atoms) < 2:
            return 0.0
        
        # Calculate distance between first and last terminal atoms
        pos1 = conf.GetAtomPosition(terminal_atoms[0])
        pos2 = conf.GetAtomPosition(terminal_atoms[-1])
        
        distance = math.sqrt(
            (pos1.x - pos2.x)**2 + 
            (pos1.y - pos2.y)**2 + 
            (pos1.z - pos2.z)**2
        )
        
        return distance
    
    def _calculate_average_bond_length(self, mol: Chem.Mol) -> float:
        """Calculate average bond length in the molecule."""
        if mol.GetNumConformers() == 0:
            return 1.5  # Default C-C bond length
        
        conf = mol.GetConformer()
        bond_lengths = []
        
        for bond in mol.GetBonds():
            atom1_idx = bond.GetBeginAtomIdx()
            atom2_idx = bond.GetEndAtomIdx()
            
            pos1 = conf.GetAtomPosition(atom1_idx)
            pos2 = conf.GetAtomPosition(atom2_idx)
            
            length = math.sqrt(
                (pos1.x - pos2.x)**2 + 
                (pos1.y - pos2.y)**2 + 
                (pos1.z - pos2.z)**2
            )
            bond_lengths.append(length)
        
        return np.mean(bond_lengths) if bond_lengths else 1.5


class PolymerDescriptorCalculator:
    """Main calculator for all polymer-specific descriptors."""
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize the polymer descriptor calculator.
        
        Args:
            use_gpu: Whether to use GPU acceleration when available
        """
        self.flexibility_calculator = PolymerChainFlexibilityCalculator(use_gpu)
    
    def calculate_all_descriptors(self, polymer_smiles: str) -> PolymerDescriptors:
        """
        Calculate all polymer-specific descriptors.
        
        Args:
            polymer_smiles: SMILES of the polymer
            
        Returns:
            PolymerDescriptors object with all calculated values
        """
        try:
            # Chain flexibility descriptors
            persistence_length = self.flexibility_calculator.calculate_persistence_length(polymer_smiles)
            radius_of_gyration = self.flexibility_calculator.calculate_radius_of_gyration(polymer_smiles)
            end_to_end_distance = self.flexibility_calculator.calculate_end_to_end_distance(polymer_smiles)
            characteristic_ratio = self.flexibility_calculator.calculate_characteristic_ratio(polymer_smiles)
            kuhn_length = persistence_length * 2  # Approximation
            
            # Simplified thermal properties (basic implementations)
            glass_transition_temperature = self._predict_glass_transition_temperature(polymer_smiles)
            melting_temperature = glass_transition_temperature + 100.0  # Simple approximation
            thermal_decomposition_temperature = glass_transition_temperature + 200.0
            
            # Simplified mechanical properties
            elastic_modulus_predictor = self._predict_elastic_modulus(polymer_smiles)
            brittleness_index = self._predict_brittleness_index(polymer_smiles)
            toughness_index = 1.0 - brittleness_index  # Simple inverse relationship
            hardness_predictor = elastic_modulus_predictor * 0.1
            
            # Simplified processing properties
            viscosity_predictor = self._predict_viscosity(polymer_smiles)
            solubility_parameter = self._predict_solubility_parameter(polymer_smiles)
            melt_flow_index_predictor = 1000.0 / viscosity_predictor if viscosity_predictor > 0 else 1.0
            processability_index = self._predict_processability_index(polymer_smiles)
            
            # Additional properties
            crystallinity_index = self._calculate_crystallinity_index(polymer_smiles)
            chain_entanglement_density = self._calculate_chain_entanglement_density(polymer_smiles)
            free_volume_fraction = self._calculate_free_volume_fraction(polymer_smiles)
            
            return PolymerDescriptors(
                persistence_length=persistence_length,
                radius_of_gyration=radius_of_gyration,
                end_to_end_distance=end_to_end_distance,
                characteristic_ratio=characteristic_ratio,
                kuhn_length=kuhn_length,
                glass_transition_temperature=glass_transition_temperature,
                melting_temperature=melting_temperature,
                thermal_decomposition_temperature=thermal_decomposition_temperature,
                elastic_modulus_predictor=elastic_modulus_predictor,
                brittleness_index=brittleness_index,
                toughness_index=toughness_index,
                hardness_predictor=hardness_predictor,
                viscosity_predictor=viscosity_predictor,
                solubility_parameter=solubility_parameter,
                melt_flow_index_predictor=melt_flow_index_predictor,
                processability_index=processability_index,
                crystallinity_index=crystallinity_index,
                chain_entanglement_density=chain_entanglement_density,
                free_volume_fraction=free_volume_fraction
            )
            
        except Exception as e:
            raise ChemiAccelError(f"Error calculating polymer descriptors: {str(e)}")
    
    def _predict_glass_transition_temperature(self, polymer_smiles: str) -> float:
        """Predict glass transition temperature using simple group contribution."""
        mol = parse_smiles_safe(polymer_smiles)
        if mol is None:
            return 298.15
        
        # Base temperature
        base_tg = 298.15  # K
        
        # Simple group contributions
        tg_contribution = 0.0
        
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                if atom.GetIsAromatic():
                    tg_contribution += 100.0  # Aromatic carbons increase Tg
                else:
                    num_h = atom.GetTotalNumHs()
                    if num_h == 2:
                        tg_contribution -= 17.0  # CH2 groups decrease Tg
                    elif num_h == 1:
                        tg_contribution += 10.0   # CHR groups increase Tg
            elif atom.GetSymbol() == 'O':
                tg_contribution += 50.0  # Oxygen increases Tg
            elif atom.GetSymbol() == 'N':
                tg_contribution += 80.0  # Nitrogen increases Tg
        
        # Molecular weight correction
        mw = Descriptors.MolWt(mol)
        mw_correction = min(50.0, mw / 1000.0 * 20.0)
        
        predicted_tg = base_tg + tg_contribution + mw_correction
        
        return max(200.0, predicted_tg)
    
    def _predict_elastic_modulus(self, polymer_smiles: str) -> float:
        """Predict elastic modulus."""
        mol = parse_smiles_safe(polymer_smiles)
        if mol is None:
            return 1.0
        
        # Base modulus
        base_modulus = 1.0  # GPa
        
        # Aromatic content increases modulus
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        aromatic_factor = 1.0 + aromatic_rings * 0.5
        
        # Flexibility decreases modulus
        rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        total_bonds = mol.GetNumBonds()
        
        if total_bonds > 0:
            flexibility_factor = 1.0 - (rotatable_bonds / total_bonds) * 0.3
        else:
            flexibility_factor = 1.0
        
        predicted_modulus = base_modulus * aromatic_factor * flexibility_factor
        
        return max(0.1, predicted_modulus)
    
    def _predict_brittleness_index(self, polymer_smiles: str) -> float:
        """Predict brittleness index."""
        mol = parse_smiles_safe(polymer_smiles)
        if mol is None:
            return 0.5
        
        brittleness_score = 0.5  # Base score
        
        # Aromatic content increases brittleness
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        aromatic_contribution = min(0.3, aromatic_rings * 0.1)
        
        # Flexibility decreases brittleness
        rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        total_bonds = mol.GetNumBonds()
        
        if total_bonds > 0:
            flexibility_contribution = -(rotatable_bonds / total_bonds) * 0.3
        else:
            flexibility_contribution = 0.0
        
        brittleness_index = brittleness_score + aromatic_contribution + flexibility_contribution
        
        return max(0.0, min(1.0, brittleness_index))
    
    def _predict_viscosity(self, polymer_smiles: str) -> float:
        """Predict melt viscosity."""
        mol = parse_smiles_safe(polymer_smiles)
        if mol is None:
            return 1000.0
        
        # Base viscosity
        base_viscosity = 1000.0  # Pa·s
        
        # Molecular weight effect
        mw = Descriptors.MolWt(mol)
        mw_factor = (mw / 1000.0) ** 1.5
        
        # Branching effect
        branching_factor = 1.0
        for atom in mol.GetAtoms():
            if atom.GetDegree() > 2:
                branching_factor *= 0.9  # Branching reduces viscosity
        
        predicted_viscosity = base_viscosity * mw_factor * branching_factor
        
        return predicted_viscosity
    
    def _predict_solubility_parameter(self, polymer_smiles: str) -> float:
        """Predict Hildebrand solubility parameter."""
        mol = parse_smiles_safe(polymer_smiles)
        if mol is None:
            return 20.0
        
        # Simple group contribution
        group_contribution = 0.0
        
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                if atom.GetTotalNumHs() == 2:
                    group_contribution += 4940  # CH2
                elif atom.GetTotalNumHs() == 3:
                    group_contribution += 4710  # CH3
                elif atom.GetIsAromatic():
                    group_contribution += 7630  # Aromatic CH
            elif atom.GetSymbol() == 'O':
                if atom.GetTotalNumHs() > 0:
                    group_contribution += 29800  # OH
                else:
                    group_contribution += 3350   # O
        
        # Estimate molar volume
        mw = Descriptors.MolWt(mol)
        molar_volume = mw / 1.0  # Assume density ~1 g/cm³
        
        if molar_volume > 0:
            solubility_parameter = np.sqrt(group_contribution / molar_volume)
        else:
            solubility_parameter = 20.0
        
        return solubility_parameter
    
    def _predict_processability_index(self, polymer_smiles: str) -> float:
        """Predict processability index."""
        mol = parse_smiles_safe(polymer_smiles)
        if mol is None:
            return 0.5
        
        # Base processability
        processability = 0.7
        
        # Thermal stability (simplified)
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        thermal_stability = min(1.0, 0.5 + aromatic_rings * 0.1)
        
        # Flow properties (inverse of viscosity)
        viscosity = self._predict_viscosity(polymer_smiles)
        flow_score = min(1.0, 1000.0 / viscosity)
        
        processability_index = (processability + thermal_stability * 0.3 + flow_score * 0.3) / 1.6
        
        return max(0.0, min(1.0, processability_index))
    
    def _calculate_crystallinity_index(self, polymer_smiles: str) -> float:
        """Calculate crystallinity index."""
        mol = parse_smiles_safe(polymer_smiles)
        if mol is None:
            return 0.5
        
        crystallinity = 0.5  # Base value
        
        # Aromatic content promotes crystallinity
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        aromatic_bonus = min(0.3, aromatic_rings * 0.1)
        
        # Branching reduces crystallinity
        branching_penalty = 0.0
        for atom in mol.GetAtoms():
            if atom.GetDegree() > 2:
                branching_penalty += 0.05
        
        crystallinity_index = crystallinity + aromatic_bonus - branching_penalty
        
        return max(0.0, min(1.0, crystallinity_index))
    
    def _calculate_chain_entanglement_density(self, polymer_smiles: str) -> float:
        """Calculate chain entanglement density."""
        mol = parse_smiles_safe(polymer_smiles)
        if mol is None:
            return 0.5
        
        # Longer chains have more entanglements
        num_heavy_atoms = mol.GetNumHeavyAtoms()
        length_factor = min(1.0, num_heavy_atoms / 100.0)
        
        # Branching reduces entanglements
        branching_factor = 1.0
        for atom in mol.GetAtoms():
            if atom.GetDegree() > 2:
                branching_factor *= 0.95
        
        entanglement_density = length_factor * branching_factor
        
        return max(0.0, min(1.0, entanglement_density))
    
    def _calculate_free_volume_fraction(self, polymer_smiles: str) -> float:
        """Calculate free volume fraction."""
        mol = parse_smiles_safe(polymer_smiles)
        if mol is None:
            return 0.1
        
        # Base free volume
        base_free_volume = 0.1  # 10%
        
        # Flexibility increases free volume
        rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        total_bonds = mol.GetNumBonds()
        
        if total_bonds > 0:
            flexibility_contribution = (rotatable_bonds / total_bonds) * 0.05
        else:
            flexibility_contribution = 0.0
        
        # Branching increases free volume
        branching_contribution = 0.0
        for atom in mol.GetAtoms():
            if atom.GetDegree() > 2:
                branching_contribution += 0.01
        
        free_volume_fraction = base_free_volume + flexibility_contribution + branching_contribution
        
        return min(0.3, free_volume_fraction)  # Cap at 30%