"""
Inter-component interaction modeling for complex mixtures.

Quantifies and predicts molecular interactions including:
- Hydrogen bonding
- π-π stacking
- Van der Waals interactions
- Electrostatic interactions
- Polymer-solvent interactions
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcNumHBA, CalcNumHBD

from ..core.backend import Backend
from ..core.exceptions import ChemiAccelError
from .core import ComponentData, MixtureData


@dataclass
class MolecularInteraction:
    """Data structure for molecular interactions."""

    component_i: int
    component_j: int
    interaction_type: str
    strength: float
    distance: Optional[float] = None
    geometry: Optional[Dict[str, Any]] = None
    energy: Optional[float] = None

    @property
    def is_self_interaction(self) -> bool:
        """Whether this is a self-interaction."""
        return self.component_i == self.component_j


class InteractionModel(ABC):
    """Abstract base class for interaction models."""

    @abstractmethod
    def calculate_interaction_energy(
        self, mol_i: Chem.Mol, mol_j: Chem.Mol, distance: float = 5.0
    ) -> float:
        """Calculate pairwise interaction energy."""
        pass

    @abstractmethod
    def get_interaction_descriptors(
        self, mol_i: Chem.Mol, mol_j: Chem.Mol
    ) -> Dict[str, float]:
        """Get descriptors relevant to this interaction type."""
        pass


class HydrogenBondingModel(InteractionModel):
    """Model for hydrogen bonding interactions."""

    def __init__(self, backend: Backend):
        self.backend = backend
        # Hydrogen bonding parameters
        self.donor_acceptor_cutoff = 3.5  # Angstroms
        self.angle_cutoff = 120.0  # degrees

    def calculate_interaction_energy(
        self, mol_i: Chem.Mol, mol_j: Chem.Mol, distance: float = 5.0
    ) -> float:
        """Calculate hydrogen bonding energy."""

        # Get hydrogen bond donors and acceptors
        hbd_i = CalcNumHBD(mol_i)
        hba_i = CalcNumHBA(mol_i)
        hbd_j = CalcNumHBD(mol_j)
        hba_j = CalcNumHBA(mol_j)

        # Calculate potential hydrogen bonds
        potential_bonds = min(hbd_i, hba_j) + min(hbd_j, hba_i)

        if potential_bonds == 0:
            return 0.0

        # Empirical hydrogen bonding energy (kJ/mol)
        base_energy = -20.0  # Typical H-bond strength
        distance_factor = np.exp(-distance / 3.0)  # Distance dependence

        return potential_bonds * base_energy * distance_factor

    def get_interaction_descriptors(
        self, mol_i: Chem.Mol, mol_j: Chem.Mol
    ) -> Dict[str, float]:
        """Get hydrogen bonding descriptors."""

        return {
            "hbd_i": CalcNumHBD(mol_i),
            "hba_i": CalcNumHBA(mol_i),
            "hbd_j": CalcNumHBD(mol_j),
            "hba_j": CalcNumHBA(mol_j),
            "h_bond_potential": min(CalcNumHBD(mol_i), CalcNumHBA(mol_j))
            + min(CalcNumHBD(mol_j), CalcNumHBA(mol_i)),
        }


class PiStackingModel(InteractionModel):
    """Model for π-π stacking interactions."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.stacking_distance = 3.4  # Typical π-π stacking distance

    def calculate_interaction_energy(
        self, mol_i: Chem.Mol, mol_j: Chem.Mol, distance: float = 5.0
    ) -> float:
        """Calculate π-π stacking energy."""

        # Count aromatic rings
        aromatic_rings_i = self._count_aromatic_rings(mol_i)
        aromatic_rings_j = self._count_aromatic_rings(mol_j)

        if aromatic_rings_i == 0 or aromatic_rings_j == 0:
            return 0.0

        # Calculate stacking potential
        stacking_pairs = min(aromatic_rings_i, aromatic_rings_j)

        # Empirical π-π stacking energy (kJ/mol)
        base_energy = -8.0  # Typical π-π interaction strength
        distance_factor = np.exp(-((distance - self.stacking_distance) ** 2) / 2.0)

        return stacking_pairs * base_energy * distance_factor

    def get_interaction_descriptors(
        self, mol_i: Chem.Mol, mol_j: Chem.Mol
    ) -> Dict[str, float]:
        """Get π-π stacking descriptors."""

        aromatic_i = self._count_aromatic_rings(mol_i)
        aromatic_j = self._count_aromatic_rings(mol_j)

        return {
            "aromatic_rings_i": aromatic_i,
            "aromatic_rings_j": aromatic_j,
            "pi_stacking_potential": min(aromatic_i, aromatic_j),
            "conjugated_systems_i": self._count_conjugated_systems(mol_i),
            "conjugated_systems_j": self._count_conjugated_systems(mol_j),
        }

    def _count_aromatic_rings(self, mol: Chem.Mol) -> int:
        """Count number of aromatic rings."""
        return rdMolDescriptors.CalcNumAromaticRings(mol)

    def _count_conjugated_systems(self, mol: Chem.Mol) -> int:
        """Count conjugated systems (simplified)."""
        # This is a simplified implementation
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        return aromatic_atoms // 6  # Approximate benzene rings


class VanDerWaalsModel(InteractionModel):
    """Model for van der Waals interactions."""

    def __init__(self, backend: Backend):
        self.backend = backend
        # Lennard-Jones parameters (simplified)
        self.epsilon = 0.5  # kJ/mol
        self.sigma = 3.5  # Angstroms

    def calculate_interaction_energy(
        self, mol_i: Chem.Mol, mol_j: Chem.Mol, distance: float = 5.0
    ) -> float:
        """Calculate van der Waals energy using Lennard-Jones potential."""

        # Get molecular volumes as proxy for size
        vol_i = self._calculate_molecular_volume(mol_i)
        vol_j = self._calculate_molecular_volume(mol_j)

        # Effective interaction strength
        epsilon_eff = self.epsilon * np.sqrt(vol_i * vol_j) / 100.0

        # Lennard-Jones potential
        sigma_over_r = self.sigma / distance
        sigma_over_r_6 = sigma_over_r**6
        sigma_over_r_12 = sigma_over_r_6**2

        energy = 4 * epsilon_eff * (sigma_over_r_12 - sigma_over_r_6)

        return energy

    def get_interaction_descriptors(
        self, mol_i: Chem.Mol, mol_j: Chem.Mol
    ) -> Dict[str, float]:
        """Get van der Waals descriptors."""

        return {
            "molecular_volume_i": self._calculate_molecular_volume(mol_i),
            "molecular_volume_j": self._calculate_molecular_volume(mol_j),
            "polarizability_i": self._estimate_polarizability(mol_i),
            "polarizability_j": self._estimate_polarizability(mol_j),
            "surface_area_i": rdMolDescriptors.CalcTPSA(mol_i),
            "surface_area_j": rdMolDescriptors.CalcTPSA(mol_j),
        }

    def _calculate_molecular_volume(self, mol: Chem.Mol) -> float:
        """Estimate molecular volume."""
        # Use molar volume approximation
        try:
            return Descriptors.MolWt(mol) / 0.6  # Approximate density
        except:
            return 100.0  # Default value

    def _estimate_polarizability(self, mol: Chem.Mol) -> float:
        """Estimate molecular polarizability."""
        # Simplified estimation based on atom count
        return mol.GetNumAtoms() * 1.5  # Approximate polarizability per atom


class ElectrostaticModel(InteractionModel):
    """Model for electrostatic interactions."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.dielectric_constant = 1.0  # Vacuum
        self.coulomb_constant = 8.99e9  # N⋅m²/C²

    def calculate_interaction_energy(
        self, mol_i: Chem.Mol, mol_j: Chem.Mol, distance: float = 5.0
    ) -> float:
        """Calculate electrostatic interaction energy."""

        # Get partial charges (simplified)
        charge_i = self._estimate_partial_charge(mol_i)
        charge_j = self._estimate_partial_charge(mol_j)

        if abs(charge_i) < 1e-6 and abs(charge_j) < 1e-6:
            return 0.0

        # Coulomb's law (converted to kJ/mol)
        energy = (self.coulomb_constant * charge_i * charge_j) / (
            4 * np.pi * self.dielectric_constant * distance * 1e-10
        )

        # Convert to kJ/mol
        energy *= 6.022e23 / 1000.0

        return energy

    def get_interaction_descriptors(
        self, mol_i: Chem.Mol, mol_j: Chem.Mol
    ) -> Dict[str, float]:
        """Get electrostatic descriptors."""

        return {
            "partial_charge_i": self._estimate_partial_charge(mol_i),
            "partial_charge_j": self._estimate_partial_charge(mol_j),
            "dipole_moment_i": self._estimate_dipole_moment(mol_i),
            "dipole_moment_j": self._estimate_dipole_moment(mol_j),
            "quadrupole_i": self._estimate_quadrupole(mol_i),
            "quadrupole_j": self._estimate_quadrupole(mol_j),
        }

    def _estimate_partial_charge(self, mol: Chem.Mol) -> float:
        """Estimate net partial charge."""
        # Simplified estimation based on electronegativity differences
        formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        return float(formal_charge)

    def _estimate_dipole_moment(self, mol: Chem.Mol) -> float:
        """Estimate molecular dipole moment."""
        # Very simplified estimation
        return Descriptors.TPSA(mol) / 100.0

    def _estimate_quadrupole(self, mol: Chem.Mol) -> float:
        """Estimate quadrupole moment."""
        # Placeholder implementation
        return 0.0


class PolymerSolventModel(InteractionModel):
    """Specialized model for polymer-solvent interactions."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.flory_huggins_params = {}

    def calculate_interaction_energy(
        self, mol_i: Chem.Mol, mol_j: Chem.Mol, distance: float = 5.0
    ) -> float:
        """Calculate polymer-solvent interaction using Flory-Huggins theory."""

        # Determine which is polymer and which is solvent
        is_polymer_i = self._is_polymer(mol_i)
        is_polymer_j = self._is_polymer(mol_j)

        if not (is_polymer_i or is_polymer_j):
            return 0.0  # Neither is a polymer

        # Get Flory-Huggins interaction parameter
        chi = self._get_flory_huggins_parameter(mol_i, mol_j)

        # Calculate interaction energy per unit volume
        RT = 8.314 * 298.15  # J/mol at room temperature
        energy = chi * RT / 1000.0  # Convert to kJ/mol

        return energy

    def get_interaction_descriptors(
        self, mol_i: Chem.Mol, mol_j: Chem.Mol
    ) -> Dict[str, float]:
        """Get polymer-solvent descriptors."""

        return {
            "is_polymer_i": float(self._is_polymer(mol_i)),
            "is_polymer_j": float(self._is_polymer(mol_j)),
            "solubility_parameter_i": self._estimate_solubility_parameter(mol_i),
            "solubility_parameter_j": self._estimate_solubility_parameter(mol_j),
            "chain_length_i": self._estimate_chain_length(mol_i),
            "chain_length_j": self._estimate_chain_length(mol_j),
            "flory_huggins_chi": self._get_flory_huggins_parameter(mol_i, mol_j),
        }

    def _is_polymer(self, mol: Chem.Mol) -> bool:
        """Check if molecule is a polymer (simplified)."""
        # Simplified check based on molecular weight and structure
        mw = Descriptors.MolWt(mol)
        return mw > 1000.0  # Arbitrary cutoff

    def _estimate_solubility_parameter(self, mol: Chem.Mol) -> float:
        """Estimate Hansen solubility parameter."""
        # Simplified group contribution method
        return np.sqrt(Descriptors.MolLogP(mol) + 20.0)

    def _estimate_chain_length(self, mol: Chem.Mol) -> float:
        """Estimate polymer chain length."""
        if self._is_polymer(mol):
            return mol.GetNumAtoms() / 10.0  # Rough estimate
        return 1.0

    def _get_flory_huggins_parameter(self, mol_i: Chem.Mol, mol_j: Chem.Mol) -> float:
        """Get Flory-Huggins interaction parameter."""
        # Simplified calculation based on solubility parameters
        delta_i = self._estimate_solubility_parameter(mol_i)
        delta_j = self._estimate_solubility_parameter(mol_j)

        V_ref = 100.0  # Reference molar volume
        RT = 8.314 * 298.15

        chi = V_ref * (delta_i - delta_j) ** 2 / RT
        return chi


class InteractionModeler:
    """
    GPU-accelerated inter-component interaction modeling system.

    Quantifies various types of molecular interactions and their
    impact on mixture properties.
    """

    def __init__(self, backend: Backend):
        self.backend = backend

        # Initialize interaction models
        self.models = {
            "hydrogen_bonding": HydrogenBondingModel(backend),
            "pi_stacking": PiStackingModel(backend),
            "van_der_waals": VanDerWaalsModel(backend),
            "electrostatic": ElectrostaticModel(backend),
            "polymer_solvent": PolymerSolventModel(backend),
        }

        self.default_models = ["hydrogen_bonding", "van_der_waals", "electrostatic"]

    def calculate_interactions(
        self,
        mixture: MixtureData,
        interaction_types: Optional[List[str]] = None,
        include_self_interactions: bool = True,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> List[MolecularInteraction]:
        """
        Calculate all pairwise interactions in mixture.

        Args:
            mixture: Mixture data
            interaction_types: Types of interactions to calculate
            include_self_interactions: Whether to include i-i interactions
            temperature: Temperature for calculations

        Returns:
            List of molecular interactions
        """

        interaction_types = interaction_types or self.default_models
        temperature = temperature or mixture.temperature or 298.15

        # Convert SMILES to molecules
        molecules = []
        for comp in mixture.components:
            mol = Chem.MolFromSmiles(comp.smiles)
            if mol is None:
                raise ChemiAccelError(f"Invalid SMILES: {comp.smiles}")
            molecules.append(mol)

        # Calculate all pairwise interactions
        interactions = []
        n_comp = len(molecules)

        for i in range(n_comp):
            for j in range(i if include_self_interactions else i + 1, n_comp):
                for interaction_type in interaction_types:
                    if interaction_type in self.models:
                        interaction = self._calculate_pairwise_interaction(
                            molecules[i],
                            molecules[j],
                            i,
                            j,
                            interaction_type,
                            temperature,
                        )
                        if interaction:
                            interactions.append(interaction)

        return interactions

    def _calculate_pairwise_interaction(
        self,
        mol_i: Chem.Mol,
        mol_j: Chem.Mol,
        idx_i: int,
        idx_j: int,
        interaction_type: str,
        temperature: float,
    ) -> Optional[MolecularInteraction]:
        """Calculate single pairwise interaction."""

        model = self.models[interaction_type]

        # Estimate intermolecular distance (simplified)
        distance = self._estimate_interaction_distance(mol_i, mol_j)

        # Calculate interaction energy
        energy = model.calculate_interaction_energy(mol_i, mol_j, distance)

        # Skip very weak interactions
        if abs(energy) < 0.1:  # kJ/mol threshold
            return None

        # Get descriptors
        descriptors = model.get_interaction_descriptors(mol_i, mol_j)

        return MolecularInteraction(
            component_i=idx_i,
            component_j=idx_j,
            interaction_type=interaction_type,
            strength=abs(energy),
            distance=distance,
            energy=energy,
            geometry=descriptors,
        )

    def _estimate_interaction_distance(self, mol_i: Chem.Mol, mol_j: Chem.Mol) -> float:
        """Estimate typical interaction distance between molecules."""
        # Simplified estimation based on molecular size
        size_i = np.sqrt(mol_i.GetNumAtoms())
        size_j = np.sqrt(mol_j.GetNumAtoms())
        return 3.0 + 0.5 * (size_i + size_j)  # Angstroms

    def build_interaction_matrix(
        self, interactions: List[MolecularInteraction], n_components: int
    ) -> np.ndarray:
        """
        Build interaction matrix from calculated interactions.

        Args:
            interactions: List of molecular interactions
            n_components: Number of components

        Returns:
            Symmetric interaction matrix
        """

        matrix = np.zeros((n_components, n_components))

        for interaction in interactions:
            i, j = interaction.component_i, interaction.component_j
            energy = interaction.energy

            matrix[i, j] = energy
            if i != j:  # Make symmetric
                matrix[j, i] = energy

        return matrix

    def calculate_mixture_interaction_energy(
        self, mixture: MixtureData, interactions: List[MolecularInteraction]
    ) -> float:
        """Calculate total mixture interaction energy."""

        total_energy = 0.0

        for interaction in interactions:
            i, j = interaction.component_i, interaction.component_j
            x_i = mixture.components[i].mol_fraction
            x_j = mixture.components[j].mol_fraction

            # Weight by composition
            if i == j:
                weight = x_i * x_i  # Self-interaction
            else:
                weight = 2 * x_i * x_j  # Cross-interaction (factor of 2)

            total_energy += weight * interaction.energy

        return total_energy

    def predict_mixing_behavior(
        self, mixture: MixtureData, interactions: List[MolecularInteraction]
    ) -> Dict[str, Any]:
        """
        Predict mixing behavior based on interactions.

        Args:
            mixture: Mixture data
            interactions: Calculated interactions

        Returns:
            Mixing behavior predictions
        """

        # Build interaction matrix
        interaction_matrix = self.build_interaction_matrix(
            interactions, mixture.n_components
        )

        # Calculate interaction energy
        total_energy = self.calculate_mixture_interaction_energy(mixture, interactions)

        # Analyze interaction patterns
        interaction_analysis = self._analyze_interaction_patterns(interactions)

        # Predict properties
        predictions = {
            "total_interaction_energy": total_energy,
            "mixing_enthalpy": total_energy,  # Simplified
            "miscibility_score": self._calculate_miscibility_score(interaction_matrix),
            "dominant_interactions": interaction_analysis["dominant_types"],
            "interaction_strength_distribution": interaction_analysis[
                "strength_distribution"
            ],
            "phase_separation_tendency": self._predict_phase_separation(
                interaction_matrix
            ),
        }

        return predictions

    def _analyze_interaction_patterns(
        self, interactions: List[MolecularInteraction]
    ) -> Dict[str, Any]:
        """Analyze patterns in molecular interactions."""

        if not interactions:
            return {"dominant_types": [], "strength_distribution": {}}

        # Group by interaction type
        by_type = {}
        for interaction in interactions:
            itype = interaction.interaction_type
            if itype not in by_type:
                by_type[itype] = []
            by_type[itype].append(interaction.strength)

        # Find dominant interaction types
        avg_strengths = {
            itype: np.mean(strengths) for itype, strengths in by_type.items()
        }
        dominant_types = sorted(
            avg_strengths.keys(), key=lambda x: avg_strengths[x], reverse=True
        )

        # Strength distribution
        all_strengths = [interaction.strength for interaction in interactions]
        strength_distribution = {
            "mean": np.mean(all_strengths),
            "std": np.std(all_strengths),
            "min": np.min(all_strengths),
            "max": np.max(all_strengths),
        }

        return {
            "dominant_types": dominant_types[:3],  # Top 3
            "strength_distribution": strength_distribution,
            "type_contributions": avg_strengths,
        }

    def _calculate_miscibility_score(self, interaction_matrix: np.ndarray) -> float:
        """Calculate miscibility score from interaction matrix."""

        if interaction_matrix.shape[0] < 2:
            return 1.0  # Single component is always miscible

        # Analyze off-diagonal elements (cross-interactions)
        n = interaction_matrix.shape[0]
        cross_interactions = []

        for i in range(n):
            for j in range(i + 1, n):
                cross_interactions.append(interaction_matrix[i, j])

        if not cross_interactions:
            return 1.0

        # Favorable (negative) interactions promote miscibility
        avg_cross_interaction = np.mean(cross_interactions)
        miscibility_score = 1.0 / (1.0 + np.exp(avg_cross_interaction / 5.0))  # Sigmoid

        return miscibility_score

    def _predict_phase_separation(self, interaction_matrix: np.ndarray) -> float:
        """Predict tendency for phase separation."""

        miscibility = self._calculate_miscibility_score(interaction_matrix)
        return 1.0 - miscibility

    def visualize_interactions(
        self, interactions: List[MolecularInteraction], mixture: MixtureData
    ) -> Dict[str, Any]:
        """Generate visualization data for interactions."""

        # This would integrate with the visualization module
        # Return data structures suitable for plotting

        return {
            "interaction_network": self._build_network_data(interactions, mixture),
            "energy_landscape": self._build_energy_landscape(interactions),
            "interaction_heatmap": self.build_interaction_matrix(
                interactions, mixture.n_components
            ),
        }

    def _build_network_data(
        self, interactions: List[MolecularInteraction], mixture: MixtureData
    ) -> Dict[str, Any]:
        """Build network representation of interactions."""

        nodes = []
        for i, comp in enumerate(mixture.components):
            nodes.append(
                {"id": i, "label": comp.smiles, "size": comp.mol_fraction * 100}
            )

        edges = []
        for interaction in interactions:
            edges.append(
                {
                    "source": interaction.component_i,
                    "target": interaction.component_j,
                    "weight": interaction.strength,
                    "type": interaction.interaction_type,
                    "energy": interaction.energy,
                }
            )

        return {"nodes": nodes, "edges": edges}

    def _build_energy_landscape(
        self, interactions: List[MolecularInteraction]
    ) -> Dict[str, Any]:
        """Build energy landscape visualization data."""

        energy_data = []
        for interaction in interactions:
            energy_data.append(
                {
                    "pair": f"{interaction.component_i}-{interaction.component_j}",
                    "type": interaction.interaction_type,
                    "energy": interaction.energy,
                    "strength": interaction.strength,
                }
            )

        return {"energy_points": energy_data}
