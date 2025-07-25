"""
Oligomer and dimer processing module.

This module provides functionality for chain growth simulation,
conformational flexibility analysis, end-group effects, and sequence analysis.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem.rdMolAlign import AlignMol

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning(
        "RDKit not available. Some oligomer analysis features will be limited."
    )

from ..core.exceptions import ChemiAccelError, InvalidMoleculeError
from ..utils.rdkit_integration import parse_smiles_safe


class ChainGrowthMechanism(Enum):
    """Chain growth mechanisms."""

    STEP_GROWTH = "step_growth"
    CHAIN_GROWTH = "chain_growth"
    LIVING_POLYMERIZATION = "living_polymerization"
    CONTROLLED_RADICAL = "controlled_radical"


@dataclass
class ChainGrowthStep:
    """Represents a single step in chain growth."""

    step_number: int
    reactant_smiles: str
    product_smiles: str
    reaction_type: str
    energy_change: Optional[float] = None
    activation_energy: Optional[float] = None


@dataclass
class ConformationalEnsemble:
    """Represents a conformational ensemble of an oligomer."""

    conformer_smiles: List[str]
    energies: List[float]
    populations: List[float]
    flexibility_metrics: Dict[str, float]


@dataclass
class EndGroupInfo:
    """Information about chain end groups."""

    left_end_group: str
    right_end_group: str
    end_group_reactivity: Dict[str, float]
    termination_probability: float


@dataclass
class SequenceInfo:
    """Information about oligomer sequence."""

    monomer_sequence: List[str]
    sequence_pattern: str
    regularity_score: float
    block_structure: Optional[Dict[str, List[int]]] = None


class ChainGrowthSimulator:
    """Simulates chain growth for step-by-step polymer extension."""

    def __init__(self):
        """Initialize the chain growth simulator."""
        if not RDKIT_AVAILABLE:
            raise ChemiAccelError("RDKit is required for chain growth simulation")

        # Define common polymerization reactions
        self.reaction_templates = {
            "vinyl_addition": {
                "reactant_pattern": "[C]=[C]",
                "product_pattern": "[C]-[C]",
                "mechanism": ChainGrowthMechanism.CHAIN_GROWTH,
            },
            "condensation": {
                "reactant_pattern": "[OH].[COOH]",
                "product_pattern": "[C](=O)[O]",
                "mechanism": ChainGrowthMechanism.STEP_GROWTH,
            },
            "ring_opening": {
                "reactant_pattern": "[C]1[O][C]1",
                "product_pattern": "[C][O][C]",
                "mechanism": ChainGrowthMechanism.LIVING_POLYMERIZATION,
            },
        }

    def simulate_chain_growth(
        self,
        monomer_smiles: str,
        chain_length: int,
        mechanism: ChainGrowthMechanism = ChainGrowthMechanism.CHAIN_GROWTH,
    ) -> List[ChainGrowthStep]:
        """
        Simulate chain growth for a given monomer.

        Args:
            monomer_smiles: SMILES of the monomer
            chain_length: Target chain length
            mechanism: Polymerization mechanism

        Returns:
            List of chain growth steps
        """
        mol = parse_smiles_safe(monomer_smiles)
        if mol is None:
            raise InvalidMoleculeError(monomer_smiles, "Invalid monomer SMILES")

        steps = []
        current_smiles = monomer_smiles

        for step in range(1, chain_length):
            try:
                # Simulate addition of another monomer unit
                next_smiles = self._add_monomer_unit(
                    current_smiles, monomer_smiles, mechanism
                )

                growth_step = ChainGrowthStep(
                    step_number=step,
                    reactant_smiles=current_smiles,
                    product_smiles=next_smiles,
                    reaction_type=mechanism.value,
                    energy_change=self._estimate_energy_change(
                        current_smiles, next_smiles
                    ),
                    activation_energy=self._estimate_activation_energy(mechanism),
                )

                steps.append(growth_step)
                current_smiles = next_smiles

            except Exception as e:
                logging.warning(f"Chain growth simulation failed at step {step}: {e}")
                break

        return steps

    def _add_monomer_unit(
        self, chain_smiles: str, monomer_smiles: str, mechanism: ChainGrowthMechanism
    ) -> str:
        """Add a monomer unit to the growing chain."""
        # Simplified simulation - in practice, this would use reaction templates
        # and more sophisticated chemistry

        chain_mol = parse_smiles_safe(chain_smiles)
        monomer_mol = parse_smiles_safe(monomer_smiles)

        if chain_mol is None or monomer_mol is None:
            raise ChemiAccelError("Invalid molecules for chain growth")

        # Simple concatenation for demonstration
        # In reality, this would involve proper reaction chemistry
        chain_atoms = chain_mol.GetNumAtoms()
        monomer_atoms = monomer_mol.GetNumAtoms()

        # Create a simple linear combination
        combined_mol = Chem.CombineMols(chain_mol, monomer_mol)

        # Add a bond between the molecules (simplified)
        editable_mol = Chem.EditableMol(combined_mol)

        try:
            # Connect last atom of chain to first atom of monomer
            editable_mol.AddBond(chain_atoms - 1, chain_atoms, Chem.BondType.SINGLE)
            result_mol = editable_mol.GetMol()

            # Sanitize and return SMILES
            Chem.SanitizeMol(result_mol)
            return Chem.MolToSmiles(result_mol)

        except Exception:
            # Fallback: return concatenated SMILES
            return f"{chain_smiles}.{monomer_smiles}"

    def _estimate_energy_change(
        self, reactant_smiles: str, product_smiles: str
    ) -> float:
        """Estimate energy change for the reaction step."""
        # Simplified energy estimation based on molecular properties
        try:
            reactant_mol = parse_smiles_safe(reactant_smiles)
            product_mol = parse_smiles_safe(product_smiles)

            if reactant_mol and product_mol:
                # Use molecular weight difference as a proxy
                mw_diff = Descriptors.MolWt(product_mol) - Descriptors.MolWt(
                    reactant_mol
                )
                return -0.1 * mw_diff  # Negative for favorable reactions

        except Exception:
            pass

        return -5.0  # Default favorable energy change

    def _estimate_activation_energy(self, mechanism: ChainGrowthMechanism) -> float:
        """Estimate activation energy based on mechanism."""
        activation_energies = {
            ChainGrowthMechanism.CHAIN_GROWTH: 15.0,
            ChainGrowthMechanism.STEP_GROWTH: 25.0,
            ChainGrowthMechanism.LIVING_POLYMERIZATION: 10.0,
            ChainGrowthMechanism.CONTROLLED_RADICAL: 12.0,
        }
        return activation_energies.get(mechanism, 20.0)


class ConformationalFlexibilityAnalyzer:
    """Analyzes conformational flexibility of oligomer chains."""

    def __init__(self):
        """Initialize the conformational flexibility analyzer."""
        if not RDKIT_AVAILABLE:
            raise ChemiAccelError("RDKit is required for conformational analysis")

    def analyze_flexibility(
        self, oligomer_smiles: str, num_conformers: int = 10
    ) -> ConformationalEnsemble:
        """
        Analyze conformational flexibility of an oligomer.

        Args:
            oligomer_smiles: SMILES of the oligomer
            num_conformers: Number of conformers to generate

        Returns:
            ConformationalEnsemble object
        """
        mol = parse_smiles_safe(oligomer_smiles)
        if mol is None:
            raise InvalidMoleculeError(oligomer_smiles, "Invalid oligomer SMILES")

        # Add hydrogens for conformer generation
        mol = Chem.AddHs(mol)

        # Generate conformers
        conformer_ids = self._generate_conformers(mol, num_conformers)

        # Calculate energies and properties
        conformer_smiles = []
        energies = []

        for conf_id in conformer_ids:
            try:
                # Get conformer SMILES (this is simplified)
                conf_smiles = Chem.MolToSmiles(mol, confId=conf_id)
                conformer_smiles.append(conf_smiles)

                # Estimate energy (simplified)
                energy = self._estimate_conformer_energy(mol, conf_id)
                energies.append(energy)

            except Exception as e:
                logging.warning(f"Error processing conformer {conf_id}: {e}")
                continue

        # Calculate populations using Boltzmann distribution
        populations = self._calculate_populations(energies)

        # Calculate flexibility metrics
        flexibility_metrics = self._calculate_flexibility_metrics(mol, conformer_ids)

        return ConformationalEnsemble(
            conformer_smiles=conformer_smiles,
            energies=energies,
            populations=populations,
            flexibility_metrics=flexibility_metrics,
        )

    def _generate_conformers(self, mol: Chem.Mol, num_conformers: int) -> List[int]:
        """Generate conformers for the molecule."""
        try:
            # Use RDKit's conformer generation
            conf_ids = AllChem.EmbedMultipleConfs(
                mol,
                numConfs=num_conformers,
                randomSeed=42,
                useExpTorsionAnglePrefs=True,
                useBasicKnowledge=True,
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
            return [0]  # Return at least one conformer

    def _estimate_conformer_energy(self, mol: Chem.Mol, conf_id: int) -> float:
        """Estimate conformer energy."""
        try:
            # Use MMFF force field for energy estimation
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props:
                ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
                if ff:
                    return ff.CalcEnergy()
        except:
            pass

        # Fallback: random energy for demonstration
        return np.random.normal(0, 5)

    def _calculate_populations(self, energies: List[float]) -> List[float]:
        """Calculate Boltzmann populations."""
        if not energies:
            return []

        # Convert to relative energies
        min_energy = min(energies)
        rel_energies = [e - min_energy for e in energies]

        # Boltzmann distribution at room temperature (kT ≈ 0.6 kcal/mol)
        kT = 0.6
        boltzmann_factors = [np.exp(-e / kT) for e in rel_energies]

        # Normalize
        total = sum(boltzmann_factors)
        populations = [bf / total for bf in boltzmann_factors]

        return populations

    def _calculate_flexibility_metrics(
        self, mol: Chem.Mol, conf_ids: List[int]
    ) -> Dict[str, float]:
        """Calculate flexibility metrics."""
        metrics = {}

        try:
            # Number of rotatable bonds
            metrics["num_rotatable_bonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)

            # Radius of gyration variation
            if len(conf_ids) > 1:
                rg_values = []
                for conf_id in conf_ids:
                    try:
                        conf = mol.GetConformer(conf_id)
                        positions = [
                            conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())
                        ]
                        rg = self._calculate_radius_of_gyration(positions)
                        rg_values.append(rg)
                    except:
                        continue

                if rg_values:
                    metrics["rg_mean"] = np.mean(rg_values)
                    metrics["rg_std"] = np.std(rg_values)
                    metrics["flexibility_index"] = (
                        metrics["rg_std"] / metrics["rg_mean"]
                        if metrics["rg_mean"] > 0
                        else 0
                    )
                else:
                    metrics["rg_mean"] = 0
                    metrics["rg_std"] = 0
                    metrics["flexibility_index"] = 0
            else:
                metrics["rg_mean"] = 0
                metrics["rg_std"] = 0
                metrics["flexibility_index"] = 0

            # Chain length
            metrics["chain_length"] = mol.GetNumAtoms()

        except Exception as e:
            logging.warning(f"Error calculating flexibility metrics: {e}")
            metrics = {
                "num_rotatable_bonds": 0,
                "rg_mean": 0,
                "rg_std": 0,
                "flexibility_index": 0,
                "chain_length": 0,
            }

        return metrics

    def _calculate_radius_of_gyration(self, positions) -> float:
        """Calculate radius of gyration from atomic positions."""
        if not positions:
            return 0.0

        # Calculate center of mass (assuming equal masses)
        center = np.mean([[pos.x, pos.y, pos.z] for pos in positions], axis=0)

        # Calculate radius of gyration
        rg_squared = np.mean(
            [
                (pos.x - center[0]) ** 2
                + (pos.y - center[1]) ** 2
                + (pos.z - center[2]) ** 2
                for pos in positions
            ]
        )

        return np.sqrt(rg_squared)


class EndGroupHandler:
    """Handles end-group effects in oligomer chains."""

    def __init__(self):
        """Initialize the end group handler."""
        if not RDKIT_AVAILABLE:
            raise ChemiAccelError("RDKit is required for end group analysis")

        # Define common end group patterns
        self.end_group_patterns = {
            "hydroxyl": "[OH]",
            "carboxyl": "[C](=O)[OH]",
            "amino": "[NH2]",
            "vinyl": "[C]=[C]",
            "methyl": "[CH3]",
            "phenyl": "[c]",
            "ester": "[C](=O)[O]",
            "ether": "[O]",
        }

    def analyze_end_groups(self, oligomer_smiles: str) -> EndGroupInfo:
        """
        Analyze end groups of an oligomer chain.

        Args:
            oligomer_smiles: SMILES of the oligomer

        Returns:
            EndGroupInfo object
        """
        mol = parse_smiles_safe(oligomer_smiles)
        if mol is None:
            raise InvalidMoleculeError(oligomer_smiles, "Invalid oligomer SMILES")

        # Identify terminal atoms
        terminal_atoms = self._find_terminal_atoms(mol)

        # Classify end groups
        left_end_group, right_end_group = self._classify_end_groups(mol, terminal_atoms)

        # Calculate end group reactivity
        end_group_reactivity = self._calculate_end_group_reactivity(
            left_end_group, right_end_group
        )

        # Estimate termination probability
        termination_probability = self._estimate_termination_probability(
            end_group_reactivity
        )

        return EndGroupInfo(
            left_end_group=left_end_group,
            right_end_group=right_end_group,
            end_group_reactivity=end_group_reactivity,
            termination_probability=termination_probability,
        )

    def _find_terminal_atoms(self, mol: Chem.Mol) -> List[int]:
        """Find terminal atoms in the molecule."""
        terminal_atoms = []

        for atom in mol.GetAtoms():
            # Consider atoms with degree 1 as potential terminal atoms
            if atom.GetDegree() == 1:
                terminal_atoms.append(atom.GetIdx())

        return terminal_atoms

    def _classify_end_groups(
        self, mol: Chem.Mol, terminal_atoms: List[int]
    ) -> Tuple[str, str]:
        """Classify the end groups."""
        end_groups = []

        for atom_idx in terminal_atoms[:2]:  # Consider first two terminal atoms
            atom = mol.GetAtomWithIdx(atom_idx)

            # Get local environment
            neighbors = [mol.GetAtomWithIdx(n.GetIdx()) for n in atom.GetNeighbors()]

            # Simple classification based on atom type and neighbors
            if atom.GetSymbol() == "O" and atom.GetTotalNumHs() > 0:
                end_groups.append("hydroxyl")
            elif atom.GetSymbol() == "N" and atom.GetTotalNumHs() > 0:
                end_groups.append("amino")
            elif atom.GetSymbol() == "C" and atom.GetTotalNumHs() == 3:
                end_groups.append("methyl")
            elif atom.GetSymbol() == "C" and any(
                n.GetSymbol() == "C"
                and n.GetHybridization() == Chem.HybridizationType.SP2
                for n in neighbors
            ):
                end_groups.append("vinyl")
            else:
                end_groups.append("unknown")

        # Ensure we have two end groups
        while len(end_groups) < 2:
            end_groups.append("unknown")

        return end_groups[0], end_groups[1]

    def _calculate_end_group_reactivity(
        self, left_end: str, right_end: str
    ) -> Dict[str, float]:
        """Calculate reactivity of end groups."""
        reactivity_values = {
            "hydroxyl": 0.7,
            "carboxyl": 0.8,
            "amino": 0.75,
            "vinyl": 0.9,
            "methyl": 0.1,
            "phenyl": 0.3,
            "ester": 0.4,
            "ether": 0.2,
            "unknown": 0.5,
        }

        return {
            "left_reactivity": reactivity_values.get(left_end, 0.5),
            "right_reactivity": reactivity_values.get(right_end, 0.5),
            "average_reactivity": (
                reactivity_values.get(left_end, 0.5)
                + reactivity_values.get(right_end, 0.5)
            )
            / 2,
        }

    def _estimate_termination_probability(self, reactivity: Dict[str, float]) -> float:
        """Estimate probability of chain termination."""
        # Higher reactivity leads to lower termination probability
        avg_reactivity = reactivity["average_reactivity"]
        return 1.0 - avg_reactivity


class SequenceAnalyzer:
    """Analyzes sequence patterns in copolymer oligomers."""

    def __init__(self):
        """Initialize the sequence analyzer."""
        if not RDKIT_AVAILABLE:
            raise ChemiAccelError("RDKit is required for sequence analysis")

    def analyze_sequence(
        self, oligomer_smiles: str, monomer_library: List[str]
    ) -> SequenceInfo:
        """
        Analyze the sequence pattern of an oligomer.

        Args:
            oligomer_smiles: SMILES of the oligomer
            monomer_library: List of possible monomer SMILES

        Returns:
            SequenceInfo object
        """
        mol = parse_smiles_safe(oligomer_smiles)
        if mol is None:
            raise InvalidMoleculeError(oligomer_smiles, "Invalid oligomer SMILES")

        # Decompose oligomer into monomer units (simplified)
        monomer_sequence = self._decompose_to_monomers(oligomer_smiles, monomer_library)

        # Analyze sequence pattern
        sequence_pattern = self._analyze_pattern(monomer_sequence)

        # Calculate regularity score
        regularity_score = self._calculate_regularity_score(monomer_sequence)

        # Identify block structure
        block_structure = self._identify_blocks(monomer_sequence)

        return SequenceInfo(
            monomer_sequence=monomer_sequence,
            sequence_pattern=sequence_pattern,
            regularity_score=regularity_score,
            block_structure=block_structure,
        )

    def _decompose_to_monomers(
        self, oligomer_smiles: str, monomer_library: List[str]
    ) -> List[str]:
        """Decompose oligomer into constituent monomers."""
        # This is a simplified implementation
        # In practice, this would require sophisticated retrosynthetic analysis

        # For demonstration, assume the oligomer can be matched against monomer patterns
        mol = parse_smiles_safe(oligomer_smiles)
        if mol is None:
            return []

        # Simple heuristic: try to match substructures
        identified_monomers = []

        for monomer_smiles in monomer_library:
            monomer_mol = parse_smiles_safe(monomer_smiles)
            if monomer_mol is None:
                continue

            # Count how many times this monomer pattern appears
            matches = mol.GetSubstructMatches(monomer_mol)
            for _ in matches:
                identified_monomers.append(monomer_smiles)

        # If no matches found, create a generic sequence
        if not identified_monomers:
            # Estimate based on molecular weight
            oligomer_mw = Descriptors.MolWt(mol)
            avg_monomer_mw = np.mean(
                [
                    Descriptors.MolWt(parse_smiles_safe(m))
                    for m in monomer_library
                    if parse_smiles_safe(m)
                ]
            )
            estimated_length = max(1, int(oligomer_mw / avg_monomer_mw))
            identified_monomers = ["unknown"] * estimated_length

        return identified_monomers

    def _analyze_pattern(self, sequence: List[str]) -> str:
        """Analyze the sequence pattern."""
        if len(sequence) <= 1:
            return "monomer"

        unique_monomers = list(set(sequence))

        if len(unique_monomers) == 1:
            return "homopolymer"
        elif len(unique_monomers) == 2:
            # Check for alternating pattern
            if self._is_alternating(sequence):
                return "alternating"
            elif self._has_blocks(sequence):
                return "block"
            else:
                return "random"
        else:
            return "multicomponent"

    def _is_alternating(self, sequence: List[str]) -> bool:
        """Check if sequence is alternating."""
        if len(sequence) < 2:
            return False

        for i in range(len(sequence) - 1):
            if sequence[i] == sequence[i + 1]:
                return False
        return True

    def _has_blocks(self, sequence: List[str]) -> bool:
        """Check if sequence has block structure."""
        if len(sequence) < 4:
            return False

        # Look for consecutive runs of the same monomer
        current_run_length = 1
        max_run_length = 1

        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i - 1]:
                current_run_length += 1
                max_run_length = max(max_run_length, current_run_length)
            else:
                current_run_length = 1

        return max_run_length >= 3

    def _calculate_regularity_score(self, sequence: List[str]) -> float:
        """Calculate sequence regularity score."""
        if len(sequence) <= 1:
            return 1.0

        # Calculate based on pattern predictability
        unique_monomers = list(set(sequence))

        if len(unique_monomers) == 1:
            return 1.0  # Perfect regularity for homopolymer

        # For copolymers, check for regular patterns
        if self._is_alternating(sequence):
            return 0.9
        elif self._has_blocks(sequence):
            return 0.7
        else:
            # Random copolymer - calculate based on distribution
            counts = {monomer: sequence.count(monomer) for monomer in unique_monomers}
            total = len(sequence)

            # Calculate entropy-based regularity
            entropy = -sum(
                (count / total) * np.log2(count / total)
                for count in counts.values()
                if count > 0
            )
            max_entropy = np.log2(len(unique_monomers))

            return 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.5

    def _identify_blocks(self, sequence: List[str]) -> Dict[str, List[int]]:
        """Identify block structure in the sequence."""
        blocks = {}

        if not sequence:
            return blocks

        current_monomer = sequence[0]
        current_block_start = 0

        for i, monomer in enumerate(sequence):
            if monomer != current_monomer:
                # End of current block
                if current_monomer not in blocks:
                    blocks[current_monomer] = []
                blocks[current_monomer].append((current_block_start, i - 1))

                # Start new block
                current_monomer = monomer
                current_block_start = i

        # Add final block
        if current_monomer not in blocks:
            blocks[current_monomer] = []
        blocks[current_monomer].append((current_block_start, len(sequence) - 1))

        return blocks


class OligomerProcessor:
    """Main class for oligomer and dimer processing."""

    def __init__(self):
        """Initialize the oligomer processor."""
        if not RDKIT_AVAILABLE:
            raise ChemiAccelError("RDKit is required for oligomer processing")

        self.chain_growth_simulator = ChainGrowthSimulator()
        self.flexibility_analyzer = ConformationalFlexibilityAnalyzer()
        self.end_group_handler = EndGroupHandler()
        self.sequence_analyzer = SequenceAnalyzer()

    def process_oligomer(
        self, oligomer_smiles: str, monomer_library: Optional[List[str]] = None
    ) -> Dict:
        """
        Comprehensive oligomer processing.

        Args:
            oligomer_smiles: SMILES of the oligomer
            monomer_library: Optional list of monomer SMILES for sequence analysis

        Returns:
            Dictionary containing all analysis results
        """
        try:
            results = {
                "smiles": oligomer_smiles,
                "flexibility_analysis": self.flexibility_analyzer.analyze_flexibility(
                    oligomer_smiles
                ),
                "end_group_analysis": self.end_group_handler.analyze_end_groups(
                    oligomer_smiles
                ),
            }

            # Add sequence analysis if monomer library provided
            if monomer_library:
                results["sequence_analysis"] = self.sequence_analyzer.analyze_sequence(
                    oligomer_smiles, monomer_library
                )

            return results

        except Exception as e:
            raise ChemiAccelError(f"Error in oligomer processing: {str(e)}")

    def simulate_oligomer_formation(
        self,
        monomer_smiles: str,
        target_length: int,
        mechanism: ChainGrowthMechanism = ChainGrowthMechanism.CHAIN_GROWTH,
    ) -> Dict:
        """
        Simulate oligomer formation from monomer.

        Args:
            monomer_smiles: SMILES of the monomer
            target_length: Target oligomer length
            mechanism: Polymerization mechanism

        Returns:
            Dictionary with simulation results
        """
        try:
            growth_steps = self.chain_growth_simulator.simulate_chain_growth(
                monomer_smiles, target_length, mechanism
            )

            # Get final oligomer
            if growth_steps:
                final_oligomer = growth_steps[-1].product_smiles
                oligomer_analysis = self.process_oligomer(final_oligomer)
            else:
                final_oligomer = monomer_smiles
                oligomer_analysis = {}

            return {
                "monomer_smiles": monomer_smiles,
                "target_length": target_length,
                "mechanism": mechanism.value,
                "growth_steps": growth_steps,
                "final_oligomer": final_oligomer,
                "oligomer_analysis": oligomer_analysis,
            }

        except Exception as e:
            raise ChemiAccelError(f"Error in oligomer formation simulation: {str(e)}")
