"""
Biological Transport and Metabolism Integration

This module provides models for simulating active transport mechanisms,
efflux pumps, influx transporters, and metabolic transformations in
biological systems for comprehensive transport analysis.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import cupy as cp

    CUDA_AVAILABLE = True
except ImportError:
    cp = None
    CUDA_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap

    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    jit = None
    vmap = None
    JAX_AVAILABLE = False

from ..core.exceptions import ChemiAccelError
from ..core.molecular import MolecularData
from .simulation_engine import TransportResult


class TransportMechanism(Enum):
    """Types of transport mechanisms."""

    PASSIVE_DIFFUSION = "passive_diffusion"
    FACILITATED_DIFFUSION = "facilitated_diffusion"
    ACTIVE_TRANSPORT = "active_transport"
    EFFLUX_PUMP = "efflux_pump"
    INFLUX_TRANSPORTER = "influx_transporter"
    ENDOCYTOSIS = "endocytosis"
    TRANSCYTOSIS = "transcytosis"


class MetabolismType(Enum):
    """Types of metabolic processes."""

    PHASE_I_OXIDATION = "phase_i_oxidation"
    PHASE_I_REDUCTION = "phase_i_reduction"
    PHASE_I_HYDROLYSIS = "phase_i_hydrolysis"
    PHASE_II_GLUCURONIDATION = "phase_ii_glucuronidation"
    PHASE_II_SULFATION = "phase_ii_sulfation"
    PHASE_II_ACETYLATION = "phase_ii_acetylation"
    PHASE_II_METHYLATION = "phase_ii_methylation"
    ENZYMATIC_CLEAVAGE = "enzymatic_cleavage"


@dataclass
class TransporterProperties:
    """Properties of biological transporters."""

    name: str
    type: TransportMechanism
    km: float  # Michaelis constant (μM)
    vmax: float  # Maximum transport rate (pmol/min/cm²)
    ki: Optional[float] = None  # Inhibition constant (μM)
    expression_level: float = 1.0  # Relative expression (0-1)
    tissue_specificity: List[str] = field(default_factory=list)
    substrate_specificity: Dict[str, float] = field(default_factory=dict)
    inhibitors: List[str] = field(default_factory=list)


@dataclass
class EnzymeProperties:
    """Properties of metabolic enzymes."""

    name: str
    enzyme_class: str  # e.g., "CYP3A4", "UGT1A1"
    km: float  # Michaelis constant (μM)
    vmax: float  # Maximum metabolic rate (pmol/min/mg protein)
    ki: Optional[float] = None  # Inhibition constant (μM)
    expression_level: float = 1.0  # Relative expression
    tissue_distribution: Dict[str, float] = field(default_factory=dict)
    substrate_specificity: Dict[str, float] = field(default_factory=dict)
    metabolic_pathways: List[MetabolismType] = field(default_factory=list)


@dataclass
class BiologicalSystemConfig:
    """Configuration for biological transport system."""

    transporters: List[TransporterProperties] = field(default_factory=list)
    enzymes: List[EnzymeProperties] = field(default_factory=list)
    tissue_barriers: List[str] = field(default_factory=list)
    ph_conditions: Dict[str, float] = field(default_factory=dict)
    protein_binding: Dict[str, float] = field(default_factory=dict)
    blood_flow_rates: Dict[str, float] = field(default_factory=dict)


@dataclass
class BiologicalTransportResult:
    """Results from biological transport simulation."""

    concentration_profiles: Dict[str, np.ndarray]
    transporter_fluxes: Dict[str, np.ndarray]
    metabolite_formation: Dict[str, np.ndarray]
    clearance_rates: Dict[str, float]
    bioavailability: float
    tissue_distribution: Dict[str, float]
    elimination_pathways: Dict[str, float]
    drug_interactions: List[str]
    simulation_time: float
    metadata: Dict[str, Any]


class TransporterModel(ABC):
    """Abstract base class for transporter models."""

    @abstractmethod
    def calculate_flux(
        self, concentration: float, transporter: TransporterProperties
    ) -> float:
        """Calculate transport flux."""
        pass

    @abstractmethod
    def is_substrate(self, molecule: str, transporter: TransporterProperties) -> bool:
        """Check if molecule is a substrate."""
        pass


class MichaelisMentenTransporter(TransporterModel):
    """Michaelis-Menten kinetics for transporter-mediated transport."""

    def calculate_flux(
        self, concentration: float, transporter: TransporterProperties
    ) -> float:
        """
        Calculate transport flux using Michaelis-Menten kinetics.

        Args:
            concentration: Substrate concentration (μM)
            transporter: Transporter properties

        Returns:
            Transport flux (pmol/min/cm²)
        """
        km = transporter.km
        vmax = transporter.vmax * transporter.expression_level

        # Michaelis-Menten equation: v = (Vmax * [S]) / (Km + [S])
        flux = (vmax * concentration) / (km + concentration)

        return flux

    def is_substrate(self, molecule: str, transporter: TransporterProperties) -> bool:
        """Check if molecule is a substrate for the transporter."""
        return molecule in transporter.substrate_specificity


class EffluxPumpModel(TransporterModel):
    """Model for efflux pump transport."""

    def calculate_flux(
        self, concentration: float, transporter: TransporterProperties
    ) -> float:
        """Calculate efflux pump transport flux."""
        # Efflux pumps remove molecules from cells
        flux = self._michaelis_menten_flux(concentration, transporter)

        # Efflux is in opposite direction (negative flux for cellular export)
        return -flux

    def _michaelis_menten_flux(
        self, concentration: float, transporter: TransporterProperties
    ) -> float:
        """Calculate Michaelis-Menten flux."""
        km = transporter.km
        vmax = transporter.vmax * transporter.expression_level

        return (vmax * concentration) / (km + concentration)

    def is_substrate(self, molecule: str, transporter: TransporterProperties) -> bool:
        """Check if molecule is an efflux pump substrate."""
        return molecule in transporter.substrate_specificity


class InfluxTransporterModel(TransporterModel):
    """Model for influx transporter."""

    def calculate_flux(
        self, concentration: float, transporter: TransporterProperties
    ) -> float:
        """Calculate influx transporter flux."""
        # Influx transporters bring molecules into cells
        km = transporter.km
        vmax = transporter.vmax * transporter.expression_level

        flux = (vmax * concentration) / (km + concentration)

        return flux

    def is_substrate(self, molecule: str, transporter: TransporterProperties) -> bool:
        """Check if molecule is an influx transporter substrate."""
        return molecule in transporter.substrate_specificity


class MetabolismModel(ABC):
    """Abstract base class for metabolism models."""

    @abstractmethod
    def calculate_metabolic_rate(
        self, concentration: float, enzyme: EnzymeProperties
    ) -> float:
        """Calculate metabolic rate."""
        pass

    @abstractmethod
    def predict_metabolites(self, molecule: str, enzyme: EnzymeProperties) -> List[str]:
        """Predict metabolite formation."""
        pass


class CytochromeP450Model(MetabolismModel):
    """Model for cytochrome P450 metabolism."""

    def __init__(self):
        self.reaction_patterns = {
            "CYP3A4": [
                ("CC", "C(O)C"),  # Hydroxylation
                ("c1ccccc1", "c1ccc(O)cc1"),  # Aromatic hydroxylation
            ],
            "CYP2D6": [
                ("CN", "C(O)N"),  # N-demethylation
                ("CC(C)", "CC(O)C"),  # Aliphatic hydroxylation
            ],
            "CYP1A2": [
                ("c1ccccc1", "c1ccc(O)cc1"),  # Aromatic hydroxylation
                ("CN", "CO"),  # N-demethylation
            ],
        }

    def calculate_metabolic_rate(
        self, concentration: float, enzyme: EnzymeProperties
    ) -> float:
        """
        Calculate metabolic rate using Michaelis-Menten kinetics.

        Args:
            concentration: Substrate concentration (μM)
            enzyme: Enzyme properties

        Returns:
            Metabolic rate (pmol/min/mg protein)
        """
        km = enzyme.km
        vmax = enzyme.vmax * enzyme.expression_level

        # Michaelis-Menten equation
        rate = (vmax * concentration) / (km + concentration)

        return rate

    def predict_metabolites(self, molecule: str, enzyme: EnzymeProperties) -> List[str]:
        """Predict metabolite formation from CYP metabolism."""
        metabolites = []

        if enzyme.enzyme_class in self.reaction_patterns:
            patterns = self.reaction_patterns[enzyme.enzyme_class]

            for substrate_pattern, product_pattern in patterns:
                if substrate_pattern in molecule:
                    # Simple pattern replacement (simplified)
                    metabolite = molecule.replace(substrate_pattern, product_pattern, 1)
                    metabolites.append(metabolite)

        return metabolites


class UGTModel(MetabolismModel):
    """Model for UDP-glucuronosyltransferase (UGT) metabolism."""

    def calculate_metabolic_rate(
        self, concentration: float, enzyme: EnzymeProperties
    ) -> float:
        """Calculate UGT metabolic rate."""
        km = enzyme.km
        vmax = enzyme.vmax * enzyme.expression_level

        return (vmax * concentration) / (km + concentration)

    def predict_metabolites(self, molecule: str, enzyme: EnzymeProperties) -> List[str]:
        """Predict glucuronide conjugates."""
        metabolites = []

        # Simplified glucuronidation prediction
        if "O" in molecule:  # Molecules with OH groups
            # Add glucuronic acid conjugate (simplified representation)
            glucuronide = molecule + "-Gluc"
            metabolites.append(glucuronide)

        return metabolites


class BiologicalTransportSimulator:
    """Simulator for active transport and metabolism integration."""

    def __init__(self, backend: str = "auto"):
        self.backend = self._select_backend(backend)
        self.logger = logging.getLogger(__name__)

        # Initialize transporter models
        self.transporter_models = {
            TransportMechanism.ACTIVE_TRANSPORT: MichaelisMentenTransporter(),
            TransportMechanism.EFFLUX_PUMP: EffluxPumpModel(),
            TransportMechanism.INFLUX_TRANSPORTER: InfluxTransporterModel(),
        }

        # Initialize metabolism models
        self.metabolism_models = {"CYP": CytochromeP450Model(), "UGT": UGTModel()}

        # Default biological system components
        self._initialize_default_system()

    def _select_backend(self, backend: str) -> str:
        """Select computational backend."""
        if backend == "auto":
            if CUDA_AVAILABLE:
                return "cuda"
            elif JAX_AVAILABLE:
                return "jax"
            else:
                return "cpu"
        return backend

    def _initialize_default_system(self):
        """Initialize default biological system components."""
        # Default transporters
        self.default_transporters = [
            TransporterProperties(
                name="P-gp",
                type=TransportMechanism.EFFLUX_PUMP,
                km=50.0,
                vmax=100.0,
                expression_level=1.0,
                tissue_specificity=["intestine", "liver", "kidney", "brain"],
                substrate_specificity={"default": 0.5},
            ),
            TransporterProperties(
                name="OATP1B1",
                type=TransportMechanism.INFLUX_TRANSPORTER,
                km=20.0,
                vmax=80.0,
                expression_level=1.0,
                tissue_specificity=["liver"],
                substrate_specificity={"default": 0.3},
            ),
            TransporterProperties(
                name="OCT1",
                type=TransportMechanism.INFLUX_TRANSPORTER,
                km=30.0,
                vmax=60.0,
                expression_level=1.0,
                tissue_specificity=["liver", "kidney"],
                substrate_specificity={"default": 0.4},
            ),
        ]

        # Default enzymes
        self.default_enzymes = [
            EnzymeProperties(
                name="CYP3A4",
                enzyme_class="CYP3A4",
                km=25.0,
                vmax=200.0,
                expression_level=1.0,
                tissue_distribution={"liver": 1.0, "intestine": 0.3},
                substrate_specificity={"default": 0.6},
                metabolic_pathways=[MetabolismType.PHASE_I_OXIDATION],
            ),
            EnzymeProperties(
                name="CYP2D6",
                enzyme_class="CYP2D6",
                km=15.0,
                vmax=100.0,
                expression_level=1.0,
                tissue_distribution={"liver": 1.0, "brain": 0.1},
                substrate_specificity={"default": 0.4},
                metabolic_pathways=[MetabolismType.PHASE_I_OXIDATION],
            ),
            EnzymeProperties(
                name="UGT1A1",
                enzyme_class="UGT1A1",
                km=40.0,
                vmax=150.0,
                expression_level=1.0,
                tissue_distribution={"liver": 1.0, "intestine": 0.5},
                substrate_specificity={"default": 0.3},
                metabolic_pathways=[MetabolismType.PHASE_II_GLUCURONIDATION],
            ),
        ]

    def simulate_biological_transport(
        self,
        molecules: List[str],
        system_config: Optional[BiologicalSystemConfig] = None,
        dose: float = 1.0,
        simulation_time: float = 24.0,
        time_points: int = 100,
    ) -> BiologicalTransportResult:
        """
        Simulate biological transport including active transport and metabolism.

        Args:
            molecules: List of SMILES strings
            system_config: Biological system configuration
            dose: Administered dose (mg)
            simulation_time: Simulation duration (hours)
            time_points: Number of time points

        Returns:
            BiologicalTransportResult: Comprehensive biological transport results
        """
        start_time = np.time.time()

        self.logger.info(
            f"Simulating biological transport for {len(molecules)} molecules"
        )

        # Use default system if not provided
        if system_config is None:
            system_config = BiologicalSystemConfig(
                transporters=self.default_transporters,
                enzymes=self.default_enzymes,
                tissue_barriers=["intestinal", "hepatic", "renal", "blood_brain"],
                ph_conditions={"stomach": 1.5, "intestine": 7.4, "blood": 7.4},
                protein_binding={"plasma": 0.95},
                blood_flow_rates={"liver": 1.5, "kidney": 1.2, "brain": 0.5},  # L/min
            )

        # Initialize time grid
        time_grid = np.linspace(0, simulation_time, time_points)

        # Simulate each molecule
        results = {}
        for mol_smiles in molecules:
            mol_result = self._simulate_single_molecule(
                mol_smiles, system_config, dose, time_grid
            )
            results[mol_smiles] = mol_result

        # Aggregate results
        aggregated_result = self._aggregate_results(results, time_grid)
        aggregated_result.simulation_time = np.time.time() - start_time

        return aggregated_result

    def _simulate_single_molecule(
        self,
        mol_smiles: str,
        system_config: BiologicalSystemConfig,
        dose: float,
        time_grid: np.ndarray,
    ) -> Dict[str, Any]:
        """Simulate transport and metabolism for a single molecule."""

        # Initialize compartment concentrations
        compartments = ["gut", "liver", "plasma", "kidney", "brain", "other"]
        concentrations = {comp: np.zeros(len(time_grid)) for comp in compartments}

        # Set initial dose in gut compartment
        concentrations["gut"][0] = dose  # Simplified

        # Initialize flux tracking
        transporter_fluxes = {}
        metabolite_formation = {}

        # Simulate over time
        for i in range(1, len(time_grid)):
            dt = time_grid[i] - time_grid[i - 1]

            # Calculate transporter-mediated fluxes
            for transporter in system_config.transporters:
                flux = self._calculate_transporter_flux(
                    mol_smiles, concentrations, transporter, i - 1
                )

                flux_key = f"{transporter.name}_{transporter.type.value}"
                if flux_key not in transporter_fluxes:
                    transporter_fluxes[flux_key] = np.zeros(len(time_grid))
                transporter_fluxes[flux_key][i] = flux

            # Calculate metabolic clearance
            for enzyme in system_config.enzymes:
                metabolic_rate = self._calculate_metabolic_clearance(
                    mol_smiles, concentrations, enzyme, i - 1
                )

                metabolite_key = f"{enzyme.name}_metabolites"
                if metabolite_key not in metabolite_formation:
                    metabolite_formation[metabolite_key] = np.zeros(len(time_grid))
                metabolite_formation[metabolite_key][i] = metabolic_rate

            # Update concentrations based on fluxes and metabolism
            concentrations = self._update_concentrations(
                concentrations, transporter_fluxes, metabolite_formation, i, dt
            )

        return {
            "concentrations": concentrations,
            "transporter_fluxes": transporter_fluxes,
            "metabolite_formation": metabolite_formation,
        }

    def _calculate_transporter_flux(
        self,
        mol_smiles: str,
        concentrations: Dict[str, np.ndarray],
        transporter: TransporterProperties,
        time_idx: int,
    ) -> float:
        """Calculate flux through a specific transporter."""

        # Check if molecule is a substrate
        if not self.transporter_models[transporter.type].is_substrate(
            mol_smiles, transporter
        ):
            return 0.0

        # Get relevant concentration based on transporter location
        if "liver" in transporter.tissue_specificity:
            concentration = concentrations["liver"][time_idx]
        elif "intestine" in transporter.tissue_specificity:
            concentration = concentrations["gut"][time_idx]
        else:
            concentration = concentrations["plasma"][time_idx]

        # Calculate flux using appropriate model
        flux = self.transporter_models[transporter.type].calculate_flux(
            concentration, transporter
        )

        return flux

    def _calculate_metabolic_clearance(
        self,
        mol_smiles: str,
        concentrations: Dict[str, np.ndarray],
        enzyme: EnzymeProperties,
        time_idx: int,
    ) -> float:
        """Calculate metabolic clearance rate."""

        # Check if molecule is a substrate
        if mol_smiles not in enzyme.substrate_specificity:
            return 0.0

        # Get concentration in primary tissue for enzyme
        primary_tissue = max(enzyme.tissue_distribution.items(), key=lambda x: x[1])[0]

        if primary_tissue == "liver":
            concentration = concentrations["liver"][time_idx]
        else:
            concentration = concentrations["plasma"][time_idx]

        # Calculate metabolic rate
        enzyme_class = (
            enzyme.enzyme_class.split("_")[0]
            if "_" in enzyme.enzyme_class
            else enzyme.enzyme_class
        )

        if enzyme_class in self.metabolism_models:
            rate = self.metabolism_models[enzyme_class].calculate_metabolic_rate(
                concentration, enzyme
            )
        else:
            # Default Michaelis-Menten kinetics
            rate = (enzyme.vmax * enzyme.expression_level * concentration) / (
                enzyme.km + concentration
            )

        return rate

    def _update_concentrations(
        self,
        concentrations: Dict[str, np.ndarray],
        transporter_fluxes: Dict[str, np.ndarray],
        metabolite_formation: Dict[str, np.ndarray],
        time_idx: int,
        dt: float,
    ) -> Dict[str, np.ndarray]:
        """Update compartment concentrations based on fluxes and metabolism."""

        # Simple first-order kinetics for concentration updates
        for compartment in concentrations:
            # Add influx from transporters
            influx = 0.0
            for flux_key, flux_array in transporter_fluxes.items():
                if flux_array[time_idx] > 0 and compartment in flux_key.lower():
                    influx += flux_array[time_idx]

            # Subtract efflux and metabolism
            efflux = 0.0
            metabolism = 0.0

            for flux_key, flux_array in transporter_fluxes.items():
                if flux_array[time_idx] < 0 and compartment in flux_key.lower():
                    efflux += abs(flux_array[time_idx])

            for metab_key, metab_array in metabolite_formation.items():
                if compartment in metab_key.lower():
                    metabolism += metab_array[time_idx]

            # Update concentration
            net_change = (influx - efflux - metabolism) * dt
            concentrations[compartment][time_idx] = max(
                0, concentrations[compartment][time_idx - 1] + net_change
            )

        return concentrations

    def _aggregate_results(
        self, results: Dict[str, Dict[str, Any]], time_grid: np.ndarray
    ) -> BiologicalTransportResult:
        """Aggregate results from multiple molecules."""

        # Average concentration profiles
        all_concentrations = {}
        all_transporter_fluxes = {}
        all_metabolite_formation = {}

        for mol_smiles, mol_result in results.items():
            for compartment, conc_profile in mol_result["concentrations"].items():
                if compartment not in all_concentrations:
                    all_concentrations[compartment] = []
                all_concentrations[compartment].append(conc_profile)

            for flux_key, flux_profile in mol_result["transporter_fluxes"].items():
                if flux_key not in all_transporter_fluxes:
                    all_transporter_fluxes[flux_key] = []
                all_transporter_fluxes[flux_key].append(flux_profile)

            for metab_key, metab_profile in mol_result["metabolite_formation"].items():
                if metab_key not in all_metabolite_formation:
                    all_metabolite_formation[metab_key] = []
                all_metabolite_formation[metab_key].append(metab_profile)

        # Average profiles
        avg_concentrations = {
            comp: np.mean(profiles, axis=0)
            for comp, profiles in all_concentrations.items()
        }
        avg_transporter_fluxes = {
            flux: np.mean(profiles, axis=0)
            for flux, profiles in all_transporter_fluxes.items()
        }
        avg_metabolite_formation = {
            metab: np.mean(profiles, axis=0)
            for metab, profiles in all_metabolite_formation.items()
        }

        # Calculate derived properties
        bioavailability = self._calculate_bioavailability(avg_concentrations)
        clearance_rates = self._calculate_clearance_rates(avg_concentrations, time_grid)
        tissue_distribution = self._calculate_tissue_distribution(avg_concentrations)
        elimination_pathways = self._analyze_elimination_pathways(
            avg_metabolite_formation
        )

        metadata = {
            "molecules": list(results.keys()),
            "time_grid": time_grid,
            "n_molecules": len(results),
        }

        return BiologicalTransportResult(
            concentration_profiles=avg_concentrations,
            transporter_fluxes=avg_transporter_fluxes,
            metabolite_formation=avg_metabolite_formation,
            clearance_rates=clearance_rates,
            bioavailability=bioavailability,
            tissue_distribution=tissue_distribution,
            elimination_pathways=elimination_pathways,
            drug_interactions=[],  # Simplified
            simulation_time=0.0,  # Will be set by caller
            metadata=metadata,
        )

    def _calculate_bioavailability(
        self, concentrations: Dict[str, np.ndarray]
    ) -> float:
        """Calculate bioavailability from plasma concentration profile."""
        if "plasma" in concentrations:
            # AUC-based bioavailability calculation
            plasma_conc = concentrations["plasma"]
            auc = np.trapz(plasma_conc, dx=1.0)  # Simplified

            # Normalize to theoretical maximum (simplified)
            max_possible_auc = np.sum(plasma_conc) if len(plasma_conc) > 0 else 1.0
            bioavailability = (
                min(1.0, auc / max_possible_auc) if max_possible_auc > 0 else 0.0
            )

            return bioavailability
        else:
            return 0.0

    def _calculate_clearance_rates(
        self, concentrations: Dict[str, np.ndarray], time_grid: np.ndarray
    ) -> Dict[str, float]:
        """Calculate clearance rates for different organs."""
        clearance_rates = {}

        for organ in ["liver", "kidney"]:
            if organ in concentrations:
                conc_profile = concentrations[organ]

                # Calculate clearance as elimination rate constant
                if len(conc_profile) > 1 and np.max(conc_profile) > 0:
                    # Fit exponential decay (simplified)
                    max_conc = np.max(conc_profile)
                    final_conc = conc_profile[-1]

                    if final_conc > 0 and max_conc > final_conc:
                        k_elimination = -np.log(final_conc / max_conc) / time_grid[-1]
                        clearance_rates[f"{organ}_clearance"] = k_elimination
                    else:
                        clearance_rates[f"{organ}_clearance"] = 0.0
                else:
                    clearance_rates[f"{organ}_clearance"] = 0.0

        return clearance_rates

    def _calculate_tissue_distribution(
        self, concentrations: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Calculate tissue distribution coefficients."""
        tissue_distribution = {}

        if "plasma" in concentrations:
            plasma_auc = np.trapz(concentrations["plasma"], dx=1.0)

            for tissue, conc_profile in concentrations.items():
                if tissue != "plasma":
                    tissue_auc = np.trapz(conc_profile, dx=1.0)

                    if plasma_auc > 0:
                        kp = tissue_auc / plasma_auc  # Tissue-to-plasma ratio
                        tissue_distribution[tissue] = kp
                    else:
                        tissue_distribution[tissue] = 0.0

        return tissue_distribution

    def _analyze_elimination_pathways(
        self, metabolite_formation: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Analyze relative contribution of elimination pathways."""
        elimination_pathways = {}

        # Calculate total metabolite formation for each pathway
        total_elimination = 0.0
        pathway_contributions = {}

        for pathway, formation_profile in metabolite_formation.items():
            pathway_contribution = np.trapz(formation_profile, dx=1.0)
            pathway_contributions[pathway] = pathway_contribution
            total_elimination += pathway_contribution

        # Calculate relative contributions
        if total_elimination > 0:
            for pathway, contribution in pathway_contributions.items():
                elimination_pathways[pathway] = contribution / total_elimination
        else:
            elimination_pathways = {
                pathway: 0.0 for pathway in metabolite_formation.keys()
            }

        return elimination_pathways


class BiologicalTransportModel:
    """
    Comprehensive model for active transport and metabolism integration.

    This class provides a unified interface for simulating the impact of
    active transport mechanisms and metabolic transformations on molecular
    distribution and clearance in biological systems.
    """

    def __init__(self, backend: str = "auto"):
        """
        Initialize the biological transport model.

        Args:
            backend: Computational backend ('cuda', 'jax', 'cpu', 'auto')
        """
        self.backend = backend
        self.logger = logging.getLogger(__name__)

        # Initialize simulator
        self.simulator = BiologicalTransportSimulator(backend)

    def simulate_drug_disposition(
        self,
        molecules: Union[List[str], List[MolecularData]],
        dose: float = 1.0,
        route: str = "oral",
        system_config: Optional[BiologicalSystemConfig] = None,
    ) -> BiologicalTransportResult:
        """
        Simulate complete drug disposition including absorption, distribution, metabolism, and excretion.

        Args:
            molecules: List of SMILES strings or MolecularData objects
            dose: Administered dose (mg)
            route: Administration route ('oral', 'iv', 'topical')
            system_config: Biological system configuration

        Returns:
            BiologicalTransportResult: Complete ADME simulation results
        """
        # Convert MolecularData to SMILES if needed
        if molecules and isinstance(molecules[0], MolecularData):
            molecule_smiles = [mol.smiles for mol in molecules]
        else:
            molecule_smiles = molecules

        self.logger.info(
            f"Simulating drug disposition for {len(molecule_smiles)} molecules"
        )

        # Adjust simulation parameters based on route
        if route == "oral":
            simulation_time = 48.0  # hours
        elif route == "iv":
            simulation_time = 24.0  # hours
        else:
            simulation_time = 72.0  # hours for topical

        return self.simulator.simulate_biological_transport(
            molecule_smiles, system_config, dose, simulation_time
        )

    def analyze_drug_interactions(
        self,
        molecules: List[str],
        system_config: Optional[BiologicalSystemConfig] = None,
    ) -> Dict[str, Any]:
        """
        Analyze potential drug-drug interactions through transporters and enzymes.

        Args:
            molecules: List of SMILES strings
            system_config: Biological system configuration

        Returns:
            Dictionary containing interaction analysis
        """
        self.logger.info("Analyzing drug-drug interactions")

        interactions = {
            "transporter_interactions": [],
            "metabolic_interactions": [],
            "competitive_inhibition": [],
            "induction_potential": [],
        }

        # Use default system if not provided
        if system_config is None:
            system_config = BiologicalSystemConfig(
                transporters=self.simulator.default_transporters,
                enzymes=self.simulator.default_enzymes,
            )

        # Analyze transporter interactions
        for transporter in system_config.transporters:
            substrates = []
            inhibitors = []

            for mol in molecules:
                if self.simulator.transporter_models[transporter.type].is_substrate(
                    mol, transporter
                ):
                    substrates.append(mol)
                if mol in transporter.inhibitors:
                    inhibitors.append(mol)

            if len(substrates) > 1:
                interactions["transporter_interactions"].append(
                    {
                        "transporter": transporter.name,
                        "competing_substrates": substrates,
                        "potential_inhibitors": inhibitors,
                    }
                )

        # Analyze metabolic interactions
        for enzyme in system_config.enzymes:
            substrates = []

            for mol in molecules:
                if mol in enzyme.substrate_specificity:
                    substrates.append(mol)

            if len(substrates) > 1:
                interactions["metabolic_interactions"].append(
                    {"enzyme": enzyme.name, "competing_substrates": substrates}
                )

        return interactions

    def predict_bioavailability(
        self,
        molecules: List[str],
        system_config: Optional[BiologicalSystemConfig] = None,
    ) -> Dict[str, float]:
        """
        Predict oral bioavailability considering transport and metabolism.

        Args:
            molecules: List of SMILES strings
            system_config: Biological system configuration

        Returns:
            Dictionary mapping molecules to predicted bioavailability
        """
        self.logger.info("Predicting bioavailability")

        bioavailabilities = {}

        for mol in molecules:
            # Simulate oral administration
            result = self.simulate_drug_disposition(
                [mol], dose=1.0, route="oral", system_config=system_config
            )
            bioavailabilities[mol] = result.bioavailability

        return bioavailabilities

    def optimize_dosing_regimen(
        self,
        molecule: str,
        target_plasma_concentration: float,
        system_config: Optional[BiologicalSystemConfig] = None,
    ) -> Dict[str, Any]:
        """
        Optimize dosing regimen to achieve target plasma concentrations.

        Args:
            molecule: SMILES string
            target_plasma_concentration: Target concentration (μM)
            system_config: Biological system configuration

        Returns:
            Dictionary containing optimal dosing recommendations
        """
        self.logger.info(
            f"Optimizing dosing regimen for target concentration {target_plasma_concentration} μM"
        )

        # Test different doses
        test_doses = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]  # mg
        dose_results = {}

        for dose in test_doses:
            result = self.simulate_drug_disposition(
                [molecule], dose=dose, system_config=system_config
            )

            # Get maximum plasma concentration
            if "plasma" in result.concentration_profiles:
                max_conc = np.max(result.concentration_profiles["plasma"])
                dose_results[dose] = max_conc

        # Find optimal dose through interpolation
        if dose_results:
            doses = list(dose_results.keys())
            concentrations = list(dose_results.values())

            # Linear interpolation to find dose for target concentration
            optimal_dose = np.interp(target_plasma_concentration, concentrations, doses)

            return {
                "optimal_dose": optimal_dose,
                "dose_response_data": dose_results,
                "target_concentration": target_plasma_concentration,
                "predicted_max_concentration": target_plasma_concentration,
            }
        else:
            return {"optimal_dose": 1.0, "error": "Could not determine optimal dose"}
