"""
Core classes for mixture and formulation design.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from ..core.backend import Backend
from ..core.exceptions import ChemiAccelError
from ..core.molecular import MolecularData


@dataclass
class ComponentData:
    """Data structure for individual mixture components."""

    smiles: str
    mol_fraction: float
    mass_fraction: Optional[float] = None
    volume_fraction: Optional[float] = None
    molecular_weight: Optional[float] = None
    density: Optional[float] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    interactions: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate missing properties."""
        if self.smiles:
            mol = Chem.MolFromSmiles(self.smiles)
            if mol and self.molecular_weight is None:
                self.molecular_weight = Descriptors.MolWt(mol)


@dataclass
class MixtureData:
    """Data structure for mixture systems."""

    components: List[ComponentData]
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    mixture_properties: Dict[str, Any] = field(default_factory=dict)
    phase_data: Dict[str, Any] = field(default_factory=dict)
    interaction_matrix: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate and normalize mixture data."""
        if not self.components:
            raise ValueError("Mixture must contain at least one component")

        # Normalize mole fractions
        total_moles = sum(comp.mol_fraction for comp in self.components)
        if abs(total_moles - 1.0) > 1e-6:
            for comp in self.components:
                comp.mol_fraction /= total_moles

    @property
    def n_components(self) -> int:
        """Number of components in mixture."""
        return len(self.components)

    @property
    def smiles_list(self) -> List[str]:
        """List of SMILES strings for all components."""
        return [comp.smiles for comp in self.components]

    @property
    def composition_vector(self) -> np.ndarray:
        """Composition as numpy array."""
        return np.array([comp.mol_fraction for comp in self.components])


class MixtureCalculator:
    """
    Main calculator for mixture and formulation design tasks.

    Provides unified interface for:
    - Phase behavior prediction
    - Interaction modeling
    - Property calculation
    - Inverse design
    """

    def __init__(
        self,
        backend: Optional[Union[str, Backend]] = None,
        temperature: Optional[float] = None,
        pressure: Optional[float] = None,
        use_gpu: bool = True,
    ):
        """
        Initialize mixture calculator.

        Args:
            backend: Computational backend ('cuda', 'jax', 'cpu')
            temperature: Default temperature (K)
            pressure: Default pressure (Pa)
            use_gpu: Whether to use GPU acceleration
        """
        self.backend = self._setup_backend(backend, use_gpu)
        self.default_temperature = temperature or 298.15
        self.default_pressure = pressure or 101325.0

        # Initialize submodules
        self._phase_predictor = None
        self._interaction_modeler = None
        self._inverse_designer = None
        self._process_integrator = None
        self._visualizer = None

    def _setup_backend(
        self, backend: Optional[Union[str, Backend]], use_gpu: bool
    ) -> Backend:
        """Setup computational backend."""
        if isinstance(backend, Backend):
            return backend

        if backend is None:
            backend = "cuda" if use_gpu else "cpu"

        from ..core.backend import get_backend

        return get_backend(backend)

    def create_mixture(
        self,
        components: List[Union[str, Tuple[str, float]]],
        temperature: Optional[float] = None,
        pressure: Optional[float] = None,
        **kwargs,
    ) -> MixtureData:
        """
        Create mixture data structure.

        Args:
            components: List of SMILES or (SMILES, mole_fraction) tuples
            temperature: Temperature (K)
            pressure: Pressure (Pa)
            **kwargs: Additional component properties

        Returns:
            MixtureData object
        """
        comp_data = []

        for i, comp in enumerate(components):
            if isinstance(comp, str):
                # Equal mole fractions if not specified
                smiles = comp
                mol_frac = 1.0 / len(components)
            elif isinstance(comp, (list, tuple)) and len(comp) == 2:
                smiles, mol_frac = comp
            else:
                raise ValueError(f"Invalid component format: {comp}")

            comp_data.append(
                ComponentData(
                    smiles=smiles,
                    mol_fraction=mol_frac,
                    **kwargs.get(f"component_{i}", {}),
                )
            )

        return MixtureData(
            components=comp_data,
            temperature=temperature or self.default_temperature,
            pressure=pressure or self.default_pressure,
        )

    def calculate_mixture_properties(
        self, mixture: MixtureData, properties: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate mixture properties.

        Args:
            mixture: Mixture data
            properties: List of properties to calculate

        Returns:
            Dictionary of calculated properties
        """
        if properties is None:
            properties = [
                "molecular_weight_avg",
                "density_mixture",
                "viscosity_mixture",
                "surface_tension",
                "dielectric_constant",
            ]

        results = {}

        # Average molecular weight
        if "molecular_weight_avg" in properties:
            mw_avg = sum(
                comp.mol_fraction * (comp.molecular_weight or 0)
                for comp in mixture.components
            )
            results["molecular_weight_avg"] = mw_avg

        # Additional properties would be calculated here
        # using specific mixing rules and models

        return results

    @property
    def phase_predictor(self):
        """Get phase behavior predictor."""
        if self._phase_predictor is None:
            from .phase_behavior import PhaseBehaviorPredictor

            self._phase_predictor = PhaseBehaviorPredictor(self.backend)
        return self._phase_predictor

    @property
    def interaction_modeler(self):
        """Get interaction modeler."""
        if self._interaction_modeler is None:
            from .interactions import InteractionModeler

            self._interaction_modeler = InteractionModeler(self.backend)
        return self._interaction_modeler

    @property
    def inverse_designer(self):
        """Get inverse designer."""
        if self._inverse_designer is None:
            from .inverse_design import MixtureInverseDesigner

            self._inverse_designer = MixtureInverseDesigner(self.backend)
        return self._inverse_designer

    @property
    def process_integrator(self):
        """Get dynamic process integrator."""
        if self._process_integrator is None:
            from .dynamic_processes import DynamicProcessIntegrator

            self._process_integrator = DynamicProcessIntegrator(self.backend)
        return self._process_integrator

    @property
    def visualizer(self):
        """Get visualization interface."""
        if self._visualizer is None:
            from .visualization import MixtureVisualizer

            self._visualizer = MixtureVisualizer()
        return self._visualizer

    def predict_phase_behavior(
        self,
        mixture: MixtureData,
        temperature_range: Optional[Tuple[float, float]] = None,
        pressure_range: Optional[Tuple[float, float]] = None,
        **kwargs,
    ):
        """Predict phase behavior."""
        return self.phase_predictor.predict_phase_equilibrium(
            mixture, temperature_range, pressure_range, **kwargs
        )

    def model_interactions(
        self,
        mixture: MixtureData,
        interaction_types: Optional[List[str]] = None,
        **kwargs,
    ):
        """Model inter-component interactions."""
        return self.interaction_modeler.calculate_interactions(
            mixture, interaction_types, **kwargs
        )

    def design_optimal_mixture(
        self,
        target_properties: Dict[str, float],
        component_library: List[str],
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Design optimal mixture composition."""
        return self.inverse_designer.optimize_composition(
            target_properties, component_library, constraints, **kwargs
        )

    def integrate_dynamic_processes(
        self, mixture: MixtureData, process_conditions: Dict[str, Any], **kwargs
    ):
        """Integrate with dynamic process simulation."""
        return self.process_integrator.simulate_process(
            mixture, process_conditions, **kwargs
        )
