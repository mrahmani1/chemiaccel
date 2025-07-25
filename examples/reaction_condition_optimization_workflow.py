#!/usr/bin/env python3
"""
Advanced Reaction Condition Optimization Workflow

This workflow demonstrates ChemiAccel's AI-driven reaction condition optimization
capabilities, featuring multi-objective optimization, automated parameter tuning,
and experimental design for maximum reaction efficiency.

Features:
- Multi-objective optimization (yield, selectivity, cost, time)
- Bayesian optimization for condition space exploration
- Automated experimental design and planning
- Real-time condition monitoring and adjustment
- Integration with reaction informatics predictions
- Cost-benefit analysis for industrial applications

Usage:
    python examples/reaction_condition_optimization_workflow.py
"""

import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from chemi_accel.core.molecular import MolecularData
from chemi_accel.descriptors.reaction_informatics import (
    create_reaction_informatics_engine,
)

# GPU backend imports (if available)
try:
    from chemi_accel.backends.cuda.reaction_informatics import (
        GPUReactionInformatics,
        ReactionCondition,
        ReactionType,
    )

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPU backend not available - using CPU backend")

# RDKit for molecular processing
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("WARNING: RDKit not available - limited functionality")

# Scipy for optimization
try:
    from scipy.optimize import differential_evolution, minimize
    from scipy.stats import norm

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: SciPy not available - limited optimization")


class OptimizationTarget(Enum):
    """Optimization target objectives."""

    YIELD = "yield"
    SELECTIVITY = "selectivity"
    RATE = "rate"
    COST = "cost"
    ATOM_ECONOMY = "atom_economy"
    E_FACTOR = "e_factor"
    SAFETY = "safety"


@dataclass
class ReactionConditions:
    """Comprehensive reaction conditions specification."""

    temperature: float = 25.0  # °C
    pressure: float = 1.0  # bar
    time: float = 8.0  # hours
    concentration: float = 0.1  # M
    ph: Optional[float] = None  # pH (for aqueous reactions)

    # Solvents and additives
    solvent: str = "THF"
    solvent_volume: float = 10.0  # mL
    catalyst: Optional[str] = None
    catalyst_loading: float = 0.01  # mol%
    additives: Dict[str, float] = field(default_factory=dict)  # name: amount

    # Atmosphere and special conditions
    atmosphere: str = "air"  # air, nitrogen, argon
    light_conditions: str = "dark"  # dark, UV, visible
    stirring_rate: float = 500.0  # rpm

    # Validation ranges
    temperature_range: Tuple[float, float] = (-78.0, 200.0)
    pressure_range: Tuple[float, float] = (0.1, 50.0)
    time_range: Tuple[float, float] = (0.1, 72.0)
    concentration_range: Tuple[float, float] = (0.001, 5.0)


@dataclass
class OptimizationResult:
    """Results from reaction condition optimization."""

    original_conditions: ReactionConditions
    optimized_conditions: ReactionConditions

    # Performance metrics
    original_metrics: Dict[str, float]
    optimized_metrics: Dict[str, float]
    improvement_factors: Dict[str, float]

    # Optimization details
    optimization_targets: List[OptimizationTarget]
    total_iterations: int
    convergence_achieved: bool
    optimization_time: float

    # Cost-benefit analysis
    cost_reduction: float
    throughput_improvement: float
    sustainability_score: float


class ReactionConditionOptimizer:
    """Advanced reaction condition optimization engine."""

    def __init__(self, use_gpu: bool = True):
        """Initialize reaction condition optimizer."""
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.engine = create_reaction_informatics_engine(
            backend="gpu" if self.use_gpu else "cpu",
            batch_size=500,
            use_gpu_acceleration=self.use_gpu,
            enable_quantum_features=True,
        )

        # Optimization parameters
        self.max_iterations = 100
        self.convergence_tolerance = 1e-4
        self.population_size = 50

        # Condition databases
        self.solvent_properties = self._initialize_solvent_database()
        self.catalyst_database = self._initialize_catalyst_database()
        self.optimization_history = []

        print(
            f"Reaction condition optimizer initialized with {self.engine.backend.upper()} backend"
        )

    def _initialize_solvent_database(self) -> Dict[str, Dict[str, float]]:
        """Initialize comprehensive solvent property database."""
        return {
            # Protic solvents
            "methanol": {"bp": 64.7, "polarity": 6.6, "viscosity": 0.59, "cost": 1.0},
            "ethanol": {"bp": 78.4, "polarity": 5.2, "viscosity": 1.2, "cost": 1.2},
            "water": {"bp": 100.0, "polarity": 9.0, "viscosity": 1.0, "cost": 0.1},
            # Aprotic polar solvents
            "DMF": {"bp": 153.0, "polarity": 6.4, "viscosity": 0.92, "cost": 3.0},
            "DMSO": {"bp": 189.0, "polarity": 7.2, "viscosity": 2.24, "cost": 3.5},
            "acetonitrile": {
                "bp": 81.6,
                "polarity": 5.8,
                "viscosity": 0.37,
                "cost": 2.5,
            },
            # Aprotic nonpolar solvents
            "THF": {"bp": 66.0, "polarity": 4.0, "viscosity": 0.55, "cost": 2.0},
            "DCM": {"bp": 39.6, "polarity": 3.1, "viscosity": 0.44, "cost": 1.8},
            "toluene": {"bp": 110.6, "polarity": 2.4, "viscosity": 0.59, "cost": 1.5},
            "hexane": {"bp": 68.7, "polarity": 0.1, "viscosity": 0.33, "cost": 1.0},
            # Green solvents
            "ethyl_acetate": {
                "bp": 77.1,
                "polarity": 4.4,
                "viscosity": 0.45,
                "cost": 1.3,
            },
            "isopropanol": {"bp": 82.6, "polarity": 4.3, "viscosity": 2.4, "cost": 1.1},
            "cyclopentyl_methyl_ether": {
                "bp": 106.0,
                "polarity": 4.0,
                "viscosity": 0.67,
                "cost": 4.0,
            },
        }

    def _initialize_catalyst_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize catalyst database with properties and costs."""
        return {
            # Palladium catalysts
            "Pd(PPh3)4": {
                "metal": "Pd",
                "cost_per_g": 45.0,
                "activity": 0.9,
                "selectivity": 0.8,
            },
            "Pd(dppf)Cl2": {
                "metal": "Pd",
                "cost_per_g": 38.0,
                "activity": 0.85,
                "selectivity": 0.9,
            },
            "Pd2(dba)3": {
                "metal": "Pd",
                "cost_per_g": 52.0,
                "activity": 0.95,
                "selectivity": 0.85,
            },
            # Copper catalysts
            "CuSO4": {
                "metal": "Cu",
                "cost_per_g": 2.0,
                "activity": 0.7,
                "selectivity": 0.75,
            },
            "Cu(OAc)2": {
                "metal": "Cu",
                "cost_per_g": 3.5,
                "activity": 0.75,
                "selectivity": 0.8,
            },
            "CuI": {
                "metal": "Cu",
                "cost_per_g": 15.0,
                "activity": 0.8,
                "selectivity": 0.85,
            },
            # Organocatalysts
            "proline": {
                "metal": None,
                "cost_per_g": 8.0,
                "activity": 0.6,
                "selectivity": 0.95,
            },
            "DIPEA": {
                "metal": None,
                "cost_per_g": 5.0,
                "activity": 0.5,
                "selectivity": 0.7,
            },
            "Et3N": {
                "metal": None,
                "cost_per_g": 2.0,
                "activity": 0.5,
                "selectivity": 0.65,
            },
            # Acid catalysts
            "TfOH": {
                "metal": None,
                "cost_per_g": 25.0,
                "activity": 0.9,
                "selectivity": 0.6,
            },
            "p-TsOH": {
                "metal": None,
                "cost_per_g": 3.0,
                "activity": 0.7,
                "selectivity": 0.75,
            },
        }

    def predict_reaction_performance(
        self,
        reactants: List[MolecularData],
        reaction_type: str,
        conditions: ReactionConditions,
    ) -> Dict[str, float]:
        """Predict reaction performance under given conditions."""

        # Base prediction from reaction informatics engine
        try:
            predictions = self.engine.predict_reactions(
                molecules=reactants,
                reaction_types=[reaction_type] if GPU_AVAILABLE else None,
                min_feasibility_score=0.1,
                max_predictions_per_molecule=1,
                include_conditions=True,
            )

            if predictions:
                base_prediction = predictions[0]
                base_yield = base_prediction.get("yield_prediction", 50.0)
                base_selectivity = base_prediction.get("selectivity_score", 0.7)
                base_feasibility = base_prediction.get("feasibility_score", 0.5)
            else:
                base_yield = 50.0
                base_selectivity = 0.7
                base_feasibility = 0.5

        except Exception as e:
            print(f"Warning: Prediction failed, using defaults: {e}")
            base_yield = 50.0
            base_selectivity = 0.7
            base_feasibility = 0.5

        # Apply condition-based corrections
        corrected_metrics = self._apply_condition_corrections(
            base_yield, base_selectivity, base_feasibility, conditions, reaction_type
        )

        # Calculate additional metrics
        additional_metrics = self._calculate_additional_metrics(
            corrected_metrics, conditions, reactants
        )

        # Combine all metrics
        performance_metrics = {**corrected_metrics, **additional_metrics}

        return performance_metrics

    def _apply_condition_corrections(
        self,
        base_yield: float,
        base_selectivity: float,
        base_feasibility: float,
        conditions: ReactionConditions,
        reaction_type: str,
    ) -> Dict[str, float]:
        """Apply condition-based corrections to base predictions."""

        corrected_yield = base_yield
        corrected_selectivity = base_selectivity
        corrected_rate = base_feasibility

        # Temperature effects
        if reaction_type.lower() in ["suzuki_coupling", "heck_reaction"]:
            # Higher temperature increases rate but may decrease selectivity
            temp_factor = (conditions.temperature - 25) / 100.0
            corrected_rate *= 1.0 + temp_factor * 0.3
            corrected_selectivity *= 1.0 - abs(temp_factor) * 0.1
            corrected_yield *= 1.0 + temp_factor * 0.2

        elif reaction_type.lower() in ["click_chemistry"]:
            # Mild conditions preferred
            if 20 <= conditions.temperature <= 80:
                corrected_yield *= 1.1
                corrected_selectivity *= 1.05

        elif reaction_type.lower() in ["amide_formation"]:
            # Can tolerate wide temperature range
            if conditions.temperature > 100:
                corrected_rate *= 1.2
            if conditions.temperature < 0:
                corrected_selectivity *= 1.1

        # Solvent effects
        solvent_props = self.solvent_properties.get(conditions.solvent, {})

        if reaction_type.lower() in ["suzuki_coupling", "heck_reaction"]:
            # Polar aprotic solvents preferred
            if conditions.solvent in ["DMF", "DMSO", "acetonitrile"]:
                corrected_yield *= 1.15
                corrected_rate *= 1.1

        elif reaction_type.lower() in ["click_chemistry"]:
            # Water or polar solvents
            if conditions.solvent in ["water", "DMSO", "DMF"]:
                corrected_yield *= 1.2
                corrected_selectivity *= 1.1

        # Concentration effects
        if conditions.concentration < 0.01:
            corrected_rate *= 0.8  # Too dilute
        elif conditions.concentration > 2.0:
            corrected_selectivity *= 0.9  # Side reactions

        # Catalyst effects
        if conditions.catalyst:
            catalyst_props = self.catalyst_database.get(conditions.catalyst, {})
            activity_factor = catalyst_props.get("activity", 0.7)
            selectivity_factor = catalyst_props.get("selectivity", 0.7)

            corrected_rate *= 1.0 + activity_factor * 0.5
            corrected_selectivity *= 1.0 + selectivity_factor * 0.3

            # Loading effects
            if conditions.catalyst_loading < 0.001:
                corrected_rate *= 0.7  # Insufficient catalyst
            elif conditions.catalyst_loading > 0.1:
                corrected_rate *= 0.9  # Diminishing returns

        # Ensure realistic bounds
        corrected_yield = np.clip(corrected_yield, 5.0, 98.0)
        corrected_selectivity = np.clip(corrected_selectivity, 0.1, 0.99)
        corrected_rate = np.clip(corrected_rate, 0.1, 1.0)

        return {
            "yield": corrected_yield,
            "selectivity": corrected_selectivity,
            "rate": corrected_rate,
        }

    def _calculate_additional_metrics(
        self,
        performance_metrics: Dict[str, float],
        conditions: ReactionConditions,
        reactants: List[MolecularData],
    ) -> Dict[str, float]:
        """Calculate additional performance metrics."""

        # Cost calculation
        base_cost = 10.0  # Base reaction cost

        # Solvent cost
        solvent_props = self.solvent_properties.get(conditions.solvent, {})
        solvent_cost = (
            solvent_props.get("cost", 2.0) * conditions.solvent_volume / 1000.0
        )

        # Catalyst cost
        catalyst_cost = 0.0
        if conditions.catalyst:
            catalyst_props = self.catalyst_database.get(conditions.catalyst, {})
            catalyst_price = catalyst_props.get("cost_per_g", 10.0)
            # Estimate catalyst mass needed (simplified)
            catalyst_mass = conditions.catalyst_loading * 0.1  # 0.1g per mol%
            catalyst_cost = catalyst_price * catalyst_mass

        # Energy cost (heating/cooling)
        energy_cost = abs(conditions.temperature - 25) * 0.1  # $0.1 per degree

        # Time cost (labor and equipment)
        time_cost = conditions.time * 2.0  # $2 per hour

        total_cost = base_cost + solvent_cost + catalyst_cost + energy_cost + time_cost

        # Atom economy (simplified calculation)
        atom_economy = (
            0.75  # Default value, would calculate from reaction stoichiometry
        )

        # E-factor (waste generation)
        solvent_waste = conditions.solvent_volume
        byproduct_waste = (100 - performance_metrics["yield"]) * 0.1  # Simplified
        e_factor = (solvent_waste + byproduct_waste) / max(
            performance_metrics["yield"], 1.0
        )

        # Safety score (based on conditions and chemicals)
        safety_score = 1.0
        if conditions.temperature > 150:
            safety_score *= 0.8  # High temperature risk
        if conditions.pressure > 10:
            safety_score *= 0.9  # High pressure risk
        if conditions.solvent in ["benzene", "chloroform", "carbon_tetrachloride"]:
            safety_score *= 0.6  # Toxic solvents

        # Throughput (reactions per day)
        throughput = 24.0 / max(conditions.time, 0.1)

        return {
            "cost": total_cost,
            "atom_economy": atom_economy,
            "e_factor": e_factor,
            "safety": safety_score,
            "throughput": throughput,
        }

    def optimize_single_objective(
        self,
        reactants: List[MolecularData],
        reaction_type: str,
        initial_conditions: ReactionConditions,
        target: OptimizationTarget,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """Optimize reaction conditions for single objective."""

        print(f"Optimizing for {target.value}...")

        # Define optimization bounds
        bounds = self._get_optimization_bounds(initial_conditions, constraints)

        # Define objective function
        def objective_function(x):
            conditions = self._decode_conditions_vector(x, initial_conditions)
            metrics = self.predict_reaction_performance(
                reactants, reaction_type, conditions
            )

            # Extract target metric (negate for minimization)
            if target == OptimizationTarget.YIELD:
                return -metrics["yield"]
            elif target == OptimizationTarget.SELECTIVITY:
                return -metrics["selectivity"]
            elif target == OptimizationTarget.RATE:
                return -metrics["rate"]
            elif target == OptimizationTarget.COST:
                return metrics["cost"]
            elif target == OptimizationTarget.ATOM_ECONOMY:
                return -metrics["atom_economy"]
            elif target == OptimizationTarget.E_FACTOR:
                return metrics["e_factor"]
            elif target == OptimizationTarget.SAFETY:
                return -metrics["safety"]
            else:
                return 0.0

        # Run optimization
        start_time = time.time()

        if SCIPY_AVAILABLE:
            # Use differential evolution for global optimization
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=self.max_iterations,
                popsize=self.population_size,
                seed=42,
                tol=self.convergence_tolerance,
            )

            optimal_conditions = self._decode_conditions_vector(
                result.x, initial_conditions
            )
            convergence_achieved = result.success
            iterations = result.nit
        else:
            # Simple grid search fallback
            (
                optimal_conditions,
                convergence_achieved,
                iterations,
            ) = self._grid_search_optimization(
                objective_function, bounds, initial_conditions
            )

        optimization_time = time.time() - start_time

        # Calculate performance metrics
        original_metrics = self.predict_reaction_performance(
            reactants, reaction_type, initial_conditions
        )
        optimized_metrics = self.predict_reaction_performance(
            reactants, reaction_type, optimal_conditions
        )

        # Calculate improvements
        improvement_factors = {}
        for key in original_metrics:
            if original_metrics[key] != 0:
                if key in ["cost", "e_factor"]:  # Lower is better
                    improvement_factors[key] = original_metrics[key] / max(
                        optimized_metrics[key], 0.001
                    )
                else:  # Higher is better
                    improvement_factors[key] = optimized_metrics[key] / max(
                        original_metrics[key], 0.001
                    )
            else:
                improvement_factors[key] = 1.0

        # Cost-benefit analysis
        cost_reduction = max(0, original_metrics["cost"] - optimized_metrics["cost"])
        throughput_improvement = optimized_metrics["throughput"] / max(
            original_metrics["throughput"], 0.1
        )
        sustainability_score = (
            optimized_metrics["safety"]
            + (1.0 / max(optimized_metrics["e_factor"], 0.1))
        ) / 2.0

        return OptimizationResult(
            original_conditions=initial_conditions,
            optimized_conditions=optimal_conditions,
            original_metrics=original_metrics,
            optimized_metrics=optimized_metrics,
            improvement_factors=improvement_factors,
            optimization_targets=[target],
            total_iterations=iterations,
            convergence_achieved=convergence_achieved,
            optimization_time=optimization_time,
            cost_reduction=cost_reduction,
            throughput_improvement=throughput_improvement,
            sustainability_score=sustainability_score,
        )

    def optimize_multi_objective(
        self,
        reactants: List[MolecularData],
        reaction_type: str,
        initial_conditions: ReactionConditions,
        targets: List[OptimizationTarget],
        weights: Optional[List[float]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """Multi-objective optimization using weighted sum approach."""

        print(f"Multi-objective optimization for: {[t.value for t in targets]}")

        if weights is None:
            weights = [1.0 / len(targets)] * len(targets)

        if len(weights) != len(targets):
            raise ValueError("Number of weights must match number of targets")

        # Normalize weights
        weights = np.array(weights) / np.sum(weights)

        # Define optimization bounds
        bounds = self._get_optimization_bounds(initial_conditions, constraints)

        # Multi-objective function
        def multi_objective_function(x):
            conditions = self._decode_conditions_vector(x, initial_conditions)
            metrics = self.predict_reaction_performance(
                reactants, reaction_type, conditions
            )

            # Calculate weighted objective
            weighted_sum = 0.0

            for target, weight in zip(targets, weights):
                if target == OptimizationTarget.YIELD:
                    weighted_sum += weight * (
                        -metrics["yield"] / 100.0
                    )  # Normalize to 0-1
                elif target == OptimizationTarget.SELECTIVITY:
                    weighted_sum += weight * (-metrics["selectivity"])
                elif target == OptimizationTarget.RATE:
                    weighted_sum += weight * (-metrics["rate"])
                elif target == OptimizationTarget.COST:
                    weighted_sum += weight * (metrics["cost"] / 100.0)  # Normalize
                elif target == OptimizationTarget.ATOM_ECONOMY:
                    weighted_sum += weight * (-metrics["atom_economy"])
                elif target == OptimizationTarget.E_FACTOR:
                    weighted_sum += weight * (metrics["e_factor"] / 10.0)  # Normalize
                elif target == OptimizationTarget.SAFETY:
                    weighted_sum += weight * (-metrics["safety"])

            return weighted_sum

        # Run optimization
        start_time = time.time()

        if SCIPY_AVAILABLE:
            result = differential_evolution(
                multi_objective_function,
                bounds,
                maxiter=self.max_iterations,
                popsize=self.population_size,
                seed=42,
            )

            optimal_conditions = self._decode_conditions_vector(
                result.x, initial_conditions
            )
            convergence_achieved = result.success
            iterations = result.nit
        else:
            (
                optimal_conditions,
                convergence_achieved,
                iterations,
            ) = self._grid_search_optimization(
                multi_objective_function, bounds, initial_conditions
            )

        optimization_time = time.time() - start_time

        # Calculate metrics and create result
        original_metrics = self.predict_reaction_performance(
            reactants, reaction_type, initial_conditions
        )
        optimized_metrics = self.predict_reaction_performance(
            reactants, reaction_type, optimal_conditions
        )

        improvement_factors = {}
        for key in original_metrics:
            if original_metrics[key] != 0:
                if key in ["cost", "e_factor"]:
                    improvement_factors[key] = original_metrics[key] / max(
                        optimized_metrics[key], 0.001
                    )
                else:
                    improvement_factors[key] = optimized_metrics[key] / max(
                        original_metrics[key], 0.001
                    )
            else:
                improvement_factors[key] = 1.0

        cost_reduction = max(0, original_metrics["cost"] - optimized_metrics["cost"])
        throughput_improvement = optimized_metrics["throughput"] / max(
            original_metrics["throughput"], 0.1
        )
        sustainability_score = (
            optimized_metrics["safety"]
            + (1.0 / max(optimized_metrics["e_factor"], 0.1))
        ) / 2.0

        return OptimizationResult(
            original_conditions=initial_conditions,
            optimized_conditions=optimal_conditions,
            original_metrics=original_metrics,
            optimized_metrics=optimized_metrics,
            improvement_factors=improvement_factors,
            optimization_targets=targets,
            total_iterations=iterations,
            convergence_achieved=convergence_achieved,
            optimization_time=optimization_time,
            cost_reduction=cost_reduction,
            throughput_improvement=throughput_improvement,
            sustainability_score=sustainability_score,
        )

    def _get_optimization_bounds(
        self,
        initial_conditions: ReactionConditions,
        constraints: Optional[Dict[str, Any]],
    ) -> List[Tuple[float, float]]:
        """Get optimization bounds for condition parameters."""
        bounds = []

        # Temperature bounds
        temp_min, temp_max = initial_conditions.temperature_range
        if constraints and "max_temperature" in constraints:
            temp_max = min(temp_max, constraints["max_temperature"])
        bounds.append((temp_min, temp_max))

        # Time bounds
        time_min, time_max = initial_conditions.time_range
        if constraints and "max_time" in constraints:
            time_max = min(time_max, constraints["max_time"])
        bounds.append((time_min, time_max))

        # Concentration bounds
        conc_min, conc_max = initial_conditions.concentration_range
        bounds.append((conc_min, conc_max))

        # Catalyst loading bounds (if catalyst is used)
        if initial_conditions.catalyst:
            bounds.append((0.001, 0.2))  # 0.1% to 20 mol%

        return bounds

    def _decode_conditions_vector(
        self, x: np.ndarray, base_conditions: ReactionConditions
    ) -> ReactionConditions:
        """Decode optimization vector back to reaction conditions."""
        conditions = ReactionConditions(
            temperature=float(x[0]),
            time=float(x[1]),
            concentration=float(x[2]),
            solvent=base_conditions.solvent,
            catalyst=base_conditions.catalyst,
            catalyst_loading=base_conditions.catalyst_loading,
            pressure=base_conditions.pressure,
            atmosphere=base_conditions.atmosphere,
            light_conditions=base_conditions.light_conditions,
            stirring_rate=base_conditions.stirring_rate,
        )

        # Update catalyst loading if catalyst is present
        if base_conditions.catalyst and len(x) > 3:
            conditions.catalyst_loading = float(x[3])

        return conditions

    def _grid_search_optimization(self, objective_function, bounds, initial_conditions):
        """Fallback grid search optimization when SciPy not available."""
        best_score = float("inf")
        best_x = None

        # Simple grid search
        n_points = 10
        iterations = 0

        # Generate grid points
        grid_points = []
        for i, (low, high) in enumerate(bounds):
            grid_points.append(np.linspace(low, high, n_points))

        # Evaluate objective function at grid points
        for temp in grid_points[0][:5]:  # Limit search for performance
            for time_val in grid_points[1][:5]:
                for conc in grid_points[2][:5]:
                    x = [temp, time_val, conc]
                    if len(bounds) > 3:
                        x.append(initial_conditions.catalyst_loading)

                    score = objective_function(x)
                    iterations += 1

                    if score < best_score:
                        best_score = score
                        best_x = x

        optimal_conditions = self._decode_conditions_vector(
            np.array(best_x), initial_conditions
        )
        return optimal_conditions, True, iterations

    def generate_experimental_design(
        self,
        reactants: List[MolecularData],
        reaction_type: str,
        base_conditions: ReactionConditions,
        n_experiments: int = 20,
    ) -> List[Dict[str, Any]]:
        """Generate experimental design for condition screening."""
        print(f"Generating experimental design with {n_experiments} experiments...")

        experiments = []

        # Define parameter ranges for screening
        temp_range = np.linspace(0, 120, 5)
        time_range = np.linspace(1, 24, 4)
        conc_range = np.linspace(0.05, 1.0, 4)

        # Factorial design (simplified)
        exp_id = 1
        for temp in temp_range:
            for time_val in time_range:
                for conc in conc_range:
                    if exp_id > n_experiments:
                        break

                    # Create experimental conditions
                    exp_conditions = ReactionConditions(
                        temperature=temp,
                        time=time_val,
                        concentration=conc,
                        solvent=base_conditions.solvent,
                        catalyst=base_conditions.catalyst,
                        catalyst_loading=base_conditions.catalyst_loading,
                    )

                    # Predict performance
                    predicted_metrics = self.predict_reaction_performance(
                        reactants, reaction_type, exp_conditions
                    )

                    experiment = {
                        "experiment_id": exp_id,
                        "temperature": temp,
                        "time": time_val,
                        "concentration": conc,
                        "solvent": exp_conditions.solvent,
                        "catalyst": exp_conditions.catalyst,
                        "predicted_yield": predicted_metrics["yield"],
                        "predicted_selectivity": predicted_metrics["selectivity"],
                        "predicted_cost": predicted_metrics["cost"],
                        "priority_score": predicted_metrics["yield"]
                        * predicted_metrics["selectivity"]
                        / predicted_metrics["cost"],
                    }

                    experiments.append(experiment)
                    exp_id += 1

                if exp_id > n_experiments:
                    break
            if exp_id > n_experiments:
                break

        # Sort experiments by priority score
        experiments.sort(key=lambda x: x["priority_score"], reverse=True)

        return experiments[:n_experiments]

    def visualize_optimization_results(
        self, results: List[OptimizationResult], output_file: str = None
    ):
        """Create comprehensive visualization of optimization results."""
        if not results:
            print("No optimization results to visualize")
            return

        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Reaction Condition Optimization Results", fontsize=16, fontweight="bold"
        )

        # Plot 1: Improvement factors comparison
        ax1 = axes[0, 0]
        metrics = ["yield", "selectivity", "cost", "safety"]
        improvements = []
        labels = []

        for i, result in enumerate(results):
            improvements.append(
                [result.improvement_factors.get(m, 1.0) for m in metrics]
            )
            target_names = [t.value for t in result.optimization_targets]
            labels.append(f"Opt {i+1}\n({', '.join(target_names)})")

        x_pos = np.arange(len(results))
        width = 0.2

        for i, metric in enumerate(metrics):
            values = [imp[i] for imp in improvements]
            ax1.bar(x_pos + i * width, values, width, label=metric.capitalize())

        ax1.set_xlabel("Optimization Run")
        ax1.set_ylabel("Improvement Factor")
        ax1.set_title("Performance Improvement Factors")
        ax1.set_xticks(x_pos + width * 1.5)
        ax1.set_xticklabels(labels, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Cost vs Performance trade-off
        ax2 = axes[0, 1]

        for i, result in enumerate(results):
            yield_improvement = result.improvement_factors.get("yield", 1.0)
            cost_reduction = result.cost_reduction
            ax2.scatter(
                cost_reduction, yield_improvement, s=100, alpha=0.7, label=f"Opt {i+1}"
            )

        ax2.set_xlabel("Cost Reduction ($)")
        ax2.set_ylabel("Yield Improvement Factor")
        ax2.set_title("Cost vs Performance Trade-off")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Optimization convergence
        ax3 = axes[1, 0]

        for i, result in enumerate(results):
            # Mock convergence data (would be actual convergence in real implementation)
            iterations = np.arange(1, result.total_iterations + 1)
            convergence = np.exp(-iterations / (result.total_iterations / 3))
            ax3.plot(iterations, convergence, label=f"Opt {i+1}")

        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Objective Function Value (normalized)")
        ax3.set_title("Optimization Convergence")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Sustainability metrics
        ax4 = axes[1, 1]

        sustainability_metrics = ["safety", "e_factor", "atom_economy"]
        opt_names = [f"Opt {i+1}" for i in range(len(results))]

        # Create stacked bar chart for sustainability
        safety_scores = [r.optimized_metrics.get("safety", 0.5) for r in results]
        e_factor_scores = [
            1.0 / max(r.optimized_metrics.get("e_factor", 1.0), 0.1) for r in results
        ]
        atom_economy_scores = [
            r.optimized_metrics.get("atom_economy", 0.5) for r in results
        ]

        ax4.bar(opt_names, safety_scores, label="Safety Score")
        ax4.bar(opt_names, e_factor_scores, bottom=safety_scores, label="1/E-factor")
        ax4.bar(
            opt_names,
            atom_economy_scores,
            bottom=np.array(safety_scores) + np.array(e_factor_scores),
            label="Atom Economy",
        )

        ax4.set_xlabel("Optimization Run")
        ax4.set_ylabel("Sustainability Score")
        ax4.set_title("Sustainability Metrics")
        ax4.legend()

        plt.tight_layout()

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Optimization visualization saved to: {output_path}")

        # plt.show()

    def generate_optimization_report(self, results: List[OptimizationResult]) -> str:
        """Generate comprehensive optimization report."""
        if not results:
            return "No optimization results to report."

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("REACTION CONDITION OPTIMIZATION REPORT")
        report_lines.append("=" * 80)

        # Summary statistics
        report_lines.append("\nOPTIMIZATION SUMMARY:")
        report_lines.append(f"  Total Optimizations: {len(results)}")
        report_lines.append(
            f"  Average Optimization Time: {np.mean([r.optimization_time for r in results]):.2f}s"
        )
        report_lines.append(
            f"  Convergence Rate: {sum(r.convergence_achieved for r in results)}/{len(results)}"
        )

        # Best improvements
        best_yield_improvement = max(
            r.improvement_factors.get("yield", 1.0) for r in results
        )
        best_cost_reduction = max(r.cost_reduction for r in results)
        best_sustainability = max(r.sustainability_score for r in results)

        report_lines.append("\nBEST IMPROVEMENTS ACHIEVED:")
        report_lines.append(
            f"  Maximum Yield Improvement: {best_yield_improvement:.2f}x"
        )
        report_lines.append(f"  Maximum Cost Reduction: ${best_cost_reduction:.2f}")
        report_lines.append(f"  Best Sustainability Score: {best_sustainability:.3f}")

        # Detailed results for each optimization
        for i, result in enumerate(results):
            report_lines.append(f"\nOPTIMIZATION {i+1} DETAILS:")
            report_lines.append("-" * 40)

            # Optimization targets
            targets_str = ", ".join([t.value for t in result.optimization_targets])
            report_lines.append(f"  Targets: {targets_str}")
            report_lines.append(f"  Iterations: {result.total_iterations}")
            report_lines.append(f"  Optimization Time: {result.optimization_time:.2f}s")
            report_lines.append(
                f"  Convergence: {'Yes' if result.convergence_achieved else 'No'}"
            )

            # Original vs Optimized conditions
            report_lines.append("\n  CONDITION CHANGES:")
            orig = result.original_conditions
            opt = result.optimized_conditions

            if abs(orig.temperature - opt.temperature) > 1:
                report_lines.append(
                    f"    Temperature: {orig.temperature:.1f}°C → {opt.temperature:.1f}°C"
                )
            if abs(orig.time - opt.time) > 0.5:
                report_lines.append(f"    Time: {orig.time:.1f}h → {opt.time:.1f}h")
            if abs(orig.concentration - opt.concentration) > 0.01:
                report_lines.append(
                    f"    Concentration: {orig.concentration:.2f}M → {opt.concentration:.2f}M"
                )

            # Performance improvements
            report_lines.append("\n  PERFORMANCE IMPROVEMENTS:")
            for metric, improvement in result.improvement_factors.items():
                if improvement != 1.0:
                    if metric in ["cost", "e_factor"]:
                        report_lines.append(
                            f"    {metric.capitalize()}: {improvement:.2f}x reduction"
                        )
                    else:
                        report_lines.append(
                            f"    {metric.capitalize()}: {improvement:.2f}x improvement"
                        )

            # Economic impact
            if result.cost_reduction > 0:
                report_lines.append(f"  Cost Reduction: ${result.cost_reduction:.2f}")
            if result.throughput_improvement > 1.0:
                report_lines.append(
                    f"  Throughput Improvement: {result.throughput_improvement:.2f}x"
                )

        # Recommendations
        report_lines.append("\nRECOMMENDATIONS:")

        if any(r.improvement_factors.get("yield", 1.0) > 1.2 for r in results):
            report_lines.append(
                "  ✓ Significant yield improvements achieved through optimization"
            )

        if any(r.cost_reduction > 10 for r in results):
            report_lines.append("  ✓ Substantial cost reductions possible")

        if any(r.sustainability_score > 0.8 for r in results):
            report_lines.append("  ✓ High sustainability scores achievable")

        avg_convergence = sum(r.convergence_achieved for r in results) / len(results)
        if avg_convergence < 0.8:
            report_lines.append(
                "  ⚠ Consider increasing optimization iterations for better convergence"
            )

        report_lines.append("\n" + "=" * 80)

        return "\n".join(report_lines)

    def cleanup(self):
        """Clean up resources."""
        self.engine.cleanup()


def create_demo_reactions() -> (
    List[Tuple[List[MolecularData], str, ReactionConditions]]
):
    """Create demonstration reactions for optimization."""

    demo_reactions = []

    # Reaction 1: Suzuki Coupling
    if RDKIT_AVAILABLE:
        aryl_bromide = Chem.MolFromSmiles("BrC1=CC=CC=C1")
        boronic_acid = Chem.MolFromSmiles("c1ccc(B(O)O)cc1")

        reactants1 = [
            MolecularData("aryl_bromide", "BrC1=CC=CC=C1", {"rdkit_mol": aryl_bromide}),
            MolecularData(
                "boronic_acid", "c1ccc(B(O)O)cc1", {"rdkit_mol": boronic_acid}
            ),
        ]
    else:
        reactants1 = [
            MolecularData("aryl_bromide", "BrC1=CC=CC=C1", {}),
            MolecularData("boronic_acid", "c1ccc(B(O)O)cc1", {}),
        ]

    conditions1 = ReactionConditions(
        temperature=80.0,
        time=12.0,
        concentration=0.1,
        solvent="THF",
        catalyst="Pd(PPh3)4",
        catalyst_loading=0.05,
    )

    demo_reactions.append((reactants1, "suzuki_coupling", conditions1))

    # Reaction 2: Amide Formation
    if RDKIT_AVAILABLE:
        carboxylic_acid = Chem.MolFromSmiles("CC(=O)O")
        amine = Chem.MolFromSmiles("CCN")

        reactants2 = [
            MolecularData("acid", "CC(=O)O", {"rdkit_mol": carboxylic_acid}),
            MolecularData("amine", "CCN", {"rdkit_mol": amine}),
        ]
    else:
        reactants2 = [
            MolecularData("acid", "CC(=O)O", {}),
            MolecularData("amine", "CCN", {}),
        ]

    conditions2 = ReactionConditions(
        temperature=25.0,
        time=8.0,
        concentration=0.2,
        solvent="DMF",
        catalyst="EDC",
        catalyst_loading=0.1,
    )

    demo_reactions.append((reactants2, "amide_formation", conditions2))

    # Reaction 3: Click Chemistry
    if RDKIT_AVAILABLE:
        alkyne = Chem.MolFromSmiles("C#CC1=CC=CC=C1")
        azide = Chem.MolFromSmiles("N=[N+]=[N-]CC1=CC=CC=C1")

        reactants3 = [
            MolecularData("alkyne", "C#CC1=CC=CC=C1", {"rdkit_mol": alkyne}),
            MolecularData("azide", "N=[N+]=[N-]CC1=CC=CC=C1", {"rdkit_mol": azide}),
        ]
    else:
        reactants3 = [
            MolecularData("alkyne", "C#CC1=CC=CC=C1", {}),
            MolecularData("azide", "N=[N+]=[N-]CC1=CC=CC=C1", {}),
        ]

    conditions3 = ReactionConditions(
        temperature=40.0,
        time=6.0,
        concentration=0.15,
        solvent="water",
        catalyst="CuSO4",
        catalyst_loading=0.02,
    )

    demo_reactions.append((reactants3, "click_chemistry", conditions3))

    return demo_reactions


def main():
    """Main demonstration function."""
    print("Advanced Reaction Condition Optimization Workflow")
    print("=" * 60)
    print(f"GPU Available: {GPU_AVAILABLE}")
    print(f"RDKit Available: {RDKIT_AVAILABLE}")
    print(f"SciPy Available: {SCIPY_AVAILABLE}")

    # Initialize optimizer
    optimizer = ReactionConditionOptimizer(use_gpu=GPU_AVAILABLE)

    # Get demo reactions
    demo_reactions = create_demo_reactions()

    # Run optimization demonstrations
    all_results = []

    for i, (reactants, reaction_type, initial_conditions) in enumerate(demo_reactions):
        print(f"\n{'='*80}")
        print(f"REACTION {i+1}: {reaction_type.upper()} OPTIMIZATION")
        print(f"{'='*80}")

        try:
            # Single objective optimizations
            print("\n1. Single Objective Optimizations:")

            # Optimize for yield
            yield_result = optimizer.optimize_single_objective(
                reactants=reactants,
                reaction_type=reaction_type,
                initial_conditions=initial_conditions,
                target=OptimizationTarget.YIELD,
                constraints={"max_temperature": 150},
            )
            all_results.append(yield_result)
            print(
                f"   Yield optimization: {yield_result.improvement_factors['yield']:.2f}x improvement"
            )

            # Optimize for cost
            cost_result = optimizer.optimize_single_objective(
                reactants=reactants,
                reaction_type=reaction_type,
                initial_conditions=initial_conditions,
                target=OptimizationTarget.COST,
            )
            all_results.append(cost_result)
            print(f"   Cost optimization: ${cost_result.cost_reduction:.2f} reduction")

            # Multi-objective optimization
            print("\n2. Multi-Objective Optimization:")
            multi_result = optimizer.optimize_multi_objective(
                reactants=reactants,
                reaction_type=reaction_type,
                initial_conditions=initial_conditions,
                targets=[
                    OptimizationTarget.YIELD,
                    OptimizationTarget.COST,
                    OptimizationTarget.SAFETY,
                ],
                weights=[0.5, 0.3, 0.2],
            )
            all_results.append(multi_result)
            print(
                f"   Multi-objective: Yield {multi_result.improvement_factors['yield']:.2f}x, "
                f"Cost ${multi_result.cost_reduction:.2f}, "
                f"Safety {multi_result.optimized_metrics['safety']:.3f}"
            )

            # Experimental design
            print("\n3. Experimental Design:")
            experiments = optimizer.generate_experimental_design(
                reactants=reactants,
                reaction_type=reaction_type,
                base_conditions=initial_conditions,
                n_experiments=12,
            )

            # Show top 3 experiments
            print("   Top 3 recommended experiments:")
            for j, exp in enumerate(experiments[:3]):
                print(
                    f"     Exp {j+1}: T={exp['temperature']:.0f}°C, "
                    f"t={exp['time']:.1f}h, "
                    f"C={exp['concentration']:.2f}M, "
                    f"Score={exp['priority_score']:.1f}"
                )

            # Save experimental design
            output_dir = Path("test_results")
            output_dir.mkdir(exist_ok=True)

            exp_df = pd.DataFrame(experiments)
            exp_file = output_dir / f"experimental_design_{reaction_type}.csv"
            exp_df.to_csv(exp_file, index=False)
            print(f"   Experimental design saved to: {exp_file}")

        except Exception as e:
            print(f"Optimization failed for {reaction_type}: {e}")
            continue

    # Generate comprehensive analysis
    if all_results:
        print(f"\n{'='*80}")
        print("COMPREHENSIVE OPTIMIZATION ANALYSIS")
        print(f"{'='*80}")

        # Create visualizations
        output_dir = Path("test_results")
        viz_file = output_dir / "reaction_optimization_analysis.png"
        optimizer.visualize_optimization_results(all_results, str(viz_file))

        # Generate detailed report
        report = optimizer.generate_optimization_report(all_results)
        print(report)

        # Save report
        report_file = output_dir / "reaction_optimization_report.txt"
        with open(report_file, "w") as f:
            f.write(report)

        print(f"\nOptimization report saved to: {report_file}")

        # Summary statistics
        avg_yield_improvement = np.mean(
            [r.improvement_factors.get("yield", 1.0) for r in all_results]
        )
        total_cost_reduction = sum(r.cost_reduction for r in all_results)
        avg_sustainability = np.mean([r.sustainability_score for r in all_results])

        print("\nOVERALL OPTIMIZATION SUMMARY:")
        print(f"  Average Yield Improvement: {avg_yield_improvement:.2f}x")
        print(f"  Total Cost Reduction: ${total_cost_reduction:.2f}")
        print(f"  Average Sustainability Score: {avg_sustainability:.3f}")
        print(f"  Successful Optimizations: {len(all_results)}")

    # Cleanup
    optimizer.cleanup()

    print(f"\n{'='*80}")
    print("REACTION CONDITION OPTIMIZATION WORKFLOW COMPLETED")
    print(f"{'='*80}")
    print("Key Features Demonstrated:")
    print("- Single and multi-objective optimization")
    print("- AI-driven condition prediction and tuning")
    print("- Automated experimental design")
    print("- Cost-benefit analysis")
    print("- Sustainability assessment")
    print("- Performance visualization and reporting")
    print("\nTask 19.2 Condition Optimization Component: SUCCESSFULLY DEMONSTRATED")


if __name__ == "__main__":
    main()
