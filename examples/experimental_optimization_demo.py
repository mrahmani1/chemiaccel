"""
Multi-Objective Experimental Optimization & Pareto Front Navigation Demo.

This example demonstrates the comprehensive experimental optimization capabilities
including multi-objective optimization, Pareto front navigation, and interactive
trade-off analysis for chemical synthesis and polymer science applications.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chemi_accel.ai.experimental_optimization import (  # Core classes; Data structures; Enums; Utility functions
    ExperimentalCondition,
    ExperimentalConstraint,
    ExperimentalObjective,
    ExperimentalOptimizer,
    ExperimentalSpace,
    ObjectiveType,
    OptimizationAlgorithm,
    ParetoFront,
    ParetoNavigator,
    SurrogateExperimentalModel,
    analyze_pareto_trade_offs,
    create_chemical_synthesis_objectives,
    create_polymer_synthesis_objectives,
    create_standard_experimental_space,
    find_optimal_experimental_conditions,
    run_experimental_optimization,
)

# Check for optional dependencies
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import plotly

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ExperimentalOptimizationDemo:
    """Demonstration of experimental optimization capabilities."""

    def __init__(self, use_gpu: bool = True, verbose: bool = True):
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        self.verbose = verbose
        self.results = {}

        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Using CPU-only implementations.")

        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Interactive visualization disabled.")

    def _log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            logger.info(message)

    async def demo_chemical_synthesis_optimization(self) -> Dict[str, Any]:
        """Demonstrate chemical synthesis optimization."""
        self._log("=== Chemical Synthesis Optimization Demo ===")

        # Create objectives for chemical synthesis
        objectives = create_chemical_synthesis_objectives()
        self._log(
            f"Created {len(objectives)} objectives: {[obj.name for obj in objectives]}"
        )

        # Create experimental space
        experimental_space = ExperimentalSpace(
            parameters={
                "temperature": (60.0, 120.0),  # °C
                "pressure": (1.0, 5.0),  # bar
                "concentration": (0.1, 2.0),  # M
                "time": (1.0, 12.0),  # hours
                "ph": (6.0, 10.0),  # pH units
            },
            discrete_parameters={
                "catalyst": ["Pd/C", "Pt/C", "Rh/C", "Ru/C"],
                "solvent": ["DMF", "THF", "DCM", "EtOH", "H2O"],
            },
            constraints=[
                ExperimentalConstraint(
                    name="temperature_safety",
                    parameter="temperature",
                    max_value=110.0,
                    penalty_weight=2.0,
                ),
                ExperimentalConstraint(
                    name="pressure_limit",
                    parameter="pressure",
                    max_value=4.0,
                    penalty_weight=1.5,
                ),
            ],
        )

        # Create and train surrogate model
        model = SurrogateExperimentalModel(use_gpu=self.use_gpu)

        # Generate synthetic training data
        training_conditions = []
        training_outcomes = []

        for i in range(20):
            # Random experimental conditions
            condition = ExperimentalCondition(
                parameters={
                    "temperature": np.random.uniform(60, 120),
                    "pressure": np.random.uniform(1, 5),
                    "concentration": np.random.uniform(0.1, 2.0),
                    "time": np.random.uniform(1, 12),
                    "ph": np.random.uniform(6, 10),
                }
            )

            # Synthetic outcomes (simplified models)
            temp = condition.parameters["temperature"]
            conc = condition.parameters["concentration"]
            time_hrs = condition.parameters["time"]

            # Yield model (optimal around 90°C)
            yield_val = (
                60 + 30 * np.exp(-((temp - 90) ** 2) / 400) + np.random.normal(0, 5)
            )
            yield_val = np.clip(yield_val, 0, 100)

            # Selectivity model (decreases with temperature and time)
            selectivity = 95 - 0.2 * temp - 1.5 * time_hrs + np.random.normal(0, 3)
            selectivity = np.clip(selectivity, 50, 100)

            # Reaction time (given parameter)
            reaction_time = time_hrs

            # Cost model (increases with pressure, temperature, and expensive catalysts)
            base_cost = 20 + 0.5 * temp + 5 * condition.parameters["pressure"]
            cost = base_cost + np.random.uniform(-5, 5)

            outcome = {
                "yield": yield_val,
                "selectivity": selectivity,
                "time": reaction_time,
                "total_cost": cost,
            }

            training_conditions.append(condition)
            training_outcomes.append(outcome)

        # Update model with training data
        model.update(training_conditions, training_outcomes)
        self._log(
            f"Trained surrogate model with {len(training_conditions)} data points"
        )

        # Run optimization
        optimizer = ExperimentalOptimizer(
            experimental_space=experimental_space,
            objectives=objectives,
            model=model,
            algorithm=OptimizationAlgorithm.NSGA_II,
            population_size=50,
            max_generations=20,
        )

        pareto_front = optimizer.optimize()
        self._log(
            f"Optimization complete. Pareto front size: {len(pareto_front.points)}"
        )
        self._log(f"Hypervolume: {pareto_front.hypervolume:.4f}")

        # Analyze results
        navigator = ParetoNavigator(pareto_front)
        analysis = navigator.get_trade_off_analysis()
        knee_points = navigator.find_knee_points()

        self._log(f"Found {len(knee_points)} knee points")
        self._log("Trade-off analysis:")
        for obj_name, obj_range in analysis["objective_ranges"].items():
            self._log(f"  {obj_name}: {obj_range['min']:.2f} - {obj_range['max']:.2f}")

        # Find optimal conditions
        optimal_conditions = find_optimal_experimental_conditions(pareto_front)
        self._log("Optimal experimental conditions:")
        for param, value in optimal_conditions.parameters.items():
            self._log(f"  {param}: {value:.2f}")

        return {
            "pareto_front_size": len(pareto_front.points),
            "hypervolume": pareto_front.hypervolume,
            "knee_points": len(knee_points),
            "optimal_conditions": optimal_conditions.parameters,
            "trade_off_analysis": analysis,
        }

    async def demo_polymer_synthesis_optimization(self) -> Dict[str, Any]:
        """Demonstrate polymer synthesis optimization."""
        self._log("=== Polymer Synthesis Optimization Demo ===")

        # Create polymer-specific objectives
        objectives = create_polymer_synthesis_objectives()
        self._log(
            f"Created {len(objectives)} polymer objectives: {[obj.name for obj in objectives]}"
        )

        # Polymer-specific experimental space
        experimental_space = ExperimentalSpace(
            parameters={
                "temperature": (40.0, 100.0),  # °C
                "initiator_conc": (0.01, 0.5),  # M
                "monomer_conc": (1.0, 8.0),  # M
                "time": (0.5, 24.0),  # hours
                "pressure": (1.0, 3.0),  # bar
            },
            discrete_parameters={
                "initiator": ["AIBN", "BPO", "KPS"],
                "solvent": ["Bulk", "Toluene", "DMF", "Water"],
                "atmosphere": ["N2", "Ar", "Air"],
            },
            constraints=[
                ExperimentalConstraint(
                    name="thermal_stability",
                    parameter="temperature",
                    max_value=95.0,
                    penalty_weight=3.0,
                ),
                ExperimentalConstraint(
                    name="initiator_efficiency",
                    parameter="initiator_conc",
                    min_value=0.02,
                    max_value=0.3,
                    penalty_weight=1.0,
                ),
            ],
        )

        # Create and train polymer model
        model = SurrogateExperimentalModel(use_gpu=self.use_gpu)

        # Generate polymer training data
        training_conditions = []
        training_outcomes = []

        for i in range(25):
            condition = ExperimentalCondition(
                parameters={
                    "temperature": np.random.uniform(40, 100),
                    "initiator_conc": np.random.uniform(0.01, 0.5),
                    "monomer_conc": np.random.uniform(1, 8),
                    "time": np.random.uniform(0.5, 24),
                    "pressure": np.random.uniform(1, 3),
                }
            )

            # Polymer-specific outcome models
            temp = condition.parameters["temperature"]
            init_conc = condition.parameters["initiator_conc"]
            mon_conc = condition.parameters["monomer_conc"]
            time_hrs = condition.parameters["time"]

            # Molecular weight model (Mn)
            # Higher temperature and initiator -> lower Mn
            log_mn = (
                4.5 - 0.02 * temp - 1.0 * np.log(init_conc) + 0.1 * np.log(mon_conc)
            )
            mn = np.exp(log_mn + np.random.normal(0, 0.1)) * 1000
            mn = np.clip(mn, 5000, 200000)

            # Polydispersity (PDI)
            # Lower with controlled conditions
            pdi_base = 1.5 + 0.01 * temp + 0.5 * init_conc
            pdi = pdi_base + np.random.normal(0, 0.1)
            pdi = np.clip(pdi, 1.1, 4.0)

            # Conversion
            # Higher with temperature and time
            conversion_rate = 0.1 * temp + 5 * init_conc + 0.5 * time_hrs
            conversion = 20 + conversion_rate + np.random.normal(0, 5)
            conversion = np.clip(conversion, 10, 95)

            # Processing time
            processing_time = time_hrs

            outcome = {
                "mn": mn,
                "pdi": pdi,
                "monomer_conversion": conversion,
                "polymerization_time": processing_time,
            }

            training_conditions.append(condition)
            training_outcomes.append(outcome)

        model.update(training_conditions, training_outcomes)
        self._log(f"Trained polymer model with {len(training_conditions)} data points")

        # Run polymer optimization
        optimizer = ExperimentalOptimizer(
            experimental_space=experimental_space,
            objectives=objectives,
            model=model,
            algorithm=OptimizationAlgorithm.NSGA_II,
            population_size=60,
            max_generations=25,
        )

        pareto_front = optimizer.optimize()
        self._log(
            f"Polymer optimization complete. Pareto front size: {len(pareto_front.points)}"
        )

        # Analyze polymer results
        navigator = ParetoNavigator(pareto_front)
        analysis = navigator.get_trade_off_analysis()

        # Filter for target molecular weight
        target_mn_points = navigator.filter_by_parameter("temperature", 60.0, 85.0)
        self._log(f"Points in moderate temperature range: {len(target_mn_points)}")

        # Find conditions near target Mn (50,000 g/mol)
        mn_filtered = []
        for point in pareto_front.points:
            predictions = model.predict(point.conditions)
            if predictions.get("mn", 0) and 45000 <= predictions["mn"] <= 55000:
                mn_filtered.append(point)

        self._log(f"Points near target Mn (50k g/mol): {len(mn_filtered)}")

        return {
            "pareto_front_size": len(pareto_front.points),
            "hypervolume": pareto_front.hypervolume,
            "moderate_temp_points": len(target_mn_points),
            "target_mn_points": len(mn_filtered),
            "trade_off_analysis": analysis,
        }

    async def demo_interactive_pareto_navigation(self) -> Dict[str, Any]:
        """Demonstrate interactive Pareto front navigation."""
        self._log("=== Interactive Pareto Front Navigation Demo ===")

        # Use results from chemical synthesis optimization
        objectives = create_chemical_synthesis_objectives()
        experimental_space = create_standard_experimental_space()
        model = SurrogateExperimentalModel(use_gpu=self.use_gpu)

        # Quick training with synthetic data
        conditions = [
            ExperimentalCondition(
                parameters={
                    "temperature": 80 + i * 5,
                    "pressure": 2 + i * 0.3,
                    "concentration": 0.5 + i * 0.1,
                    "time": 4 + i * 0.5,
                    "ph": 7 + i * 0.2,
                }
            )
            for i in range(10)
        ]

        outcomes = [
            {
                "yield": 70 + i * 2 + np.random.normal(0, 2),
                "selectivity": 85 - i * 0.5 + np.random.normal(0, 1),
                "time": 4 + i * 0.5,
                "total_cost": 40 + i * 3 + np.random.normal(0, 2),
            }
            for i in range(10)
        ]

        model.update(conditions, outcomes)

        # Generate larger Pareto front
        optimizer = ExperimentalOptimizer(
            experimental_space=experimental_space,
            objectives=objectives,
            model=model,
            population_size=80,
            max_generations=15,
        )

        pareto_front = optimizer.optimize()
        navigator = ParetoNavigator(pareto_front)

        self._log(f"Generated Pareto front with {len(pareto_front.points)} points")

        # Demonstrate filtering capabilities
        # Filter by temperature range
        temp_filtered = navigator.filter_by_parameter("temperature", 70.0, 90.0)
        self._log(f"Points in temperature range 70-90°C: {len(temp_filtered)}")

        # Filter by high yield objective
        yield_filtered = navigator.filter_by_objective(
            0, 75.0, 100.0
        )  # Assuming yield is first objective
        self._log(f"Points with yield > 75%: {len(yield_filtered)}")

        # Find knee points with different preference weights
        equal_weights = [1.0, 1.0, 1.0, 1.0]
        yield_focused = [2.0, 1.0, 0.5, 0.8]
        cost_focused = [1.0, 1.0, 1.0, 2.0]

        knee_equal = navigator.find_knee_points(equal_weights)
        knee_yield = navigator.find_knee_points(yield_focused)
        knee_cost = navigator.find_knee_points(cost_focused)

        self._log(f"Knee points - Equal weights: {len(knee_equal)}")
        self._log(f"Knee points - Yield focused: {len(knee_yield)}")
        self._log(f"Knee points - Cost focused: {len(knee_cost)}")

        # Comprehensive trade-off analysis
        analysis = navigator.get_trade_off_analysis()

        self._log("Objective correlations:")
        for correlation_key, correlation_value in analysis["correlations"].items():
            self._log(f"  {correlation_key}: {correlation_value:.3f}")

        self._log("Diversity metrics:")
        for metric_key, metric_value in analysis["diversity_metrics"].items():
            self._log(f"  {metric_key}: {metric_value:.3f}")

        # Create interactive plot if Plotly is available
        plot_created = False
        if PLOTLY_AVAILABLE:
            try:
                plot = navigator.create_interactive_plot()
                if plot is not None:
                    # Save plot to HTML file for viewing
                    plot_file = Path("pareto_front_interactive.html")
                    plot.write_html(str(plot_file))
                    self._log(f"Interactive plot saved to {plot_file}")
                    plot_created = True
            except Exception as e:
                self._log(f"Could not create interactive plot: {e}")

        return {
            "total_points": len(pareto_front.points),
            "temperature_filtered": len(temp_filtered),
            "yield_filtered": len(yield_filtered),
            "knee_points_equal": len(knee_equal),
            "knee_points_yield": len(knee_yield),
            "knee_points_cost": len(knee_cost),
            "plot_created": plot_created,
            "correlations": analysis["correlations"],
            "diversity_metrics": analysis["diversity_metrics"],
        }

    async def demo_multi_algorithm_comparison(self) -> Dict[str, Any]:
        """Compare different optimization algorithms."""
        self._log("=== Multi-Algorithm Comparison Demo ===")

        objectives = create_chemical_synthesis_objectives()
        experimental_space = create_standard_experimental_space()

        # Create consistent training data
        model = SurrogateExperimentalModel(use_gpu=self.use_gpu)

        # Larger training dataset for better comparison
        training_conditions = []
        training_outcomes = []

        np.random.seed(42)  # For reproducible comparison

        for i in range(30):
            condition = ExperimentalCondition(
                parameters={
                    "temperature": np.random.uniform(20, 200),
                    "pressure": np.random.uniform(1, 10),
                    "concentration": np.random.uniform(0.01, 2.0),
                    "time": np.random.uniform(0.5, 24),
                    "ph": np.random.uniform(1, 14),
                }
            )

            # More complex synthetic models
            temp = condition.parameters["temperature"]
            press = condition.parameters["pressure"]
            conc = condition.parameters["concentration"]
            time_val = condition.parameters["time"]
            ph = condition.parameters["ph"]

            # Multi-modal yield function
            yield_val = (
                50
                + 20 * np.sin(temp / 30)
                + 15 * np.cos(press)
                + 10 * np.exp(-((conc - 1.0) ** 2))
                + np.random.normal(0, 3)
            )
            yield_val = np.clip(yield_val, 20, 95)

            # Complex selectivity model
            selectivity = (
                80
                + 10 * np.tanh((ph - 7) / 2)
                - 0.1 * temp
                + 5 * np.exp(-time_val / 5)
                + np.random.normal(0, 2)
            )
            selectivity = np.clip(selectivity, 50, 98)

            outcome = {
                "yield": yield_val,
                "selectivity": selectivity,
                "time": time_val,
                "total_cost": 30
                + 0.3 * temp
                + 2 * press
                + 10 * conc
                + np.random.normal(0, 3),
            }

            training_conditions.append(condition)
            training_outcomes.append(outcome)

        model.update(training_conditions, training_outcomes)

        # Test different algorithms
        algorithms = [
            OptimizationAlgorithm.NSGA_II,
            OptimizationAlgorithm.NSGA_III,
            OptimizationAlgorithm.MOEA_D,
        ]

        results = {}

        for algorithm in algorithms:
            self._log(f"Testing {algorithm.value.upper()} algorithm...")

            optimizer = ExperimentalOptimizer(
                experimental_space=experimental_space,
                objectives=objectives,
                model=model,
                algorithm=algorithm,
                population_size=40,
                max_generations=15,
            )

            pareto_front = optimizer.optimize()

            # Calculate metrics
            navigator = ParetoNavigator(pareto_front)
            analysis = navigator.get_trade_off_analysis()

            results[algorithm.value] = {
                "pareto_size": len(pareto_front.points),
                "hypervolume": pareto_front.hypervolume,
                "spread": analysis["diversity_metrics"]["spread"],
                "mean_distance": analysis["diversity_metrics"]["mean_distance"],
            }

            self._log(
                f"  {algorithm.value}: {len(pareto_front.points)} points, "
                f"HV = {pareto_front.hypervolume:.4f}"
            )

        # Compare results
        best_hv_algorithm = max(results.keys(), key=lambda k: results[k]["hypervolume"])
        best_size_algorithm = max(
            results.keys(), key=lambda k: results[k]["pareto_size"]
        )

        self._log(
            f"Best hypervolume: {best_hv_algorithm} ({results[best_hv_algorithm]['hypervolume']:.4f})"
        )
        self._log(
            f"Most diverse front: {best_size_algorithm} ({results[best_size_algorithm]['pareto_size']} points)"
        )

        return {
            "algorithm_results": results,
            "best_hypervolume": best_hv_algorithm,
            "best_diversity": best_size_algorithm,
        }

    async def demo_constraint_optimization(self) -> Dict[str, Any]:
        """Demonstrate constraint-based optimization."""
        self._log("=== Constraint-Based Optimization Demo ===")

        objectives = create_chemical_synthesis_objectives()

        # Create experimental space with strict constraints
        constrained_space = ExperimentalSpace(
            parameters={
                "temperature": (50.0, 150.0),
                "pressure": (1.0, 8.0),
                "concentration": (0.1, 1.5),
                "time": (1.0, 10.0),
                "ph": (5.0, 12.0),
            },
            discrete_parameters={
                "catalyst": ["Pd/C", "Pt/C"],
                "solvent": ["DMF", "THF", "EtOH"],
            },
            constraints=[
                # Safety constraints
                ExperimentalConstraint(
                    name="high_temp_pressure_safety",
                    parameter="temperature",
                    max_value=120.0,
                    penalty_weight=5.0,
                    constraint_type="safety",
                ),
                ExperimentalConstraint(
                    name="pressure_safety",
                    parameter="pressure",
                    max_value=6.0,
                    penalty_weight=3.0,
                    constraint_type="safety",
                ),
                # Process constraints
                ExperimentalConstraint(
                    name="concentration_feasibility",
                    parameter="concentration",
                    min_value=0.2,
                    max_value=1.2,
                    penalty_weight=2.0,
                    constraint_type="process",
                ),
                # Environmental constraints
                ExperimentalConstraint(
                    name="ph_environmental",
                    parameter="ph",
                    min_value=6.0,
                    max_value=10.0,
                    penalty_weight=1.5,
                    constraint_type="environmental",
                ),
            ],
        )

        # Train model
        model = SurrogateExperimentalModel(use_gpu=self.use_gpu)

        # Generate training data within constraints
        valid_conditions = []
        valid_outcomes = []

        for _ in range(50):
            while True:
                condition = ExperimentalCondition(
                    parameters={
                        "temperature": np.random.uniform(50, 150),
                        "pressure": np.random.uniform(1, 8),
                        "concentration": np.random.uniform(0.1, 1.5),
                        "time": np.random.uniform(1, 10),
                        "ph": np.random.uniform(5, 12),
                    }
                )

                # Check if condition satisfies constraints
                satisfies_constraints = True
                for constraint in constrained_space.constraints:
                    param_value = condition.parameters.get(constraint.parameter)
                    if param_value is None:
                        continue

                    if (
                        constraint.min_value is not None
                        and param_value < constraint.min_value
                    ) or (
                        constraint.max_value is not None
                        and param_value > constraint.max_value
                    ):
                        satisfies_constraints = False
                        break

                if satisfies_constraints:
                    break

            # Generate outcome
            temp = condition.parameters["temperature"]
            press = condition.parameters["pressure"]
            conc = condition.parameters["concentration"]
            time_val = condition.parameters["time"]

            # Constrained optimization typically leads to different optima
            yield_val = (
                60 + 25 * np.exp(-((temp - 100) ** 2) / 800) + np.random.normal(0, 4)
            )
            yield_val = np.clip(yield_val, 30, 92)

            selectivity = (
                90
                - 0.15 * temp
                - 2 * press
                + 5 * np.exp(-time_val / 3)
                + np.random.normal(0, 2)
            )
            selectivity = np.clip(selectivity, 60, 95)

            outcome = {
                "yield": yield_val,
                "selectivity": selectivity,
                "time": time_val,
                "total_cost": 25
                + 0.2 * temp
                + 1.5 * press
                + 8 * conc
                + np.random.normal(0, 2),
            }

            valid_conditions.append(condition)
            valid_outcomes.append(outcome)

        model.update(valid_conditions, valid_outcomes)
        self._log(
            f"Trained constrained model with {len(valid_conditions)} valid conditions"
        )

        # Run constrained optimization
        optimizer = ExperimentalOptimizer(
            experimental_space=constrained_space,
            objectives=objectives,
            model=model,
            population_size=60,
            max_generations=20,
        )

        pareto_front = optimizer.optimize()

        # Verify constraint satisfaction
        constraint_violations = 0
        for point in pareto_front.points:
            if not point.conditions.constraints_satisfied:
                constraint_violations += 1

        violation_rate = (
            constraint_violations / len(pareto_front.points)
            if pareto_front.points
            else 0
        )

        self._log("Constrained optimization complete")
        self._log(f"Pareto front size: {len(pareto_front.points)}")
        self._log(
            f"Constraint violations: {constraint_violations}/{len(pareto_front.points)} ({violation_rate:.1%})"
        )

        # Analyze constrained vs unconstrained trade-offs
        navigator = ParetoNavigator(pareto_front)
        analysis = navigator.get_trade_off_analysis()

        # Find feasible knee points
        feasible_points = [
            p for p in pareto_front.points if p.conditions.constraints_satisfied
        ]
        if feasible_points:
            feasible_front = ParetoFront(points=feasible_points, objectives=objectives)
            feasible_navigator = ParetoNavigator(feasible_front)
            feasible_knees = feasible_navigator.find_knee_points()
            self._log(f"Feasible knee points: {len(feasible_knees)}")
        else:
            feasible_knees = []

        return {
            "pareto_front_size": len(pareto_front.points),
            "constraint_violations": constraint_violations,
            "violation_rate": violation_rate,
            "feasible_points": len(feasible_points),
            "feasible_knee_points": len(feasible_knees),
            "hypervolume": pareto_front.hypervolume,
        }

    async def run_all_demos(self) -> Dict[str, Any]:
        """Run all demonstration scenarios."""
        self._log("Starting Experimental Optimization Demonstrations")
        self._log("=" * 60)

        all_results = {}

        try:
            # Demo 1: Chemical Synthesis Optimization
            all_results[
                "chemical_synthesis"
            ] = await self.demo_chemical_synthesis_optimization()

            # Demo 2: Polymer Synthesis Optimization
            all_results[
                "polymer_synthesis"
            ] = await self.demo_polymer_synthesis_optimization()

            # Demo 3: Interactive Pareto Navigation
            all_results[
                "pareto_navigation"
            ] = await self.demo_interactive_pareto_navigation()

            # Demo 4: Multi-Algorithm Comparison
            all_results[
                "algorithm_comparison"
            ] = await self.demo_multi_algorithm_comparison()

            # Demo 5: Constraint-Based Optimization
            all_results[
                "constraint_optimization"
            ] = await self.demo_constraint_optimization()

            self._log("=" * 60)
            self._log("All demonstrations completed successfully!")

            # Summary statistics
            total_pareto_points = sum(
                result.get("pareto_front_size", 0)
                for result in all_results.values()
                if isinstance(result, dict)
            )

            self._log(f"Total Pareto points generated: {total_pareto_points}")

            # Best performing algorithm
            if "algorithm_comparison" in all_results:
                best_alg = all_results["algorithm_comparison"]["best_hypervolume"]
                self._log(f"Best performing algorithm: {best_alg}")

            return all_results

        except Exception as e:
            logger.error(f"Error during demonstrations: {e}")
            all_results["error"] = str(e)
            return all_results


async def main():
    """Main demonstration function."""
    print("Multi-Objective Experimental Optimization & Pareto Front Navigation Demo")
    print("=" * 80)

    # Initialize demo
    demo = ExperimentalOptimizationDemo(use_gpu=TORCH_AVAILABLE, verbose=True)

    # Run all demonstrations
    results = await demo.run_all_demos()

    # Save results
    results_file = Path("experimental_optimization_demo_results.json")

    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj

    serializable_results = convert_numpy(results)

    with open(results_file, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)

    print(f"\nResults saved to {results_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("DEMONSTRATION SUMMARY")
    print("=" * 80)

    for demo_name, demo_results in results.items():
        if isinstance(demo_results, dict) and "error" not in demo_results:
            print(f"\n{demo_name.replace('_', ' ').title()}:")
            if "pareto_front_size" in demo_results:
                print(f"  Pareto Front Size: {demo_results['pareto_front_size']}")
            if "hypervolume" in demo_results:
                print(f"  Hypervolume: {demo_results['hypervolume']:.4f}")
            if "best_hypervolume" in demo_results:
                print(f"  Best Algorithm: {demo_results['best_hypervolume']}")
            if "constraint_violations" in demo_results:
                violations = demo_results["constraint_violations"]
                total = demo_results["pareto_front_size"]
                print(f"  Constraint Satisfaction: {total - violations}/{total}")

    print("\nTechnology Stack:")
    print(f"  PyTorch: {'Available' if TORCH_AVAILABLE else 'Not Available'}")
    print(f"  Plotly: {'Available' if PLOTLY_AVAILABLE else 'Not Available'}")

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
