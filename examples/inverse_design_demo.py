"""
Comprehensive Inverse Design for Sustainable Materials Demo.

This example demonstrates the inverse design capabilities for sustainable materials
including multi-objective optimization, GPU acceleration, and sustainability metrics
optimization for polymer design applications.
"""

import asyncio
import json
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chemi_accel.sustainability.inverse_design import (  # Core classes; Data structures; Utility functions
    ConstraintType,
    MaterialParameter,
    OptimizationAlgorithm,
    OptimizationConfig,
    OptimizationConstraint,
    OptimizationObjective,
    SustainabilityPropertyPredictor,
    SustainabilityTarget,
    SustainableInverseDesigner,
    create_performance_constraints,
    create_polymer_design_parameters,
    create_sustainability_objectives,
    design_sustainable_material,
)

# Check for optional dependencies
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class InverseDesignDemo:
    """Comprehensive inverse design demonstration."""

    def __init__(self, use_gpu: bool = True, verbose: bool = True):
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        self.verbose = verbose
        self.results = {}
        self.output_dir = None

        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Using CPU-only implementations.")

        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Visualization features disabled.")

    def _log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            logger.info(message)

    async def demo_basic_material_design(self) -> Dict[str, Any]:
        """Demonstrate basic sustainable material design."""
        self._log("=== Basic Sustainable Material Design Demo ===")

        # Configure optimization
        config = OptimizationConfig(
            algorithm=OptimizationAlgorithm.GENETIC_ALGORITHM,
            max_iterations=50,
            population_size=20,
            use_gpu=self.use_gpu,
            random_seed=42,
        )

        designer = SustainableInverseDesigner(config)

        # Define design parameters
        parameters = [
            MaterialParameter(
                name="monomer_ratio",
                min_value=0.3,
                max_value=0.9,
                description="Primary monomer fraction",
            ),
            MaterialParameter(
                name="crosslink_density",
                min_value=0.0,
                max_value=0.2,
                description="Crosslink density",
            ),
            MaterialParameter(
                name="stabilizer_content",
                min_value=0.001,
                max_value=0.05,
                description="UV stabilizer content",
            ),
            MaterialParameter(
                name="filler_content",
                min_value=0.0,
                max_value=0.3,
                description="Reinforcing filler content",
            ),
        ]

        # Define sustainability objectives
        objectives = [
            SustainabilityTarget(
                objective=OptimizationObjective.MINIMIZE_DEGRADATION_RATE,
                weight=1.0,
                minimize=True,
            ),
            SustainabilityTarget(
                objective=OptimizationObjective.MAXIMIZE_RECYCLABILITY,
                weight=1.0,
                minimize=False,
            ),
            SustainabilityTarget(
                objective=OptimizationObjective.MINIMIZE_CARBON_FOOTPRINT,
                weight=0.8,
                minimize=True,
            ),
        ]

        # Define performance constraints
        constraints = [
            OptimizationConstraint(
                name="minimum_strength",
                constraint_type=ConstraintType.INEQUALITY,
                function=lambda props, params: 25.0 - props.get("tensile_strength", 0),
                description="Minimum tensile strength of 25 MPa",
            ),
            OptimizationConstraint(
                name="maximum_density",
                constraint_type=ConstraintType.INEQUALITY,
                function=lambda props, params: params.get("filler_content", 0) - 0.25,
                description="Maximum filler content of 25%",
            ),
        ]

        self._log("Running basic material design optimization...")

        # Run design optimization
        result = await designer.design_material(parameters, objectives, constraints)

        # Analyze results
        analysis = {
            "optimization_success": result.success,
            "optimal_parameters": result.optimal_parameters,
            "objective_values": {
                obj.value: val for obj, val in result.objective_values.items()
            },
            "sustainability_scores": result.sustainability_scores,
            "predicted_properties": result.predicted_properties,
            "computation_time": result.computation_time,
            "convergence_info": result.convergence_metrics,
        }

        self._log(f"Basic design completed in {result.computation_time:.2f} seconds")
        self._log(
            f"Overall sustainability score: {result.sustainability_scores.get('overall_sustainability', 0):.3f}"
        )

        return {
            "scenario": "basic_material_design",
            "design_result": result,
            "analysis": analysis,
            "recommendations": self._generate_basic_recommendations(result),
        }

    async def demo_multi_objective_optimization(self) -> Dict[str, Any]:
        """Demonstrate multi-objective optimization with Pareto front."""
        self._log("=== Multi-Objective Optimization Demo ===")

        # Configure for multi-objective optimization
        config = OptimizationConfig(
            algorithm=OptimizationAlgorithm.MULTI_OBJECTIVE_GENETIC,
            max_iterations=100,
            population_size=50,
            use_gpu=self.use_gpu,
            random_seed=123,
        )

        designer = SustainableInverseDesigner(config)

        # Use standard parameters and objectives
        parameters = create_polymer_design_parameters()
        objectives = create_sustainability_objectives()
        constraints = create_performance_constraints()

        self._log("Running multi-objective optimization...")

        # Generate multiple solutions for Pareto analysis
        pareto_solutions = await designer.multi_objective_design(
            parameters, objectives, constraints, n_solutions=10
        )

        # Analyze Pareto front
        trade_off_analysis = designer.analyze_trade_offs(pareto_solutions)

        # Extract key metrics from solutions
        solution_data = []
        for i, solution in enumerate(pareto_solutions):
            if solution.success:
                solution_data.append(
                    {
                        "solution_id": f"solution_{i+1}",
                        "parameters": solution.optimal_parameters,
                        "objectives": {
                            obj.value: val
                            for obj, val in solution.objective_values.items()
                        },
                        "sustainability_score": solution.sustainability_scores.get(
                            "overall_sustainability", 0
                        ),
                        "properties": solution.predicted_properties,
                    }
                )

        self._log(f"Generated {len(solution_data)} Pareto-optimal solutions")

        return {
            "scenario": "multi_objective_optimization",
            "pareto_solutions": solution_data,
            "trade_off_analysis": trade_off_analysis,
            "pareto_front": trade_off_analysis.get("pareto_front", []),
            "objective_correlations": trade_off_analysis.get(
                "objective_correlations", {}
            ),
            "recommendations": trade_off_analysis.get("recommendations", []),
        }

    async def demo_constraint_optimization(self) -> Dict[str, Any]:
        """Demonstrate optimization with complex constraints."""
        self._log("=== Constraint-Based Optimization Demo ===")

        config = OptimizationConfig(
            algorithm=OptimizationAlgorithm.BAYESIAN_OPTIMIZATION,
            max_iterations=75,
            use_gpu=self.use_gpu,
            random_seed=456,
        )

        designer = SustainableInverseDesigner(config)

        # Define specialized parameters for high-performance application
        parameters = [
            MaterialParameter(
                name="high_modulus_filler",
                min_value=0.1,
                max_value=0.4,
                description="High modulus filler content",
            ),
            MaterialParameter(
                name="toughening_agent",
                min_value=0.02,
                max_value=0.15,
                description="Impact modifier content",
            ),
            MaterialParameter(
                name="processing_aid",
                min_value=0.005,
                max_value=0.03,
                description="Processing aid content",
            ),
            MaterialParameter(
                name="chain_extender",
                min_value=0.0,
                max_value=0.05,
                description="Chain extender content",
            ),
        ]

        # Sustainability objectives
        objectives = [
            SustainabilityTarget(
                objective=OptimizationObjective.MINIMIZE_DEGRADATION_RATE,
                weight=1.5,
                minimize=True,
            ),
            SustainabilityTarget(
                objective=OptimizationObjective.MAXIMIZE_RECYCLABILITY,
                weight=1.2,
                minimize=False,
            ),
            SustainabilityTarget(
                objective=OptimizationObjective.MINIMIZE_ENVIRONMENTAL_IMPACT,
                weight=1.0,
                minimize=True,
            ),
        ]

        # Complex constraints for high-performance application
        constraints = [
            OptimizationConstraint(
                name="high_strength_requirement",
                constraint_type=ConstraintType.INEQUALITY,
                function=lambda props, params: 60.0 - props.get("tensile_strength", 0),
                description="Minimum tensile strength of 60 MPa",
            ),
            OptimizationConstraint(
                name="impact_resistance",
                constraint_type=ConstraintType.INEQUALITY,
                function=lambda props, params: 30.0
                - (
                    props.get("tensile_strength", 0)
                    * params.get("toughening_agent", 0)
                    * 100
                ),
                description="Minimum impact resistance",
            ),
            OptimizationConstraint(
                name="processing_constraint",
                constraint_type=ConstraintType.INEQUALITY,
                function=lambda props, params: params.get("processing_aid", 0) - 0.025,
                description="Maximum processing aid content",
            ),
            OptimizationConstraint(
                name="filler_compatibility",
                constraint_type=ConstraintType.INEQUALITY,
                function=lambda props, params: (
                    params.get("high_modulus_filler", 0)
                    + params.get("toughening_agent", 0)
                )
                - 0.45,
                description="Total additive compatibility limit",
            ),
        ]

        self._log("Running constraint-based optimization...")

        # Run constrained optimization
        result = await designer.design_material(parameters, objectives, constraints)

        # Analyze constraint satisfaction
        constraint_analysis = self._analyze_constraints(result, constraints)

        self._log(
            f"Constraint optimization completed with {len(constraints)} constraints"
        )

        return {
            "scenario": "constraint_optimization",
            "design_result": result,
            "constraint_analysis": constraint_analysis,
            "performance_metrics": self._calculate_performance_metrics(result),
            "optimization_summary": {
                "algorithm": config.algorithm.value,
                "iterations": config.max_iterations,
                "success": result.success,
                "final_score": result.sustainability_scores.get(
                    "overall_sustainability", 0
                ),
            },
        }

    async def demo_algorithm_comparison(self) -> Dict[str, Any]:
        """Demonstrate comparison of different optimization algorithms."""
        self._log("=== Optimization Algorithm Comparison Demo ===")

        # Define common problem setup
        parameters = create_polymer_design_parameters()
        objectives = [
            SustainabilityTarget(
                objective=OptimizationObjective.MINIMIZE_DEGRADATION_RATE,
                weight=1.0,
                minimize=True,
            ),
            SustainabilityTarget(
                objective=OptimizationObjective.MAXIMIZE_RECYCLABILITY,
                weight=1.0,
                minimize=False,
            ),
        ]
        constraints = []

        # Test different algorithms
        algorithms = [
            OptimizationAlgorithm.GENETIC_ALGORITHM,
            OptimizationAlgorithm.BAYESIAN_OPTIMIZATION,
            OptimizationAlgorithm.MULTI_OBJECTIVE_GENETIC,
        ]

        algorithm_results = {}

        for algorithm in algorithms:
            self._log(f"Testing {algorithm.value} algorithm...")

            config = OptimizationConfig(
                algorithm=algorithm,
                max_iterations=30,  # Reduced for comparison
                population_size=15,
                use_gpu=self.use_gpu,
                random_seed=789,
            )

            designer = SustainableInverseDesigner(config)

            try:
                result = await designer.design_material(
                    parameters, objectives, constraints
                )

                algorithm_results[algorithm.value] = {
                    "success": result.success,
                    "computation_time": result.computation_time,
                    "sustainability_score": result.sustainability_scores.get(
                        "overall_sustainability", 0
                    ),
                    "objective_values": {
                        obj.value: val for obj, val in result.objective_values.items()
                    },
                    "convergence_metrics": result.convergence_metrics,
                    "optimal_parameters": result.optimal_parameters,
                }

            except Exception as e:
                logger.warning(f"Algorithm {algorithm.value} failed: {e}")
                algorithm_results[algorithm.value] = {
                    "success": False,
                    "error": str(e),
                    "computation_time": 0.0,
                }

        # Analyze algorithm performance
        performance_analysis = self._analyze_algorithm_performance(algorithm_results)

        self._log(f"Algorithm comparison completed for {len(algorithms)} algorithms")

        return {
            "scenario": "algorithm_comparison",
            "algorithm_results": algorithm_results,
            "performance_analysis": performance_analysis,
            "recommendations": self._generate_algorithm_recommendations(
                performance_analysis
            ),
        }

    async def demo_property_prediction_validation(self) -> Dict[str, Any]:
        """Demonstrate property prediction validation and accuracy."""
        self._log("=== Property Prediction Validation Demo ===")

        predictor = SustainabilityPropertyPredictor(use_gpu=self.use_gpu)

        # Test parameter sets with known expected behaviors
        test_cases = [
            {
                "name": "High_Stability_Polymer",
                "parameters": {
                    "monomer_ratio": 0.9,
                    "crosslink_density": 0.1,
                    "stabilizer_content": 0.05,
                    "filler_content": 0.2,
                    "polymer_purity": 0.98,
                },
                "expected_behavior": "Low degradation rate, high durability",
            },
            {
                "name": "Biodegradable_Polymer",
                "parameters": {
                    "monomer_ratio": 0.6,
                    "crosslink_density": 0.02,
                    "stabilizer_content": 0.01,
                    "filler_content": 0.05,
                    "polymer_purity": 0.85,
                },
                "expected_behavior": "Higher degradation rate, good biodegradability",
            },
            {
                "name": "Recycling_Optimized",
                "parameters": {
                    "monomer_ratio": 0.8,
                    "crosslink_density": 0.0,
                    "stabilizer_content": 0.02,
                    "filler_content": 0.1,
                    "polymer_purity": 0.95,
                },
                "expected_behavior": "High recyclability, moderate durability",
            },
            {
                "name": "Low_Impact_Design",
                "parameters": {
                    "monomer_ratio": 0.7,
                    "crosslink_density": 0.05,
                    "stabilizer_content": 0.015,
                    "filler_content": 0.15,
                    "polymer_purity": 0.92,
                },
                "expected_behavior": "Balanced sustainability profile",
            },
        ]

        prediction_results = {}

        for test_case in test_cases:
            self._log(f"Predicting properties for {test_case['name']}...")

            properties = await predictor.predict_properties(test_case["parameters"])

            prediction_results[test_case["name"]] = {
                "parameters": test_case["parameters"],
                "predicted_properties": properties,
                "expected_behavior": test_case["expected_behavior"],
                "sustainability_metrics": {
                    "degradation_resistance": 1.0
                    / (properties.get("degradation_rate", 0.001) + 1e-6),
                    "recyclability": properties.get("recyclability_score", 0.5),
                    "environmental_impact": 1.0
                    / (properties.get("carbon_footprint", 2.0) + 1e-6),
                    "overall_score": self._calculate_case_sustainability_score(
                        properties
                    ),
                },
            }

        # Validate predictions against expected behaviors
        validation_analysis = self._validate_predictions(prediction_results)

        self._log(
            f"Property prediction validation completed for {len(test_cases)} test cases"
        )

        return {
            "scenario": "property_prediction_validation",
            "test_cases": prediction_results,
            "validation_analysis": validation_analysis,
            "prediction_quality": self._assess_prediction_quality(prediction_results),
        }

    def _generate_basic_recommendations(self, result) -> List[str]:
        """Generate recommendations for basic design results."""
        recommendations = []

        if result.success:
            sustainability_score = result.sustainability_scores.get(
                "overall_sustainability", 0
            )

            if sustainability_score > 0.8:
                recommendations.append("Excellent sustainability profile achieved")
            elif sustainability_score > 0.6:
                recommendations.append("Good sustainability with room for improvement")

                # Specific recommendations based on objectives
                obj_values = {
                    obj.value: val for obj, val in result.objective_values.items()
                }

                if obj_values.get("minimize_degradation_rate", 1.0) > 0.5:
                    recommendations.append(
                        "Consider increasing stabilizer content for better durability"
                    )

                if obj_values.get("maximize_recyclability", 0.0) < 0.6:
                    recommendations.append(
                        "Reduce crosslink density to improve recyclability"
                    )

                if obj_values.get("minimize_carbon_footprint", 1.0) > 2.0:
                    recommendations.append(
                        "Optimize manufacturing process to reduce carbon footprint"
                    )
            else:
                recommendations.append(
                    "Sustainability goals not met - consider parameter adjustment"
                )

            # Parameter-specific recommendations
            params = result.optimal_parameters
            if params.get("stabilizer_content", 0) < 0.02:
                recommendations.append(
                    "Consider higher stabilizer content for UV protection"
                )

            if params.get("filler_content", 0) > 0.2:
                recommendations.append("High filler content may impact recyclability")
        else:
            recommendations.append(
                "Optimization failed - try different algorithm or constraints"
            )

        return recommendations

    def _analyze_constraints(self, result, constraints) -> Dict[str, Any]:
        """Analyze constraint satisfaction."""
        if not result.success:
            return {"analysis": "Optimization failed", "violations": {}}

        analysis = {
            "total_constraints": len(constraints),
            "satisfied": 0,
            "violated": 0,
            "constraint_details": {},
        }

        for constraint in constraints:
            violation = result.constraint_violations.get(constraint.name, 0.0)

            if abs(violation) <= constraint.tolerance:
                analysis["satisfied"] += 1
                status = "satisfied"
            else:
                analysis["violated"] += 1
                status = "violated"

            analysis["constraint_details"][constraint.name] = {
                "status": status,
                "violation": violation,
                "description": constraint.description,
            }

        analysis["satisfaction_rate"] = (
            analysis["satisfied"] / analysis["total_constraints"]
        )

        return analysis

    def _calculate_performance_metrics(self, result) -> Dict[str, float]:
        """Calculate performance metrics from design result."""
        if not result.success:
            return {"error": "Optimization failed"}

        props = result.predicted_properties

        return {
            "strength_to_weight": props.get("tensile_strength", 0)
            / (1.0 + props.get("density", 1.0)),
            "durability_index": 1.0 / (props.get("degradation_rate", 0.001) + 1e-6),
            "eco_efficiency": props.get("recyclability_score", 0.5)
            / (props.get("carbon_footprint", 2.0) + 1e-6),
            "overall_performance": result.sustainability_scores.get(
                "overall_sustainability", 0
            ),
        }

    def _analyze_algorithm_performance(self, algorithm_results) -> Dict[str, Any]:
        """Analyze performance across different algorithms."""
        successful_results = {
            alg: result
            for alg, result in algorithm_results.items()
            if result.get("success", False)
        }

        if not successful_results:
            return {"error": "No algorithms succeeded"}

        analysis = {
            "success_rate": len(successful_results) / len(algorithm_results),
            "performance_ranking": [],
            "speed_ranking": [],
            "quality_ranking": [],
        }

        # Rank by computation time (speed)
        speed_sorted = sorted(
            successful_results.items(), key=lambda x: x[1]["computation_time"]
        )
        analysis["speed_ranking"] = [alg for alg, _ in speed_sorted]

        # Rank by sustainability score (quality)
        quality_sorted = sorted(
            successful_results.items(),
            key=lambda x: x[1]["sustainability_score"],
            reverse=True,
        )
        analysis["quality_ranking"] = [alg for alg, _ in quality_sorted]

        # Overall performance (weighted combination)
        performance_scores = {}
        for alg, result in successful_results.items():
            # Normalize and combine speed and quality
            speed_score = 1.0 / (result["computation_time"] + 1.0)
            quality_score = result["sustainability_score"]
            performance_scores[alg] = 0.3 * speed_score + 0.7 * quality_score

        performance_sorted = sorted(
            performance_scores.items(), key=lambda x: x[1], reverse=True
        )
        analysis["performance_ranking"] = [alg for alg, _ in performance_sorted]

        return analysis

    def _generate_algorithm_recommendations(self, analysis) -> List[str]:
        """Generate recommendations based on algorithm comparison."""
        recommendations = []

        if "error" in analysis:
            recommendations.append(
                "No algorithms succeeded - check problem formulation"
            )
            return recommendations

        if analysis["success_rate"] < 1.0:
            recommendations.append(
                f"Success rate: {analysis['success_rate']:.1%} - some algorithms failed"
            )

        if analysis["performance_ranking"]:
            best_overall = analysis["performance_ranking"][0]
            recommendations.append(f"Best overall algorithm: {best_overall}")

        if analysis["speed_ranking"]:
            fastest = analysis["speed_ranking"][0]
            recommendations.append(f"Fastest algorithm: {fastest}")

        if analysis["quality_ranking"]:
            highest_quality = analysis["quality_ranking"][0]
            recommendations.append(f"Highest quality results: {highest_quality}")

        # Specific recommendations
        if "genetic_algorithm" in analysis["performance_ranking"][:2]:
            recommendations.append(
                "Genetic algorithms perform well for this problem type"
            )

        if "bayesian_optimization" in analysis["quality_ranking"][:2]:
            recommendations.append(
                "Bayesian optimization provides high-quality solutions"
            )

        return recommendations

    def _validate_predictions(self, prediction_results) -> Dict[str, Any]:
        """Validate predictions against expected behaviors."""
        validation = {
            "cases_analyzed": len(prediction_results),
            "validation_results": {},
            "overall_consistency": 0.0,
        }

        consistency_scores = []

        for case_name, case_data in prediction_results.items():
            props = case_data["predicted_properties"]
            expected = case_data["expected_behavior"]

            # Check consistency with expected behavior
            consistency_score = 0.0
            checks = 0

            if "low degradation" in expected.lower():
                if props.get("degradation_rate", 1.0) < 0.002:
                    consistency_score += 1.0
                checks += 1

            if "high degradation" in expected.lower():
                if props.get("degradation_rate", 0.0) > 0.003:
                    consistency_score += 1.0
                checks += 1

            if "high recyclability" in expected.lower():
                if props.get("recyclability_score", 0.0) > 0.7:
                    consistency_score += 1.0
                checks += 1

            if "biodegradability" in expected.lower():
                if (
                    props.get("recyclability_score", 0.0) > 0.6
                ):  # Biodegradable materials often recyclable
                    consistency_score += 1.0
                checks += 1

            if checks > 0:
                consistency_score /= checks

            validation["validation_results"][case_name] = {
                "consistency_score": consistency_score,
                "expected_behavior": expected,
                "key_predictions": {
                    "degradation_rate": props.get("degradation_rate", 0),
                    "recyclability_score": props.get("recyclability_score", 0),
                    "carbon_footprint": props.get("carbon_footprint", 0),
                },
            }

            consistency_scores.append(consistency_score)

        validation["overall_consistency"] = (
            np.mean(consistency_scores) if consistency_scores else 0.0
        )

        return validation

    def _calculate_case_sustainability_score(self, properties) -> float:
        """Calculate sustainability score for a test case."""
        scores = []

        # Degradation resistance
        if "degradation_rate" in properties:
            scores.append(1.0 / (properties["degradation_rate"] + 1e-6))

        # Recyclability
        if "recyclability_score" in properties:
            scores.append(properties["recyclability_score"])

        # Environmental impact
        if "carbon_footprint" in properties:
            scores.append(1.0 / (properties["carbon_footprint"] + 1e-6))

        return np.mean(scores) if scores else 0.5

    def _assess_prediction_quality(self, prediction_results) -> Dict[str, Any]:
        """Assess overall prediction quality."""
        assessment = {
            "property_ranges": {},
            "prediction_consistency": 0.0,
            "outliers": [],
        }

        # Analyze property ranges
        all_properties = {}
        for case_data in prediction_results.values():
            for prop_name, prop_value in case_data["predicted_properties"].items():
                if prop_name not in all_properties:
                    all_properties[prop_name] = []
                all_properties[prop_name].append(prop_value)

        for prop_name, values in all_properties.items():
            if values:
                assessment["property_ranges"][prop_name] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                }

        # Check for consistency in sustainability metrics
        sustainability_scores = [
            case_data["sustainability_metrics"]["overall_score"]
            for case_data in prediction_results.values()
        ]

        if sustainability_scores:
            assessment["prediction_consistency"] = 1.0 - (
                np.std(sustainability_scores) / (np.mean(sustainability_scores) + 1e-6)
            )

        return assessment

    def create_optimization_visualization(self, results: Dict[str, Any]) -> str:
        """Create comprehensive optimization visualization."""

        if not PLOTLY_AVAILABLE:
            self._log("Plotly not available. Skipping visualization.")
            return "Visualization not available"

        # Create multi-scenario dashboard
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Parameter Optimization Results",
                "Sustainability Scores Comparison",
                "Algorithm Performance",
                "Prediction Validation",
            ),
            specs=[
                [{"type": "bar"}, {"type": "radar"}],
                [{"type": "scatter"}, {"type": "heatmap"}],
            ],
        )

        # Plot 1: Parameter optimization results
        if "basic_material_design" in results:
            basic_result = results["basic_material_design"]["design_result"]
            if basic_result.success:
                param_names = list(basic_result.optimal_parameters.keys())
                param_values = list(basic_result.optimal_parameters.values())

                fig.add_trace(
                    go.Bar(x=param_names, y=param_values, name="Optimal Parameters"),
                    row=1,
                    col=1,
                )

        # Plot 2: Sustainability scores radar
        if "multi_objective_optimization" in results:
            solutions = results["multi_objective_optimization"]["pareto_solutions"]
            if solutions:
                for i, solution in enumerate(solutions[:3]):  # Show top 3 solutions
                    metrics = [
                        "Degradation Resistance",
                        "Recyclability",
                        "Low Impact",
                        "Overall",
                    ]
                    values = [
                        1.0
                        / (
                            solution["objectives"].get(
                                "minimize_degradation_rate", 0.001
                            )
                            + 1e-6
                        ),
                        solution["objectives"].get("maximize_recyclability", 0.5),
                        1.0
                        / (
                            solution["objectives"].get("minimize_carbon_footprint", 2.0)
                            + 1e-6
                        ),
                        solution["sustainability_score"],
                    ]

                    fig.add_trace(
                        go.Scatterpolar(
                            r=values,
                            theta=metrics,
                            fill="toself",
                            name=f"Solution {i+1}",
                            line=dict(width=2),
                        ),
                        row=1,
                        col=2,
                    )

        # Plot 3: Algorithm performance
        if "algorithm_comparison" in results:
            alg_results = results["algorithm_comparison"]["algorithm_results"]
            algorithms = []
            times = []
            scores = []

            for alg, result in alg_results.items():
                if result.get("success", False):
                    algorithms.append(alg.replace("_", " ").title())
                    times.append(result["computation_time"])
                    scores.append(result["sustainability_score"])

            if algorithms:
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=scores,
                        mode="markers+text",
                        text=algorithms,
                        textposition="top center",
                        name="Algorithm Performance",
                        marker=dict(size=12),
                    ),
                    row=2,
                    col=1,
                )

        # Plot 4: Prediction validation heatmap
        if "property_prediction_validation" in results:
            validation = results["property_prediction_validation"][
                "validation_analysis"
            ]
            if "validation_results" in validation:
                case_names = list(validation["validation_results"].keys())
                consistency_scores = [
                    result["consistency_score"]
                    for result in validation["validation_results"].values()
                ]

                # Create simple heatmap representation
                fig.add_trace(
                    go.Heatmap(
                        z=[consistency_scores],
                        x=case_names,
                        y=["Consistency"],
                        colorscale="RdYlGn",
                        showscale=True,
                    ),
                    row=2,
                    col=2,
                )

        fig.update_layout(
            title="Comprehensive Inverse Design Analysis Dashboard",
            height=800,
            showlegend=True,
        )

        # Update polar chart
        fig.update_polars(radialaxis=dict(visible=True, range=[0, 1]), row=1, col=2)

        # Update axis labels
        fig.update_xaxes(title_text="Computation Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Sustainability Score", row=2, col=1)

        # Save to file
        if self.output_dir:
            viz_path = self.output_dir / "inverse_design_dashboard.html"
            fig.write_html(str(viz_path))
            return str(viz_path)

        return "Dashboard created (no output directory specified)"

    async def run_all_demos(self) -> Dict[str, Any]:
        """Run all inverse design demonstration scenarios."""
        self._log("Starting Comprehensive Inverse Design Demonstrations")
        self._log("=" * 80)

        # Create output directory
        self.output_dir = Path(tempfile.mkdtemp(prefix="inverse_design_demo_"))
        self._log(f"Output directory: {self.output_dir}")

        all_results = {}

        try:
            # Demo 1: Basic Material Design
            all_results[
                "basic_material_design"
            ] = await self.demo_basic_material_design()

            # Demo 2: Multi-Objective Optimization
            all_results[
                "multi_objective_optimization"
            ] = await self.demo_multi_objective_optimization()

            # Demo 3: Constraint-Based Optimization
            all_results[
                "constraint_optimization"
            ] = await self.demo_constraint_optimization()

            # Demo 4: Algorithm Comparison
            all_results["algorithm_comparison"] = await self.demo_algorithm_comparison()

            # Demo 5: Property Prediction Validation
            all_results[
                "property_prediction_validation"
            ] = await self.demo_property_prediction_validation()

            self._log("=" * 80)
            self._log("All demonstrations completed successfully!")

            # Create visualization
            viz_path = self.create_optimization_visualization(all_results)
            all_results["visualization_path"] = viz_path

            # Generate summary report
            summary_report = self.generate_summary_report(all_results)
            all_results["summary_report"] = summary_report

            return all_results

        except Exception as e:
            logger.error(f"Error during demonstrations: {e}")
            all_results["error"] = str(e)
            return all_results

    def generate_summary_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary report."""

        report = {
            "executive_summary": {},
            "key_findings": {},
            "performance_metrics": {},
            "recommendations": {},
        }

        # Executive summary
        successful_demos = sum(
            1
            for key, result in results.items()
            if key != "visualization_path"
            and key != "error"
            and isinstance(result, dict)
            and result.get("design_result", {}).get("success", True)
        )

        report["executive_summary"] = {
            "total_demonstrations": 5,
            "successful_demonstrations": successful_demos,
            "algorithms_tested": len(
                results.get("algorithm_comparison", {}).get("algorithm_results", {})
            ),
            "optimization_approaches": [
                "single_objective",
                "multi_objective",
                "constrained",
                "comparative",
            ],
            "prediction_validation_cases": len(
                results.get("property_prediction_validation", {}).get("test_cases", {})
            ),
        }

        # Key findings
        if (
            "basic_material_design" in results
            and results["basic_material_design"]["design_result"].success
        ):
            basic_result = results["basic_material_design"]
            report["key_findings"]["basic_design"] = {
                "sustainability_score": basic_result[
                    "design_result"
                ].sustainability_scores.get("overall_sustainability", 0),
                "computation_time": basic_result["design_result"].computation_time,
                "optimal_parameters": basic_result["design_result"].optimal_parameters,
            }

        if "multi_objective_optimization" in results:
            pareto_data = results["multi_objective_optimization"]
            report["key_findings"]["multi_objective"] = {
                "pareto_solutions_found": len(pareto_data.get("pareto_solutions", [])),
                "trade_off_correlations": pareto_data.get("objective_correlations", {}),
                "pareto_front_size": len(pareto_data.get("pareto_front", [])),
            }

        if "algorithm_comparison" in results:
            alg_data = results["algorithm_comparison"]
            performance_analysis = alg_data.get("performance_analysis", {})
            report["key_findings"]["algorithm_performance"] = {
                "success_rate": performance_analysis.get("success_rate", 0),
                "best_overall": performance_analysis.get("performance_ranking", [None])[
                    0
                ],
                "fastest": performance_analysis.get("speed_ranking", [None])[0],
                "highest_quality": performance_analysis.get("quality_ranking", [None])[
                    0
                ],
            }

        # Performance metrics
        total_computation_time = 0
        for key, result in results.items():
            if isinstance(result, dict) and "design_result" in result:
                total_computation_time += result["design_result"].computation_time

        report["performance_metrics"] = {
            "total_computation_time": total_computation_time,
            "gpu_acceleration_used": self.use_gpu and TORCH_AVAILABLE,
            "average_sustainability_score": self._calculate_average_sustainability_score(
                results
            ),
        }

        # Recommendations
        recommendations = []

        if (
            report["key_findings"]
            .get("algorithm_performance", {})
            .get("success_rate", 0)
            > 0.8
        ):
            recommendations.append(
                "Multiple optimization algorithms work well for sustainable material design"
            )

        if (
            report["key_findings"]
            .get("multi_objective", {})
            .get("pareto_solutions_found", 0)
            > 5
        ):
            recommendations.append(
                "Multi-objective optimization provides diverse design alternatives"
            )

        if report["performance_metrics"]["gpu_acceleration_used"]:
            recommendations.append(
                "GPU acceleration successfully enabled for improved performance"
            )
        else:
            recommendations.append(
                "Consider enabling GPU acceleration for better performance"
            )

        avg_score = report["performance_metrics"]["average_sustainability_score"]
        if avg_score > 0.7:
            recommendations.append(
                "Achieved high sustainability scores across optimization scenarios"
            )
        elif avg_score > 0.5:
            recommendations.append(
                "Moderate sustainability achieved - consider constraint refinement"
            )
        else:
            recommendations.append(
                "Low sustainability scores - review problem formulation"
            )

        report["recommendations"] = recommendations

        return report

    def _calculate_average_sustainability_score(self, results: Dict[str, Any]) -> float:
        """Calculate average sustainability score across all successful optimizations."""
        scores = []

        for key, result in results.items():
            if isinstance(result, dict) and "design_result" in result:
                design_result = result["design_result"]
                if design_result.success:
                    score = design_result.sustainability_scores.get(
                        "overall_sustainability", 0
                    )
                    scores.append(score)

        return np.mean(scores) if scores else 0.0


async def main():
    """Main demonstration function."""
    print("Comprehensive Inverse Design for Sustainable Materials Demo")
    print("=" * 80)

    # Initialize demo
    demo = InverseDesignDemo(use_gpu=TORCH_AVAILABLE, verbose=True)

    # Run all demonstrations
    results = await demo.run_all_demos()

    # Save results
    results_file = Path("inverse_design_demo_results.json")

    # Convert any non-serializable objects
    def convert_for_json(obj):
        if hasattr(obj, "value"):  # Handle Enum objects
            return obj.value
        elif isinstance(obj, (datetime, np.integer, np.floating)):
            return str(obj)
        elif isinstance(obj, dict):
            return {key: convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj

    serializable_results = convert_for_json(results)

    with open(results_file, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)

    print(f"\nResults saved to {results_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("DEMONSTRATION SUMMARY")
    print("=" * 80)

    if "summary_report" in results:
        report = results["summary_report"]

        print("\nExecutive Summary:")
        exec_summary = report.get("executive_summary", {})
        print(f"  Total Demonstrations: {exec_summary.get('total_demonstrations', 0)}")
        print(
            f"  Successful Demonstrations: {exec_summary.get('successful_demonstrations', 0)}"
        )
        print(f"  Algorithms Tested: {exec_summary.get('algorithms_tested', 0)}")

        print("\nKey Findings:")
        findings = report.get("key_findings", {})

        if "basic_design" in findings:
            basic = findings["basic_design"]
            print(
                f"  Basic Design Sustainability Score: {basic.get('sustainability_score', 0):.3f}"
            )

        if "multi_objective" in findings:
            multi_obj = findings["multi_objective"]
            print(
                f"  Pareto Solutions Found: {multi_obj.get('pareto_solutions_found', 0)}"
            )

        if "algorithm_performance" in findings:
            alg_perf = findings["algorithm_performance"]
            print(f"  Algorithm Success Rate: {alg_perf.get('success_rate', 0):.1%}")
            print(f"  Best Overall Algorithm: {alg_perf.get('best_overall', 'N/A')}")

        print("\nPerformance Metrics:")
        perf_metrics = report.get("performance_metrics", {})
        print(
            f"  Total Computation Time: {perf_metrics.get('total_computation_time', 0):.2f} seconds"
        )
        print(
            f"  GPU Acceleration: {'Enabled' if perf_metrics.get('gpu_acceleration_used', False) else 'Disabled'}"
        )
        print(
            f"  Average Sustainability Score: {perf_metrics.get('average_sustainability_score', 0):.3f}"
        )

        print("\nRecommendations:")
        recommendations = report.get("recommendations", [])
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

    if "visualization_path" in results:
        print(f"\nVisualization: {results['visualization_path']}")

    print("\nTechnology Stack:")
    print(f"  PyTorch: {'Available' if TORCH_AVAILABLE else 'Not Available'}")
    print(f"  Plotly: {'Available' if PLOTLY_AVAILABLE else 'Not Available'}")
    print(f"  GPU Acceleration: {'Enabled' if demo.use_gpu else 'Disabled'}")

    print("\nInverse design demonstration completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
