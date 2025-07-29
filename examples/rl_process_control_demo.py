"""
Comprehensive Demonstration of RL Process Control Framework

This script demonstrates the full capabilities of ChemiAccel's RL process control
framework through various chemical process optimization scenarios.

Key demonstrations:
1. Multi-step synthesis optimization with RL
2. Continuous manufacturing process control
3. Cost-aware experimental planning
4. Real-time feedback and model updates
5. Constraint-based optimization
6. Multi-objective process optimization
"""

import asyncio
import json
import logging
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import ChemiAccel RL components
from src.chemi_accel.ai.rl_process_control import (
    CUPY_AVAILABLE,
    JAX_AVAILABLE,
    TORCH_AVAILABLE,
    AgentType,
    ConstraintType,
    ContinuousManufacturingEnvironment,
    CostAwareAcquisitionFunction,
    EnvironmentType,
    ExperimentalCost,
    MultiStepSynthesisEnvironment,
    ProcessConstraint,
    ProcessControlTrainer,
    RealTimeData,
    RealTimeDataManager,
    RLConfig,
    create_cost_aware_acquisition_function,
    create_cost_constraints,
    create_quality_constraints,
    create_real_time_data_manager,
    create_safety_constraints,
    energy_cost_model,
    material_cost_model,
    train_continuous_manufacturing_agent,
    train_synthesis_agent,
)

if TORCH_AVAILABLE:
    from src.chemi_accel.ai.rl_process_control import PPOAgent

# Try to import optional plotting libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Some visualizations will be skipped.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RLProcessControlDemo:
    """Comprehensive demonstration of RL process control capabilities."""

    def __init__(self, use_gpu: bool = True, verbose: bool = True):
        """Initialize the demonstration."""
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        self.verbose = verbose
        self.results = {}

        if self.verbose:
            print("=" * 80)
            print("ChemiAccel RL Process Control Framework Demo")
            print("=" * 80)
            print(f"GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
            print(
                f"Available backends: PyTorch={TORCH_AVAILABLE}, JAX={JAX_AVAILABLE}, CuPy={CUPY_AVAILABLE}"
            )
            print()

    async def run_all_demos(self):
        """Run all demonstration scenarios."""
        demos = [
            ("Multi-Step Synthesis Optimization", self.demo_synthesis_optimization),
            ("Continuous Manufacturing Control", self.demo_continuous_manufacturing),
            ("Cost-Aware Experimental Planning", self.demo_cost_aware_planning),
            ("Real-Time Feedback Integration", self.demo_real_time_feedback),
            ("Constraint-Based Optimization", self.demo_constraint_optimization),
            ("Multi-Objective Process Control", self.demo_multi_objective_control),
            ("Scalability and Performance", self.demo_scalability_performance),
        ]

        for demo_name, demo_func in demos:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Running: {demo_name}")
                print(f"{'='*60}")

            try:
                result = await demo_func()
                self.results[demo_name] = result

                if self.verbose:
                    print(f"✓ {demo_name} completed successfully")
                    if result and isinstance(result, dict):
                        if "best_reward" in result:
                            print(f"  Best reward: {result['best_reward']:.4f}")
                        if "training_time" in result:
                            print(f"  Training time: {result['training_time']:.2f}s")
                        if "total_episodes" in result:
                            print(f"  Episodes: {result['total_episodes']}")

            except Exception as e:
                if self.verbose:
                    print(f"✗ {demo_name} failed: {e}")
                logger.error(f"Demo {demo_name} failed: {e}")
                self.results[demo_name] = None

        # Generate summary report
        self.generate_summary_report()

    async def demo_synthesis_optimization(self) -> Dict[str, Any]:
        """Demonstrate multi-step synthesis optimization with RL."""

        if self.verbose:
            print("Training RL agent for multi-step synthesis optimization...")

        if not TORCH_AVAILABLE:
            print("PyTorch not available. Skipping RL training demo.")
            return {"status": "skipped", "reason": "PyTorch not available"}

        # Define custom reaction network for drug synthesis
        reaction_network = {
            "reactions": [
                # Step 1: A + B -> C (coupling reaction)
                {"reactants": ["A", "B"], "products": ["C"], "rate_constant": 0.15},
                # Step 2: C -> D (cyclization)
                {"reactants": ["C"], "products": ["D"], "rate_constant": 0.08},
                # Step 3: D -> target (final transformation)
                {"reactants": ["D"], "products": ["target"], "rate_constant": 0.12},
                # Side reactions
                {"reactants": ["A"], "products": ["waste1"], "rate_constant": 0.02},
                {"reactants": ["C"], "products": ["waste2"], "rate_constant": 0.03},
                {
                    "reactants": ["D", "A"],
                    "products": ["waste3"],
                    "rate_constant": 0.01,
                },
            ],
            "initial_concentrations": {
                "A": 1.5,
                "B": 1.2,
                "C": 0.0,
                "D": 0.0,
                "target": 0.0,
                "waste1": 0.0,
                "waste2": 0.0,
                "waste3": 0.0,
            },
        }

        # Configuration for synthesis optimization
        config = RLConfig(
            agent_type=AgentType.PPO,
            environment_type=EnvironmentType.MULTI_STEP_SYNTHESIS,
            max_episodes=50,
            max_steps_per_episode=25,
            learning_rate=3e-4,
            batch_size=32,
            use_gpu=self.use_gpu,
            device="auto" if self.use_gpu else "cpu",
        )

        # Create constraints for safe operation
        constraints = [
            ProcessConstraint(
                name="temperature",
                constraint_type=ConstraintType.SAFETY,
                min_value=298.15,  # 25°C
                max_value=373.15,  # 100°C
                target_value=323.15,  # 50°C
                tolerance=10.0,
                violation_penalty=500.0,
            ),
            ProcessConstraint(
                name="target",
                constraint_type=ConstraintType.QUALITY,
                min_value=0.8,  # Minimum target concentration
                violation_penalty=200.0,
            ),
            ProcessConstraint(
                name="selectivity",
                constraint_type=ConstraintType.QUALITY,
                min_value=0.85,  # Minimum selectivity
                target_value=0.95,
                tolerance=0.05,
                violation_penalty=150.0,
            ),
        ]

        # Train synthesis agent
        start_time = time.time()
        agent, training_summary = await train_synthesis_agent(
            reaction_network=reaction_network, config=config, constraints=constraints
        )
        training_time = time.time() - start_time

        if self.verbose:
            print(f"  Training completed in {training_time:.2f}s")
            print(f"  Best synthesis reward: {training_summary['best_reward']:.4f}")
            print(f"  Final average reward: {training_summary['average_reward']:.4f}")

            # Show convergence info
            if training_summary.get("convergence_episode"):
                print(
                    f"  Converged at episode: {training_summary['convergence_episode']}"
                )

        # Test trained agent
        env = MultiStepSynthesisEnvironment(config, reaction_network)
        for constraint in constraints:
            env.add_constraint(constraint)

        test_results = await self._test_synthesis_agent(agent, env, num_episodes=5)

        result = {
            **training_summary,
            "training_time": training_time,
            "test_results": test_results,
            "reaction_network": reaction_network,
            "constraints": len(constraints),
        }

        return result

    async def demo_continuous_manufacturing(self) -> Dict[str, Any]:
        """Demonstrate continuous manufacturing process control."""

        if self.verbose:
            print("Training RL agent for continuous manufacturing control...")

        if not TORCH_AVAILABLE:
            print("PyTorch not available. Skipping RL training demo.")
            return {"status": "skipped", "reason": "PyTorch not available"}

        # Custom process configuration for pharmaceutical manufacturing
        process_config = {
            "reactor_volume": 2000.0,  # L
            "residence_time": 7200.0,  # 2 hours
            "heat_capacity": 4184.0,  # J/kg/K
            "heat_transfer_coeff": 1200.0,  # W/m²/K
            "nominal_flow_rate": 150.0,  # L/h
            "nominal_temperature": 345.0,  # K
            "nominal_pressure": 2.5e5,  # Pa
            "target_conversion": 0.92,
            "target_throughput": 120.0,  # kg/h
        }

        # Configuration for continuous control
        config = RLConfig(
            agent_type=AgentType.PPO,
            environment_type=EnvironmentType.CONTINUOUS_MANUFACTURING,
            max_episodes=80,
            max_steps_per_episode=100,  # Longer episodes for continuous process
            learning_rate=1e-4,  # Lower learning rate for stability
            batch_size=64,
            use_gpu=self.use_gpu,
            device="auto" if self.use_gpu else "cpu",
            cost_weight=0.15,
            constraint_weight=0.8,
        )

        # Process constraints for manufacturing
        constraints = (
            create_safety_constraints()
            + create_quality_constraints()
            + [
                ProcessConstraint(
                    name="throughput",
                    constraint_type=ConstraintType.QUALITY,
                    min_value=100.0,  # kg/h
                    target_value=120.0,
                    tolerance=5.0,
                    violation_penalty=100.0,
                ),
                ProcessConstraint(
                    name="fouling",
                    constraint_type=ConstraintType.QUALITY,
                    max_value=0.3,  # Maximum allowable fouling
                    violation_penalty=300.0,
                ),
            ]
        )

        # Train manufacturing agent
        start_time = time.time()
        agent, training_summary = await train_continuous_manufacturing_agent(
            process_config=process_config, config=config, constraints=constraints
        )
        training_time = time.time() - start_time

        if self.verbose:
            print(f"  Training completed in {training_time:.2f}s")
            print(f"  Best manufacturing reward: {training_summary['best_reward']:.4f}")
            print(f"  Episodes trained: {training_summary['total_episodes']}")

        # Test manufacturing agent
        env = ContinuousManufacturingEnvironment(config, process_config)
        for constraint in constraints:
            env.add_constraint(constraint)

        test_results = await self._test_manufacturing_agent(agent, env, num_episodes=3)

        result = {
            **training_summary,
            "training_time": training_time,
            "test_results": test_results,
            "process_config": process_config,
            "constraints": len(constraints),
        }

        return result

    async def demo_cost_aware_planning(self) -> Dict[str, Any]:
        """Demonstrate cost-aware experimental planning."""

        if self.verbose:
            print("Demonstrating cost-aware experimental planning...")

        # Define comprehensive cost models
        def advanced_material_cost_model(action: np.ndarray) -> ExperimentalCost:
            """Advanced material cost model with economies of scale."""
            flow_rate = abs(action[0]) if len(action) > 0 else 0.1
            temp_change = abs(action[1]) if len(action) > 1 else 0.0
            catalyst = abs(action[2]) if len(action) > 2 else 0.0

            # Base material costs
            reagent_cost = flow_rate * 25.0  # $25 per unit flow

            # Catalyst cost (expensive but catalytic)
            catalyst_cost = catalyst * 200.0  # $200 per unit catalyst

            # Temperature-dependent solvent costs
            solvent_cost = (1.0 + temp_change / 100.0) * 15.0

            # Energy cost for heating/cooling
            energy_cost = temp_change * 8.0

            # Equipment wear (higher for extreme conditions)
            equipment_cost = (1.0 + temp_change / 50.0) * 3.0

            # Labor cost (constant per experiment)
            labor_cost = 50.0

            # Environmental cost (waste disposal, emissions)
            env_cost = (reagent_cost + solvent_cost) * 0.1

            return ExperimentalCost(
                material_cost=reagent_cost + catalyst_cost + solvent_cost,
                energy_cost=energy_cost,
                time_cost=0.0,  # Time cost handled separately
                equipment_cost=equipment_cost,
                labor_cost=labor_cost,
                environmental_cost=env_cost,
            )

        def time_dependent_cost_model(action: np.ndarray) -> ExperimentalCost:
            """Time-dependent cost model."""
            complexity = np.sum(np.abs(action))

            # Time cost increases with complexity
            base_time = 2.0  # 2 hours base time
            complex_time = complexity * 0.5  # Additional time for complex experiments
            total_time = base_time + complex_time

            # Time cost: $30/hour
            time_cost = total_time * 30.0

            return ExperimentalCost(time_cost=time_cost)

        # Cost models dictionary
        cost_models = {
            "materials": advanced_material_cost_model,
            "time": time_dependent_cost_model,
            "energy": energy_cost_model,
        }

        # Comprehensive constraints
        constraints = (
            create_safety_constraints()
            + create_quality_constraints()
            + create_cost_constraints(max_cost_per_step=200.0)
        )

        # Create cost-aware acquisition function
        config = RLConfig(cost_weight=0.2, constraint_weight=0.6)
        acq_func = create_cost_aware_acquisition_function(
            cost_models=cost_models, constraints=constraints, config=config
        )

        # Test acquisition function on experimental candidates
        n_candidates = 50
        candidates = np.random.uniform(-1, 1, (n_candidates, 3))

        # Simulate surrogate model predictions
        surrogate_mean = np.random.uniform(0.3, 0.9, n_candidates)
        surrogate_variance = np.random.uniform(0.05, 0.3, n_candidates)
        best_value = 0.6

        # Evaluate cost-aware acquisition
        acquisition_values = acq_func.evaluate(
            candidates, surrogate_mean, surrogate_variance, best_value
        )

        # Find best candidates
        top_indices = np.argsort(acquisition_values)[-5:]
        top_candidates = candidates[top_indices]
        top_acq_values = acquisition_values[top_indices]

        # Calculate costs for top candidates
        cost_analysis = []
        for i, candidate in enumerate(top_candidates):
            total_cost = 0.0
            cost_breakdown = {}

            for name, cost_model in cost_models.items():
                cost = cost_model(candidate)
                cost_value = cost.total_cost()
                total_cost += cost_value
                cost_breakdown[name] = cost_value

            cost_analysis.append(
                {
                    "candidate": candidate.tolist(),
                    "acquisition_value": float(top_acq_values[i]),
                    "total_cost": total_cost,
                    "cost_breakdown": cost_breakdown,
                }
            )

        if self.verbose:
            print(f"  Evaluated {n_candidates} experimental candidates")
            print("  Top 5 candidates by cost-aware acquisition:")
            for i, analysis in enumerate(cost_analysis):
                print(
                    f"    {i+1}. Acq: {analysis['acquisition_value']:.4f}, "
                    f"Cost: ${analysis['total_cost']:.2f}"
                )

        result = {
            "num_candidates": n_candidates,
            "num_cost_models": len(cost_models),
            "num_constraints": len(constraints),
            "top_candidates": cost_analysis,
            "cost_models": list(cost_models.keys()),
            "constraint_types": [c.constraint_type.value for c in constraints],
        }

        return result

    async def demo_real_time_feedback(self) -> Dict[str, Any]:
        """Demonstrate real-time feedback and model updates."""

        if self.verbose:
            print("Demonstrating real-time feedback integration...")

        # Create real-time data manager
        config = RLConfig(
            data_buffer_size=500, model_update_frequency=20, prediction_window=10
        )

        data_manager = create_real_time_data_manager(config)

        # Add custom preprocessing for chemical data
        def chemical_data_preprocessing(data: RealTimeData) -> RealTimeData:
            """Custom preprocessing for chemical measurement data."""
            processed_measurements = {}

            for key, value in data.measurements.items():
                # Convert units if needed
                if key == "temperature_celsius":
                    processed_measurements["temperature"] = (
                        float(value) + 273.15
                    )  # Convert to Kelvin
                elif key == "pressure_bar":
                    processed_measurements["pressure"] = (
                        float(value) * 1e5
                    )  # Convert to Pa
                elif key in ["yield", "selectivity", "conversion"]:
                    # Ensure percentages are in decimal form
                    val = float(value)
                    processed_measurements[key] = val / 100.0 if val > 1.0 else val
                else:
                    processed_measurements[key] = value

            return RealTimeData(
                timestamp=data.timestamp,
                experiment_id=data.experiment_id,
                measurements=processed_measurements,
                metadata=data.metadata,
            )

        data_manager.add_preprocessing(chemical_data_preprocessing)

        # Add validation for chemical data
        def chemical_data_validation(data: RealTimeData) -> bool:
            """Validate chemical measurement data."""
            required_fields = ["temperature", "pressure"]

            # Check required fields
            for field in required_fields:
                if field not in data.measurements:
                    return False

            # Check reasonable ranges
            temp = data.measurements.get("temperature", 0)
            if not (250.0 <= temp <= 500.0):  # Reasonable temperature range in K
                return False

            pressure = data.measurements.get("pressure", 0)
            if not (1e4 <= pressure <= 1e7):  # Reasonable pressure range in Pa
                return False

            return True

        data_manager.add_validation_rule(chemical_data_validation)

        # Start real-time streaming
        await data_manager.start_streaming()

        # Simulate real-time experimental data
        experiment_data = []
        base_time = time.time()

        for i in range(50):
            # Simulate varying experimental conditions
            temp_c = 25 + i * 2 + np.random.normal(0, 5)  # Temperature trend with noise
            pressure_bar = 2.0 + np.random.normal(0, 0.2)  # Pressure with noise
            yield_pct = 70 + i * 0.5 + np.random.normal(0, 5)  # Improving yield trend
            selectivity_pct = 85 + np.random.normal(0, 3)  # Selectivity with noise

            data_point = RealTimeData(
                timestamp=base_time + i * 60,  # 1 minute intervals
                experiment_id=f"rt_exp_{i:03d}",
                measurements={
                    "temperature_celsius": temp_c,
                    "pressure_bar": pressure_bar,
                    "yield": yield_pct,
                    "selectivity": selectivity_pct,
                    "conversion": min(95, 80 + i * 0.3),
                    "cost": 45.0 + np.random.uniform(-10, 15),
                },
                metadata={
                    "operator": "demo_system",
                    "batch_id": f"batch_{i//10}",
                    "instrument": "demo_reactor",
                },
            )

            # Ingest data with small delay to simulate real-time
            await asyncio.sleep(0.01)  # Small delay for simulation
            await data_manager.ingest_data(data_point)
            experiment_data.append(data_point)

        # Wait for processing
        await asyncio.sleep(0.1)

        # Get data statistics
        stats = data_manager.get_data_statistics()

        # Get recent data for analysis
        recent_data = data_manager.get_recent_data(20)

        # Stop streaming
        await data_manager.stop_streaming()

        if self.verbose:
            print(f"  Processed {stats['buffer_size']} real-time data points")
            print(
                f"  Data ingestion rate: {stats.get('data_rate', 0):.2f} points/second"
            )
            print(f"  Pending model updates: {stats['pending_updates']}")

            if stats.get("measurement_stats"):
                temp_stats = stats["measurement_stats"].get("temperature", {})
                if temp_stats:
                    print(
                        f"  Temperature range: {temp_stats.get('min', 0):.1f} - {temp_stats.get('max', 0):.1f} K"
                    )

                yield_stats = stats["measurement_stats"].get("yield", {})
                if yield_stats:
                    print(f"  Average yield: {yield_stats.get('mean', 0):.2f}")

        result = {
            "total_data_points": len(experiment_data),
            "buffer_size": stats["buffer_size"],
            "data_rate": stats.get("data_rate", 0),
            "measurement_statistics": stats.get("measurement_stats", {}),
            "preprocessing_functions": len(data_manager.preprocessing_functions),
            "validation_rules": len(data_manager.validation_rules),
            "recent_data_count": len(recent_data),
        }

        return result

    async def demo_constraint_optimization(self) -> Dict[str, Any]:
        """Demonstrate constraint-based optimization."""

        if self.verbose:
            print("Demonstrating constraint-based optimization...")

        # Create comprehensive constraint set for pharmaceutical process
        constraints = [
            # Safety constraints (hard limits)
            ProcessConstraint(
                name="temperature",
                constraint_type=ConstraintType.SAFETY,
                min_value=273.15,  # 0°C
                max_value=473.15,  # 200°C
                violation_penalty=1000.0,
                weight=2.0,
            ),
            ProcessConstraint(
                name="pressure",
                constraint_type=ConstraintType.SAFETY,
                min_value=1e5,  # 1 bar
                max_value=15e5,  # 15 bar
                violation_penalty=1000.0,
                weight=2.0,
            ),
            ProcessConstraint(
                name="pH",
                constraint_type=ConstraintType.SAFETY,
                min_value=2.0,
                max_value=12.0,
                violation_penalty=500.0,
                weight=1.5,
            ),
            # Quality constraints (targets with tolerance)
            ProcessConstraint(
                name="yield",
                constraint_type=ConstraintType.QUALITY,
                min_value=0.70,  # Minimum 70% yield
                target_value=0.90,  # Target 90% yield
                tolerance=0.05,  # ±5% tolerance
                violation_penalty=200.0,
                weight=1.0,
            ),
            ProcessConstraint(
                name="purity",
                constraint_type=ConstraintType.QUALITY,
                min_value=0.95,  # Minimum 95% purity
                target_value=0.98,  # Target 98% purity
                tolerance=0.01,  # ±1% tolerance
                violation_penalty=300.0,
                weight=1.2,
            ),
            ProcessConstraint(
                name="selectivity",
                constraint_type=ConstraintType.QUALITY,
                min_value=0.85,  # Minimum 85% selectivity
                target_value=0.95,  # Target 95% selectivity
                tolerance=0.03,  # ±3% tolerance
                violation_penalty=150.0,
                weight=1.0,
            ),
            # Economic constraints
            ProcessConstraint(
                name="cost_per_kg",
                constraint_type=ConstraintType.COST,
                max_value=500.0,  # Maximum $500/kg
                target_value=300.0,  # Target $300/kg
                tolerance=50.0,  # ±$50/kg tolerance
                violation_penalty=100.0,
                weight=0.8,
            ),
            ProcessConstraint(
                name="energy_consumption",
                constraint_type=ConstraintType.COST,
                max_value=100.0,  # Maximum 100 kWh/kg
                target_value=60.0,  # Target 60 kWh/kg
                tolerance=10.0,  # ±10 kWh/kg tolerance
                violation_penalty=80.0,
                weight=0.6,
            ),
            # Environmental constraints
            ProcessConstraint(
                name="waste_generation",
                constraint_type=ConstraintType.ENVIRONMENTAL,
                max_value=0.2,  # Maximum 20% waste
                target_value=0.05,  # Target 5% waste
                tolerance=0.02,  # ±2% tolerance
                violation_penalty=120.0,
                weight=0.7,
            ),
            ProcessConstraint(
                name="solvent_usage",
                constraint_type=ConstraintType.ENVIRONMENTAL,
                max_value=5.0,  # Maximum 5 L/kg product
                target_value=2.0,  # Target 2 L/kg product
                tolerance=0.5,  # ±0.5 L/kg tolerance
                violation_penalty=60.0,
                weight=0.5,
            ),
        ]

        # Test constraint evaluation on simulated process data
        n_test_points = 100
        constraint_results = []

        for i in range(n_test_points):
            # Generate random process conditions
            measurements = {
                "temperature": np.random.uniform(250, 500),  # K
                "pressure": np.random.uniform(0.5e5, 20e5),  # Pa
                "pH": np.random.uniform(1, 14),
                "yield": np.random.uniform(0.5, 1.0),
                "purity": np.random.uniform(0.8, 1.0),
                "selectivity": np.random.uniform(0.7, 1.0),
                "cost_per_kg": np.random.uniform(200, 600),  # $/kg
                "energy_consumption": np.random.uniform(40, 120),  # kWh/kg
                "waste_generation": np.random.uniform(0, 0.3),
                "solvent_usage": np.random.uniform(1, 8),  # L/kg
            }

            # Evaluate all constraints
            total_penalty = 0.0
            satisfied_constraints = 0
            constraint_violations = []

            for constraint in constraints:
                if constraint.name in measurements:
                    satisfied, penalty = constraint.evaluate(
                        measurements[constraint.name]
                    )
                    total_penalty += penalty

                    if satisfied:
                        satisfied_constraints += 1
                    else:
                        constraint_violations.append(
                            {
                                "name": constraint.name,
                                "type": constraint.constraint_type.value,
                                "value": measurements[constraint.name],
                                "penalty": penalty,
                            }
                        )

            constraint_results.append(
                {
                    "measurements": measurements,
                    "total_penalty": total_penalty,
                    "satisfied_constraints": satisfied_constraints,
                    "total_constraints": len(constraints),
                    "constraint_satisfaction_rate": satisfied_constraints
                    / len(constraints),
                    "violations": constraint_violations,
                }
            )

        # Analyze constraint satisfaction
        satisfaction_rates = [
            r["constraint_satisfaction_rate"] for r in constraint_results
        ]
        total_penalties = [r["total_penalty"] for r in constraint_results]

        # Find best operating conditions (highest satisfaction, lowest penalty)
        best_idx = max(
            range(len(constraint_results)),
            key=lambda i: constraint_results[i]["constraint_satisfaction_rate"]
            - constraint_results[i]["total_penalty"] / 1000.0,
        )

        best_conditions = constraint_results[best_idx]

        # Analyze constraint types
        constraint_type_analysis = {}
        for constraint_type in ConstraintType:
            type_constraints = [
                c for c in constraints if c.constraint_type == constraint_type
            ]
            if type_constraints:
                constraint_type_analysis[constraint_type.value] = {
                    "count": len(type_constraints),
                    "avg_penalty": np.mean(
                        [c.violation_penalty for c in type_constraints]
                    ),
                    "avg_weight": np.mean([c.weight for c in type_constraints]),
                }

        if self.verbose:
            print(
                f"  Evaluated {n_test_points} process conditions against {len(constraints)} constraints"
            )
            print(
                f"  Average constraint satisfaction: {np.mean(satisfaction_rates):.2%}"
            )
            print(
                f"  Best satisfaction rate: {best_conditions['constraint_satisfaction_rate']:.2%}"
            )
            print(f"  Best conditions penalty: {best_conditions['total_penalty']:.2f}")

            print("  Constraint types:")
            for ctype, analysis in constraint_type_analysis.items():
                print(
                    f"    {ctype}: {analysis['count']} constraints, "
                    f"avg penalty: {analysis['avg_penalty']:.1f}"
                )

        result = {
            "num_test_points": n_test_points,
            "num_constraints": len(constraints),
            "average_satisfaction_rate": float(np.mean(satisfaction_rates)),
            "best_satisfaction_rate": float(
                best_conditions["constraint_satisfaction_rate"]
            ),
            "best_total_penalty": float(best_conditions["total_penalty"]),
            "best_conditions": best_conditions["measurements"],
            "constraint_type_analysis": constraint_type_analysis,
            "constraint_distribution": {
                ctype.value: len([c for c in constraints if c.constraint_type == ctype])
                for ctype in ConstraintType
            },
        }

        return result

    async def demo_multi_objective_control(self) -> Dict[str, Any]:
        """Demonstrate multi-objective process control."""

        if self.verbose:
            print("Demonstrating multi-objective process control...")

        # Define multi-objective optimization problem for pharmaceutical process
        # Objectives: maximize yield, maximize purity, minimize cost, minimize time

        def multi_objective_process_function(x: np.ndarray) -> np.ndarray:
            """Multi-objective process function.

            Args:
                x: Process variables [temperature, pressure, concentration, flow_rate]

            Returns:
                Objectives [yield, purity, -cost, -time] (negative for minimization)
            """
            temp, pressure, conc, flow = x

            # Yield model (higher temp and pressure generally better, but with optimum)
            temp_factor = np.exp(
                -((temp - 350) ** 2) / (2 * 30**2)
            )  # Optimum at 350K
            pressure_factor = 1.0 - np.exp(-pressure / 2e5)  # Asymptotic increase
            conc_factor = conc / (conc + 0.5)  # Michaelis-Menten type
            yield_obj = 0.95 * temp_factor * pressure_factor * conc_factor

            # Purity model (depends on selectivity, affected by temperature)
            selectivity = 0.98 * np.exp(-abs(temp - 340) / 50)  # Optimal at 340K
            conversion = min(0.99, 0.8 + 0.1 * pressure / 1e5)
            purity_obj = selectivity * conversion

            # Cost model (materials, energy, time)
            material_cost = conc * flow * 50  # $50 per unit
            energy_cost = (temp - 298) * flow * 0.1  # Heating cost
            pressure_cost = (pressure - 1e5) * flow * 0.001  # Compression cost
            total_cost = material_cost + energy_cost + pressure_cost
            cost_obj = -total_cost  # Negative for maximization

            # Time model (reaction rate dependent)
            rate_constant = 0.1 * np.exp(-5000 / (8.314 * temp))  # Arrhenius
            reaction_time = -np.log(1 - 0.95) / (rate_constant * conc)  # 95% conversion
            time_obj = -reaction_time  # Negative for maximization

            return np.array([yield_obj, purity_obj, cost_obj, time_obj])

        # Generate Pareto front through sampling
        n_samples = 200
        pareto_solutions = []

        # Parameter bounds
        temp_bounds = (300, 400)  # K
        pressure_bounds = (1e5, 10e5)  # Pa
        conc_bounds = (0.5, 3.0)  # M
        flow_bounds = (10, 100)  # L/h

        # Latin Hypercube sampling for better coverage
        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(d=4, seed=42)
        unit_samples = sampler.random(n_samples)

        # Scale to actual bounds
        samples = np.zeros_like(unit_samples)
        samples[:, 0] = (
            unit_samples[:, 0] * (temp_bounds[1] - temp_bounds[0]) + temp_bounds[0]
        )
        samples[:, 1] = (
            unit_samples[:, 1] * (pressure_bounds[1] - pressure_bounds[0])
            + pressure_bounds[0]
        )
        samples[:, 2] = (
            unit_samples[:, 2] * (conc_bounds[1] - conc_bounds[0]) + conc_bounds[0]
        )
        samples[:, 3] = (
            unit_samples[:, 3] * (flow_bounds[1] - flow_bounds[0]) + flow_bounds[0]
        )

        # Evaluate objectives for all samples
        all_objectives = np.array(
            [multi_objective_process_function(x) for x in samples]
        )

        # Find Pareto front
        pareto_indices = self._find_pareto_front(all_objectives)
        pareto_objectives = all_objectives[pareto_indices]
        pareto_parameters = samples[pareto_indices]

        # Analyze Pareto front
        pareto_analysis = {
            "num_pareto_solutions": len(pareto_indices),
            "pareto_fraction": len(pareto_indices) / n_samples,
            "objective_ranges": {
                "yield": {
                    "min": float(pareto_objectives[:, 0].min()),
                    "max": float(pareto_objectives[:, 0].max()),
                },
                "purity": {
                    "min": float(pareto_objectives[:, 1].min()),
                    "max": float(pareto_objectives[:, 1].max()),
                },
                "cost": {
                    "min": float(-pareto_objectives[:, 2].max()),
                    "max": float(-pareto_objectives[:, 2].min()),
                },
                "time": {
                    "min": float(-pareto_objectives[:, 3].max()),
                    "max": float(-pareto_objectives[:, 3].min()),
                },
            },
        }

        # Find compromise solutions
        # 1. Maximum yield solution
        max_yield_idx = np.argmax(pareto_objectives[:, 0])
        max_yield_solution = {
            "parameters": pareto_parameters[max_yield_idx].tolist(),
            "objectives": pareto_objectives[max_yield_idx].tolist(),
            "yield": float(pareto_objectives[max_yield_idx, 0]),
            "purity": float(pareto_objectives[max_yield_idx, 1]),
        }

        # 2. Best balance solution (using simple weighted sum)
        weights = np.array([0.3, 0.3, 0.2, 0.2])  # Balanced weights
        normalized_objectives = (pareto_objectives - pareto_objectives.min(axis=0)) / (
            pareto_objectives.max(axis=0) - pareto_objectives.min(axis=0)
        )
        weighted_scores = np.sum(normalized_objectives * weights, axis=1)
        best_balance_idx = np.argmax(weighted_scores)
        best_balance_solution = {
            "parameters": pareto_parameters[best_balance_idx].tolist(),
            "objectives": pareto_objectives[best_balance_idx].tolist(),
            "weighted_score": float(weighted_scores[best_balance_idx]),
        }

        # 3. Minimum cost solution
        min_cost_idx = np.argmax(pareto_objectives[:, 2])  # Remember cost is negative
        min_cost_solution = {
            "parameters": pareto_parameters[min_cost_idx].tolist(),
            "objectives": pareto_objectives[min_cost_idx].tolist(),
            "cost": float(-pareto_objectives[min_cost_idx, 2]),
        }

        if self.verbose:
            print(f"  Generated {n_samples} process candidates")
            print(
                f"  Found {len(pareto_indices)} Pareto-optimal solutions ({pareto_analysis['pareto_fraction']:.1%})"
            )
            print(
                f"  Yield range: {pareto_analysis['objective_ranges']['yield']['min']:.3f} - {pareto_analysis['objective_ranges']['yield']['max']:.3f}"
            )
            print(
                f"  Purity range: {pareto_analysis['objective_ranges']['purity']['min']:.3f} - {pareto_analysis['objective_ranges']['purity']['max']:.3f}"
            )
            print(
                f"  Best yield solution: {max_yield_solution['yield']:.3f} yield, {max_yield_solution['purity']:.3f} purity"
            )
            print(
                f"  Best balance solution score: {best_balance_solution['weighted_score']:.3f}"
            )
            print(f"  Minimum cost solution: ${min_cost_solution['cost']:.2f}")

        result = {
            "num_samples": n_samples,
            "pareto_analysis": pareto_analysis,
            "compromise_solutions": {
                "max_yield": max_yield_solution,
                "best_balance": best_balance_solution,
                "min_cost": min_cost_solution,
            },
            "pareto_front_size": len(pareto_indices),
            "hypervolume": self._calculate_hypervolume(pareto_objectives),
        }

        return result

    async def demo_scalability_performance(self) -> Dict[str, Any]:
        """Demonstrate scalability and performance characteristics."""

        if self.verbose:
            print("Demonstrating scalability and performance...")

        scalability_results = {}

        # Test 1: Environment step performance
        if self.verbose:
            print("  Testing environment step performance...")

        config = RLConfig()
        env = MultiStepSynthesisEnvironment(config)

        step_times = []
        for _ in range(100):
            state = env.reset()
            action = np.random.uniform(-1, 1, env.action_dim)

            start_time = time.time()
            env.step(action)
            step_time = time.time() - start_time
            step_times.append(step_time)

        scalability_results["environment_step_performance"] = {
            "mean_step_time": float(np.mean(step_times)),
            "std_step_time": float(np.std(step_times)),
            "steps_per_second": float(1.0 / np.mean(step_times)),
            "num_steps_tested": len(step_times),
        }

        # Test 2: Cost model performance
        if self.verbose:
            print("  Testing cost model performance...")

        cost_times = []
        for _ in range(1000):
            action = np.random.uniform(-1, 1, 4)

            start_time = time.time()
            cost = material_cost_model(action)
            cost_time = time.time() - start_time
            cost_times.append(cost_time)

        scalability_results["cost_model_performance"] = {
            "mean_cost_time": float(np.mean(cost_times)),
            "evaluations_per_second": float(1.0 / np.mean(cost_times)),
            "num_evaluations_tested": len(cost_times),
        }

        # Test 3: Constraint evaluation performance
        if self.verbose:
            print("  Testing constraint evaluation performance...")

        constraints = create_safety_constraints() + create_quality_constraints()
        constraint_times = []

        for _ in range(500):
            measurements = {
                "temperature": np.random.uniform(300, 400),
                "pressure": np.random.uniform(1e5, 5e5),
                "yield": np.random.uniform(0.7, 1.0),
                "purity": np.random.uniform(0.9, 1.0),
            }

            start_time = time.time()
            for constraint in constraints:
                if constraint.name in measurements:
                    constraint.evaluate(measurements[constraint.name])
            constraint_time = time.time() - start_time
            constraint_times.append(constraint_time)

        scalability_results["constraint_evaluation_performance"] = {
            "mean_constraint_time": float(np.mean(constraint_times)),
            "evaluations_per_second": float(1.0 / np.mean(constraint_times)),
            "num_constraints": len(constraints),
            "num_evaluations_tested": len(constraint_times),
        }

        # Test 4: Data manager throughput
        if self.verbose:
            print("  Testing data manager throughput...")

        data_manager = create_real_time_data_manager()

        throughput_start = time.time()
        for i in range(100):
            data_point = RealTimeData(
                timestamp=time.time(),
                experiment_id=f"perf_test_{i}",
                measurements={"temperature": 350.0, "yield": 0.85},
            )
            await data_manager.ingest_data(data_point)

        throughput_time = time.time() - throughput_start

        scalability_results["data_manager_throughput"] = {
            "total_time": throughput_time,
            "data_points_per_second": 100.0 / throughput_time,
            "buffer_size": len(data_manager.data_buffer),
        }

        # Test 5: Memory usage estimation
        import sys

        # Create large environment to test memory
        large_config = RLConfig(data_buffer_size=10000)
        large_env = ContinuousManufacturingEnvironment(large_config)
        large_data_manager = create_real_time_data_manager(large_config)

        # Estimate memory usage (simplified)
        memory_estimate = {
            "environment_state_size": sys.getsizeof(large_env.current_state),
            "data_buffer_capacity": large_config.data_buffer_size,
            "estimated_memory_mb": (large_config.data_buffer_size * 1000)
            / (1024 * 1024),  # Rough estimate
        }

        scalability_results["memory_usage"] = memory_estimate

        if self.verbose:
            print(
                f"  Environment performance: {scalability_results['environment_step_performance']['steps_per_second']:.0f} steps/sec"
            )
            print(
                f"  Cost model performance: {scalability_results['cost_model_performance']['evaluations_per_second']:.0f} evals/sec"
            )
            print(
                f"  Constraint performance: {scalability_results['constraint_evaluation_performance']['evaluations_per_second']:.0f} evals/sec"
            )
            print(
                f"  Data throughput: {scalability_results['data_manager_throughput']['data_points_per_second']:.0f} points/sec"
            )
            print(
                f"  Estimated memory usage: {memory_estimate['estimated_memory_mb']:.1f} MB"
            )

        return scalability_results

    async def _test_synthesis_agent(
        self, agent, env, num_episodes: int = 3
    ) -> Dict[str, Any]:
        """Test synthesis agent performance."""
        test_rewards = []
        test_lengths = []
        test_info = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0.0
            episode_length = 0
            episode_info = {"product_concentrations": [], "selectivities": []}

            for step in range(20):  # Limit test episode length
                action = await agent.select_action(state, training=False)
                next_state, reward, done, info = env.step(action)

                state = next_state
                episode_reward += reward
                episode_length += 1

                episode_info["product_concentrations"].append(
                    info.get("product_concentration", 0.0)
                )
                episode_info["selectivities"].append(info.get("selectivity", 0.0))

                if done:
                    break

            test_rewards.append(episode_reward)
            test_lengths.append(episode_length)
            test_info.append(episode_info)

        return {
            "mean_reward": float(np.mean(test_rewards)),
            "std_reward": float(np.std(test_rewards)),
            "mean_length": float(np.mean(test_lengths)),
            "rewards": test_rewards,
            "final_product_concentrations": [
                info["product_concentrations"][-1] for info in test_info
            ],
            "final_selectivities": [info["selectivities"][-1] for info in test_info],
        }

    async def _test_manufacturing_agent(
        self, agent, env, num_episodes: int = 3
    ) -> Dict[str, Any]:
        """Test manufacturing agent performance."""
        test_rewards = []
        test_lengths = []
        test_throughputs = []
        test_qualities = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0.0
            episode_length = 0

            for step in range(50):  # Limit test episode length
                action = await agent.select_action(state, training=False)
                next_state, reward, done, info = env.step(action)

                state = next_state
                episode_reward += reward
                episode_length += 1

                if done:
                    break

            test_rewards.append(episode_reward)
            test_lengths.append(episode_length)
            test_throughputs.append(state[13])  # Throughput
            test_qualities.append(state[10])  # Quality

        return {
            "mean_reward": float(np.mean(test_rewards)),
            "std_reward": float(np.std(test_rewards)),
            "mean_length": float(np.mean(test_lengths)),
            "mean_throughput": float(np.mean(test_throughputs)),
            "mean_quality": float(np.mean(test_qualities)),
            "rewards": test_rewards,
        }

    def _find_pareto_front(self, objectives: np.ndarray) -> np.ndarray:
        """Find Pareto front indices."""
        n_points = len(objectives)
        pareto_indices = []

        for i in range(n_points):
            is_pareto = True
            for j in range(n_points):
                if i != j:
                    # Check if point j dominates point i
                    if np.all(objectives[j] >= objectives[i]) and np.any(
                        objectives[j] > objectives[i]
                    ):
                        is_pareto = False
                        break

            if is_pareto:
                pareto_indices.append(i)

        return np.array(pareto_indices)

    def _calculate_hypervolume(self, pareto_objectives: np.ndarray) -> float:
        """Calculate hypervolume (simplified 2D case)."""
        if len(pareto_objectives) == 0:
            return 0.0

        # Sort by first objective
        sorted_indices = np.argsort(pareto_objectives[:, 0])
        sorted_objectives = pareto_objectives[sorted_indices]

        # Calculate area under the curve (2D approximation)
        hypervolume = 0.0
        ref_point = np.zeros(pareto_objectives.shape[1])  # Origin as reference

        for i in range(len(sorted_objectives)):
            if i == 0:
                width = sorted_objectives[i, 0] - ref_point[0]
            else:
                width = sorted_objectives[i, 0] - sorted_objectives[i - 1, 0]

            height = sorted_objectives[i, 1] - ref_point[1]
            hypervolume += width * height

        return float(hypervolume)

    def generate_summary_report(self):
        """Generate comprehensive summary report."""

        if self.verbose:
            print("\n" + "=" * 80)
            print("RL PROCESS CONTROL DEMO SUMMARY REPORT")
            print("=" * 80)

        successful_demos = [
            name for name, result in self.results.items() if result is not None
        ]
        failed_demos = [name for name, result in self.results.items() if result is None]

        print(
            f"\nSuccessful demonstrations: {len(successful_demos)}/{len(self.results)}"
        )

        if successful_demos:
            print("\n✓ Completed demonstrations:")
            for demo in successful_demos:
                print(f"  - {demo}")

        if failed_demos:
            print("\n✗ Failed demonstrations:")
            for demo in failed_demos:
                print(f"  - {demo}")

        # Performance summary
        print("\nPerformance Summary:")
        print(f"GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
        print(f"PyTorch available: {TORCH_AVAILABLE}")
        print(f"JAX available: {JAX_AVAILABLE}")
        print(f"CuPy available: {CUPY_AVAILABLE}")

        # Training results summary
        if "Multi-Step Synthesis Optimization" in self.results:
            result = self.results["Multi-Step Synthesis Optimization"]
            if result and "best_reward" in result:
                print(f"Best synthesis reward: {result['best_reward']:.4f}")

        if "Continuous Manufacturing Control" in self.results:
            result = self.results["Continuous Manufacturing Control"]
            if result and "best_reward" in result:
                print(f"Best manufacturing reward: {result['best_reward']:.4f}")

        # Cost analysis summary
        if "Cost-Aware Experimental Planning" in self.results:
            result = self.results["Cost-Aware Experimental Planning"]
            if result and "top_candidates" in result:
                top_cost = result["top_candidates"][0]["total_cost"]
                print(f"Best experimental cost: ${top_cost:.2f}")

        # Constraint satisfaction summary
        if "Constraint-Based Optimization" in self.results:
            result = self.results["Constraint-Based Optimization"]
            if result and "average_satisfaction_rate" in result:
                satisfaction = result["average_satisfaction_rate"]
                print(f"Average constraint satisfaction: {satisfaction:.2%}")

        # Real-time data summary
        if "Real-Time Feedback Integration" in self.results:
            result = self.results["Real-Time Feedback Integration"]
            if result and "total_data_points" in result:
                data_points = result["total_data_points"]
                print(f"Real-time data points processed: {data_points}")

        # Performance summary
        if "Scalability and Performance" in self.results:
            result = self.results["Scalability and Performance"]
            if result and "environment_step_performance" in result:
                steps_per_sec = result["environment_step_performance"][
                    "steps_per_second"
                ]
                print(f"Environment performance: {steps_per_sec:.0f} steps/sec")

        print("\n" + "=" * 80)
        print("Demo completed successfully!")
        print("=" * 80)

    def save_results(self, filepath: str = "rl_demo_results.json"):
        """Save demonstration results to file."""

        # Convert numpy arrays and other non-serializable objects
        serializable_results = {}
        for key, value in self.results.items():
            if value is not None:
                serializable_results[key] = self._make_serializable(value)

        with open(filepath, "w") as f:
            json.dump(serializable_results, f, indent=2)

        if self.verbose:
            print(f"\nResults saved to {filepath}")

    def _make_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        else:
            return obj


async def main():
    """Run the comprehensive RL process control demonstration."""

    print("Starting ChemiAccel RL Process Control Demo...")
    print("This may take several minutes to complete all demonstrations.\n")

    # Create demo instance
    demo = RLProcessControlDemo(use_gpu=True, verbose=True)  # Enable GPU if available

    # Run all demonstrations
    await demo.run_all_demos()

    # Save results
    demo.save_results("rl_process_control_results.json")

    print("\nDemo completed! Check the results above for performance metrics.")
    print("Results have been saved to 'rl_process_control_results.json'")


if __name__ == "__main__":
    # Check requirements
    if not TORCH_AVAILABLE:
        print("Warning: PyTorch not available. Some RL demonstrations will be skipped.")
        print("Install PyTorch to see full RL capabilities.")

    try:
        from scipy.stats import qmc
    except ImportError:
        print(
            "Warning: SciPy not available. Some advanced sampling methods will be skipped."
        )

    # Run the demo
    asyncio.run(main())
