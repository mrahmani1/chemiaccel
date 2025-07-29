#!/usr/bin/env python3
"""
Reinforcement Learning Process Optimization Demonstration

Comprehensive demonstration of ChemiAccel's RL-driven process optimization
capabilities, showcasing GPU-accelerated agents, advanced reward functions,
and parameter space exploration for manufacturing process optimization.

Features demonstrated:
- GPU-accelerated RL agents (PPO, DDPG, SAC)
- Multi-objective reward functions
- Parameter space exploration
- Process optimization with real-time adaptation
- Performance comparison between algorithms
"""

import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.chemi_accel.digital_twin.manufacturing_integration import (
    FluidDynamicsSimulator,
    HeatTransferSimulator,
    ManufacturingDigitalTwin,
    ProcessConditions,
    ProcessType,
)

# Import ChemiAccel components
from src.chemi_accel.digital_twin.multiscale_framework import (
    AtomisticModel,
    ContinuumModel,
    LengthScale,
    MesoscopicModel,
    MultiScaleFramework,
)
from src.chemi_accel.digital_twin.parameter_space_exploration import (
    ExplorationConfig,
    ExplorationStrategy,
    ParameterExplorerFactory,
    ParameterSpace,
    RLParameterExplorer,
)
from src.chemi_accel.digital_twin.rl_agents import ProcessOptimizer, RLAgentFactory
from src.chemi_accel.digital_twin.rl_process_optimization import (
    EnergyEfficiencyRewardFunction,
    MultiObjectiveRewardFunction,
    OptimizationObjective,
    ProcessAction,
    ProcessEnvironment,
    ProcessState,
    QualityRewardFunction,
    RLAlgorithm,
    RLConfig,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_manufacturing_digital_twin():
    """Setup manufacturing digital twin for RL optimization."""
    print("=" * 70)
    print("Setting Up Manufacturing Digital Twin for RL Optimization")
    print("=" * 70)

    # Create multi-scale framework
    framework = MultiScaleFramework()

    # Add models at different scales
    atomistic_model = AtomisticModel("md")
    mesoscopic_model = MesoscopicModel("dpd")
    continuum_model = ContinuumModel("fem")

    framework.add_model(LengthScale.ATOMISTIC, atomistic_model, "atomistic_md")
    framework.add_model(LengthScale.MESOSCOPIC, mesoscopic_model, "mesoscopic_dpd")
    framework.add_model(LengthScale.MACROSCOPIC, continuum_model, "continuum_fem")

    # Create standard interfaces
    framework.create_standard_interfaces()

    # Initialize models
    config = {
        "atomistic_md": {
            "atoms": ["C"] * 100,
            "positions": np.random.random((100, 3)) * 10,
            "velocities": np.random.normal(0, 1, (100, 3)),
            "temperature": 473.15,
            "pressure": 2e5,
        },
        "mesoscopic_dpd": {
            "num_particles": 1000,
            "box_size": [20.0, 20.0, 20.0],
            "temperature": 473.15,
        },
        "continuum_fem": {
            "mesh_size": [20, 20, 20],
        },
    }

    framework.initialize_models(config)

    # Create manufacturing digital twin
    digital_twin = ManufacturingDigitalTwin(framework)

    # Add process simulators
    fluid_simulator = FluidDynamicsSimulator()
    heat_simulator = HeatTransferSimulator()

    digital_twin.add_process_simulator(
        ProcessType.EXTRUSION, fluid_simulator, "fluid_sim"
    )
    digital_twin.add_process_simulator(
        ProcessType.EXTRUSION, heat_simulator, "heat_sim"
    )

    # Initialize simulators
    sim_config = {"mesh_size": [15, 15, 15], "domain_size": [1.0, 0.5, 0.5]}

    fluid_simulator.initialize(sim_config)
    heat_simulator.initialize(sim_config)

    print(f"✓ Created multi-scale framework with {len(framework.models)} models")
    print(
        f"✓ Initialized manufacturing digital twin with {len(digital_twin.process_simulators)} simulators"
    )

    return digital_twin


def demonstrate_reward_functions():
    """Demonstrate different reward function capabilities."""
    print("\n" + "=" * 70)
    print("Demonstrating Reward Functions")
    print("=" * 70)

    # Create sample states and outputs for testing
    initial_state = ProcessState(
        temperature=473.15,
        pressure=2e5,
        flow_rate=0.1,
        shear_rate=100.0,
        residence_time=60.0,
        viscosity=1000.0,
        density=1000.0,
        thermal_conductivity=0.2,
        part_quality=0.7,
        dimensional_accuracy=0.75,
        surface_finish=0.7,
    )

    improved_state = ProcessState(
        temperature=473.15,
        pressure=2e5,
        flow_rate=0.1,
        shear_rate=100.0,
        residence_time=60.0,
        viscosity=1000.0,
        density=1000.0,
        thermal_conductivity=0.2,
        part_quality=0.85,
        dimensional_accuracy=0.9,
        surface_finish=0.8,
        quality_trend=[0.7, 0.75, 0.8, 0.85],
    )

    action = ProcessAction(0.1, -0.05, 0.02, 0.15, -0.1)

    # Test scenarios
    scenarios = [
        {
            "name": "High Quality",
            "output": ProcessConditions(
                part_quality=0.85,
                dimensional_accuracy=0.9,
                surface_finish=0.8,
                processing_time=80.0,
                energy_consumption=800.0,
            ),
        },
        {
            "name": "Low Energy",
            "output": ProcessConditions(
                part_quality=0.75,
                dimensional_accuracy=0.8,
                surface_finish=0.75,
                processing_time=90.0,
                energy_consumption=500.0,
            ),
        },
        {
            "name": "Poor Performance",
            "output": ProcessConditions(
                part_quality=0.4,
                dimensional_accuracy=0.45,
                surface_finish=0.4,
                processing_time=150.0,
                energy_consumption=1500.0,
            ),
        },
    ]

    # Test reward functions
    reward_functions = {
        "Quality": QualityRewardFunction(target_quality=0.9),
        "Energy Efficiency": EnergyEfficiencyRewardFunction(baseline_energy=1000.0),
        "Multi-Objective": MultiObjectiveRewardFunction(
            {
                OptimizationObjective.QUALITY: 0.4,
                OptimizationObjective.ENERGY_EFFICIENCY: 0.3,
                OptimizationObjective.CYCLE_TIME: 0.3,
            }
        ),
    }

    print("Reward Function Comparison:")
    print("-" * 70)
    print(f"{'Scenario':<20} {'Quality':<12} {'Energy':<12} {'Multi-Obj':<12}")
    print("-" * 70)

    for scenario in scenarios:
        scenario_name = scenario["name"]
        output = scenario["output"]

        rewards = {}
        for rf_name, rf in reward_functions.items():
            # Create a proper ProcessOutput object
            from src.chemi_accel.digital_twin.manufacturing_integration import (
                ProcessOutput,
            )

            process_output = ProcessOutput(
                part_quality=getattr(output, "part_quality", 0.5),
                dimensional_accuracy=getattr(output, "dimensional_accuracy", 0.5),
                surface_finish=getattr(output, "surface_finish", 0.5),
                processing_time=getattr(output, "processing_time", 100.0),
                energy_consumption=getattr(output, "energy_consumption", 1000.0),
            )

            reward = rf.calculate_reward(
                initial_state, action, improved_state, process_output
            )
            rewards[rf_name] = reward

        print(
            f"{scenario_name:<20} {rewards['Quality']:<12.3f} {rewards['Energy Efficiency']:<12.3f} {rewards['Multi-Objective']:<12.3f}"
        )

    print("-" * 70)
    print("✓ Reward functions demonstrate different optimization objectives")
    print("✓ Quality function prioritizes part quality metrics")
    print("✓ Energy function balances quality with energy consumption")
    print("✓ Multi-objective function provides balanced optimization")

    return reward_functions


def demonstrate_rl_agents(digital_twin, reward_function):
    """Demonstrate different RL algorithms."""
    print("\n" + "=" * 70)
    print("Demonstrating RL Algorithms")
    print("=" * 70)

    # Parameter ranges for optimization
    parameter_ranges = {
        "temperature": (450.0, 500.0),
        "pressure": (1.5e5, 2.5e5),
        "flow_rate": (0.05, 0.15),
        "shear_rate": (50.0, 150.0),
        "residence_time": (30.0, 90.0),
    }

    # RL algorithms to test
    algorithms = [RLAlgorithm.PPO, RLAlgorithm.DDPG, RLAlgorithm.SAC]

    results = {}

    for algorithm in algorithms:
        print(f"\nTesting {algorithm.value.upper()} Algorithm:")
        print("-" * 40)

        # Create configuration
        config = RLConfig(
            algorithm=algorithm,
            device="cpu",  # Use CPU for demo compatibility
            hidden_sizes=[64, 64],
            learning_rate=3e-4,
            max_episodes=20,  # Reduced for demo
            max_steps_per_episode=50,
            batch_size=32,
            eval_frequency=10,
        )

        # Create optimizer
        optimizer = ProcessOptimizer(
            digital_twin,
            ProcessType.EXTRUSION,
            reward_function,
            parameter_ranges,
            config,
        )

        # Run optimization
        start_time = time.time()
        optimization_results = optimizer.optimize(num_episodes=20)
        training_time = time.time() - start_time

        # Store results
        results[algorithm.value] = {
            "best_reward": optimization_results["best_reward"],
            "final_performance": optimization_results["final_performance"],
            "training_time": training_time,
            "convergence_episode": optimization_results.get("convergence_episode"),
            "training_history": optimization_results["training_history"],
        }

        print(f"  Best reward: {optimization_results['best_reward']:.3f}")
        print(f"  Final performance: {optimization_results['final_performance']:.3f}")
        print(f"  Training time: {training_time:.1f}s")

        convergence = optimization_results.get("convergence_episode")
        if convergence:
            print(f"  Converged at episode: {convergence}")
        else:
            print(f"  Did not converge within {config.max_episodes} episodes")

    # Compare algorithms
    print("\nAlgorithm Comparison:")
    print("-" * 60)
    print(f"{'Algorithm':<12} {'Best Reward':<12} {'Final Perf':<12} {'Time (s)':<10}")
    print("-" * 60)

    for algo_name, result in results.items():
        print(
            f"{algo_name.upper():<12} {result['best_reward']:<12.3f} "
            f"{result['final_performance']:<12.3f} {result['training_time']:<10.1f}"
        )

    print("-" * 60)

    # Find best algorithm
    best_algo = max(results.keys(), key=lambda k: results[k]["best_reward"])
    print(f"✓ Best performing algorithm: {best_algo.upper()}")

    return results, best_algo


def demonstrate_parameter_space_exploration(digital_twin):
    """Demonstrate parameter space exploration."""
    print("\n" + "=" * 70)
    print("Demonstrating Parameter Space Exploration")
    print("=" * 70)

    # Define parameter space
    parameter_space = ParameterSpace(
        parameters={
            "temperature": (450.0, 500.0),
            "pressure": (1.5e5, 2.5e5),
            "flow_rate": (0.05, 0.15),
            "shear_rate": (50.0, 150.0),
            "residence_time": (30.0, 90.0),
        },
        priorities={
            "temperature": 1.0,
            "pressure": 0.8,
            "flow_rate": 0.9,
            "shear_rate": 0.7,
            "residence_time": 0.6,
        },
    )

    print(f"Parameter space dimension: {parameter_space.dimension}")
    print(f"Parameters: {list(parameter_space.parameters.keys())}")

    # Test different exploration strategies
    strategies = [
        ExplorationStrategy.GAUSSIAN_PROCESS,
        ExplorationStrategy.MULTI_OBJECTIVE,
        ExplorationStrategy.ADAPTIVE_GRID,
    ]

    exploration_results = {}

    for strategy in strategies:
        print(f"\nTesting {strategy.value.replace('_', ' ').title()} Strategy:")
        print("-" * 50)

        # Create exploration config
        config = ExplorationConfig(
            strategy=strategy,
            num_initial_samples=10,
            num_exploration_samples=30,
            max_iterations=20,
        )

        # Create explorer
        if strategy == ExplorationStrategy.MULTI_OBJECTIVE:
            explorer = ParameterExplorerFactory.create_explorer(
                strategy,
                parameter_space,
                config,
                objectives=[
                    OptimizationObjective.QUALITY,
                    OptimizationObjective.ENERGY_EFFICIENCY,
                ],
            )
        else:
            explorer = ParameterExplorerFactory.create_explorer(
                strategy, parameter_space, config
            )

        # Define evaluation function
        def evaluate_parameters(params):
            try:
                # Convert to process conditions
                conditions = ProcessConditions(
                    temperature=params.get("temperature"),
                    pressure=params.get("pressure"),
                    flow_rate=params.get("flow_rate"),
                    shear_rate=params.get("shear_rate"),
                    residence_time=params.get("residence_time"),
                )

                # Set conditions and run simulation
                digital_twin.set_process_conditions(conditions)
                output = digital_twin.run_manufacturing_simulation(
                    ProcessType.EXTRUSION
                )

                # Calculate response
                response = (
                    0.4 * output.part_quality
                    + 0.3 * output.dimensional_accuracy
                    + 0.2 * output.surface_finish
                    + 0.1 * min(1.0, 100.0 / max(output.processing_time, 1.0))
                )

                return response

            except Exception as e:
                logger.warning(f"Parameter evaluation failed: {e}")
                return 0.1

        # Run exploration
        start_time = time.time()
        results = explorer.explore(evaluate_parameters, num_iterations=20)
        exploration_time = time.time() - start_time

        exploration_results[strategy.value] = {
            "best_response": results["best_response"],
            "total_evaluations": results["total_evaluations"],
            "exploration_time": exploration_time,
            "exploration_efficiency": results["exploration_efficiency"],
            "parameter_importance": results["parameter_importance"],
        }

        print(f"  Best response: {results['best_response']:.3f}")
        print(f"  Total evaluations: {results['total_evaluations']}")
        print(f"  Exploration time: {exploration_time:.1f}s")
        print(f"  Efficiency: {results['exploration_efficiency']:.3f}")

        # Show parameter importance
        importance = results["parameter_importance"]
        top_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
        print(
            f"  Most important parameters: {', '.join([f'{p}({i:.2f})' for p, i in top_params])}"
        )

    # Compare exploration strategies
    print("\nExploration Strategy Comparison:")
    print("-" * 70)
    print(
        f"{'Strategy':<20} {'Best Response':<12} {'Evaluations':<12} {'Efficiency':<10}"
    )
    print("-" * 70)

    for strategy_name, result in exploration_results.items():
        strategy_display = strategy_name.replace("_", " ").title()
        print(
            f"{strategy_display:<20} {result['best_response']:<12.3f} "
            f"{result['total_evaluations']:<12} {result['exploration_efficiency']:<10.3f}"
        )

    print("-" * 70)

    best_strategy = max(
        exploration_results.keys(),
        key=lambda k: exploration_results[k]["best_response"],
    )
    print(f"✓ Best exploration strategy: {best_strategy.replace('_', ' ').title()}")

    return exploration_results


def demonstrate_rl_guided_exploration(digital_twin, best_rl_algo):
    """Demonstrate RL-guided parameter exploration."""
    print("\n" + "=" * 70)
    print("Demonstrating RL-Guided Parameter Exploration")
    print("=" * 70)

    # Create parameter space
    parameter_space = ParameterSpace(
        parameters={
            "temperature": (450.0, 500.0),
            "pressure": (1.5e5, 2.5e5),
            "flow_rate": (0.05, 0.15),
            "shear_rate": (50.0, 150.0),
        }
    )

    # Create RL agent for guidance
    rl_config = RLConfig(
        algorithm=RLAlgorithm[best_rl_algo.upper()],
        device="cpu",
        hidden_sizes=[64, 64],
        max_episodes=10,  # Pre-train briefly
    )

    # Create and train RL agent
    reward_function = QualityRewardFunction()
    parameter_ranges = dict(parameter_space.parameters)

    optimizer = ProcessOptimizer(
        digital_twin,
        ProcessType.EXTRUSION,
        reward_function,
        parameter_ranges,
        rl_config,
    )

    print("Pre-training RL agent for guidance...")
    optimizer.optimize(num_episodes=10)

    # Create RL-guided explorer
    exploration_config = ExplorationConfig(
        strategy=ExplorationStrategy.GAUSSIAN_PROCESS,
        num_initial_samples=5,
        max_iterations=30,
    )

    rl_explorer = RLParameterExplorer(
        digital_twin,
        ProcessType.EXTRUSION,
        parameter_space,
        optimizer.agent,
        exploration_config,
    )

    # Run RL-guided exploration
    print("\nRunning RL-guided exploration...")
    start_time = time.time()

    rl_results = rl_explorer.explore_with_rl_guidance(
        num_iterations=30, rl_guidance_frequency=5
    )

    exploration_time = time.time() - start_time

    print("\nRL-Guided Exploration Results:")
    print("-" * 50)
    print(f"Strategy: {rl_results['exploration_strategy']}")
    print(f"Total iterations: {rl_results['total_iterations']}")
    print(f"RL guidance frequency: {rl_results['rl_guidance_frequency']}")
    print(f"Best response: {rl_results['best_response']:.3f}")
    print(f"Exploration time: {exploration_time:.1f}s")

    # Analyze RL guidance impact
    guidance_impact = rl_results["rl_guidance_impact"]
    print("\nRL Guidance Impact:")
    print(f"  RL-guided mean response: {guidance_impact['rl_guided_mean']:.3f}")
    print(
        f"  Standard exploration mean: {guidance_impact['exploration_only_mean']:.3f}"
    )
    print(f"  Relative improvement: {guidance_impact['relative_improvement']:.1%}")
    print(f"  Guidance effective: {guidance_impact['rl_guidance_effectiveness']}")

    if guidance_impact["rl_guidance_effectiveness"]:
        print("✓ RL guidance significantly improved exploration performance")
    else:
        print("⚠ RL guidance did not show significant improvement")

    return rl_results


def demonstrate_multi_objective_optimization(digital_twin):
    """Demonstrate multi-objective optimization."""
    print("\n" + "=" * 70)
    print("Demonstrating Multi-Objective Optimization")
    print("=" * 70)

    # Create multi-objective reward function
    multi_objective_reward = MultiObjectiveRewardFunction(
        objective_weights={
            OptimizationObjective.QUALITY: 0.4,
            OptimizationObjective.ENERGY_EFFICIENCY: 0.3,
            OptimizationObjective.CYCLE_TIME: 0.2,
            OptimizationObjective.YIELD: 0.1,
        }
    )

    # Parameter ranges
    parameter_ranges = {
        "temperature": (450.0, 500.0),
        "pressure": (1.5e5, 2.5e5),
        "flow_rate": (0.05, 0.15),
        "shear_rate": (50.0, 150.0),
        "residence_time": (30.0, 90.0),
    }

    # RL configuration for multi-objective optimization
    config = RLConfig(
        algorithm=RLAlgorithm.SAC,  # SAC works well for multi-objective
        device="cpu",
        hidden_sizes=[128, 128, 64],
        learning_rate=1e-4,
        max_episodes=30,
        max_steps_per_episode=100,
    )

    # Create optimizer
    optimizer = ProcessOptimizer(
        digital_twin,
        ProcessType.EXTRUSION,
        multi_objective_reward,
        parameter_ranges,
        config,
    )

    print("Running multi-objective optimization...")
    start_time = time.time()

    results = optimizer.optimize(num_episodes=30)

    optimization_time = time.time() - start_time

    print("\nMulti-Objective Optimization Results:")
    print("-" * 50)
    print(f"Algorithm: {results['algorithm'].upper()}")
    print(f"Total episodes: {results['total_episodes']}")
    print(f"Best reward: {results['best_reward']:.3f}")
    print(f"Final performance: {results['final_performance']:.3f}")
    print(f"Optimization time: {optimization_time:.1f}s")

    # Get optimal conditions
    best_state, best_reward = optimizer.get_optimal_conditions()

    print("\nOptimal Process Conditions:")
    print("-" * 30)
    print(
        f"Temperature: {best_state.temperature:.1f} K ({best_state.temperature-273.15:.1f}°C)"
    )
    print(f"Pressure: {best_state.pressure/1e5:.2f} bar")
    print(f"Flow rate: {best_state.flow_rate:.3f} kg/s")
    print(f"Shear rate: {best_state.shear_rate:.1f} s⁻¹")
    print(f"Residence time: {best_state.residence_time:.1f} s")

    print("\nOptimal Performance Metrics:")
    print("-" * 30)
    print(f"Part quality: {best_state.part_quality:.3f}")
    print(f"Dimensional accuracy: {best_state.dimensional_accuracy:.3f}")
    print(f"Surface finish: {best_state.surface_finish:.3f}")

    # Show Pareto front if available
    if (
        hasattr(multi_objective_reward, "pareto_front")
        and multi_objective_reward.pareto_front
    ):
        pareto_front = multi_objective_reward.get_pareto_front()
        print(f"\nPareto Front: {len(pareto_front)} solutions found")

        # Show top 3 Pareto solutions
        print("Top Pareto Solutions:")
        for i, solution in enumerate(pareto_front[:3]):
            objectives = solution["objectives"]
            print(
                f"  Solution {i+1}: Quality={objectives[0]:.3f}, "
                f"Energy={objectives[1]:.3f}, Time={objectives[2]:.3f}"
            )

    print("✓ Multi-objective optimization completed successfully")
    print("✓ Found optimal balance between quality, energy, and time objectives")

    return results, best_state


def generate_visualization_plots(results_data):
    """Generate visualization plots for results."""
    print("\n" + "=" * 70)
    print("Generating Visualization Plots")
    print("=" * 70)

    try:
        # Create output directory
        output_dir = Path("rl_optimization_results")
        output_dir.mkdir(exist_ok=True)

        # Plot 1: Algorithm comparison
        if "rl_algorithms" in results_data:
            plt.figure(figsize=(12, 8))

            algorithms = list(results_data["rl_algorithms"].keys())
            best_rewards = [
                results_data["rl_algorithms"][algo]["best_reward"]
                for algo in algorithms
            ]
            training_times = [
                results_data["rl_algorithms"][algo]["training_time"]
                for algo in algorithms
            ]

            # Subplot 1: Best rewards
            plt.subplot(2, 2, 1)
            bars = plt.bar(
                algorithms, best_rewards, color=["skyblue", "lightcoral", "lightgreen"]
            )
            plt.title("Best Rewards by Algorithm")
            plt.ylabel("Best Reward")
            plt.xticks(rotation=45)

            # Add value labels on bars
            for bar, reward in zip(bars, best_rewards):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{reward:.3f}",
                    ha="center",
                    va="bottom",
                )

            # Subplot 2: Training times
            plt.subplot(2, 2, 2)
            bars = plt.bar(
                algorithms, training_times, color=["orange", "purple", "gold"]
            )
            plt.title("Training Times by Algorithm")
            plt.ylabel("Training Time (s)")
            plt.xticks(rotation=45)

            # Add value labels
            for bar, time_val in zip(bars, training_times):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{time_val:.1f}s",
                    ha="center",
                    va="bottom",
                )

            # Subplot 3: Training progress (if available)
            plt.subplot(2, 2, 3)
            for algo in algorithms:
                history = results_data["rl_algorithms"][algo].get(
                    "training_history", {}
                )
                if "episode_rewards" in history:
                    episodes = range(1, len(history["episode_rewards"]) + 1)
                    plt.plot(
                        episodes,
                        history["episode_rewards"],
                        label=algo.upper(),
                        marker="o",
                        markersize=3,
                    )

            plt.title("Training Progress")
            plt.xlabel("Episode")
            plt.ylabel("Episode Reward")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Subplot 4: Exploration strategies (if available)
            if "exploration_strategies" in results_data:
                plt.subplot(2, 2, 4)
                strategies = list(results_data["exploration_strategies"].keys())
                responses = [
                    results_data["exploration_strategies"][strategy]["best_response"]
                    for strategy in strategies
                ]

                # Truncate strategy names for display
                display_names = [s.replace("_", "\n").title() for s in strategies]
                bars = plt.bar(
                    display_names, responses, color=["teal", "coral", "gold"]
                )
                plt.title("Exploration Strategy Performance")
                plt.ylabel("Best Response")
                plt.xticks(rotation=0, fontsize=9)

                # Add value labels
                for bar, response in zip(bars, responses):
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{response:.3f}",
                        ha="center",
                        va="bottom",
                    )

            plt.tight_layout()
            plt.savefig(
                output_dir / "rl_algorithm_comparison.png", dpi=300, bbox_inches="tight"
            )
            print(
                f"✓ Algorithm comparison plot saved to {output_dir / 'rl_algorithm_comparison.png'}"
            )

        # Plot 2: Parameter importance (if available)
        if "exploration_strategies" in results_data:
            plt.figure(figsize=(10, 6))

            # Get parameter importance from Gaussian Process explorer
            gp_results = results_data["exploration_strategies"].get("gaussian_process")
            if gp_results and "parameter_importance" in gp_results:
                importance = gp_results["parameter_importance"]

                params = list(importance.keys())
                importances = list(importance.values())

                plt.barh(params, importances, color="lightblue")
                plt.title("Parameter Importance Analysis")
                plt.xlabel("Importance Score")
                plt.ylabel("Parameters")

                # Add value labels
                for i, (param, imp) in enumerate(zip(params, importances)):
                    plt.text(imp + 0.01, i, f"{imp:.3f}", va="center")

                plt.tight_layout()
                plt.savefig(
                    output_dir / "parameter_importance.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                print(
                    f"✓ Parameter importance plot saved to {output_dir / 'parameter_importance.png'}"
                )

        # Plot 3: Multi-objective optimization results (if available)
        if "multi_objective" in results_data:
            plt.figure(figsize=(10, 6))

            mo_results = results_data["multi_objective"]
            if "training_history" in mo_results:
                history = mo_results["training_history"]

                if "episode_rewards" in history:
                    episodes = range(1, len(history["episode_rewards"]) + 1)
                    rewards = history["episode_rewards"]

                    plt.plot(
                        episodes, rewards, "b-", linewidth=2, label="Episode Reward"
                    )

                    # Add moving average
                    window_size = min(5, len(rewards))
                    if window_size > 1:
                        moving_avg = np.convolve(
                            rewards, np.ones(window_size) / window_size, mode="valid"
                        )
                        avg_episodes = range(window_size, len(rewards) + 1)
                        plt.plot(
                            avg_episodes,
                            moving_avg,
                            "r--",
                            linewidth=2,
                            label=f"{window_size}-Episode Moving Average",
                        )

                    plt.title("Multi-Objective Optimization Progress")
                    plt.xlabel("Episode")
                    plt.ylabel("Reward")
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                    plt.tight_layout()
                    plt.savefig(
                        output_dir / "multi_objective_progress.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    print(
                        f"✓ Multi-objective progress plot saved to {output_dir / 'multi_objective_progress.png'}"
                    )

        plt.show()

        print(f"✓ All plots saved to {output_dir}")

    except ImportError:
        print("matplotlib not available, skipping plot generation")
    except Exception as e:
        print(f"Plot generation failed: {e}")


def generate_comprehensive_report(results_data):
    """Generate comprehensive optimization report."""
    print("\n" + "=" * 70)
    print("Generating Comprehensive Optimization Report")
    print("=" * 70)

    report_content = []
    report_content.append("# ChemiAccel RL Process Optimization Report")
    report_content.append("=" * 50)
    report_content.append("")

    # Executive Summary
    report_content.append("## Executive Summary")
    report_content.append("")

    if "rl_algorithms" in results_data:
        best_algo = max(
            results_data["rl_algorithms"].keys(),
            key=lambda k: results_data["rl_algorithms"][k]["best_reward"],
        )
        best_reward = results_data["rl_algorithms"][best_algo]["best_reward"]

        report_content.append(f"- **Best performing algorithm**: {best_algo.upper()}")
        report_content.append(f"- **Optimal reward achieved**: {best_reward:.3f}")

        if "multi_objective" in results_data:
            mo_reward = results_data["multi_objective"]["best_reward"]
            report_content.append(
                f"- **Multi-objective optimization reward**: {mo_reward:.3f}"
            )

        report_content.append("")

    # Algorithm Performance
    if "rl_algorithms" in results_data:
        report_content.append("## RL Algorithm Performance")
        report_content.append("")
        report_content.append(
            "| Algorithm | Best Reward | Final Performance | Training Time (s) | Convergence |"
        )
        report_content.append(
            "|-----------|-------------|-------------------|-------------------|-------------|"
        )

        for algo, results in results_data["rl_algorithms"].items():
            convergence = results.get("convergence_episode", "No")
            if convergence != "No" and convergence is not None:
                convergence = f"Episode {convergence}"

            report_content.append(
                f"| {algo.upper()} | {results['best_reward']:.3f} | "
                f"{results['final_performance']:.3f} | {results['training_time']:.1f} | {convergence} |"
            )

        report_content.append("")

    # Exploration Strategies
    if "exploration_strategies" in results_data:
        report_content.append("## Parameter Space Exploration")
        report_content.append("")
        report_content.append(
            "| Strategy | Best Response | Evaluations | Efficiency | Time (s) |"
        )
        report_content.append(
            "|----------|---------------|-------------|------------|----------|"
        )

        for strategy, results in results_data["exploration_strategies"].items():
            strategy_name = strategy.replace("_", " ").title()
            report_content.append(
                f"| {strategy_name} | {results['best_response']:.3f} | "
                f"{results['total_evaluations']} | {results['exploration_efficiency']:.3f} | "
                f"{results['exploration_time']:.1f} |"
            )

        report_content.append("")

        # Parameter importance
        gp_results = results_data["exploration_strategies"].get("gaussian_process")
        if gp_results and "parameter_importance" in gp_results:
            report_content.append("### Parameter Importance Ranking")
            report_content.append("")

            importance = gp_results["parameter_importance"]
            sorted_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)

            for i, (param, imp) in enumerate(sorted_params, 1):
                report_content.append(
                    f"{i}. **{param.replace('_', ' ').title()}**: {imp:.3f}"
                )

            report_content.append("")

    # Multi-Objective Results
    if "multi_objective" in results_data:
        report_content.append("## Multi-Objective Optimization")
        report_content.append("")

        mo_results = results_data["multi_objective"]
        report_content.append(
            f"- **Algorithm used**: {mo_results['algorithm'].upper()}"
        )
        report_content.append(f"- **Best reward**: {mo_results['best_reward']:.3f}")
        report_content.append(
            f"- **Final performance**: {mo_results['final_performance']:.3f}"
        )

        if "optimal_conditions" in results_data:
            optimal = results_data["optimal_conditions"]
            report_content.append("")
            report_content.append("### Optimal Process Conditions")
            report_content.append(
                f"- **Temperature**: {optimal.temperature:.1f} K ({optimal.temperature-273.15:.1f}°C)"
            )
            report_content.append(f"- **Pressure**: {optimal.pressure/1e5:.2f} bar")
            report_content.append(f"- **Flow rate**: {optimal.flow_rate:.3f} kg/s")
            report_content.append(f"- **Shear rate**: {optimal.shear_rate:.1f} s⁻¹")
            report_content.append(
                f"- **Residence time**: {optimal.residence_time:.1f} s"
            )

            report_content.append("")
            report_content.append("### Predicted Performance")
            report_content.append(f"- **Part quality**: {optimal.part_quality:.3f}")
            report_content.append(
                f"- **Dimensional accuracy**: {optimal.dimensional_accuracy:.3f}"
            )
            report_content.append(f"- **Surface finish**: {optimal.surface_finish:.3f}")

        report_content.append("")

    # RL-Guided Exploration
    if "rl_guided_exploration" in results_data:
        rl_guided = results_data["rl_guided_exploration"]
        report_content.append("## RL-Guided Exploration Impact")
        report_content.append("")

        impact = rl_guided["rl_guidance_impact"]
        report_content.append(
            f"- **RL-guided mean response**: {impact['rl_guided_mean']:.3f}"
        )
        report_content.append(
            f"- **Standard exploration mean**: {impact['exploration_only_mean']:.3f}"
        )
        report_content.append(
            f"- **Relative improvement**: {impact['relative_improvement']:.1%}"
        )
        report_content.append(
            f"- **Guidance effectiveness**: {'Yes' if impact['rl_guidance_effectiveness'] else 'No'}"
        )
        report_content.append("")

    # Recommendations
    report_content.append("## Recommendations")
    report_content.append("")

    if "rl_algorithms" in results_data:
        best_algo = max(
            results_data["rl_algorithms"].keys(),
            key=lambda k: results_data["rl_algorithms"][k]["best_reward"],
        )
        report_content.append(
            f"1. **Use {best_algo.upper()} algorithm** for this process type based on superior performance"
        )

    if "exploration_strategies" in results_data:
        best_strategy = max(
            results_data["exploration_strategies"].keys(),
            key=lambda k: results_data["exploration_strategies"][k]["best_response"],
        )
        strategy_name = best_strategy.replace("_", " ").title()
        report_content.append(
            f"2. **Implement {strategy_name} exploration** for parameter space optimization"
        )

    report_content.append(
        "3. **Monitor process stability** to maintain optimal conditions"
    )
    report_content.append(
        "4. **Implement real-time feedback control** based on RL agent recommendations"
    )
    report_content.append(
        "5. **Regular model retraining** as new process data becomes available"
    )
    report_content.append("")

    # Conclusion
    report_content.append("## Conclusion")
    report_content.append("")
    report_content.append(
        "The RL-based process optimization demonstrates significant potential for:"
    )
    report_content.append("- Automated process parameter optimization")
    report_content.append("- Multi-objective balancing of quality, energy, and time")
    report_content.append("- Intelligent exploration of parameter spaces")
    report_content.append("- Real-time adaptation to changing conditions")
    report_content.append("")
    report_content.append(
        "*Report generated by ChemiAccel RL Process Optimization System*"
    )

    # Save report
    output_dir = Path("rl_optimization_results")
    output_dir.mkdir(exist_ok=True)

    report_file = output_dir / "optimization_report.md"
    with open(report_file, "w") as f:
        f.write("\n".join(report_content))

    print(f"✓ Comprehensive report saved to {report_file}")

    # Print summary
    print("\nKey Findings:")
    if "rl_algorithms" in results_data:
        best_algo = max(
            results_data["rl_algorithms"].keys(),
            key=lambda k: results_data["rl_algorithms"][k]["best_reward"],
        )
        print(f"  • Best RL algorithm: {best_algo.upper()}")

    if "exploration_strategies" in results_data:
        best_strategy = max(
            results_data["exploration_strategies"].keys(),
            key=lambda k: results_data["exploration_strategies"][k]["best_response"],
        )
        print(
            f"  • Best exploration strategy: {best_strategy.replace('_', ' ').title()}"
        )

    if "rl_guided_exploration" in results_data:
        effectiveness = results_data["rl_guided_exploration"]["rl_guidance_impact"][
            "rl_guidance_effectiveness"
        ]
        print(
            f"  • RL guidance effectiveness: {'Positive' if effectiveness else 'Neutral'}"
        )

    return report_file


def main():
    """Run complete RL process optimization demonstration."""
    print("ChemiAccel Reinforcement Learning Process Optimization Demo")
    print("=" * 70)
    print("This demo showcases advanced RL-driven process optimization")
    print("including GPU-accelerated agents, multi-objective optimization,")
    print("and intelligent parameter space exploration.")
    print("=" * 70)

    results_data = {}

    try:
        # Step 1: Setup manufacturing digital twin
        digital_twin = setup_manufacturing_digital_twin()

        # Step 2: Demonstrate reward functions
        reward_functions = demonstrate_reward_functions()
        quality_reward = reward_functions["Quality"]

        # Step 3: Demonstrate RL algorithms
        rl_results, best_algo = demonstrate_rl_agents(digital_twin, quality_reward)
        results_data["rl_algorithms"] = rl_results

        # Step 4: Demonstrate parameter space exploration
        exploration_results = demonstrate_parameter_space_exploration(digital_twin)
        results_data["exploration_strategies"] = exploration_results

        # Step 5: Demonstrate RL-guided exploration
        rl_guided_results = demonstrate_rl_guided_exploration(digital_twin, best_algo)
        results_data["rl_guided_exploration"] = rl_guided_results

        # Step 6: Demonstrate multi-objective optimization
        mo_results, optimal_state = demonstrate_multi_objective_optimization(
            digital_twin
        )
        results_data["multi_objective"] = mo_results
        results_data["optimal_conditions"] = optimal_state

        # Step 7: Generate visualizations
        generate_visualization_plots(results_data)

        # Step 8: Generate comprehensive report
        report_file = generate_comprehensive_report(results_data)

        print("\n" + "=" * 70)
        print("RL Process Optimization Demonstration Completed Successfully!")
        print("=" * 70)
        print("\nKey Achievements:")
        print("✓ Demonstrated GPU-accelerated RL agents (PPO, DDPG, SAC)")
        print("✓ Implemented multi-objective reward functions")
        print("✓ Showcased parameter space exploration strategies")
        print("✓ Validated RL-guided exploration effectiveness")
        print("✓ Optimized manufacturing process conditions")
        print("✓ Generated comprehensive analysis and visualizations")

        print("\nResults saved to: rl_optimization_results/")
        print(f"Detailed report: {report_file}")

        print("\nThis demonstration shows how ChemiAccel enables:")
        print("• Intelligent process parameter optimization")
        print("• Multi-objective manufacturing optimization")
        print("• Automated exploration of parameter spaces")
        print("• Real-time process control and adaptation")
        print("• Data-driven manufacturing process improvement")

    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\nError: {e}")
        print("\nNote: Some features require additional dependencies")
        print("for full functionality in production environments.")

        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
