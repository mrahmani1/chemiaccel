"""
Comprehensive Demonstration of GPU-Accelerated Bayesian Optimization Framework

This script demonstrates the full capabilities of ChemiAccel's Bayesian optimization
framework through various chemical and materials science optimization scenarios.

Key demonstrations:
1. Molecular property optimization
2. Chemical process optimization
3. Multi-objective drug discovery
4. High-dimensional polymer property optimization
5. GPU acceleration benchmarks
6. Real-time optimization monitoring
"""

import asyncio
import time
import warnings
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import ChemiAccel Bayesian optimization components
from src.chemi_accel.ai.bayesian_optimization import (
    CUPY_AVAILABLE,
    JAX_AVAILABLE,
    TORCH_AVAILABLE,
    AcquisitionFunction,
    BayesianOptimizationConfig,
    BayesianOptimizer,
    OptimizationResult,
    OptimizationStrategy,
    ParameterSpace,
    SurrogateModelType,
    branin_function,
    create_example_parameter_spaces,
    multi_objective_drug_discovery_optimization,
    optimize_molecular_properties,
    optimize_process_conditions,
    synthetic_chemical_objective,
)
from src.chemi_accel.ai.multi_objective_optimization import ObjectiveConfig

# Try to import optional plotting libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Some visualizations will be skipped.")


class BayesianOptimizationDemo:
    """Comprehensive demonstration of Bayesian optimization capabilities."""

    def __init__(self, use_gpu: bool = True, verbose: bool = True):
        """Initialize the demonstration."""
        self.use_gpu = use_gpu and (TORCH_AVAILABLE or JAX_AVAILABLE)
        self.verbose = verbose
        self.results = {}

        if self.verbose:
            print("=" * 80)
            print("ChemiAccel GPU-Accelerated Bayesian Optimization Framework Demo")
            print("=" * 80)
            print(f"GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
            print(
                f"Available backends: PyTorch={TORCH_AVAILABLE}, JAX={JAX_AVAILABLE}, CuPy={CUPY_AVAILABLE}"
            )
            print()

    async def run_all_demos(self):
        """Run all demonstration scenarios."""
        demos = [
            ("Molecular Property Optimization", self.demo_molecular_optimization),
            ("Chemical Process Optimization", self.demo_process_optimization),
            (
                "Multi-Objective Drug Discovery",
                self.demo_multi_objective_drug_discovery,
            ),
            (
                "High-Dimensional Polymer Optimization",
                self.demo_high_dimensional_optimization,
            ),
            ("GPU Performance Benchmarks", self.demo_gpu_benchmarks),
            ("Acquisition Function Comparison", self.demo_acquisition_functions),
            ("Real-time Optimization Monitoring", self.demo_real_time_monitoring),
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
                    if result and hasattr(result, "get_summary"):
                        summary = result.get_summary()
                        print(
                            f"  Best objective: {summary.get('best_objective', 'N/A'):.4f}"
                        )
                        print(f"  Total time: {summary.get('total_time', 0):.2f}s")
                        print(f"  Iterations: {summary.get('n_iterations', 0)}")

            except Exception as e:
                if self.verbose:
                    print(f"✗ {demo_name} failed: {e}")
                self.results[demo_name] = None

        # Generate summary report
        self.generate_summary_report()

    async def demo_molecular_optimization(self) -> OptimizationResult:
        """Demonstrate molecular property optimization."""

        if self.verbose:
            print("Optimizing molecular descriptors for drug-likeness...")

        # Define parameter space for molecular descriptors
        parameter_spaces = [
            ParameterSpace("molecular_weight", (150, 500), "continuous"),
            ParameterSpace("logp", (-2, 5), "continuous"),
            ParameterSpace("tpsa", (20, 140), "continuous"),
            ParameterSpace("hbd", (0, 5), "discrete"),
            ParameterSpace("hba", (0, 10), "discrete"),
        ]

        # Drug-likeness scoring function (simplified)
        def drug_likeness_score(x):
            mw, logp, tpsa, hbd, hba = x

            # Lipinski's Rule of Five scoring
            lipinski_score = 0.0

            # Molecular weight: 150-500 is optimal
            mw_score = 1.0 - abs(mw - 325) / 175
            lipinski_score += max(0, mw_score)

            # LogP: 1-3 is optimal
            logp_score = 1.0 - abs(logp - 2.0) / 3.0
            lipinski_score += max(0, logp_score)

            # TPSA: 60-80 is optimal
            tpsa_score = 1.0 - abs(tpsa - 70) / 50
            lipinski_score += max(0, tpsa_score)

            # HBD: 0-2 is optimal
            hbd_score = 1.0 - abs(hbd - 1) / 2
            lipinski_score += max(0, hbd_score)

            # HBA: 2-6 is optimal
            hba_score = 1.0 - abs(hba - 4) / 4
            lipinski_score += max(0, hba_score)

            # Add some noise to make it more realistic
            noise = 0.1 * np.random.randn()

            return lipinski_score + noise

        config = BayesianOptimizationConfig(
            n_initial_points=10,
            max_iterations=15,
            acquisition_function=AcquisitionFunction.EXPECTED_IMPROVEMENT,
            use_gpu=self.use_gpu,
            device="auto",
            model_training_iterations=100 if self.use_gpu else 50,
        )

        result = optimize_molecular_properties(
            parameter_spaces, drug_likeness_score, config
        )

        if self.verbose:
            best_params = result.best_parameters
            print(f"  Best drug-likeness score: {result.best_objective:.4f}")
            print(f"  Optimal molecular weight: {best_params['molecular_weight']:.1f}")
            print(f"  Optimal LogP: {best_params['logp']:.2f}")
            print(f"  Optimal TPSA: {best_params['tpsa']:.1f}")

        return result

    async def demo_process_optimization(self) -> OptimizationResult:
        """Demonstrate chemical process optimization."""

        if self.verbose:
            print("Optimizing chemical reaction conditions...")

        # Process optimization function (yield maximization)
        def reaction_yield(x):
            temperature, pressure, catalyst_loading = x

            # Simplified Arrhenius-type kinetics with optimal conditions
            optimal_temp = 350  # Kelvin
            optimal_pressure = 3e5  # Pa
            optimal_catalyst = 1.5  # mol%

            # Temperature effect (Arrhenius)
            temp_effect = np.exp(-5000 / (8.314 * temperature)) * np.exp(
                -((temperature - optimal_temp) ** 2) / 2000
            )

            # Pressure effect (favorable up to optimal point)
            pressure_effect = 1.0 - abs(pressure - optimal_pressure) ** 2 / (2e5) ** 2
            pressure_effect = max(0, pressure_effect)

            # Catalyst effect (diminishing returns)
            catalyst_effect = catalyst_loading / (catalyst_loading + 0.5)

            # Combined yield with interaction terms
            yield_value = temp_effect * pressure_effect * catalyst_effect

            # Add realistic noise
            noise = 0.05 * np.random.randn()

            return yield_value + noise

        config = BayesianOptimizationConfig(
            n_initial_points=12,
            max_iterations=20,
            acquisition_function=AcquisitionFunction.UPPER_CONFIDENCE_BOUND,
            exploration_weight=2.5,
            use_gpu=self.use_gpu,
            device="auto",
        )

        result = optimize_process_conditions(
            temperature_range=(300, 400),
            pressure_range=(1e5, 5e5),
            flow_rate_range=(0.5, 3.0),  # Catalyst loading
            objective_function=reaction_yield,
            config=config,
        )

        if self.verbose:
            best_params = result.best_parameters
            print(f"  Best reaction yield: {result.best_objective:.4f}")
            print(f"  Optimal temperature: {best_params['temperature']:.1f} K")
            print(f"  Optimal pressure: {best_params['pressure']:.0f} Pa")
            print(f"  Optimal catalyst loading: {best_params['flow_rate']:.2f} mol%")

        return result

    async def demo_multi_objective_drug_discovery(self) -> OptimizationResult:
        """Demonstrate multi-objective drug discovery optimization."""

        if self.verbose:
            print("Multi-objective optimization for drug discovery...")

        parameter_spaces = [
            ParameterSpace("molecular_weight", (200, 600), "continuous"),
            ParameterSpace("logp", (-1, 4), "continuous"),
            ParameterSpace("tpsa", (40, 120), "continuous"),
            ParameterSpace("rotatable_bonds", (0, 15), "discrete"),
            ParameterSpace("aromatic_rings", (1, 4), "discrete"),
        ]

        # Multi-objective function: potency, selectivity, ADMET
        async def drug_discovery_objectives(x):
            mw, logp, tpsa, rb, ar = x

            # Potency (higher is better)
            potency = (
                0.8 - 0.001 * abs(mw - 400) ** 2 - 0.1 * abs(logp - 2.5) ** 2 + 0.1 * ar
            )
            potency += 0.05 * np.random.randn()

            # Selectivity (higher is better)
            selectivity = (
                0.9 - 0.02 * rb - 0.0005 * tpsa + 0.1 * np.exp(-abs(logp - 2.0))
            )
            selectivity += 0.05 * np.random.randn()

            # ADMET score (higher is better)
            admet = (
                0.85
                - 0.0001 * (mw - 350) ** 2
                - 0.002 * tpsa
                - 0.05 * max(0, rb - 7) ** 2
            )
            admet += 0.05 * np.random.randn()

            # Synthetic accessibility (lower is better -> negate for maximization)
            syn_acc = -(2.0 + 0.5 * ar + 0.1 * rb + 0.001 * mw)
            syn_acc += 0.1 * np.random.randn()

            return np.array([potency, selectivity, admet, syn_acc])

        objectives = [
            ObjectiveConfig("potency", minimize=False, weight=1.5),
            ObjectiveConfig("selectivity", minimize=False, weight=1.0),
            ObjectiveConfig("admet", minimize=False, weight=1.2),
            ObjectiveConfig("synthetic_accessibility", minimize=True, weight=0.8),
        ]

        config = BayesianOptimizationConfig(
            n_initial_points=15,
            max_iterations=25,
            acquisition_function=AcquisitionFunction.EXPECTED_IMPROVEMENT,
            use_gpu=self.use_gpu,
            device="auto",
            scalarization_weights=np.array([0.3, 0.25, 0.3, 0.15]),
        )

        optimizer = BayesianOptimizer(parameter_spaces, config)
        result = await optimizer.optimize(drug_discovery_objectives, objectives)

        if self.verbose:
            print(
                f"  Pareto front size: {len(result.pareto_front.pareto_front.pareto_indices)}"
            )
            print(f"  Hypervolume: {result.pareto_front.pareto_front.hypervolume:.4f}")

            # Show best compromise solution
            if len(result.pareto_front.pareto_front.pareto_indices) > 0:
                best_idx = result.pareto_front.pareto_front.pareto_indices[0]
                best_objectives = result.pareto_front.objective_values[best_idx]
                print(
                    f"  Best compromise - Potency: {best_objectives[0]:.3f}, "
                    f"Selectivity: {best_objectives[1]:.3f}, "
                    f"ADMET: {best_objectives[2]:.3f}"
                )

        return result

    async def demo_high_dimensional_optimization(self) -> OptimizationResult:
        """Demonstrate high-dimensional polymer property optimization."""

        if self.verbose:
            print("High-dimensional polymer property optimization (20D)...")

        # Create 20-dimensional parameter space for polymer properties
        parameter_spaces = []
        polymer_properties = [
            "monomer_1_fraction",
            "monomer_2_fraction",
            "monomer_3_fraction",
            "molecular_weight",
            "polydispersity",
            "crystallinity",
            "glass_transition_temp",
            "melting_temp",
            "density",
            "tensile_strength",
            "elongation_at_break",
            "youngs_modulus",
            "thermal_conductivity",
            "electrical_conductivity",
            "dielectric_constant",
            "processing_temp",
            "processing_pressure",
            "cooling_rate",
            "additive_1_loading",
            "additive_2_loading",
        ]

        bounds = [
            (0.2, 0.8),
            (0.1, 0.6),
            (0.0, 0.3),  # Monomer fractions
            (10000, 100000),
            (1.1, 3.0),
            (0.1, 0.9),  # MW, PDI, crystallinity
            (200, 400),
            (400, 600),
            (0.8, 1.4),  # Thermal properties
            (20, 100),
            (100, 1000),
            (1, 10),  # Mechanical properties
            (0.1, 2.0),
            (1e-12, 1e-6),
            (2, 8),  # Electrical properties
            (150, 250),
            (1e5, 1e7),
            (1, 100),  # Processing conditions
            (0, 5),
            (0, 2),  # Additive loadings
        ]

        for prop, bound in zip(polymer_properties, bounds):
            parameter_spaces.append(ParameterSpace(prop, bound, "continuous"))

        # Complex polymer performance function
        def polymer_performance(x):
            # Normalize inputs
            x_norm = np.array(x)

            # Performance index combining multiple factors
            performance = 0.0

            # Monomer composition effect
            mono_sum = x_norm[0] + x_norm[1] + x_norm[2]
            if abs(mono_sum - 1.0) > 0.1:  # Composition constraint
                performance -= 10 * abs(mono_sum - 1.0)

            # Molecular weight effects
            mw_score = 1.0 - abs(x_norm[3] - 50000) ** 2 / 30000**2
            performance += mw_score

            # Mechanical properties balance
            strength_modulus_ratio = x_norm[9] / max(x_norm[11], 0.1)
            performance += 0.5 * np.tanh(strength_modulus_ratio - 5)

            # Thermal stability
            thermal_score = (x_norm[6] + x_norm[7]) / 500 - 1.0
            performance += 0.3 * thermal_score

            # Processing window
            processing_score = 1.0 - abs(x_norm[15] - 200) ** 2 / 50**2
            performance += 0.2 * processing_score

            # Add realistic noise
            noise = 0.1 * np.random.randn()

            return performance + noise

        config = BayesianOptimizationConfig(
            n_initial_points=30,  # More initial points for high-D
            max_iterations=20,
            acquisition_function=AcquisitionFunction.UPPER_CONFIDENCE_BOUND,
            exploration_weight=1.5,
            use_gpu=self.use_gpu,
            device="auto",
            model_training_iterations=150,
        )

        optimizer = BayesianOptimizer(parameter_spaces, config)
        result = await optimizer.optimize(polymer_performance)

        if self.verbose:
            print(f"  Best polymer performance: {result.best_objective:.4f}")
            print(f"  Optimization in 20D space completed in {result.total_time:.1f}s")

            # Show key optimized parameters
            best_params = result.best_parameters
            print(f"  Optimal MW: {best_params['molecular_weight']:.0f}")
            print(f"  Optimal Tg: {best_params['glass_transition_temp']:.1f} K")
            print(f"  Optimal strength: {best_params['tensile_strength']:.1f} MPa")

        return result

    async def demo_gpu_benchmarks(self) -> Dict[str, Any]:
        """Benchmark GPU vs CPU performance."""

        if self.verbose:
            print("Benchmarking GPU vs CPU performance...")

        benchmarks = {}

        # Simple 2D optimization problem for benchmarking
        parameter_spaces = [
            ParameterSpace("x1", (-5, 5), "continuous"),
            ParameterSpace("x2", (-5, 5), "continuous"),
        ]

        def benchmark_objective(x):
            return branin_function(x)

        # CPU benchmark
        if self.verbose:
            print("  Running CPU benchmark...")

        config_cpu = BayesianOptimizationConfig(
            n_initial_points=10,
            max_iterations=10,
            use_gpu=False,
            device="cpu",
            model_training_iterations=100,
        )

        start_time = time.time()
        optimizer_cpu = BayesianOptimizer(parameter_spaces, config_cpu)
        result_cpu = await optimizer_cpu.optimize(benchmark_objective)
        cpu_time = time.time() - start_time

        benchmarks["cpu_time"] = cpu_time
        benchmarks["cpu_result"] = result_cpu

        # GPU benchmark (if available)
        if self.use_gpu and TORCH_AVAILABLE:
            if self.verbose:
                print("  Running GPU benchmark...")

            config_gpu = BayesianOptimizationConfig(
                n_initial_points=10,
                max_iterations=10,
                use_gpu=True,
                device="auto",
                model_training_iterations=100,
            )

            start_time = time.time()
            optimizer_gpu = BayesianOptimizer(parameter_spaces, config_gpu)
            result_gpu = await optimizer_gpu.optimize(benchmark_objective)
            gpu_time = time.time() - start_time

            benchmarks["gpu_time"] = gpu_time
            benchmarks["gpu_result"] = result_gpu
            benchmarks["speedup"] = cpu_time / gpu_time

            if self.verbose:
                print(f"  CPU time: {cpu_time:.2f}s")
                print(f"  GPU time: {gpu_time:.2f}s")
                print(f"  Speedup: {cpu_time/gpu_time:.1f}x")
        else:
            if self.verbose:
                print("  GPU benchmark skipped (not available)")

        return benchmarks

    async def demo_acquisition_functions(self) -> Dict[str, OptimizationResult]:
        """Compare different acquisition functions."""

        if self.verbose:
            print("Comparing acquisition function performance...")

        parameter_spaces = [
            ParameterSpace("x1", (-5, 5), "continuous"),
            ParameterSpace("x2", (-5, 5), "continuous"),
        ]

        def test_objective(x):
            return branin_function(x)

        acquisition_functions = [
            AcquisitionFunction.EXPECTED_IMPROVEMENT,
            AcquisitionFunction.UPPER_CONFIDENCE_BOUND,
            AcquisitionFunction.PROBABILITY_OF_IMPROVEMENT,
        ]

        results = {}

        for acq_func in acquisition_functions:
            if self.verbose:
                print(f"  Testing {acq_func.value}...")

            config = BayesianOptimizationConfig(
                n_initial_points=8,
                max_iterations=12,
                acquisition_function=acq_func,
                use_gpu=self.use_gpu,
                device="auto",
                exploration_weight=2.0
                if acq_func == AcquisitionFunction.UPPER_CONFIDENCE_BOUND
                else 0.01,
            )

            optimizer = BayesianOptimizer(parameter_spaces, config)
            result = await optimizer.optimize(test_objective)
            results[acq_func.value] = result

            if self.verbose:
                print(f"    Best objective: {result.best_objective:.4f}")
                print(f"    Total time: {result.total_time:.2f}s")

        return results

    async def demo_real_time_monitoring(self) -> OptimizationResult:
        """Demonstrate real-time optimization monitoring."""

        if self.verbose:
            print("Real-time optimization monitoring demo...")

        parameter_spaces = [
            ParameterSpace("reaction_temp", (300, 400), "continuous"),
            ParameterSpace("catalyst_conc", (0.1, 2.0), "continuous"),
            ParameterSpace("residence_time", (5, 60), "continuous"),
        ]

        # Simulate a realistic chemical process with time-varying optimum
        async def dynamic_objective(x):
            temp, catalyst, time_res = x

            # Base performance model
            base_performance = np.exp(
                -(
                    (temp - 350) ** 2 / 1000
                    + (catalyst - 1.0) ** 2 / 0.5
                    + (time_res - 30) ** 2 / 400
                )
            )

            # Add time-dependent drift (simulate changing conditions)
            time_factor = 1.0 + 0.1 * np.sin(0.1 * time.time())

            # Simulate measurement delay
            await asyncio.sleep(0.1)

            performance = base_performance * time_factor + 0.05 * np.random.randn()

            return performance

        config = BayesianOptimizationConfig(
            n_initial_points=12,
            max_iterations=15,
            acquisition_function=AcquisitionFunction.EXPECTED_IMPROVEMENT,
            use_gpu=self.use_gpu,
            device="auto",
        )

        optimizer = BayesianOptimizer(parameter_spaces, config)

        # Add monitoring callbacks
        monitoring_data = {"iterations": [], "best_objectives": [], "times": []}

        def monitor_progress(iteration, best_obj, current_time):
            monitoring_data["iterations"].append(iteration)
            monitoring_data["best_objectives"].append(best_obj)
            monitoring_data["times"].append(current_time)

            if self.verbose and iteration % 5 == 0:
                print(f"    Iteration {iteration}: Best = {best_obj:.4f}")

        # Run optimization with monitoring
        start_time = time.time()
        result = await optimizer.optimize(dynamic_objective)

        # Simulate monitoring data
        for i, obj in enumerate(result.objective_history):
            monitor_progress(
                i, max(result.objective_history[: i + 1]), time.time() - start_time
            )

        if self.verbose:
            print("  Real-time monitoring completed")
            print(f"  Final best objective: {result.best_objective:.4f}")
            print(f"  Total monitoring points: {len(monitoring_data['iterations'])}")

        # Store monitoring data in result
        result.monitoring_data = monitoring_data

        return result

    def generate_summary_report(self):
        """Generate comprehensive summary report."""

        if self.verbose:
            print("\n" + "=" * 80)
            print("BAYESIAN OPTIMIZATION DEMO SUMMARY REPORT")
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

        if (
            "GPU Performance Benchmarks" in self.results
            and self.results["GPU Performance Benchmarks"]
        ):
            benchmarks = self.results["GPU Performance Benchmarks"]
            if "speedup" in benchmarks:
                print(f"GPU speedup: {benchmarks['speedup']:.1f}x")

        # Best results summary
        print("\nOptimization Results:")
        for demo_name, result in self.results.items():
            if result and hasattr(result, "best_objective"):
                print(f"  {demo_name}: {result.best_objective:.4f}")

        print("\n" + "=" * 80)
        print("Demo completed successfully!")
        print("=" * 80)

    def plot_optimization_progress(
        self, result: OptimizationResult, title: str = "Optimization Progress"
    ):
        """Plot optimization progress."""

        if not PLOTLY_AVAILABLE:
            # Use matplotlib as fallback
            plt.figure(figsize=(10, 6))
            plt.plot(result.objective_history, "b-o", markersize=4)
            plt.xlabel("Iteration")
            plt.ylabel("Objective Value")
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.show()
            return

        # Use Plotly for interactive plots
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=list(range(len(result.objective_history))),
                y=result.objective_history,
                mode="lines+markers",
                name="Objective Value",
                line=dict(color="blue", width=2),
                marker=dict(size=6),
            )
        )

        # Add best objective line
        best_so_far = []
        current_best = -np.inf
        for obj in result.objective_history:
            if obj > current_best:
                current_best = obj
            best_so_far.append(current_best)

        fig.add_trace(
            go.Scatter(
                x=list(range(len(best_so_far))),
                y=best_so_far,
                mode="lines",
                name="Best So Far",
                line=dict(color="red", width=2, dash="dash"),
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Iteration",
            yaxis_title="Objective Value",
            hovermode="x unified",
            showlegend=True,
        )

        fig.show()


async def main():
    """Run the comprehensive Bayesian optimization demonstration."""

    print("Starting ChemiAccel Bayesian Optimization Demo...")
    print("This may take several minutes to complete all demonstrations.\n")

    # Create demo instance
    demo = BayesianOptimizationDemo(
        use_gpu=True, verbose=True  # Enable GPU if available
    )

    # Run all demonstrations
    await demo.run_all_demos()

    # Optional: Generate visualizations for select results
    if PLOTLY_AVAILABLE or True:  # Use matplotlib as fallback
        print("\nGenerating optimization progress plots...")

        # Plot molecular optimization progress
        if "Molecular Property Optimization" in demo.results:
            result = demo.results["Molecular Property Optimization"]
            if result:
                demo.plot_optimization_progress(
                    result, "Molecular Property Optimization"
                )

        # Plot process optimization progress
        if "Chemical Process Optimization" in demo.results:
            result = demo.results["Chemical Process Optimization"]
            if result:
                demo.plot_optimization_progress(result, "Chemical Process Optimization")

    print("\nDemo completed! Check the results above for performance metrics.")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
