"""
Multi-Objective Optimization Framework Demo.

This example demonstrates the comprehensive multi-objective optimization
capabilities of ChemiAccel, including Pareto front identification,
interactive visualization, and scalarization for generative model guidance.
"""

import os
import sys

import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chemi_accel.ai.multi_objective_optimization import (
    MPOptimizer,
    ObjectiveConfig,
    create_drug_discovery_objectives,
    create_polymer_objectives,
    run_multi_objective_optimization,
)


def generate_demo_molecules(n_molecules: int = 1000) -> tuple:
    """Generate synthetic molecular data for demonstration."""
    np.random.seed(42)

    # Generate molecule identifiers
    molecules = [f"DEMO_MOL_{i:04d}" for i in range(n_molecules)]

    # Generate correlated molecular properties
    # Molecular weight (150-600 Da)
    mw = np.random.lognormal(mean=np.log(300), sigma=0.3, size=n_molecules)
    mw = np.clip(mw, 150, 600)

    # LogP (correlated with MW)
    logp = 0.01 * mw + np.random.normal(0, 0.8, n_molecules)
    logp = np.clip(logp, -2, 6)

    # TPSA (inversely correlated with LogP)
    tpsa = 120 - 15 * logp + np.random.normal(0, 20, n_molecules)
    tpsa = np.clip(tpsa, 20, 200)

    # Synthetic accessibility (correlated with complexity)
    complexity = (mw - 150) / 450 + (np.abs(logp - 2.5) / 4)
    sa_score = 1 + 4 * complexity + np.random.normal(0, 0.5, n_molecules)
    sa_score = np.clip(sa_score, 1, 6)

    # Additional properties for polymer demo
    glass_transition_temp = 50 + 0.2 * mw + np.random.normal(0, 20, n_molecules)
    youngs_modulus = 1000 + 5 * mw + np.random.normal(0, 500, n_molecules)

    properties = {
        "molecular_weight": mw,
        "logp": logp,
        "tpsa": tpsa,
        "synthetic_accessibility": sa_score,
        "glass_transition_temp": glass_transition_temp,
        "youngs_modulus": youngs_modulus,
    }

    return molecules, properties


def demo_basic_multi_objective_optimization():
    """Demonstrate basic multi-objective optimization."""
    print("=" * 60)
    print("Basic Multi-Objective Optimization Demo")
    print("=" * 60)

    # Generate demo data
    molecules, properties = generate_demo_molecules(1000)

    # Define objectives for drug discovery
    objectives = [
        ObjectiveConfig(
            name="molecular_weight",
            minimize=True,
            weight=1.0,
            target=350.0,
            normalization="minmax",
        ),
        ObjectiveConfig(
            name="logp", minimize=True, weight=1.0, target=2.5, normalization="minmax"
        ),
        ObjectiveConfig(
            name="tpsa", minimize=True, weight=1.0, target=70.0, normalization="minmax"
        ),
    ]

    # Create optimizer
    optimizer = MPOptimizer(objectives=objectives, use_gpu=True)

    # Run optimization
    print("Running multi-objective optimization...")
    result = optimizer.optimize(molecules, properties)

    # Display results
    print("\nResults:")
    print(f"  Total molecules evaluated: {len(result.molecules)}")
    print(f"  Pareto optimal solutions: {len(result.pareto_front.pareto_indices)}")
    print(f"  Hypervolume: {result.pareto_front.hypervolume:.4f}")
    print(f"  Computation time: {result.convergence_metrics['computation_time']:.2f}s")

    # Show diversity metrics
    print("\nDiversity metrics:")
    for metric, value in result.diversity_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Display top Pareto solutions
    print("\nTop 5 Pareto optimal solutions:")
    pareto_molecules = result.pareto_front.pareto_molecules[:5]
    pareto_objectives = result.pareto_front.pareto_objectives[:5]

    for i, (mol, obj_vals) in enumerate(zip(pareto_molecules, pareto_objectives)):
        print(
            f"  {i+1}. {mol}: MW={obj_vals[0]:.2f}, LogP={obj_vals[1]:.2f}, TPSA={obj_vals[2]:.2f}"
        )

    return result


def demo_scalarization_functions():
    """Demonstrate scalarization functions for generative model guidance."""
    print("\n" + "=" * 60)
    print("Scalarization Functions Demo")
    print("=" * 60)

    # Generate demo data
    molecules, properties = generate_demo_molecules(500)

    # Define objectives
    objectives = [
        ObjectiveConfig(name="molecular_weight", minimize=True, normalization="minmax"),
        ObjectiveConfig(name="logp", minimize=True, normalization="minmax"),
        ObjectiveConfig(
            name="synthetic_accessibility", minimize=True, normalization="minmax"
        ),
    ]

    # Create optimizer
    optimizer = MPOptimizer(objectives=objectives)
    result = optimizer.optimize(molecules, properties)

    # Test different scalarization methods
    scalarization_methods = ["weighted_sum", "tchebycheff", "asf", "pbi"]

    print("Scalarization scores for top 5 molecules:")
    print("Molecule ID".ljust(15), end="")
    for method in scalarization_methods:
        print(method.ljust(12), end="")
    print()
    print("-" * 60)

    for i in range(5):
        mol_id = result.molecules[i]
        print(mol_id.ljust(15), end="")

        for method in scalarization_methods:
            scores = optimizer.get_scalarization_guidance(
                result.objective_values, method=method
            )
            print(f"{scores[i]:10.4f}".ljust(12), end="")
        print()

    # Demonstrate guidance for generative models
    print("\nGenerative model guidance:")
    print("Using weighted sum scalarization to rank molecules...")

    weighted_scores = optimizer.get_scalarization_guidance(
        result.objective_values, method="weighted_sum"
    )

    # Sort by scalarized score (lower is better for minimization)
    sorted_indices = np.argsort(weighted_scores)

    print("Top 5 molecules for generative model training:")
    for i in range(5):
        idx = sorted_indices[i]
        mol_id = result.molecules[idx]
        score = weighted_scores[idx]
        print(f"  {i+1}. {mol_id}: Score = {score:.4f}")


def demo_polymer_optimization():
    """Demonstrate polymer-specific multi-objective optimization."""
    print("\n" + "=" * 60)
    print("Polymer Multi-Objective Optimization Demo")
    print("=" * 60)

    # Generate demo data
    molecules, properties = generate_demo_molecules(800)

    # Use polymer-specific objectives
    objectives = create_polymer_objectives()

    # Run optimization
    result = run_multi_objective_optimization(
        molecules, properties, objectives, use_gpu=True
    )

    print("Polymer optimization results:")
    print(f"  Pareto optimal polymers: {len(result.pareto_front.pareto_indices)}")
    print(
        f"  Average Tg of Pareto solutions: {np.mean(result.pareto_front.pareto_objectives[:, 0]):.1f}°C"
    )
    print(
        f"  Average modulus of Pareto solutions: {np.mean(result.pareto_front.pareto_objectives[:, 1]):.0f} MPa"
    )

    # Show trade-offs
    print("\nProperty trade-offs in Pareto front:")
    pareto_obj = result.pareto_front.pareto_objectives

    print(
        f"  Tg range: {np.min(pareto_obj[:, 0]):.1f} - {np.max(pareto_obj[:, 0]):.1f}°C"
    )
    print(
        f"  Modulus range: {np.min(pareto_obj[:, 1]):.0f} - {np.max(pareto_obj[:, 1]):.0f} MPa"
    )
    print(
        f"  SA score range: {np.min(pareto_obj[:, 2]):.2f} - {np.max(pareto_obj[:, 2]):.2f}"
    )


def demo_large_scale_optimization():
    """Demonstrate large-scale multi-objective optimization."""
    print("\n" + "=" * 60)
    print("Large-Scale Multi-Objective Optimization Demo")
    print("=" * 60)

    # Generate large dataset
    print("Generating large dataset (10,000 molecules)...")
    molecules, properties = generate_demo_molecules(10000)

    # Define comprehensive objectives
    objectives = create_drug_discovery_objectives()

    # Run optimization with population size limit
    print("Running large-scale optimization with GPU acceleration...")

    optimizer = MPOptimizer(objectives=objectives, use_gpu=True)
    result = optimizer.optimize(molecules, properties, population_size=5000)

    print("\nLarge-scale optimization results:")
    print(f"  Dataset size: {len(molecules):,} molecules")
    print(f"  Population size: {len(result.molecules):,} molecules")
    print(f"  Pareto solutions: {len(result.pareto_front.pareto_indices):,}")
    print(
        f"  Pareto fraction: {len(result.pareto_front.pareto_indices)/len(result.molecules)*100:.2f}%"
    )
    print(f"  Computation time: {result.convergence_metrics['computation_time']:.2f}s")
    print(
        f"  Throughput: {len(result.molecules)/result.convergence_metrics['computation_time']:.0f} molecules/s"
    )

    # Performance analysis
    time_per_molecule = (
        result.convergence_metrics["computation_time"] / len(result.molecules) * 1000
    )
    print(f"  Time per molecule: {time_per_molecule:.3f} ms")


def demo_visualization_capabilities():
    """Demonstrate visualization capabilities."""
    print("\n" + "=" * 60)
    print("Visualization Capabilities Demo")
    print("=" * 60)

    # Generate demo data
    molecules, properties = generate_demo_molecules(500)

    # 2D optimization for visualization
    objectives_2d = [
        ObjectiveConfig(name="molecular_weight", minimize=True, normalization="minmax"),
        ObjectiveConfig(name="logp", minimize=True, normalization="minmax"),
    ]

    optimizer_2d = MPOptimizer(objectives=objectives_2d)
    result_2d = optimizer_2d.optimize(molecules, properties)

    # Create visualizations
    print("Creating interactive visualizations...")
    visualizations = optimizer_2d.visualize_tradeoffs(result_2d)

    if "pareto_front_2d" in visualizations:
        print("  ✓ 2D Pareto front visualization created")
        # In a real application, you would show or save the plot
        # visualizations['pareto_front_2d'].show()

    # 3D optimization
    objectives_3d = [
        ObjectiveConfig(name="molecular_weight", minimize=True, normalization="minmax"),
        ObjectiveConfig(name="logp", minimize=True, normalization="minmax"),
        ObjectiveConfig(name="tpsa", minimize=True, normalization="minmax"),
    ]

    optimizer_3d = MPOptimizer(objectives=objectives_3d)
    result_3d = optimizer_3d.optimize(molecules, properties)

    visualizations_3d = optimizer_3d.visualize_tradeoffs(result_3d)

    if "pareto_front_3d" in visualizations_3d:
        print("  ✓ 3D Pareto front visualization created")

    # High-dimensional visualization
    objectives_hd = create_drug_discovery_objectives()
    optimizer_hd = MPOptimizer(objectives=objectives_hd)
    result_hd = optimizer_hd.optimize(molecules, properties)

    visualizations_hd = optimizer_hd.visualize_tradeoffs(result_hd)

    if "parallel_coordinates" in visualizations_hd:
        print("  ✓ Parallel coordinates visualization created")

    print("Visualization demo completed!")
    print("Note: In a real application, these would be interactive Plotly figures")


def demo_performance_comparison():
    """Demonstrate CPU vs GPU performance comparison."""
    print("\n" + "=" * 60)
    print("Performance Comparison Demo (CPU vs GPU)")
    print("=" * 60)

    # Test different problem sizes
    problem_sizes = [100, 500, 1000, 2000]

    print("Problem Size | CPU Time | GPU Time | Speedup")
    print("-" * 45)

    for n_molecules in problem_sizes:
        molecules, properties = generate_demo_molecules(n_molecules)
        objectives = create_drug_discovery_objectives()

        # CPU benchmark
        optimizer_cpu = MPOptimizer(objectives=objectives, use_gpu=False)
        import time

        start_time = time.time()
        result_cpu = optimizer_cpu.optimize(molecules, properties)
        cpu_time = time.time() - start_time

        # GPU benchmark
        optimizer_gpu = MPOptimizer(objectives=objectives, use_gpu=True)
        start_time = time.time()
        result_gpu = optimizer_gpu.optimize(molecules, properties)
        gpu_time = time.time() - start_time

        speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0

        print(
            f"{n_molecules:11d} | {cpu_time:8.3f}s | {gpu_time:8.3f}s | {speedup:6.2f}x"
        )

    print("\nNote: GPU performance benefits increase with problem size")


def main():
    """Run all demonstration examples."""
    print("ChemiAccel Multi-Objective Optimization Framework Demo")
    print(
        "This demo showcases the comprehensive multi-objective optimization capabilities"
    )
    print("including GPU acceleration, visualization, and scalarization functions.\n")

    try:
        # Run all demos
        demo_basic_multi_objective_optimization()
        demo_scalarization_functions()
        demo_polymer_optimization()
        demo_large_scale_optimization()
        demo_visualization_capabilities()
        demo_performance_comparison()

        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)

        print("\nKey features demonstrated:")
        print("  ✓ GPU-accelerated Pareto front identification")
        print("  ✓ Multiple scalarization functions")
        print("  ✓ Drug discovery and polymer optimization")
        print("  ✓ Large-scale optimization (10,000+ molecules)")
        print("  ✓ Interactive visualization capabilities")
        print("  ✓ Performance comparison (CPU vs GPU)")

        print("\nNext steps:")
        print("  - Integrate with your molecular datasets")
        print("  - Customize objectives for your specific use case")
        print("  - Use scalarization for generative model guidance")
        print("  - Export Pareto fronts for further analysis")

    except Exception as e:
        print(f"Demo failed with error: {e}")
        print("This may be due to missing dependencies (CuPy, Plotly)")
        print("The framework will fall back to CPU-only mode")


if __name__ == "__main__":
    main()
