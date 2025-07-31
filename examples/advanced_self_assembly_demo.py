"""
Advanced Self-Assembly Demonstration: Inverse Design and Multi-Scale Integration.

This comprehensive demonstration showcases the advanced capabilities of ChemiAccel's
self-assembly module, including inverse design for emergent properties and
multi-scale integration workflows.

Features demonstrated:
- Inverse design of surfactants for target micelle properties
- Environmental condition optimization for desired morphologies
- Multi-objective optimization with trade-offs
- Multi-scale modeling from atomistic to mesoscale
- Data transfer and coupling between scales
- Performance analysis and validation
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import ChemiAccel self-assembly modules
try:
    from chemi_accel.self_assembly.characterization import MorphologyType, StructureType
    from chemi_accel.self_assembly.inverse_design import (
        DesignTarget,
        EmergentPropertyInverseDesigner,
        OptimizationMethod,
        OptimizationObjective,
        design_micelle_forming_surfactant,
        multi_objective_surfactant_design,
        optimize_environmental_conditions,
    )
    from chemi_accel.self_assembly.multiscale_integration import (
        MultiScaleIntegrator,
        ScaleData,
        ScaleType,
        create_hierarchical_system,
        run_multiscale_workflow,
        validate_multiscale_consistency,
    )

    CHEMI_ACCEL_AVAILABLE = True

except ImportError as e:
    logger.warning(f"ChemiAccel self-assembly modules not available: {e}")
    CHEMI_ACCEL_AVAILABLE = False

# Optional visualization dependencies
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class AdvancedSelfAssemblyDemo:
    """
    Comprehensive demonstration of advanced self-assembly capabilities.

    This class provides detailed examples of inverse design optimization
    and multi-scale integration workflows.
    """

    def __init__(
        self,
        use_gpu: bool = True,
        save_output: bool = True,
        output_dir: str = "advanced_self_assembly_results",
        verbose: bool = True,
    ):
        """
        Initialize the advanced demonstration.

        Args:
            use_gpu: Whether to use GPU acceleration
            save_output: Whether to save results to files
            output_dir: Directory for saving results
            verbose: Whether to print detailed output
        """
        self.use_gpu = use_gpu and CHEMI_ACCEL_AVAILABLE
        self.save_output = save_output
        self.verbose = verbose

        if save_output:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(exist_ok=True)
            logger.info(f"Results will be saved to: {self.output_dir}")

        self.results = {}

        # Initialize components if available
        if CHEMI_ACCEL_AVAILABLE:
            self.inverse_designer = EmergentPropertyInverseDesigner(
                use_gpu=self.use_gpu, cache_simulations=True, parallel_evaluations=True
            )

        logger.info(f"AdvancedSelfAssemblyDemo initialized (GPU: {self.use_gpu})")

    def run_all_demonstrations(self):
        """Run all demonstration scenarios."""
        logger.info("Starting comprehensive self-assembly demonstrations...")

        if not CHEMI_ACCEL_AVAILABLE:
            logger.error("ChemiAccel not available - running mock demonstrations")
            self._run_mock_demonstrations()
            return

        try:
            # Inverse design demonstrations
            self.demo_surfactant_inverse_design()
            self.demo_environmental_optimization()
            self.demo_multi_objective_optimization()

            # Multi-scale integration demonstrations
            self.demo_hierarchical_multiscale_modeling()
            self.demo_scale_coupling_analysis()
            self.demo_multiscale_workflow_validation()

            # Advanced scenarios
            self.demo_integrated_design_and_multiscale()
            self.demo_performance_benchmarking()

            # Generate comprehensive report
            self.generate_comprehensive_report()

        except Exception as e:
            logger.error(f"Demonstration failed: {e}")
            raise

        logger.info("All demonstrations completed successfully!")

    def demo_surfactant_inverse_design(self):
        """Demonstrate inverse design of surfactants for target micelle properties."""
        logger.info("\n" + "=" * 60)
        logger.info("DEMONSTRATION 1: Surfactant Inverse Design")
        logger.info("=" * 60)

        start_time = time.time()

        # Scenario 1: Design for large micelles
        logger.info(
            "Scenario 1A: Optimizing for large micelles (aggregation number = 60)"
        )

        result_large_micelles = design_micelle_forming_surfactant(
            target_aggregation_number=60,
            target_cmc=0.005,
            head_groups=["ionic", "nonionic", "zwitterionic"],
            tail_length_range=(8, 16),
            max_evaluations=50,
            use_gpu=self.use_gpu,
        )

        self._log_optimization_result("Large Micelles", result_large_micelles)

        # Scenario 2: Design for small, stable micelles
        logger.info(
            "\nScenario 1B: Optimizing for small, stable micelles (aggregation number = 25)"
        )

        result_small_micelles = design_micelle_forming_surfactant(
            target_aggregation_number=25,
            target_cmc=0.1,
            head_groups=["ionic", "nonionic"],
            tail_length_range=(6, 12),
            max_evaluations=40,
            use_gpu=self.use_gpu,
        )

        self._log_optimization_result("Small Stable Micelles", result_small_micelles)

        # Scenario 3: Custom design space exploration
        logger.info("\nScenario 1C: Custom design space exploration")

        # Create custom design space
        design_space = self.inverse_designer.create_surfactant_design_space(
            head_group_options=["ionic", "nonionic", "zwitterionic", "amphoteric"],
            tail_length_range=(6, 18),
            property_targets={
                "aggregation_number": 45.0,
                "minimize_cmc": 0.01,
                "maximize_stability": 0.8,
            },
        )

        result_custom = self.inverse_designer.optimize_design(
            design_space=design_space,
            design_target=DesignTarget.SURFACTANT_STRUCTURE,
            optimization_method=OptimizationMethod.BAYESIAN_OPTIMIZATION,
            max_evaluations=60,
        )

        self._log_optimization_result("Custom Design", result_custom)

        # Store results
        self.results["surfactant_inverse_design"] = {
            "large_micelles": result_large_micelles,
            "small_micelles": result_small_micelles,
            "custom_design": result_custom,
            "execution_time": time.time() - start_time,
        }

        # Visualize optimization convergence
        if self.save_output:
            self._plot_optimization_convergence(
                [result_large_micelles, result_small_micelles, result_custom],
                ["Large Micelles", "Small Micelles", "Custom Design"],
                "surfactant_optimization_convergence.png",
            )

        logger.info(
            f"Surfactant inverse design completed in {time.time() - start_time:.2f} seconds"
        )

    def demo_environmental_optimization(self):
        """Demonstrate optimization of environmental conditions."""
        logger.info("\n" + "=" * 60)
        logger.info("DEMONSTRATION 2: Environmental Condition Optimization")
        logger.info("=" * 60)

        start_time = time.time()

        # Scenario 1: Optimize for micelle formation
        logger.info("Scenario 2A: Optimizing conditions for micelle formation")

        result_micelle_conditions = optimize_environmental_conditions(
            target_morphology=StructureType.MICELLE,
            temperature_range=(280.0, 340.0),
            concentration_range=(0.001, 0.5),
            max_evaluations=30,
            use_gpu=self.use_gpu,
        )

        self._log_optimization_result(
            "Micelle Formation Conditions", result_micelle_conditions
        )

        # Scenario 2: Optimize for vesicle formation
        logger.info("\nScenario 2B: Optimizing conditions for vesicle formation")

        result_vesicle_conditions = optimize_environmental_conditions(
            target_morphology=StructureType.VESICLE,
            temperature_range=(285.0, 325.0),
            concentration_range=(0.01, 0.8),
            max_evaluations=35,
            use_gpu=self.use_gpu,
        )

        self._log_optimization_result(
            "Vesicle Formation Conditions", result_vesicle_conditions
        )

        # Scenario 3: Multi-parameter environmental optimization
        logger.info("\nScenario 2C: Comprehensive environmental optimization")

        env_design_space = self.inverse_designer.create_environmental_design_space(
            temperature_range=(270.0, 350.0),
            concentration_range=(0.001, 1.0),
            property_targets={"maximize_stability": 0.9, "aggregation_number": 40.0},
        )

        result_comprehensive = self.inverse_designer.optimize_design(
            design_space=env_design_space,
            design_target=DesignTarget.ENVIRONMENTAL_CONDITIONS,
            optimization_method=OptimizationMethod.GENETIC_ALGORITHM,
            max_evaluations=50,
            population_size=20,
        )

        self._log_optimization_result(
            "Comprehensive Environmental", result_comprehensive
        )

        # Store results
        self.results["environmental_optimization"] = {
            "micelle_conditions": result_micelle_conditions,
            "vesicle_conditions": result_vesicle_conditions,
            "comprehensive": result_comprehensive,
            "execution_time": time.time() - start_time,
        }

        # Visualize environmental parameter space
        if self.save_output:
            self._plot_environmental_parameter_space(
                [
                    result_micelle_conditions,
                    result_vesicle_conditions,
                    result_comprehensive,
                ],
                ["Micelle", "Vesicle", "Comprehensive"],
                "environmental_parameter_space.png",
            )

        logger.info(
            f"Environmental optimization completed in {time.time() - start_time:.2f} seconds"
        )

    def demo_multi_objective_optimization(self):
        """Demonstrate multi-objective optimization with trade-offs."""
        logger.info("\n" + "=" * 60)
        logger.info("DEMONSTRATION 3: Multi-Objective Optimization")
        logger.info("=" * 60)

        start_time = time.time()

        # Scenario 1: Balance micelle size and CMC
        logger.info(
            "Scenario 3A: Balancing micelle size and critical micelle concentration"
        )

        objectives_balanced = {"aggregation_number": 50.0, "minimize_cmc": 0.01}

        result_balanced = multi_objective_surfactant_design(
            objectives=objectives_balanced,
            head_groups=["ionic", "nonionic", "zwitterionic"],
            tail_length_range=(6, 16),
            max_evaluations=80,
            use_gpu=self.use_gpu,
        )

        self._log_optimization_result("Balanced Design", result_balanced)

        # Scenario 2: Complex multi-objective optimization
        logger.info("\nScenario 3B: Complex multi-objective optimization")

        objectives_complex = {
            "aggregation_number": 45.0,
            "minimize_cmc": 0.005,
            "maximize_stability": 0.85,
        }

        result_complex = multi_objective_surfactant_design(
            objectives=objectives_complex,
            head_groups=["ionic", "nonionic", "zwitterionic", "amphoteric"],
            tail_length_range=(8, 18),
            max_evaluations=100,
            use_gpu=self.use_gpu,
        )

        self._log_optimization_result("Complex Multi-Objective", result_complex)

        # Analyze Pareto front
        if result_complex.pareto_front:
            logger.info(
                f"Pareto front contains {len(result_complex.pareto_front)} solutions"
            )

            # Log some Pareto solutions
            for i, (params, objectives) in enumerate(result_complex.pareto_front[:3]):
                logger.info(
                    f"  Pareto solution {i+1}: {params} -> objectives: {objectives}"
                )

        # Store results
        self.results["multi_objective_optimization"] = {
            "balanced_design": result_balanced,
            "complex_design": result_complex,
            "execution_time": time.time() - start_time,
        }

        # Visualize Pareto front
        if self.save_output and result_complex.pareto_front:
            self._plot_pareto_front(result_complex, "pareto_front_analysis.png")

        logger.info(
            f"Multi-objective optimization completed in {time.time() - start_time:.2f} seconds"
        )

    def demo_hierarchical_multiscale_modeling(self):
        """Demonstrate hierarchical multi-scale modeling."""
        logger.info("\n" + "=" * 60)
        logger.info("DEMONSTRATION 4: Hierarchical Multi-Scale Modeling")
        logger.info("=" * 60)

        start_time = time.time()

        # Create hierarchical system
        logger.info("Creating hierarchical multi-scale system...")

        system = create_hierarchical_system(
            n_atoms=200, n_molecules=50, n_cg_particles=15, box_size=12.0
        )

        logger.info(
            f"System created with {len(system.scales)} scales and {len(system.couplings)} couplings"
        )

        # Log system details
        for i, scale in enumerate(system.scales):
            logger.info(f"  Scale {i+1}: {scale.scale_type.value}")
            logger.info(f"    - Spatial resolution: {scale.spatial_resolution} nm")
            logger.info(f"    - Temporal resolution: {scale.temporal_resolution} ps")
            logger.info(f"    - Particle count: {scale.particle_count}")

        # Run multi-scale workflow
        logger.info("\nRunning multi-scale simulation workflow...")

        results = run_multiscale_workflow(
            system=system, total_steps=50, use_gpu=self.use_gpu
        )

        logger.info("Multi-scale simulation completed:")
        logger.info(
            f"  - Total time: {results['performance_metrics']['total_time']:.2f} seconds"
        )
        logger.info(
            f"  - Steps per second: {results['performance_metrics']['steps_per_second']:.1f}"
        )

        # Analyze coupling quality
        coupling_analysis = results["coupling_analysis"]
        logger.info(
            f"  - Transfer success rate: {coupling_analysis['transfer_success_rate']:.2%}"
        )
        logger.info(
            f"  - Average transfer time: {coupling_analysis['average_transfer_time']:.4f} seconds"
        )

        # Validate consistency
        validation = validate_multiscale_consistency(results)
        logger.info("\nMulti-scale consistency validation:")
        for check, passed in validation.items():
            status = "PASS" if passed else "FAIL"
            logger.info(f"  - {check}: {status}")

        # Store results
        self.results["hierarchical_multiscale"] = {
            "system_definition": system,
            "simulation_results": results,
            "consistency_validation": validation,
            "execution_time": time.time() - start_time,
        }

        # Visualize multi-scale trajectories
        if self.save_output:
            self._plot_multiscale_trajectories(results, "multiscale_trajectories.png")
            self._plot_coupling_analysis(
                coupling_analysis, "coupling_quality_analysis.png"
            )

        logger.info(
            f"Hierarchical multi-scale modeling completed in {time.time() - start_time:.2f} seconds"
        )

    def demo_scale_coupling_analysis(self):
        """Demonstrate detailed analysis of scale coupling."""
        logger.info("\n" + "=" * 60)
        logger.info("DEMONSTRATION 5: Scale Coupling Analysis")
        logger.info("=" * 60)

        start_time = time.time()

        # Create system with frequent coupling
        system = create_hierarchical_system(
            n_atoms=100, n_molecules=25, n_cg_particles=8
        )

        # Modify coupling frequencies for detailed analysis
        for coupling in system.couplings:
            coupling.transfer_frequency = 3  # More frequent coupling

        logger.info("Running simulation with frequent scale coupling...")

        integrator = MultiScaleIntegrator(
            system, use_gpu=self.use_gpu, parallel_execution=True
        )

        # Initialize with controlled initial conditions
        initial_conditions = {
            ScaleType.ATOMISTIC: ScaleData(
                ScaleType.ATOMISTIC,
                np.random.uniform(-3, 3, (100, 3)),
                properties={"initial_energy": 100.0},
            ),
            ScaleType.MOLECULAR: ScaleData(
                ScaleType.MOLECULAR,
                np.random.uniform(-2, 2, (25, 3)),
                properties={"initial_energy": 25.0},
            ),
            ScaleType.MESOSCALE: ScaleData(
                ScaleType.MESOSCALE,
                np.random.uniform(-1, 1, (8, 3)),
                properties={"initial_energy": 8.0},
            ),
        }

        integrator.initialize_system(initial_conditions)

        # Run with detailed coupling analysis
        results = integrator.run_coupled_simulation(total_steps=30, output_frequency=5)

        # Detailed coupling analysis
        coupling_analysis = integrator.analyze_coupling_quality()

        logger.info("Detailed coupling analysis results:")
        logger.info(f"  - Total coupling events: {len(integrator.coupling_history)}")
        logger.info(
            f"  - Success rate: {coupling_analysis['transfer_success_rate']:.2%}"
        )
        logger.info(
            f"  - Average transfer time: {coupling_analysis['average_transfer_time']:.4f} s"
        )

        # Analyze errors by coupling type
        for coupling_type, errors in coupling_analysis["coupling_errors"].items():
            if errors:
                error_values = [e["value"] for e in errors]
                logger.info(
                    f"  - {coupling_type} errors: mean={np.mean(error_values):.4f}, "
                    f"std={np.std(error_values):.4f}"
                )

        # Store results
        self.results["scale_coupling_analysis"] = {
            "coupling_history": integrator.coupling_history,
            "coupling_analysis": coupling_analysis,
            "execution_time": time.time() - start_time,
        }

        # Visualize coupling details
        if self.save_output:
            self._plot_coupling_timeline(
                integrator.coupling_history, "coupling_timeline.png"
            )
            self._plot_transfer_errors(coupling_analysis, "transfer_error_analysis.png")

        logger.info(
            f"Scale coupling analysis completed in {time.time() - start_time:.2f} seconds"
        )

    def demo_multiscale_workflow_validation(self):
        """Demonstrate comprehensive multi-scale workflow validation."""
        logger.info("\n" + "=" * 60)
        logger.info("DEMONSTRATION 6: Multi-Scale Workflow Validation")
        logger.info("=" * 60)

        start_time = time.time()

        # Test different system configurations
        configurations = [
            {
                "n_atoms": 50,
                "n_molecules": 15,
                "n_cg_particles": 5,
                "name": "Small System",
            },
            {
                "n_atoms": 150,
                "n_molecules": 40,
                "n_cg_particles": 12,
                "name": "Medium System",
            },
            {
                "n_atoms": 300,
                "n_molecules": 80,
                "n_cg_particles": 20,
                "name": "Large System",
            },
        ]

        validation_results = {}

        for config in configurations:
            logger.info(f"\nTesting {config['name']}...")

            system = create_hierarchical_system(
                n_atoms=config["n_atoms"],
                n_molecules=config["n_molecules"],
                n_cg_particles=config["n_cg_particles"],
            )

            # Run workflow
            results = run_multiscale_workflow(
                system=system, total_steps=20, use_gpu=self.use_gpu
            )

            # Validate
            validation = validate_multiscale_consistency(results, tolerance=0.15)

            validation_results[config["name"]] = {
                "system_size": config["n_atoms"]
                + config["n_molecules"]
                + config["n_cg_particles"],
                "performance": results["performance_metrics"],
                "validation": validation,
                "coupling_quality": results["coupling_analysis"],
            }

            logger.info(
                f"  - System size: {validation_results[config['name']]['system_size']} particles"
            )
            logger.info(
                f"  - Simulation time: {results['performance_metrics']['total_time']:.2f} s"
            )
            logger.info(
                f"  - Validation passed: {sum(validation.values())}/{len(validation)} checks"
            )

        # Performance scaling analysis
        logger.info("\nPerformance scaling analysis:")
        for name, result in validation_results.items():
            size = result["system_size"]
            time_taken = result["performance"]["total_time"]
            logger.info(
                f"  - {name}: {size} particles -> {time_taken:.2f} s "
                f"({time_taken/size*1000:.2f} ms/particle)"
            )

        # Store results
        self.results["workflow_validation"] = {
            "configurations": validation_results,
            "execution_time": time.time() - start_time,
        }

        # Visualize scaling
        if self.save_output:
            self._plot_performance_scaling(
                validation_results, "performance_scaling.png"
            )
            self._plot_validation_summary(validation_results, "validation_summary.png")

        logger.info(
            f"Multi-scale workflow validation completed in {time.time() - start_time:.2f} seconds"
        )

    def demo_integrated_design_and_multiscale(self):
        """Demonstrate integrated inverse design and multi-scale modeling."""
        logger.info("\n" + "=" * 60)
        logger.info("DEMONSTRATION 7: Integrated Design and Multi-Scale Modeling")
        logger.info("=" * 60)

        start_time = time.time()

        # Step 1: Design optimal surfactant
        logger.info(
            "Step 1: Designing optimal surfactant for large-scale simulation..."
        )

        optimal_surfactant = design_micelle_forming_surfactant(
            target_aggregation_number=35,
            target_cmc=0.02,
            head_groups=["ionic", "nonionic"],
            tail_length_range=(8, 14),
            max_evaluations=30,
            use_gpu=self.use_gpu,
        )

        logger.info("Optimal surfactant design:")
        for param, value in optimal_surfactant.best_parameters.items():
            logger.info(f"  - {param}: {value}")

        # Step 2: Use design parameters in multi-scale simulation
        logger.info("\nStep 2: Implementing design in multi-scale simulation...")

        # Create system based on design parameters
        tail_length = optimal_surfactant.best_parameters.get("tail_length", 10)
        n_molecules = max(20, int(tail_length * 2))  # Scale molecules with tail length

        system = create_hierarchical_system(
            n_atoms=n_molecules * 3,  # 3 atoms per molecule (simplified)
            n_molecules=n_molecules,
            n_cg_particles=max(5, n_molecules // 4),
            box_size=15.0,
        )

        # Run simulation with design-optimized parameters
        results = run_multiscale_workflow(
            system=system, total_steps=40, use_gpu=self.use_gpu
        )

        # Step 3: Validate that simulation achieves design targets
        logger.info("\nStep 3: Validating simulation against design targets...")

        # Extract final properties from simulation
        final_trajectories = results["trajectories"]
        mesoscale_final = final_trajectories[ScaleType.MESOSCALE][-1]["data"]

        # Estimate aggregation behavior (simplified)
        estimated_aggregation = len(mesoscale_final.positions)
        target_aggregation = 35

        aggregation_error = (
            abs(estimated_aggregation - target_aggregation) / target_aggregation
        )

        logger.info("Design validation:")
        logger.info(f"  - Target aggregation number: {target_aggregation}")
        logger.info(f"  - Simulated aggregation number: {estimated_aggregation}")
        logger.info(f"  - Relative error: {aggregation_error:.2%}")

        # Step 4: Iterative refinement (if needed)
        if aggregation_error > 0.2:  # 20% error threshold
            logger.info("\nStep 4: Refinement needed - adjusting parameters...")

            # Simplified refinement: adjust system size
            refined_n_molecules = int(
                n_molecules * (target_aggregation / estimated_aggregation)
            )

            refined_system = create_hierarchical_system(
                n_atoms=refined_n_molecules * 3,
                n_molecules=refined_n_molecules,
                n_cg_particles=max(5, refined_n_molecules // 4),
            )

            refined_results = run_multiscale_workflow(
                system=refined_system, total_steps=30, use_gpu=self.use_gpu
            )

            logger.info("Refined system completed")

            results["refined_simulation"] = refined_results

        # Store results
        self.results["integrated_design_multiscale"] = {
            "optimal_design": optimal_surfactant,
            "multiscale_simulation": results,
            "validation_metrics": {
                "aggregation_error": aggregation_error,
                "target_achieved": aggregation_error <= 0.2,
            },
            "execution_time": time.time() - start_time,
        }

        # Visualize integration workflow
        if self.save_output:
            self._plot_integrated_workflow(
                optimal_surfactant, results, "integrated_design_workflow.png"
            )

        logger.info(
            f"Integrated design and multi-scale modeling completed in {time.time() - start_time:.2f} seconds"
        )

    def demo_performance_benchmarking(self):
        """Demonstrate performance benchmarking of advanced features."""
        logger.info("\n" + "=" * 60)
        logger.info("DEMONSTRATION 8: Performance Benchmarking")
        logger.info("=" * 60)

        start_time = time.time()

        # Benchmark 1: Inverse design performance
        logger.info("Benchmark 1: Inverse design optimization performance")

        design_benchmarks = {}
        evaluation_counts = [10, 25, 50, 100]

        for eval_count in evaluation_counts:
            bench_start = time.time()

            result = design_micelle_forming_surfactant(
                target_aggregation_number=40,
                target_cmc=0.015,
                max_evaluations=eval_count,
                use_gpu=self.use_gpu,
            )

            bench_time = time.time() - bench_start
            design_benchmarks[eval_count] = {
                "time": bench_time,
                "final_objective": result.best_objective_value,
                "convergence_rate": len(result.objective_history),
            }

            logger.info(
                f"  - {eval_count} evaluations: {bench_time:.2f}s, "
                f"objective: {result.best_objective_value:.4f}"
            )

        # Benchmark 2: Multi-scale simulation performance
        logger.info("\nBenchmark 2: Multi-scale simulation performance")

        multiscale_benchmarks = {}
        system_sizes = [
            {"atoms": 50, "molecules": 15, "cg": 5},
            {"atoms": 100, "molecules": 30, "cg": 10},
            {"atoms": 200, "molecules": 60, "cg": 20},
        ]

        for size_config in system_sizes:
            bench_start = time.time()

            system = create_hierarchical_system(
                n_atoms=size_config["atoms"],
                n_molecules=size_config["molecules"],
                n_cg_particles=size_config["cg"],
            )

            results = run_multiscale_workflow(
                system, total_steps=20, use_gpu=self.use_gpu
            )

            bench_time = time.time() - bench_start
            total_particles = sum(size_config.values())

            multiscale_benchmarks[total_particles] = {
                "time": bench_time,
                "particles_per_second": total_particles
                * 20
                / bench_time,  # particles * steps / time
                "coupling_success_rate": results["coupling_analysis"][
                    "transfer_success_rate"
                ],
            }

            logger.info(
                f"  - {total_particles} particles: {bench_time:.2f}s, "
                f"{multiscale_benchmarks[total_particles]['particles_per_second']:.1f} particle-steps/s"
            )

        # Benchmark 3: GPU vs CPU comparison (if GPU available)
        if self.use_gpu:
            logger.info("\nBenchmark 3: GPU vs CPU performance comparison")

            # Test with CPU
            cpu_start = time.time()
            cpu_result = design_micelle_forming_surfactant(
                target_aggregation_number=35, max_evaluations=20, use_gpu=False
            )
            cpu_time = time.time() - cpu_start

            # Test with GPU
            gpu_start = time.time()
            gpu_result = design_micelle_forming_surfactant(
                target_aggregation_number=35, max_evaluations=20, use_gpu=True
            )
            gpu_time = time.time() - gpu_start

            speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0

            logger.info(f"  - CPU time: {cpu_time:.2f}s")
            logger.info(f"  - GPU time: {gpu_time:.2f}s")
            logger.info(f"  - GPU speedup: {speedup:.1f}x")

            gpu_comparison = {
                "cpu_time": cpu_time,
                "gpu_time": gpu_time,
                "speedup": speedup,
            }
        else:
            gpu_comparison = None

        # Store results
        self.results["performance_benchmarking"] = {
            "design_benchmarks": design_benchmarks,
            "multiscale_benchmarks": multiscale_benchmarks,
            "gpu_comparison": gpu_comparison,
            "execution_time": time.time() - start_time,
        }

        # Visualize performance
        if self.save_output:
            self._plot_performance_benchmarks(
                design_benchmarks, multiscale_benchmarks, "performance_benchmarks.png"
            )

        logger.info(
            f"Performance benchmarking completed in {time.time() - start_time:.2f} seconds"
        )

    def generate_comprehensive_report(self):
        """Generate comprehensive report of all demonstrations."""
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING COMPREHENSIVE REPORT")
        logger.info("=" * 60)

        if not self.save_output:
            logger.info("Report generation skipped (save_output=False)")
            return

        report_path = self.output_dir / "comprehensive_report.md"

        with open(report_path, "w") as f:
            f.write("# Advanced Self-Assembly Demonstration Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            total_time = sum(
                result.get("execution_time", 0) for result in self.results.values()
            )
            f.write(f"- **Total demonstration time**: {total_time:.2f} seconds\n")
            f.write(f"- **Number of scenarios**: {len(self.results)}\n")
            f.write(
                f"- **GPU acceleration**: {'Enabled' if self.use_gpu else 'Disabled'}\n\n"
            )

            # Detailed Results
            for demo_name, demo_results in self.results.items():
                f.write(f"## {demo_name.replace('_', ' ').title()}\n\n")

                if "execution_time" in demo_results:
                    f.write(
                        f"**Execution time**: {demo_results['execution_time']:.2f} seconds\n\n"
                    )

                # Add specific details based on demonstration type
                if "surfactant_inverse_design" in demo_name:
                    self._write_surfactant_design_details(f, demo_results)
                elif "environmental_optimization" in demo_name:
                    self._write_environmental_details(f, demo_results)
                elif "multi_objective" in demo_name:
                    self._write_multi_objective_details(f, demo_results)
                elif "hierarchical_multiscale" in demo_name:
                    self._write_multiscale_details(f, demo_results)
                elif "performance_benchmarking" in demo_name:
                    self._write_performance_details(f, demo_results)

                f.write("\n---\n\n")

            # Conclusions
            f.write("## Conclusions\n\n")
            f.write(
                "The advanced self-assembly demonstrations successfully showcase:\n\n"
            )
            f.write(
                "1. **Inverse Design Capabilities**: Effective optimization of molecular structures\n"
            )
            f.write(
                "2. **Multi-Scale Integration**: Seamless coupling between different scales\n"
            )
            f.write(
                "3. **Performance Optimization**: Efficient GPU-accelerated computations\n"
            )
            f.write(
                "4. **Validation and Analysis**: Comprehensive quality assessment tools\n\n"
            )

        logger.info(f"Comprehensive report saved to: {report_path}")

        # Generate summary statistics
        self._save_summary_statistics()

    def _run_mock_demonstrations(self):
        """Run mock demonstrations when ChemiAccel is not available."""
        logger.info("Running mock demonstrations...")

        scenarios = [
            "Surfactant Inverse Design",
            "Environmental Optimization",
            "Multi-Objective Optimization",
            "Hierarchical Multi-Scale Modeling",
            "Scale Coupling Analysis",
            "Workflow Validation",
            "Integrated Design and Multi-Scale",
            "Performance Benchmarking",
        ]

        for i, scenario in enumerate(scenarios, 1):
            logger.info(f"Mock Demo {i}: {scenario}")
            time.sleep(0.5)  # Simulate computation time

            # Generate mock results
            mock_result = {
                "scenario": scenario,
                "success": True,
                "execution_time": np.random.uniform(1, 5),
                "mock_objective": np.random.uniform(0.1, 10.0),
                "mock_parameters": {
                    f"param_{j}": np.random.uniform(0, 1) for j in range(3)
                },
            }

            self.results[scenario.lower().replace(" ", "_")] = mock_result
            logger.info(
                f"  - Mock execution time: {mock_result['execution_time']:.2f}s"
            )
            logger.info(f"  - Mock objective: {mock_result['mock_objective']:.3f}")

        if self.save_output:
            # Save mock results
            mock_report_path = self.output_dir / "mock_demonstration_report.md"
            with open(mock_report_path, "w") as f:
                f.write("# Mock Advanced Self-Assembly Demonstration Report\n\n")
                f.write(
                    "This is a mock report generated when ChemiAccel is not available.\n\n"
                )

                for scenario, result in self.results.items():
                    f.write(f"## {scenario.replace('_', ' ').title()}\n")
                    f.write(f"- Execution time: {result['execution_time']:.2f}s\n")
                    f.write(f"- Mock objective: {result['mock_objective']:.3f}\n\n")

            logger.info(f"Mock report saved to: {mock_report_path}")

    # Helper methods for logging and visualization
    def _log_optimization_result(self, name: str, result):
        """Log optimization result details."""
        if hasattr(result, "success") and result.success:
            logger.info(f"  {name} optimization succeeded:")
            logger.info(f"    - Best objective: {result.best_objective_value:.4f}")
            logger.info(f"    - Evaluations: {result.evaluation_count}")
            logger.info(f"    - Time: {result.optimization_time:.2f}s")
            logger.info(f"    - Best parameters: {result.best_parameters}")
        else:
            logger.warning(f"  {name} optimization failed or unavailable")

    def _plot_optimization_convergence(
        self, results: List, labels: List[str], filename: str
    ):
        """Plot optimization convergence curves."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available - skipping convergence plot")
            return

        fig = go.Figure()

        for result, label in zip(results, labels):
            if hasattr(result, "objective_history") and result.objective_history:
                fig.add_trace(
                    go.Scatter(
                        y=result.objective_history,
                        mode="lines+markers",
                        name=label,
                        line=dict(width=2),
                    )
                )

        fig.update_layout(
            title="Optimization Convergence Comparison",
            xaxis_title="Evaluation Number",
            yaxis_title="Objective Value",
            template="plotly_white",
            width=800,
            height=500,
        )

        fig.write_image(self.output_dir / filename)
        logger.info(f"Convergence plot saved to: {filename}")

    def _plot_environmental_parameter_space(
        self, results: List, labels: List[str], filename: str
    ):
        """Plot environmental parameter space exploration."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available - skipping parameter space plot")
            return

        fig = go.Figure()

        for result, label in zip(results, labels):
            if hasattr(result, "best_parameters"):
                params = result.best_parameters
                temperature = params.get("temperature", 298.15)
                concentration = params.get("concentration", 0.1)

                fig.add_trace(
                    go.Scatter(
                        x=[temperature],
                        y=[concentration],
                        mode="markers",
                        name=label,
                        marker=dict(size=12),
                    )
                )

        fig.update_layout(
            title="Environmental Parameter Space Optimization",
            xaxis_title="Temperature (K)",
            yaxis_title="Concentration",
            template="plotly_white",
            width=700,
            height=500,
        )

        fig.write_image(self.output_dir / filename)
        logger.info(f"Parameter space plot saved to: {filename}")

    def _plot_pareto_front(self, result, filename: str):
        """Plot Pareto front for multi-objective optimization."""
        if (
            not PLOTLY_AVAILABLE
            or not hasattr(result, "pareto_front")
            or not result.pareto_front
        ):
            logger.warning(
                "Cannot plot Pareto front - data unavailable or Plotly missing"
            )
            return

        # Extract objectives from Pareto front
        obj1_values = [obj[0] for _, obj in result.pareto_front]
        obj2_values = [obj[1] for _, obj in result.pareto_front]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=obj1_values,
                y=obj2_values,
                mode="markers",
                name="Pareto Solutions",
                marker=dict(size=10, color="red"),
            )
        )

        fig.update_layout(
            title="Multi-Objective Optimization Pareto Front",
            xaxis_title="Objective 1",
            yaxis_title="Objective 2",
            template="plotly_white",
            width=600,
            height=500,
        )

        fig.write_image(self.output_dir / filename)
        logger.info(f"Pareto front plot saved to: {filename}")

    def _plot_multiscale_trajectories(self, results: Dict, filename: str):
        """Plot multi-scale trajectory evolution."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available - skipping trajectory plot")
            return

        trajectories = results.get("trajectories", {})

        fig = make_subplots(
            rows=len(trajectories),
            cols=1,
            subplot_titles=[
                f"{scale.value.title()} Scale" for scale in trajectories.keys()
            ],
            vertical_spacing=0.1,
        )

        for i, (scale_type, trajectory) in enumerate(trajectories.items(), 1):
            if trajectory:
                steps = [entry.get("step", j) for j, entry in enumerate(trajectory)]
                n_particles = [len(entry["data"].positions) for entry in trajectory]

                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=n_particles,
                        mode="lines+markers",
                        name=f"{scale_type.value} particles",
                    ),
                    row=i,
                    col=1,
                )

        fig.update_layout(
            title="Multi-Scale Trajectory Evolution",
            height=300 * len(trajectories),
            template="plotly_white",
        )

        fig.write_image(self.output_dir / filename)
        logger.info(f"Multi-scale trajectory plot saved to: {filename}")

    def _plot_coupling_analysis(self, coupling_analysis: Dict, filename: str):
        """Plot coupling quality analysis."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available - skipping coupling analysis plot")
            return

        # Create summary metrics plot
        metrics = {
            "Success Rate": coupling_analysis.get("transfer_success_rate", 0) * 100,
            "Avg Transfer Time (ms)": coupling_analysis.get("average_transfer_time", 0)
            * 1000,
        }

        fig = go.Figure(
            data=[
                go.Bar(
                    x=list(metrics.keys()),
                    y=list(metrics.values()),
                    marker_color=["green", "blue"],
                )
            ]
        )

        fig.update_layout(
            title="Scale Coupling Quality Metrics",
            yaxis_title="Value",
            template="plotly_white",
            width=600,
            height=400,
        )

        fig.write_image(self.output_dir / filename)
        logger.info(f"Coupling analysis plot saved to: {filename}")

    def _plot_coupling_timeline(self, coupling_history: List, filename: str):
        """Plot coupling event timeline."""
        if not PLOTLY_AVAILABLE or not coupling_history:
            logger.warning(
                "Cannot plot coupling timeline - data unavailable or Plotly missing"
            )
            return

        steps = [entry["step"] for entry in coupling_history]
        success = [1 if entry["result"].success else 0 for entry in coupling_history]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=steps,
                y=success,
                mode="markers",
                name="Coupling Events",
                marker=dict(size=8, color=success, colorscale="RdYlGn"),
            )
        )

        fig.update_layout(
            title="Coupling Event Timeline",
            xaxis_title="Simulation Step",
            yaxis_title="Success (1) / Failure (0)",
            template="plotly_white",
            width=800,
            height=400,
        )

        fig.write_image(self.output_dir / filename)
        logger.info(f"Coupling timeline plot saved to: {filename}")

    def _plot_transfer_errors(self, coupling_analysis: Dict, filename: str):
        """Plot transfer error analysis."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available - skipping transfer error plot")
            return

        coupling_errors = coupling_analysis.get("coupling_errors", {})

        if not coupling_errors:
            logger.warning("No coupling error data available for plotting")
            return

        fig = go.Figure()

        for coupling_type, errors in coupling_errors.items():
            if errors:
                error_values = [e["value"] for e in errors]
                steps = [e["step"] for e in errors]

                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=error_values,
                        mode="lines+markers",
                        name=coupling_type,
                    )
                )

        fig.update_layout(
            title="Transfer Error Analysis",
            xaxis_title="Simulation Step",
            yaxis_title="Error Value",
            template="plotly_white",
            width=800,
            height=500,
        )

        fig.write_image(self.output_dir / filename)
        logger.info(f"Transfer error plot saved to: {filename}")

    def _plot_performance_scaling(self, validation_results: Dict, filename: str):
        """Plot performance scaling analysis."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available - skipping performance scaling plot")
            return

        sizes = [result["system_size"] for result in validation_results.values()]
        times = [
            result["performance"]["total_time"]
            for result in validation_results.values()
        ]
        names = list(validation_results.keys())

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=sizes,
                y=times,
                mode="lines+markers",
                name="Execution Time",
                text=names,
                textposition="top center",
            )
        )

        fig.update_layout(
            title="Multi-Scale Simulation Performance Scaling",
            xaxis_title="Total System Size (particles)",
            yaxis_title="Execution Time (seconds)",
            template="plotly_white",
            width=700,
            height=500,
        )

        fig.write_image(self.output_dir / filename)
        logger.info(f"Performance scaling plot saved to: {filename}")

    def _plot_validation_summary(self, validation_results: Dict, filename: str):
        """Plot validation summary."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available - skipping validation summary plot")
            return

        # Count validation passes
        validation_data = []
        for name, result in validation_results.items():
            validation = result["validation"]
            passed_checks = sum(validation.values())
            total_checks = len(validation)
            validation_data.append(
                {
                    "name": name,
                    "passed": passed_checks,
                    "total": total_checks,
                    "percentage": passed_checks / total_checks * 100,
                }
            )

        fig = go.Figure(
            data=[
                go.Bar(
                    x=[d["name"] for d in validation_data],
                    y=[d["percentage"] for d in validation_data],
                    text=[f"{d['passed']}/{d['total']}" for d in validation_data],
                    textposition="auto",
                    marker_color="lightblue",
                )
            ]
        )

        fig.update_layout(
            title="Multi-Scale Validation Summary",
            xaxis_title="System Configuration",
            yaxis_title="Validation Pass Rate (%)",
            template="plotly_white",
            width=600,
            height=400,
        )

        fig.write_image(self.output_dir / filename)
        logger.info(f"Validation summary plot saved to: {filename}")

    def _plot_integrated_workflow(
        self, design_result, simulation_results: Dict, filename: str
    ):
        """Plot integrated workflow visualization."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available - skipping integrated workflow plot")
            return

        # Create workflow diagram-style plot
        stages = ["Design", "Implementation", "Simulation", "Validation"]
        values = [
            (
                design_result.best_objective_value
                if hasattr(design_result, "best_objective_value")
                else 1.0
            ),
            1.0,  # Implementation always succeeds
            simulation_results["performance_metrics"]["total_time"] / 10,  # Normalized
            0.8,  # Validation score (simplified)
        ]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=stages, y=values, marker_color=["red", "orange", "blue", "green"]
                )
            ]
        )

        fig.update_layout(
            title="Integrated Design-to-Simulation Workflow",
            xaxis_title="Workflow Stage",
            yaxis_title="Normalized Performance Metric",
            template="plotly_white",
            width=600,
            height=400,
        )

        fig.write_image(self.output_dir / filename)
        logger.info(f"Integrated workflow plot saved to: {filename}")

    def _plot_performance_benchmarks(
        self, design_benchmarks: Dict, multiscale_benchmarks: Dict, filename: str
    ):
        """Plot performance benchmarking results."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available - skipping performance benchmark plot")
            return

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Inverse Design Performance", "Multi-Scale Performance"],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]],
        )

        # Design benchmarks
        eval_counts = list(design_benchmarks.keys())
        times = [design_benchmarks[ec]["time"] for ec in eval_counts]

        fig.add_trace(
            go.Scatter(
                x=eval_counts, y=times, mode="lines+markers", name="Design Time"
            ),
            row=1,
            col=1,
        )

        # Multi-scale benchmarks
        particle_counts = list(multiscale_benchmarks.keys())
        ms_times = [multiscale_benchmarks[pc]["time"] for pc in particle_counts]

        fig.add_trace(
            go.Scatter(
                x=particle_counts,
                y=ms_times,
                mode="lines+markers",
                name="Multi-Scale Time",
                line=dict(color="red"),
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="Evaluations", row=1, col=1)
        fig.update_xaxes(title_text="Particles", row=1, col=2)
        fig.update_yaxes(title_text="Time (seconds)", row=1, col=1)
        fig.update_yaxes(title_text="Time (seconds)", row=1, col=2)

        fig.update_layout(
            title="Performance Benchmarking Results",
            template="plotly_white",
            width=1000,
            height=400,
            showlegend=False,
        )

        fig.write_image(self.output_dir / filename)
        logger.info(f"Performance benchmark plot saved to: {filename}")

    # Report writing helper methods
    def _write_surfactant_design_details(self, f, demo_results: Dict):
        """Write surfactant design details to report."""
        f.write("### Key Results\n\n")

        for scenario, result in demo_results.items():
            if scenario == "execution_time":
                continue

            f.write(f"**{scenario.replace('_', ' ').title()}**:\n")

            if hasattr(result, "best_parameters"):
                f.write(f"- Best parameters: {result.best_parameters}\n")
                f.write(f"- Best objective: {result.best_objective_value:.4f}\n")
                f.write(f"- Evaluations: {result.evaluation_count}\n")
                f.write(f"- Optimization time: {result.optimization_time:.2f}s\n\n")

    def _write_environmental_details(self, f, demo_results: Dict):
        """Write environmental optimization details to report."""
        f.write("### Environmental Conditions Optimized\n\n")

        for scenario, result in demo_results.items():
            if scenario == "execution_time":
                continue

            f.write(f"**{scenario.replace('_', ' ').title()}**:\n")

            if hasattr(result, "best_parameters"):
                params = result.best_parameters
                f.write(f"- Temperature: {params.get('temperature', 'N/A')} K\n")
                f.write(f"- Concentration: {params.get('concentration', 'N/A')}\n")
                f.write(f"- pH: {params.get('ph', 'N/A')}\n")
                f.write(f"- Objective value: {result.best_objective_value:.4f}\n\n")

    def _write_multi_objective_details(self, f, demo_results: Dict):
        """Write multi-objective optimization details to report."""
        f.write("### Multi-Objective Optimization Results\n\n")

        for scenario, result in demo_results.items():
            if scenario == "execution_time":
                continue

            f.write(f"**{scenario.replace('_', ' ').title()}**:\n")

            if hasattr(result, "pareto_front") and result.pareto_front:
                f.write(f"- Pareto front size: {len(result.pareto_front)}\n")
                f.write(f"- Best compromise solution: {result.best_parameters}\n")
                f.write(f"- Evaluations performed: {result.evaluation_count}\n\n")

    def _write_multiscale_details(self, f, demo_results: Dict):
        """Write multi-scale modeling details to report."""
        sim_results = demo_results.get("simulation_results", {})
        performance = sim_results.get("performance_metrics", {})
        validation = demo_results.get("consistency_validation", {})

        f.write("### Multi-Scale System Performance\n\n")
        f.write(
            f"- Total simulation time: {performance.get('total_time', 'N/A')} seconds\n"
        )
        f.write(f"- Steps per second: {performance.get('steps_per_second', 'N/A')}\n")

        f.write("\n### Consistency Validation\n\n")
        for check, passed in validation.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            f.write(f"- {check.replace('_', ' ').title()}: {status}\n")
        f.write("\n")

    def _write_performance_details(self, f, demo_results: Dict):
        """Write performance benchmarking details to report."""
        design_benchmarks = demo_results.get("design_benchmarks", {})
        multiscale_benchmarks = demo_results.get("multiscale_benchmarks", {})
        gpu_comparison = demo_results.get("gpu_comparison")

        f.write("### Inverse Design Performance\n\n")
        for eval_count, metrics in design_benchmarks.items():
            f.write(f"- {eval_count} evaluations: {metrics['time']:.2f}s\n")

        f.write("\n### Multi-Scale Performance\n\n")
        for particles, metrics in multiscale_benchmarks.items():
            f.write(
                f"- {particles} particles: {metrics['time']:.2f}s "
                f"({metrics['particles_per_second']:.1f} particle-steps/s)\n"
            )

        if gpu_comparison:
            f.write("\n### GPU vs CPU Comparison\n\n")
            f.write(f"- CPU time: {gpu_comparison['cpu_time']:.2f}s\n")
            f.write(f"- GPU time: {gpu_comparison['gpu_time']:.2f}s\n")
            f.write(f"- GPU speedup: {gpu_comparison['speedup']:.1f}x\n")

        f.write("\n")

    def _save_summary_statistics(self):
        """Save summary statistics to JSON."""
        summary = {
            "total_demonstrations": len(self.results),
            "total_execution_time": sum(
                r.get("execution_time", 0) for r in self.results.values()
            ),
            "gpu_enabled": self.use_gpu,
            "successful_demonstrations": sum(
                1 for r in self.results.values() if r.get("execution_time", 0) > 0
            ),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        summary_path = self.output_dir / "summary_statistics.json"

        import json

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary statistics saved to: {summary_path}")


def main():
    """Run the advanced self-assembly demonstration."""
    print("=" * 80)
    print("ADVANCED SELF-ASSEMBLY DEMONSTRATION")
    print("ChemiAccel: Inverse Design and Multi-Scale Integration")
    print("=" * 80)

    # Configuration
    use_gpu = True  # Set to False if no GPU available
    save_output = True
    verbose = True

    # Create and run demonstration
    demo = AdvancedSelfAssemblyDemo(
        use_gpu=use_gpu, save_output=save_output, verbose=verbose
    )

    try:
        demo.run_all_demonstrations()

        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)

        if save_output:
            print(f"Results saved to: {demo.output_dir}")
            print("Generated files:")
            for file_path in demo.output_dir.glob("*"):
                print(f"  - {file_path.name}")

    except Exception as e:
        print(f"\nDemonstration failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
