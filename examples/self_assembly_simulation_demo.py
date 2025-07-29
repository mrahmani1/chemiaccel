"""
Comprehensive Demonstration of Self-Assembly Simulation and Characterization.

This demo showcases the GPU-accelerated coarse-grained simulation engine and
comprehensive characterization tools for self-assembling systems including
micelles, vesicles, and liquid crystals.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add src to path for importing ChemiAccel modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chemi_accel.self_assembly.characterization import (  # Main classes; Data structures; Utility functions
    ClusteringMethod,
    MorphologyType,
    PhaseType,
    SelfAssemblyCharacterizer,
    StructureType,
    analyze_micelle_formation,
    characterize_trajectory,
)

# Import simulation and characterization modules
from chemi_accel.self_assembly.coarse_grained_simulation import (  # Main classes; Data structures; Enums; Utility functions
    BoundaryCondition,
    CoarseGrainedSimulationEngine,
    IntegratorType,
    InteractionParameters,
    InteractionType,
    Particle,
    ParticleInteractionCalculator,
    ParticleType,
    SimulationBox,
    SimulationMethod,
    SimulationParameters,
    create_amphiphilic_system,
    run_dpd_simulation,
)

# Optional plotting dependencies
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available - skipping visualization")

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SelfAssemblyDemo:
    """Comprehensive demonstration of self-assembly simulation and characterization."""

    def __init__(
        self, use_gpu: bool = True, verbose: bool = True, save_output: bool = True
    ):
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.save_output = save_output
        self.results = {}
        self.output_dir = None

        # Check GPU availability
        try:
            import cupy as cp

            if use_gpu:
                cp.cuda.Device(0).use()
                logger.info("GPU acceleration enabled")
            else:
                logger.info("Using CPU computation")
        except ImportError:
            self.use_gpu = False
            logger.warning("CuPy not available - using CPU computation")

        if save_output:
            self.output_dir = Path("self_assembly_demo_output")
            self.output_dir.mkdir(exist_ok=True)
            logger.info(f"Output will be saved to {self.output_dir}")

    def demonstrate_all(self):
        """Run all demonstration scenarios."""
        logger.info("Starting comprehensive self-assembly simulation demonstration")

        scenarios = [
            ("micelle_formation", self.demonstrate_micelle_formation),
            ("vesicle_formation", self.demonstrate_vesicle_formation),
            ("liquid_crystal_phase", self.demonstrate_liquid_crystal_phase),
            ("multi_component_system", self.demonstrate_multi_component_system),
            ("performance_comparison", self.demonstrate_performance_comparison),
        ]

        for scenario_name, scenario_func in scenarios:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running scenario: {scenario_name}")
            logger.info(f"{'='*60}")

            try:
                start_time = time.time()
                result = scenario_func()
                end_time = time.time()

                self.results[scenario_name] = {
                    "result": result,
                    "execution_time": end_time - start_time,
                    "success": True,
                }

                logger.info(
                    f"Scenario {scenario_name} completed in {end_time - start_time:.2f} seconds"
                )

            except Exception as e:
                logger.error(f"Error in scenario {scenario_name}: {e}")
                self.results[scenario_name] = {
                    "error": str(e),
                    "execution_time": 0,
                    "success": False,
                }

        self._generate_summary_report()
        logger.info("All demonstrations completed!")

    def demonstrate_micelle_formation(self) -> Dict[str, Any]:
        """Demonstrate micelle formation from amphiphilic molecules."""
        logger.info("Simulating micelle formation from amphiphilic molecules")

        # Create amphiphilic system
        particles, simulation_box, interaction_params = create_amphiphilic_system(
            n_molecules=50, head_size=1, tail_size=6, box_size=20.0
        )

        logger.info(f"Created system with {len(particles)} particles")
        logger.info(f"Box dimensions: {simulation_box.dimensions}")

        # Setup simulation parameters
        simulation_params = SimulationParameters(
            method=SimulationMethod.DISSIPATIVE_PARTICLE_DYNAMICS,
            integrator=IntegratorType.DPD_INTEGRATOR,
            timestep=0.01,
            total_time=10.0,  # 10 time units
            temperature=298.15,
            cutoff_radius=2.5,
            use_gpu=self.use_gpu,
            output_frequency=50,
        )

        # Run simulation
        logger.info("Running DPD simulation for micelle formation")
        engine = CoarseGrainedSimulationEngine(
            particles, simulation_box, simulation_params, interaction_params
        )

        trajectory = engine.run_simulation()
        logger.info(f"Simulation completed with {len(trajectory)} frames")

        # Characterize trajectory
        logger.info("Characterizing self-assembly structures")
        particle_types = [p.particle_type.name for p in particles]

        characterization_results = characterize_trajectory(
            trajectory,
            particle_types,
            use_gpu=self.use_gpu,
            clustering_method=ClusteringMethod.DENSITY_BASED,
        )

        # Analyze micelle formation
        micelle_analysis = analyze_micelle_formation(trajectory, particle_types)

        # Extract key metrics
        final_result = characterization_results[-1]
        n_micelles = len(final_result.micelles)
        avg_aggregation_number = (
            np.mean([m.aggregation_number for m in final_result.micelles])
            if final_result.micelles
            else 0
        )

        logger.info(f"Final analysis: {n_micelles} micelles detected")
        logger.info(f"Average aggregation number: {avg_aggregation_number:.1f}")

        # Visualization
        if MATPLOTLIB_AVAILABLE and self.save_output:
            self._plot_micelle_formation(
                trajectory, micelle_analysis, "micelle_formation"
            )

        return {
            "trajectory": trajectory,
            "characterization": characterization_results,
            "micelle_analysis": micelle_analysis,
            "final_micelles": n_micelles,
            "avg_aggregation_number": avg_aggregation_number,
            "system_info": {
                "n_particles": len(particles),
                "simulation_time": simulation_params.total_time,
                "box_size": simulation_box.dimensions.tolist(),
            },
        }

    def demonstrate_vesicle_formation(self) -> Dict[str, Any]:
        """Demonstrate vesicle formation with different amphiphile parameters."""
        logger.info("Simulating vesicle formation with modified interaction parameters")

        # Create system with parameters favoring vesicle formation
        particles, simulation_box, interaction_params = create_amphiphilic_system(
            n_molecules=80,
            head_size=2,  # Larger head groups
            tail_size=4,  # Shorter tails
            box_size=25.0,
        )

        # Modify interactions to favor vesicle formation
        # Weaker head-head repulsion, stronger tail-water repulsion
        interaction_params.add_interaction(
            "HEAD", "HEAD", InteractionType.DPD_CONSERVATIVE, {"a": 20.0}
        )
        interaction_params.add_interaction(
            "TAIL", "WATER", InteractionType.DPD_CONSERVATIVE, {"a": 100.0}
        )

        logger.info(f"Created vesicle-prone system with {len(particles)} particles")

        # Simulation parameters
        simulation_params = SimulationParameters(
            method=SimulationMethod.DISSIPATIVE_PARTICLE_DYNAMICS,
            timestep=0.01,
            total_time=15.0,  # Longer simulation for vesicle formation
            temperature=310.15,  # Higher temperature
            use_gpu=self.use_gpu,
            output_frequency=75,
        )

        # Run simulation
        logger.info("Running extended DPD simulation for vesicle formation")
        trajectory = run_dpd_simulation(
            particles,
            simulation_box,
            interaction_params,
            total_time=simulation_params.total_time,
            timestep=simulation_params.timestep,
            temperature=simulation_params.temperature,
            use_gpu=self.use_gpu,
        )

        # Characterize structures
        particle_types = [p.particle_type.name for p in particles]
        characterization_results = characterize_trajectory(
            trajectory, particle_types, use_gpu=self.use_gpu
        )

        # Analyze vesicle formation
        final_result = characterization_results[-1]
        n_vesicles = len(final_result.vesicles)
        vesicle_structures = [
            s
            for s in final_result.structures
            if s.structure_type == StructureType.VESICLE
        ]

        logger.info(f"Final analysis: {n_vesicles} vesicles detected")
        logger.info(f"Vesicle-type structures: {len(vesicle_structures)}")

        # Analyze structure evolution
        structure_evolution = self._analyze_structure_evolution(
            characterization_results
        )

        if MATPLOTLIB_AVAILABLE and self.save_output:
            self._plot_vesicle_formation(
                trajectory, characterization_results, "vesicle_formation"
            )

        return {
            "trajectory": trajectory,
            "characterization": characterization_results,
            "final_vesicles": n_vesicles,
            "structure_evolution": structure_evolution,
            "vesicle_characteristics": final_result.vesicles,
            "system_info": {
                "n_particles": len(particles),
                "simulation_time": simulation_params.total_time,
                "modified_interactions": True,
            },
        }

    def demonstrate_liquid_crystal_phase(self) -> Dict[str, Any]:
        """Demonstrate liquid crystal phase formation."""
        logger.info("Simulating liquid crystal phase formation")

        # Create system with rod-like particles
        particle_types = []
        particles = []

        # Create rod-like particle type
        rod_type = ParticleType(
            name="ROD", mass=2.0, radius=0.3, epsilon=1.5, sigma=3.0  # Anisotropic
        )

        # Create ordered initial configuration
        n_rods = 200
        box_size = 15.0

        for i in range(n_rods):
            # Random position
            position = np.random.uniform(0, box_size, 3)

            # Initially random velocity
            velocity = np.random.normal(0, 1, 3)

            particle = Particle(
                particle_id=i,
                particle_type=rod_type,
                position=position,
                velocity=velocity,
            )
            particles.append(particle)

        # Simulation box
        simulation_box = SimulationBox(
            dimensions=np.array([box_size, box_size, box_size]),
            boundary_conditions=[BoundaryCondition.PERIODIC] * 3,
        )

        # Interaction parameters favoring alignment
        interaction_params = InteractionParameters()
        interaction_params.add_interaction(
            "ROD", "ROD", InteractionType.LENNARD_JONES, {"epsilon": 2.0, "sigma": 2.0}
        )

        # Simulation parameters
        simulation_params = SimulationParameters(
            method=SimulationMethod.COARSE_GRAINED_MD,
            integrator=IntegratorType.VELOCITY_VERLET,
            timestep=0.005,
            total_time=8.0,
            temperature=250.0,  # Lower temperature for ordering
            use_gpu=self.use_gpu,
            output_frequency=80,
        )

        logger.info(f"Created liquid crystal system with {n_rods} rod-like particles")

        # Run simulation
        engine = CoarseGrainedSimulationEngine(
            particles, simulation_box, simulation_params, interaction_params
        )
        trajectory = engine.run_simulation()

        # Characterize liquid crystal phase
        particle_types = ["ROD"] * n_rods
        characterization_results = characterize_trajectory(
            trajectory, particle_types, use_gpu=self.use_gpu
        )

        # Analyze phase behavior
        final_result = characterization_results[-1]
        phase_type = final_result.phase_analysis.phase_type
        order_parameters = final_result.phase_analysis.order_parameters

        logger.info(f"Final phase: {phase_type}")
        logger.info(f"Order parameters: {order_parameters}")

        # Track order parameter evolution
        order_evolution = [
            result.phase_analysis.order_parameters.get("translational", 0)
            for result in characterization_results
        ]

        if MATPLOTLIB_AVAILABLE and self.save_output:
            self._plot_liquid_crystal_analysis(
                trajectory, characterization_results, "liquid_crystal"
            )

        return {
            "trajectory": trajectory,
            "characterization": characterization_results,
            "final_phase": phase_type,
            "order_parameters": order_parameters,
            "order_evolution": order_evolution,
            "liquid_crystals": final_result.liquid_crystals,
            "system_info": {
                "n_particles": n_rods,
                "particle_type": "rod-like",
                "temperature": simulation_params.temperature,
            },
        }

    def demonstrate_multi_component_system(self) -> Dict[str, Any]:
        """Demonstrate complex multi-component self-assembly."""
        logger.info("Simulating multi-component self-assembly system")

        # Create complex system with multiple molecule types
        particles = []
        particle_id = 0
        box_size = 30.0

        # Type 1: Amphiphilic molecules (surfactant)
        surfactant_head = ParticleType(
            name="SURF_HEAD", mass=1.0, radius=0.6, hydrophobic=False
        )
        surfactant_tail = ParticleType(
            name="SURF_TAIL", mass=1.0, radius=0.5, hydrophobic=True
        )

        n_surfactants = 40
        for i in range(n_surfactants):
            # Head
            position = np.random.uniform(0, box_size, 3)
            velocity = np.random.normal(0, 1, 3)
            head_particle = Particle(particle_id, surfactant_head, position, velocity)
            particles.append(head_particle)
            head_id = particle_id
            particle_id += 1

            # Tail chain
            for j in range(5):
                tail_position = (
                    position
                    + (j + 1) * np.array([0.5, 0, 0])
                    + np.random.normal(0, 0.1, 3)
                )
                tail_velocity = np.random.normal(0, 1, 3)
                tail_particle = Particle(
                    particle_id, surfactant_tail, tail_position, tail_velocity
                )
                particles.append(tail_particle)
                particle_id += 1

        # Type 2: Block copolymer
        block_a = ParticleType(name="BLOCK_A", mass=1.2, radius=0.5, hydrophobic=False)
        block_b = ParticleType(name="BLOCK_B", mass=1.2, radius=0.5, hydrophobic=True)

        n_copolymers = 30
        for i in range(n_copolymers):
            center = np.random.uniform(0, box_size, 3)

            # A block
            for j in range(4):
                position = (
                    center + j * np.array([0.4, 0, 0]) + np.random.normal(0, 0.05, 3)
                )
                velocity = np.random.normal(0, 1, 3)
                particle = Particle(particle_id, block_a, position, velocity)
                particles.append(particle)
                particle_id += 1

            # B block
            for j in range(6):
                position = (
                    center
                    + (j + 4) * np.array([0.4, 0, 0])
                    + np.random.normal(0, 0.05, 3)
                )
                velocity = np.random.normal(0, 1, 3)
                particle = Particle(particle_id, block_b, position, velocity)
                particles.append(particle)
                particle_id += 1

        # Type 3: Water/solvent
        water_type = ParticleType(name="WATER", mass=0.8, radius=0.3, hydrophobic=False)
        n_water = 300
        for i in range(n_water):
            position = np.random.uniform(0, box_size, 3)
            velocity = np.random.normal(0, 1, 3)
            particle = Particle(particle_id, water_type, position, velocity)
            particles.append(particle)
            particle_id += 1

        logger.info(f"Created multi-component system with {len(particles)} particles")
        logger.info(f"  - {n_surfactants} surfactant molecules")
        logger.info(f"  - {n_copolymers} block copolymers")
        logger.info(f"  - {n_water} water molecules")

        # Simulation box
        simulation_box = SimulationBox(
            dimensions=np.array([box_size, box_size, box_size]),
            boundary_conditions=[BoundaryCondition.PERIODIC] * 3,
        )

        # Complex interaction matrix
        interaction_params = InteractionParameters()

        # Surfactant interactions
        interaction_params.add_interaction(
            "SURF_HEAD", "SURF_HEAD", InteractionType.DPD_CONSERVATIVE, {"a": 25.0}
        )
        interaction_params.add_interaction(
            "SURF_HEAD", "SURF_TAIL", InteractionType.DPD_CONSERVATIVE, {"a": 35.0}
        )
        interaction_params.add_interaction(
            "SURF_TAIL", "SURF_TAIL", InteractionType.DPD_CONSERVATIVE, {"a": 25.0}
        )
        interaction_params.add_interaction(
            "SURF_HEAD", "WATER", InteractionType.DPD_CONSERVATIVE, {"a": 25.0}
        )
        interaction_params.add_interaction(
            "SURF_TAIL", "WATER", InteractionType.DPD_CONSERVATIVE, {"a": 80.0}
        )

        # Block copolymer interactions
        interaction_params.add_interaction(
            "BLOCK_A", "BLOCK_A", InteractionType.DPD_CONSERVATIVE, {"a": 25.0}
        )
        interaction_params.add_interaction(
            "BLOCK_B", "BLOCK_B", InteractionType.DPD_CONSERVATIVE, {"a": 25.0}
        )
        interaction_params.add_interaction(
            "BLOCK_A", "BLOCK_B", InteractionType.DPD_CONSERVATIVE, {"a": 40.0}
        )
        interaction_params.add_interaction(
            "BLOCK_A", "WATER", InteractionType.DPD_CONSERVATIVE, {"a": 30.0}
        )
        interaction_params.add_interaction(
            "BLOCK_B", "WATER", InteractionType.DPD_CONSERVATIVE, {"a": 75.0}
        )

        # Cross interactions
        interaction_params.add_interaction(
            "SURF_HEAD", "BLOCK_A", InteractionType.DPD_CONSERVATIVE, {"a": 30.0}
        )
        interaction_params.add_interaction(
            "SURF_HEAD", "BLOCK_B", InteractionType.DPD_CONSERVATIVE, {"a": 45.0}
        )
        interaction_params.add_interaction(
            "SURF_TAIL", "BLOCK_A", InteractionType.DPD_CONSERVATIVE, {"a": 50.0}
        )
        interaction_params.add_interaction(
            "SURF_TAIL", "BLOCK_B", InteractionType.DPD_CONSERVATIVE, {"a": 30.0}
        )

        # Water interactions
        interaction_params.add_interaction(
            "WATER", "WATER", InteractionType.DPD_CONSERVATIVE, {"a": 25.0}
        )

        # Simulation parameters
        simulation_params = SimulationParameters(
            method=SimulationMethod.DISSIPATIVE_PARTICLE_DYNAMICS,
            timestep=0.01,
            total_time=20.0,  # Longer simulation for complex assembly
            temperature=298.15,
            use_gpu=self.use_gpu,
            output_frequency=100,
        )

        # Run simulation
        logger.info("Running multi-component self-assembly simulation")
        engine = CoarseGrainedSimulationEngine(
            particles, simulation_box, simulation_params, interaction_params
        )
        trajectory = engine.run_simulation()

        # Comprehensive characterization
        particle_types = [p.particle_type.name for p in particles]
        characterization_results = characterize_trajectory(
            trajectory,
            particle_types,
            use_gpu=self.use_gpu,
            clustering_method=ClusteringMethod.DENSITY_BASED,
        )

        # Analyze different structure types
        final_result = characterization_results[-1]
        structure_summary = self._analyze_structure_diversity(final_result)

        logger.info("Multi-component assembly analysis:")
        for structure_type, count in structure_summary.items():
            logger.info(f"  - {structure_type}: {count}")

        if MATPLOTLIB_AVAILABLE and self.save_output:
            self._plot_multi_component_system(
                trajectory, characterization_results, "multi_component"
            )

        return {
            "trajectory": trajectory,
            "characterization": characterization_results,
            "structure_summary": structure_summary,
            "final_structures": len(final_result.structures),
            "system_composition": {
                "surfactants": n_surfactants,
                "copolymers": n_copolymers,
                "water_molecules": n_water,
                "total_particles": len(particles),
            },
        }

    def demonstrate_performance_comparison(self) -> Dict[str, Any]:
        """Demonstrate performance comparison between CPU and GPU implementations."""
        logger.info(
            "Running performance comparison between CPU and GPU implementations"
        )

        # Create test system
        particles, simulation_box, interaction_params = create_amphiphilic_system(
            n_molecules=100, head_size=1, tail_size=4, box_size=20.0
        )

        # Short simulation parameters for benchmarking
        simulation_params = SimulationParameters(
            method=SimulationMethod.DISSIPATIVE_PARTICLE_DYNAMICS,
            timestep=0.01,
            total_time=1.0,  # Short simulation for benchmarking
            temperature=298.15,
            output_frequency=50,
        )

        results = {}

        # CPU benchmark
        logger.info("Running CPU benchmark")
        simulation_params.use_gpu = False
        start_time = time.time()

        engine_cpu = CoarseGrainedSimulationEngine(
            particles, simulation_box, simulation_params, interaction_params
        )
        trajectory_cpu = engine_cpu.run_simulation()

        cpu_time = time.time() - start_time
        logger.info(f"CPU simulation completed in {cpu_time:.2f} seconds")

        results["cpu"] = {
            "execution_time": cpu_time,
            "n_frames": len(trajectory_cpu),
            "final_energy": trajectory_cpu[-1].energies["total"],
        }

        # GPU benchmark (if available)
        if self.use_gpu:
            logger.info("Running GPU benchmark")
            simulation_params.use_gpu = True
            start_time = time.time()

            try:
                engine_gpu = CoarseGrainedSimulationEngine(
                    particles, simulation_box, simulation_params, interaction_params
                )
                trajectory_gpu = engine_gpu.run_simulation()

                gpu_time = time.time() - start_time
                logger.info(f"GPU simulation completed in {gpu_time:.2f} seconds")

                speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0
                logger.info(f"GPU speedup: {speedup:.2f}x")

                results["gpu"] = {
                    "execution_time": gpu_time,
                    "n_frames": len(trajectory_gpu),
                    "final_energy": trajectory_gpu[-1].energies["total"],
                    "speedup": speedup,
                }

            except Exception as e:
                logger.warning(f"GPU benchmark failed: {e}")
                results["gpu"] = {"error": str(e)}
        else:
            logger.info("GPU not available - skipping GPU benchmark")
            results["gpu"] = {"error": "GPU not available"}

        # Characterization performance comparison
        logger.info("Comparing characterization performance")
        particle_types = [p.particle_type.name for p in particles]

        # CPU characterization
        start_time = time.time()
        char_results_cpu = characterize_trajectory(
            trajectory_cpu, particle_types, use_gpu=False
        )
        cpu_char_time = time.time() - start_time

        results["characterization"] = {
            "cpu_time": cpu_char_time,
            "n_structures_detected": len(char_results_cpu[-1].structures),
        }

        if self.use_gpu:
            try:
                start_time = time.time()
                char_results_gpu = characterize_trajectory(
                    trajectory_cpu, particle_types, use_gpu=True
                )
                gpu_char_time = time.time() - start_time

                char_speedup = (
                    cpu_char_time / gpu_char_time if gpu_char_time > 0 else 1.0
                )

                results["characterization"]["gpu_time"] = gpu_char_time
                results["characterization"]["speedup"] = char_speedup

            except Exception as e:
                logger.warning(f"GPU characterization failed: {e}")

        logger.info("Performance summary:")
        logger.info(f"  CPU simulation: {results['cpu']['execution_time']:.2f}s")
        if "execution_time" in results.get("gpu", {}):
            logger.info(f"  GPU simulation: {results['gpu']['execution_time']:.2f}s")
            logger.info(f"  Speedup: {results['gpu']['speedup']:.2f}x")

        return results

    def _analyze_structure_evolution(
        self, characterization_results: List
    ) -> Dict[str, List]:
        """Analyze how structures evolve over time."""
        evolution = {
            "micelles": [],
            "vesicles": [],
            "clusters": [],
            "total_structures": [],
        }

        for result in characterization_results:
            micelle_count = len(result.micelles)
            vesicle_count = len(result.vesicles)
            cluster_count = len(
                [
                    s
                    for s in result.structures
                    if s.structure_type == StructureType.CLUSTER
                ]
            )
            total_count = len(result.structures)

            evolution["micelles"].append(micelle_count)
            evolution["vesicles"].append(vesicle_count)
            evolution["clusters"].append(cluster_count)
            evolution["total_structures"].append(total_count)

        return evolution

    def _analyze_structure_diversity(self, final_result) -> Dict[str, int]:
        """Analyze diversity of structures in final configuration."""
        structure_counts = {}

        for structure in final_result.structures:
            structure_type = structure.structure_type.value
            structure_counts[structure_type] = (
                structure_counts.get(structure_type, 0) + 1
            )

        return structure_counts

    def _plot_micelle_formation(self, trajectory, micelle_analysis, filename_prefix):
        """Plot micelle formation analysis."""
        if not MATPLOTLIB_AVAILABLE:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Micelle count over time
        ax1.plot(micelle_analysis["times"], micelle_analysis["micelle_counts"], "bo-")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Number of Micelles")
        ax1.set_title("Micelle Formation")
        ax1.grid(True)

        # Aggregation number evolution
        ax2.plot(
            micelle_analysis["times"], micelle_analysis["aggregation_numbers"], "ro-"
        )
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Average Aggregation Number")
        ax2.set_title("Aggregation Number Evolution")
        ax2.grid(True)

        # Energy evolution
        times = [frame.time for frame in trajectory]
        energies = [frame.energies["total"] for frame in trajectory]
        ax3.plot(times, energies, "go-")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Total Energy")
        ax3.set_title("Energy Conservation")
        ax3.grid(True)

        # Final structure snapshot (2D projection)
        final_frame = trajectory[-1]
        positions = final_frame.positions
        ax4.scatter(positions[:, 0], positions[:, 1], alpha=0.6, s=20)
        ax4.set_xlabel("X Position")
        ax4.set_ylabel("Y Position")
        ax4.set_title("Final Configuration (XY View)")
        ax4.set_aspect("equal")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"{filename_prefix}_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        logger.info(f"Saved micelle formation plot as {filename_prefix}_analysis.png")

    def _plot_vesicle_formation(
        self, trajectory, characterization_results, filename_prefix
    ):
        """Plot vesicle formation analysis."""
        if not MATPLOTLIB_AVAILABLE:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Structure evolution
        times = [result.time for result in characterization_results]
        vesicle_counts = [len(result.vesicles) for result in characterization_results]
        total_structures = [
            len(result.structures) for result in characterization_results
        ]

        ax1.plot(times, vesicle_counts, "bo-", label="Vesicles")
        ax1.plot(times, total_structures, "ro-", label="Total Structures")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Count")
        ax1.set_title("Structure Formation")
        ax1.legend()
        ax1.grid(True)

        # Order parameter evolution
        order_params = [
            result.phase_analysis.order_parameters.get("translational", 0)
            for result in characterization_results
        ]
        ax2.plot(times, order_params, "go-")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Order Parameter")
        ax2.set_title("Structural Ordering")
        ax2.grid(True)

        # 3D structure visualization (final frame)
        final_positions = trajectory[-1].positions
        ax3 = fig.add_subplot(223, projection="3d")
        ax3.scatter(
            final_positions[:, 0],
            final_positions[:, 1],
            final_positions[:, 2],
            alpha=0.6,
            s=10,
        )
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        ax3.set_zlabel("Z")
        ax3.set_title("Final 3D Configuration")

        # Radial distribution (center of mass analysis)
        if characterization_results[-1].structures:
            structure = characterization_results[-1].structures[0]
            com = structure.center_of_mass
            distances = np.linalg.norm(final_positions - com, axis=1)
            ax4.hist(distances, bins=20, alpha=0.7, edgecolor="black")
            ax4.set_xlabel("Distance from COM")
            ax4.set_ylabel("Particle Count")
            ax4.set_title("Radial Distribution")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"{filename_prefix}_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        logger.info(f"Saved vesicle formation plot as {filename_prefix}_analysis.png")

    def _plot_liquid_crystal_analysis(
        self, trajectory, characterization_results, filename_prefix
    ):
        """Plot liquid crystal phase analysis."""
        if not MATPLOTLIB_AVAILABLE:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Order parameter evolution
        times = [result.time for result in characterization_results]
        order_params = [
            result.phase_analysis.order_parameters.get("translational", 0)
            for result in characterization_results
        ]

        ax1.plot(times, order_params, "bo-")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Translational Order Parameter")
        ax1.set_title("Order Parameter Evolution")
        ax1.grid(True)

        # Phase classification over time
        phases = [
            result.phase_analysis.phase_type.value
            for result in characterization_results
        ]
        unique_phases = list(set(phases))
        phase_counts = {
            phase: [phases[: i + 1].count(phase) for i in range(len(phases))]
            for phase in unique_phases
        }

        for phase, counts in phase_counts.items():
            ax2.plot(times, counts, label=phase, marker="o")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Cumulative Phase Count")
        ax2.set_title("Phase Evolution")
        ax2.legend()
        ax2.grid(True)

        # Particle distribution (XY plane)
        final_positions = trajectory[-1].positions
        ax3.scatter(final_positions[:, 0], final_positions[:, 1], alpha=0.6, s=20)
        ax3.set_xlabel("X Position")
        ax3.set_ylabel("Y Position")
        ax3.set_title("Final Configuration (XY View)")
        ax3.set_aspect("equal")

        # Energy evolution
        trajectory_times = [frame.time for frame in trajectory]
        energies = [frame.energies["total"] for frame in trajectory]
        ax4.plot(trajectory_times, energies, "go-")
        ax4.set_xlabel("Time")
        ax4.set_ylabel("Total Energy")
        ax4.set_title("Energy Evolution")
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"{filename_prefix}_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        logger.info(
            f"Saved liquid crystal analysis plot as {filename_prefix}_analysis.png"
        )

    def _plot_multi_component_system(
        self, trajectory, characterization_results, filename_prefix
    ):
        """Plot multi-component system analysis."""
        if not MATPLOTLIB_AVAILABLE:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Structure diversity over time
        times = [result.time for result in characterization_results]
        structure_counts = []

        for result in characterization_results:
            diversity = {}
            for structure in result.structures:
                stype = structure.structure_type.value
                diversity[stype] = diversity.get(stype, 0) + 1
            structure_counts.append(diversity)

        # Plot different structure types
        all_types = set()
        for counts in structure_counts:
            all_types.update(counts.keys())

        for stype in all_types:
            counts = [sc.get(stype, 0) for sc in structure_counts]
            ax1.plot(times, counts, label=stype, marker="o")

        ax1.set_xlabel("Time")
        ax1.set_ylabel("Structure Count")
        ax1.set_title("Structure Diversity Evolution")
        ax1.legend()
        ax1.grid(True)

        # Total structures and clustering metrics
        total_structures = [
            len(result.structures) for result in characterization_results
        ]
        clustered_fractions = [
            result.global_metrics["clustered_fraction"]
            for result in characterization_results
        ]

        ax2.plot(times, total_structures, "bo-", label="Total Structures")
        ax2_twin = ax2.twinx()
        ax2_twin.plot(times, clustered_fractions, "ro-", label="Clustered Fraction")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Total Structures", color="blue")
        ax2_twin.set_ylabel("Clustered Fraction", color="red")
        ax2.set_title("Assembly Metrics")
        ax2.grid(True)

        # Final configuration with particle types
        final_positions = trajectory[-1].positions
        # Color-code by position (simplified)
        colors = final_positions[:, 2]  # Use Z-coordinate for coloring
        scatter = ax3.scatter(
            final_positions[:, 0],
            final_positions[:, 1],
            c=colors,
            alpha=0.6,
            s=15,
            cmap="viridis",
        )
        ax3.set_xlabel("X Position")
        ax3.set_ylabel("Y Position")
        ax3.set_title("Final Configuration (Colored by Z)")
        plt.colorbar(scatter, ax=ax3)

        # System energy evolution
        trajectory_times = [frame.time for frame in trajectory]
        kinetic_energies = [frame.energies["kinetic"] for frame in trajectory]
        potential_energies = [frame.energies["potential"] for frame in trajectory]

        ax4.plot(trajectory_times, kinetic_energies, "r-", label="Kinetic")
        ax4.plot(trajectory_times, potential_energies, "b-", label="Potential")
        ax4.set_xlabel("Time")
        ax4.set_ylabel("Energy")
        ax4.set_title("Energy Components")
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"{filename_prefix}_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        logger.info(
            f"Saved multi-component system plot as {filename_prefix}_analysis.png"
        )

    def _generate_summary_report(self):
        """Generate a comprehensive summary report."""
        if not self.save_output:
            return

        report_path = self.output_dir / "summary_report.txt"

        with open(report_path, "w") as f:
            f.write("Self-Assembly Simulation Demonstration Summary Report\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                f"GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}\n\n"
            )

            # Overall statistics
            total_scenarios = len(self.results)
            successful_scenarios = sum(1 for r in self.results.values() if r["success"])
            total_time = sum(r["execution_time"] for r in self.results.values())

            f.write("Overall Statistics:\n")
            f.write(f"  Total scenarios: {total_scenarios}\n")
            f.write(f"  Successful scenarios: {successful_scenarios}\n")
            f.write(f"  Total execution time: {total_time:.2f} seconds\n\n")

            # Scenario details
            for scenario_name, result in self.results.items():
                f.write(f"Scenario: {scenario_name}\n")
                f.write("-" * 40 + "\n")

                if result["success"]:
                    f.write("Status: SUCCESS\n")
                    f.write(f"Execution time: {result['execution_time']:.2f} seconds\n")

                    # Scenario-specific details
                    if scenario_name == "micelle_formation":
                        data = result["result"]
                        f.write(f"Final micelles: {data['final_micelles']}\n")
                        f.write(
                            f"Average aggregation number: {data['avg_aggregation_number']:.1f}\n"
                        )
                        f.write(
                            f"System particles: {data['system_info']['n_particles']}\n"
                        )

                    elif scenario_name == "vesicle_formation":
                        data = result["result"]
                        f.write(f"Final vesicles: {data['final_vesicles']}\n")
                        f.write(f"Total structures: {data['final_structures']}\n")

                    elif scenario_name == "liquid_crystal_phase":
                        data = result["result"]
                        f.write(f"Final phase: {data['final_phase']}\n")
                        f.write(f"Order parameters: {data['order_parameters']}\n")

                    elif scenario_name == "multi_component_system":
                        data = result["result"]
                        f.write(f"Structure diversity: {data['structure_summary']}\n")
                        f.write(f"System composition: {data['system_composition']}\n")

                    elif scenario_name == "performance_comparison":
                        data = result["result"]
                        f.write(f"CPU time: {data['cpu']['execution_time']:.2f}s\n")
                        if "execution_time" in data.get("gpu", {}):
                            f.write(f"GPU time: {data['gpu']['execution_time']:.2f}s\n")
                            f.write(f"Speedup: {data['gpu']['speedup']:.2f}x\n")

                else:
                    f.write("Status: FAILED\n")
                    f.write(f"Error: {result['error']}\n")

                f.write("\n")

            f.write("End of Report\n")

        logger.info(f"Summary report saved to {report_path}")


def main():
    """Main demonstration function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Self-Assembly Simulation Demonstration"
    )
    parser.add_argument(
        "--no-gpu", action="store_true", help="Disable GPU acceleration"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output")
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save output files"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=[
            "micelle",
            "vesicle",
            "liquid_crystal",
            "multi_component",
            "performance",
            "all",
        ],
        default="all",
        help="Run specific scenario",
    )

    args = parser.parse_args()

    # Configure logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Create demo instance
    demo = SelfAssemblyDemo(
        use_gpu=not args.no_gpu, verbose=not args.quiet, save_output=not args.no_save
    )

    # Run demonstration
    if args.scenario == "all":
        demo.demonstrate_all()
    elif args.scenario == "micelle":
        demo.demonstrate_micelle_formation()
    elif args.scenario == "vesicle":
        demo.demonstrate_vesicle_formation()
    elif args.scenario == "liquid_crystal":
        demo.demonstrate_liquid_crystal_phase()
    elif args.scenario == "multi_component":
        demo.demonstrate_multi_component_system()
    elif args.scenario == "performance":
        demo.demonstrate_performance_comparison()


if __name__ == "__main__":
    main()
