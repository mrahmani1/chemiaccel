#!/usr/bin/env python3
"""
Multi-Phase Transport & Bioavailability Optimization Demo

This demo showcases the AI-driven multi-phase transport and bioavailability
optimization capabilities of ChemiAccel, including:

1. GPU-accelerated multi-phase simulation
2. Transport kinetics and partitioning analysis
3. Inverse design for optimal transport properties
4. Active transport and metabolism modeling
5. Interactive visualization of transport pathways

Example usage:
    python examples/multi_phase_transport_demo.py
"""

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chemi_accel.transport import (
    BiologicalTransportModel,
    MultiPhaseSimulator,
    TransportAnalyzer,
    TransportOptimizer,
    TransportVisualizer,
)
from chemi_accel.transport.biological_transport import (
    BiologicalSystemConfig,
    EnzymeProperties,
    TransporterProperties,
)
from chemi_accel.transport.simulation_engine import SimulationConfig, TransportResult
from chemi_accel.transport.transport_optimizer import TransportTarget
from chemi_accel.transport.transport_visualizer import VisualizationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demo_multi_phase_simulation():
    """Demonstrate GPU-accelerated multi-phase simulation."""
    logger.info("=== Multi-Phase Simulation Demo ===")

    # Initialize simulator with AI surrogate method for speed
    simulator = MultiPhaseSimulator(backend="auto", method="ai_surrogate")

    # Test molecules with different transport properties
    test_molecules = [
        "CCO",  # Ethanol - small, polar
        "CC(C)C",  # Isobutane - small, nonpolar
        "CC(=O)Nc1ccc(O)cc1",  # Acetaminophen - drug-like
        "CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen - lipophilic drug
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine - ampiphilic
    ]

    molecule_names = ["Ethanol", "Isobutane", "Acetaminophen", "Ibuprofen", "Caffeine"]

    print("\n1. Membrane Transport Simulation")
    print("-" * 50)

    results = {}
    for mol_smiles, mol_name in zip(test_molecules, molecule_names):
        print(f"\nSimulating {mol_name}...")

        # Simulate membrane transport
        result = simulator.simulate_membrane_transport(
            molecules=[mol_smiles],
            membrane_type="dppc_bilayer",
            conditions={"temperature": 310.0, "pH": 7.4},
            n_steps=50000,
            output_frequency=500,
        )

        results[mol_name] = result

        print(f"  Permeability: {result.permeability_coefficient:.2e} cm/s")
        print(f"  Simulation time: {result.simulation_time:.2f} seconds")

        # Display diffusion coefficients
        for phase, diff_coeff in result.diffusion_coefficients.items():
            print(f"  {phase} diffusion: {diff_coeff:.2e} cm²/s")

    print(f"\nCompleted simulations for {len(test_molecules)} molecules")

    # Demonstrate liquid-liquid extraction
    print("\n2. Liquid-Liquid Extraction Simulation")
    print("-" * 50)

    extraction_result = simulator.simulate_liquid_liquid_extraction(
        molecules=["CC(=O)Nc1ccc(O)cc1"],  # Acetaminophen
        solvent_system=("water", "octanol"),
        conditions={"temperature": 298.15},
    )

    print(
        f"Extraction permeability: {extraction_result.permeability_coefficient:.2e} cm/s"
    )
    print(
        f"Water-octanol partition: {extraction_result.partition_coefficients.get('water_octanol', 'N/A')}"
    )

    return results


def demo_transport_analysis():
    """Demonstrate transport kinetics and partitioning analysis."""
    logger.info("=== Transport Analysis Demo ===")

    # Initialize analyzer
    analyzer = TransportAnalyzer(backend="auto")

    # Run a simulation to analyze
    simulator = MultiPhaseSimulator(backend="auto", method="ai_surrogate")

    print("\n3. Transport Kinetics Analysis")
    print("-" * 50)

    # Simulate ibuprofen transport for detailed analysis
    ibuprofen_result = simulator.simulate_membrane_transport(
        molecules=["CC1=CC=C(C=C1)C(C)C(=O)O"],
        membrane_type="dppc_bilayer",
        conditions={"temperature": 310.0, "pH": 7.4},
        n_steps=100000,
        output_frequency=1000,
    )

    # Analyze permeability
    permeability_analysis = analyzer.analyze_permeability(ibuprofen_result)
    print("\nPermeability Analysis:")
    for key, value in permeability_analysis.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.3e}")
        else:
            print(f"  {key}: {value}")

    # Analyze diffusion
    diffusion_analysis = analyzer.analyze_diffusion(ibuprofen_result)
    print("\nDiffusion Analysis:")
    if "diffusion_coefficients" in diffusion_analysis:
        for phase, coeff in diffusion_analysis["diffusion_coefficients"].items():
            if isinstance(coeff, (int, float)):
                print(f"  {phase}: {coeff:.3e}")

    # Analyze partitioning
    partitioning_analysis = analyzer.analyze_partitioning(ibuprofen_result)
    print("\nPartitioning Analysis:")
    print(f"  Phase behavior: {partitioning_analysis.phase_behavior_classification}")
    print(
        f"  Equilibrium constants: {len(partitioning_analysis.equilibrium_constants)}"
    )

    # Generate comprehensive report
    print("\n4. Comprehensive Transport Report")
    print("-" * 50)

    comprehensive_report = analyzer.generate_transport_report(ibuprofen_result)

    print("\nKey Findings:")
    for recommendation in comprehensive_report.get("recommendations", []):
        print(f"  • {recommendation}")

    return comprehensive_report


def demo_transport_optimization():
    """Demonstrate inverse design for optimal transport properties."""
    logger.info("=== Transport Optimization Demo ===")

    print("\n5. Transport Property Optimization")
    print("-" * 50)

    # Initialize optimizer
    optimizer = TransportOptimizer(backend="auto", simulation_method="ai_surrogate")

    # Define starting molecules
    initial_molecules = [
        "CC(=O)Nc1ccc(O)cc1",  # Acetaminophen
        "CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    ]

    # Optimize for specific permeability
    print("Optimizing for permeability = 1e-6 cm/s...")

    permeability_result = optimizer.optimize_permeability(
        initial_molecules=initial_molecules,
        target_permeability=1e-6,
        permeability_range=(5e-7, 2e-6),
        n_iterations=20,  # Reduced for demo
        population_size=10,
    )

    print(
        f"Optimization completed in {permeability_result.computational_time:.2f} seconds"
    )
    print(f"Success rate: {permeability_result.success_rate:.2%}")
    print(f"Found {len(permeability_result.optimized_molecules)} optimized molecules")

    # Display top optimized molecules
    print("\nTop optimized molecules:")
    for i, (mol, props) in enumerate(
        zip(
            permeability_result.optimized_molecules[:3],
            permeability_result.predicted_properties[:3],
        )
    ):
        if "error" not in props:
            print(f"  {i+1}. {mol}")
            print(
                f"     Permeability: {props.get('permeability_coefficient', 'N/A'):.2e} cm/s"
            )

    # Optimize bioavailability
    print("\n6. Bioavailability Optimization")
    print("-" * 50)

    bioavailability_result = optimizer.optimize_bioavailability(
        initial_molecules=["CCO", "CC(=O)Nc1ccc(O)cc1"],
        target_bioavailability=0.8,
        constraints={"molecular_weight_range": (150, 500), "logp_range": (0, 4)},
        n_iterations=15,
        population_size=8,
    )

    print("Bioavailability optimization completed")
    print(f"Success rate: {bioavailability_result.success_rate:.2%}")

    # Analyze optimization results
    analysis = optimizer.analyze_optimization_results(bioavailability_result)
    print("\nOptimization Analysis:")
    if "convergence" in analysis:
        conv = analysis["convergence"]
        print(f"  Final best score: {conv.get('final_best_score', 'N/A'):.3f}")
        print(f"  Improvement: {conv.get('improvement', 'N/A'):.3f}")
        print(f"  Converged: {conv.get('converged', False)}")

    return permeability_result, bioavailability_result


def demo_biological_transport():
    """Demonstrate active transport and metabolism modeling."""
    logger.info("=== Biological Transport Demo ===")

    print("\n7. Biological Transport Simulation")
    print("-" * 50)

    # Initialize biological transport model
    bio_model = BiologicalTransportModel(backend="auto")

    # Test drug disposition simulation
    test_drugs = [
        "CC(=O)Nc1ccc(O)cc1",  # Acetaminophen
        "CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    ]

    drug_names = ["Acetaminophen", "Ibuprofen"]

    # Simulate oral drug disposition
    print("Simulating oral drug disposition...")

    disposition_result = bio_model.simulate_drug_disposition(
        molecules=test_drugs, dose=100.0, route="oral"  # mg
    )

    print(f"Bioavailability: {disposition_result.bioavailability:.2%}")
    print(f"Simulation completed in {disposition_result.simulation_time:.2f} seconds")

    # Display tissue distribution
    print("\nTissue Distribution:")
    for tissue, ratio in disposition_result.tissue_distribution.items():
        print(f"  {tissue}: {ratio:.2f}")

    # Display clearance rates
    print("\nClearance Rates:")
    for pathway, rate in disposition_result.clearance_rates.items():
        print(f"  {pathway}: {rate:.3f} h⁻¹")

    # Analyze drug interactions
    print("\n8. Drug Interaction Analysis")
    print("-" * 50)

    interaction_analysis = bio_model.analyze_drug_interactions(test_drugs)

    print(
        f"Transporter interactions found: {len(interaction_analysis['transporter_interactions'])}"
    )
    print(
        f"Metabolic interactions found: {len(interaction_analysis['metabolic_interactions'])}"
    )

    for interaction in interaction_analysis["transporter_interactions"]:
        print(
            f"  {interaction['transporter']}: {len(interaction['competing_substrates'])} substrates"
        )

    # Predict bioavailability
    print("\n9. Bioavailability Prediction")
    print("-" * 50)

    bioavailabilities = bio_model.predict_bioavailability(test_drugs)

    for drug_smiles, drug_name in zip(test_drugs, drug_names):
        ba = bioavailabilities.get(drug_smiles, 0)
        print(f"  {drug_name}: {ba:.2%}")

    return disposition_result


def demo_transport_visualization():
    """Demonstrate interactive transport pathway visualization."""
    logger.info("=== Transport Visualization Demo ===")

    print("\n10. Transport Visualization")
    print("-" * 50)

    # Run a simulation for visualization
    simulator = MultiPhaseSimulator(backend="auto", method="ai_surrogate")

    viz_result = simulator.simulate_membrane_transport(
        molecules=["CC(=O)Nc1ccc(O)cc1"],  # Acetaminophen
        membrane_type="dppc_bilayer",
        conditions={"temperature": 310.0, "pH": 7.4},
        n_steps=30000,
        output_frequency=300,
    )

    # Initialize visualizer
    print("Creating visualizations...")

    try:
        # Try Plotly first (interactive)
        visualizer = TransportVisualizer(backend="plotly")

        # Create concentration profile plot
        concentration_plot = visualizer.visualize_transport_results(viz_result)
        print("  ✓ Concentration profiles created")

        # Create pathway diagram
        pathway_diagram = visualizer.create_pathway_diagram(viz_result)
        print("  ✓ Pathway diagram created")

        # Save visualizations
        output_dir = Path("demo_outputs")
        output_dir.mkdir(exist_ok=True)

        concentration_plot.write_html(str(output_dir / "concentration_profiles.html"))
        pathway_diagram.write_html(str(output_dir / "pathway_diagram.html"))

        print(f"  ✓ Visualizations saved to {output_dir}/")

    except Exception as e:
        logger.warning(f"Plotly visualization failed: {e}")

        try:
            # Fallback to matplotlib
            visualizer = TransportVisualizer(backend="matplotlib")

            concentration_plot = visualizer.visualize_transport_results(viz_result)
            pathway_diagram = visualizer.create_pathway_diagram(viz_result)

            output_dir = Path("demo_outputs")
            output_dir.mkdir(exist_ok=True)

            concentration_plot.savefig(
                str(output_dir / "concentration_profiles.png"),
                dpi=300,
                bbox_inches="tight",
            )
            pathway_diagram.savefig(
                str(output_dir / "pathway_diagram.png"), dpi=300, bbox_inches="tight"
            )

            print("  ✓ Static visualizations created and saved")

        except Exception as e2:
            logger.error(f"All visualization backends failed: {e2}")
            print("  ✗ Visualization creation failed")

    return viz_result


def run_comprehensive_demo():
    """Run comprehensive demo of all transport capabilities."""
    logger.info("Starting Comprehensive Multi-Phase Transport Demo")

    print("=" * 80)
    print("ChemiAccel Multi-Phase Transport & Bioavailability Optimization Demo")
    print("=" * 80)

    demo_start_time = time.time()

    try:
        # 1. Multi-phase simulation
        sim_results = demo_multi_phase_simulation()

        # 2. Transport analysis
        analysis_results = demo_transport_analysis()

        # 3. Transport optimization
        opt_results = demo_transport_optimization()

        # 4. Biological transport
        bio_results = demo_biological_transport()

        # 5. Visualization
        viz_results = demo_transport_visualization()

        demo_time = time.time() - demo_start_time

        print("\n" + "=" * 80)
        print("DEMO SUMMARY")
        print("=" * 80)
        print(f"Total demo time: {demo_time:.2f} seconds")
        print(f"Molecules simulated: {len(sim_results) if sim_results else 0}")
        print("Features demonstrated:")
        print("  ✓ GPU-accelerated multi-phase simulation")
        print("  ✓ Transport kinetics and partitioning analysis")
        print("  ✓ Inverse design optimization")
        print("  ✓ Biological transport and metabolism modeling")
        print("  ✓ Interactive visualization")

        print("\nKey Capabilities Showcased:")
        print("  • Multiple simulation backends (AI surrogate, MD, DPD)")
        print("  • Comprehensive transport property analysis")
        print("  • AI-driven molecular optimization")
        print("  • ADME simulation with active transport")
        print("  • Interactive visualization and dashboards")

        print(
            "\nDemo completed successfully! Check 'demo_outputs/' for visualizations."
        )

        return {
            "simulation_results": sim_results,
            "analysis_results": analysis_results,
            "optimization_results": opt_results,
            "biological_results": bio_results,
            "visualization_results": viz_results,
        }

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    try:
        results = run_comprehensive_demo()
        print("\n🎉 Demo completed successfully!")

    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)
