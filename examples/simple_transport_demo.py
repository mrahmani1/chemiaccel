#!/usr/bin/env python3
"""
Simple Multi-Phase Transport Demo

This demo showcases core transport functionality without heavy dependencies.
"""

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Direct imports to avoid dependency issues
import sys

sys.path.insert(
    0, str(Path(__file__).parent.parent / "src" / "chemi_accel" / "transport")
)

from simulation_engine import MultiPhaseSimulator, SimulationConfig
from transport_analyzer import TransportAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demo_basic_transport():
    """Demonstrate basic transport simulation."""
    logger.info("=== Basic Transport Simulation Demo ===")

    print("Testing Multi-Phase Transport & Bioavailability Optimization")
    print("=" * 60)

    # Initialize simulator
    simulator = MultiPhaseSimulator(backend="auto", method="ai_surrogate")

    # Test molecules
    test_molecules = [
        "CCO",  # Ethanol
        "CC(=O)Nc1ccc(O)cc1",  # Acetaminophen
        "CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    ]

    molecule_names = ["Ethanol", "Acetaminophen", "Ibuprofen"]

    print("\n1. Membrane Transport Simulation")
    print("-" * 40)

    results = {}
    for mol_smiles, mol_name in zip(test_molecules, molecule_names):
        print(f"\nSimulating {mol_name}...")

        try:
            # Simulate membrane transport
            result = simulator.simulate_membrane_transport(
                molecules=[mol_smiles],
                membrane_type="dppc_bilayer",
                conditions={"temperature": 310.0, "pH": 7.4},
                n_steps=10000,  # Reduced for demo
                output_frequency=1000,
            )

            results[mol_name] = result

            print(f"  ✓ Permeability: {result.permeability_coefficient:.2e} cm/s")
            print(f"  ✓ Simulation time: {result.simulation_time:.2f} seconds")

            # Display diffusion coefficients
            for phase, diff_coeff in result.diffusion_coefficients.items():
                print(f"  ✓ {phase} diffusion: {diff_coeff:.2e} cm²/s")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    return results


def demo_transport_analysis():
    """Demonstrate transport analysis."""
    logger.info("=== Transport Analysis Demo ===")

    print("\n2. Transport Analysis")
    print("-" * 40)

    # Initialize analyzer
    analyzer = TransportAnalyzer(backend="auto")

    # Initialize simulator
    simulator = MultiPhaseSimulator(backend="auto", method="ai_surrogate")

    try:
        # Run simulation for analysis
        result = simulator.simulate_membrane_transport(
            molecules=["CC(=O)Nc1ccc(O)cc1"],  # Acetaminophen
            membrane_type="dppc_bilayer",
            conditions={"temperature": 310.0, "pH": 7.4},
            n_steps=20000,
            output_frequency=2000,
        )

        print("\nAnalyzing transport properties...")

        # Analyze permeability
        perm_analysis = analyzer.analyze_permeability(result)
        print(
            f"  ✓ Steady-state permeability: {perm_analysis.get('steady_state_permeability', 'N/A'):.2e}"
        )
        print(
            f"  ✓ Transport mechanisms: {perm_analysis.get('transport_mechanisms', 'N/A')}"
        )

        # Analyze diffusion
        diff_analysis = analyzer.analyze_diffusion(result)
        if "diffusion_coefficients" in diff_analysis:
            for phase, coeff in diff_analysis["diffusion_coefficients"].items():
                if isinstance(coeff, (int, float)):
                    print(f"  ✓ {phase}: {coeff:.2e}")

        # Analyze partitioning
        part_analysis = analyzer.analyze_partitioning(result)
        print(f"  ✓ Phase behavior: {part_analysis.phase_behavior_classification}")

        # Generate recommendations
        report = analyzer.generate_transport_report(result)
        print("\nRecommendations:")
        for rec in report.get("recommendations", []):
            print(f"  • {rec}")

        return result

    except Exception as e:
        print(f"  ✗ Analysis error: {e}")
        return None


def demo_different_conditions():
    """Demonstrate transport under different conditions."""
    print("\n3. Transport Under Different Conditions")
    print("-" * 40)

    simulator = MultiPhaseSimulator(backend="auto", method="ai_surrogate")

    conditions_to_test = [
        {"temperature": 298.15, "pH": 7.4, "name": "Room temp, physiological pH"},
        {"temperature": 310.15, "pH": 7.4, "name": "Body temp, physiological pH"},
        {"temperature": 310.15, "pH": 1.5, "name": "Body temp, gastric pH"},
    ]

    mol_smiles = "CC(=O)Nc1ccc(O)cc1"  # Acetaminophen

    for conditions in conditions_to_test:
        try:
            result = simulator.simulate_membrane_transport(
                molecules=[mol_smiles],
                membrane_type="dppc_bilayer",
                conditions=conditions,
                n_steps=5000,
                output_frequency=500,
            )

            print(f"\n{conditions['name']}:")
            print(f"  Permeability: {result.permeability_coefficient:.2e} cm/s")

        except Exception as e:
            print(f"  ✗ Error with {conditions['name']}: {e}")


def run_simple_demo():
    """Run simple transport demo."""
    start_time = time.time()

    try:
        # Basic transport simulation
        sim_results = demo_basic_transport()

        # Transport analysis
        analysis_result = demo_transport_analysis()

        # Different conditions
        demo_different_conditions()

        demo_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("DEMO SUMMARY")
        print("=" * 60)
        print(f"Total demo time: {demo_time:.2f} seconds")
        print(f"Molecules tested: {len(sim_results) if sim_results else 0}")
        print("Features demonstrated:")
        print("  ✓ Multi-phase transport simulation")
        print("  ✓ Transport kinetics analysis")
        print("  ✓ Different environmental conditions")
        print("  ✓ Permeability and diffusion coefficient calculation")
        print("  ✓ Phase behavior classification")

        print("\n🎉 Simple transport demo completed successfully!")

        return True

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        return False


if __name__ == "__main__":
    try:
        success = run_simple_demo()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Demo crashed: {e}")
        print(f"\n❌ Demo crashed: {e}")
        sys.exit(1)
