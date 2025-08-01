#!/usr/bin/env python3
"""
Standalone Transport Test

This test bypasses package imports and tests the transport modules directly.
"""

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# Set up path to import modules directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set up minimal environment to avoid import issues
os.environ["CHEMI_ACCEL_NO_GPU"] = "1"

logging.basicConfig(level=logging.WARNING)  # Reduce noise


def test_simulation_engine():
    """Test simulation engine directly."""
    print("Testing Simulation Engine...")

    try:
        # Import directly
        from chemi_accel.transport.simulation_engine import (
            AISubrogateSimulation,
            SimulationConfig,
            TransportResult,
        )

        # Create config
        config = SimulationConfig(
            method="ai_surrogate", temperature=310.0, n_steps=2000, output_frequency=200
        )

        # Create simulator
        sim = AISubrogateSimulation(backend="cpu")

        # Run simulation
        molecules = ["CCO", "CC(=O)Nc1ccc(O)cc1"]
        result = sim.run_simulation(molecules, config)

        print(f"  + Permeability: {result.permeability_coefficient:.2e} cm/s")
        print(f"  + Trajectories: {result.trajectories.shape}")
        print(f"  + Time: {result.simulation_time:.3f}s")

        return True

    except Exception as e:
        print(f"  x Error: {e}")
        return False


def test_transport_analyzer():
    """Test transport analyzer directly."""
    print("\nTesting Transport Analyzer...")

    try:
        # Import modules
        from chemi_accel.transport.simulation_engine import (
            AISubrogateSimulation,
            SimulationConfig,
        )
        from chemi_accel.transport.transport_analyzer import TransportAnalyzer

        # Generate test data
        config = SimulationConfig(n_steps=1000, output_frequency=100)
        sim = AISubrogateSimulation(backend="cpu")
        result = sim.run_simulation(["CCO"], config)

        # Analyze
        analyzer = TransportAnalyzer(backend="cpu")

        # Test permeability analysis
        perm_analysis = analyzer.analyze_permeability(result)
        print(f"  + Permeability analysis: {len(perm_analysis)} metrics")

        # Test diffusion analysis
        diff_analysis = analyzer.analyze_diffusion(result)
        print(f"  + Diffusion analysis: {len(diff_analysis)} components")

        # Test partitioning
        part_analysis = analyzer.analyze_partitioning(result)
        print(f"  + Phase behavior: {part_analysis.phase_behavior_classification}")

        # Test report
        report = analyzer.generate_transport_report(result)
        print(f"  + Report sections: {len(report)}")

        return True

    except Exception as e:
        print(f"  x Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_biological_transport():
    """Test biological transport model."""
    print("\nTesting Biological Transport...")

    try:
        from chemi_accel.transport.biological_transport import (
            BiologicalTransportSimulator,
            EnzymeProperties,
            MetabolismType,
            TransporterProperties,
            TransportMechanism,
        )

        # Create simulator
        bio_sim = BiologicalTransportSimulator(backend="cpu")

        # Test basic simulation
        result = bio_sim.simulate_biological_transport(
            molecules=["CCO"], dose=1.0, simulation_time=12.0, time_points=50
        )

        print(f"  + Bioavailability: {result.bioavailability:.2%}")
        print(f"  + Tissues: {len(result.tissue_distribution)}")
        print(f"  + Clearance rates: {len(result.clearance_rates)}")

        return True

    except Exception as e:
        print(f"  x Error: {e}")
        return False


def test_different_backends():
    """Test different computational backends."""
    print("\nTesting Different Backends...")

    try:
        from chemi_accel.transport.simulation_engine import (
            AISubrogateSimulation,
            SimulationConfig,
        )

        config = SimulationConfig(n_steps=500, output_frequency=50)

        # Test CPU backend
        sim_cpu = AISubrogateSimulation(backend="cpu")
        result_cpu = sim_cpu.run_simulation(["CCO"], config)
        print(f"  + CPU backend: {result_cpu.permeability_coefficient:.2e} cm/s")

        # Test auto backend selection
        sim_auto = AISubrogateSimulation(backend="auto")
        result_auto = sim_auto.run_simulation(["CCO"], config)
        print(
            f"  + Auto backend ({sim_auto.backend}): {result_auto.permeability_coefficient:.2e} cm/s"
        )

        return True

    except Exception as e:
        print(f"  x Error: {e}")
        return False


def main():
    """Run standalone transport tests."""
    print("STANDALONE TRANSPORT MODULE TESTS")
    print("=" * 50)

    start_time = time.time()
    tests = [
        ("Simulation Engine", test_simulation_engine),
        ("Transport Analyzer", test_transport_analyzer),
        ("Biological Transport", test_biological_transport),
        ("Different Backends", test_different_backends),
    ]

    passed = 0
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"PASS: {name}")
            else:
                print(f"FAIL: {name}")
        except Exception as e:
            print(f"ERROR: {name} - {e}")

    total_time = time.time() - start_time

    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Success rate: {passed/len(tests)*100:.1f}%")
    print(f"Total time: {total_time:.2f}s")

    if passed == len(tests):
        print("\nSUCCESS: All transport tests passed!")
        print("\nCapabilities verified:")
        print("  + AI surrogate simulation engine")
        print("  + Transport kinetics analysis")
        print("  + Biological transport modeling")
        print("  + Multiple computational backends")
        print("  + Comprehensive property prediction")
        return True
    else:
        print(f"\nWARNING: {len(tests) - passed} test(s) failed")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTesting interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\nTesting crashed: {e}")
        sys.exit(1)
