#!/usr/bin/env python3
"""
Constitutional Descriptors Demo

This script demonstrates the comprehensive constitutional descriptor calculations
available in ChemiAccel, including atom counts, bond counts, complexity measures,
information content descriptors, and synthetic accessibility scoring.
"""

import os
import sys

# Add the src directory to the path so we can import chemi_accel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import chemi_accel as ca
    from chemi_accel.descriptors.constitutional import (
        ConstitutionalDescriptorCalculator,
        calculate_constitutional_descriptors,
        get_constitutional_descriptor_info,
    )
    from chemi_accel.utils.rdkit_integration import RDKitMolecularParser
except ImportError as e:
    print(f"Error importing ChemiAccel: {e}")
    print("Make sure ChemiAccel is properly installed and RDKit is available.")
    sys.exit(1)


def main():
    """Main demonstration function."""
    print("=" * 80)
    print("ChemiAccel Constitutional Descriptors Demo")
    print("=" * 80)

    # Sample molecules representing different chemical classes
    sample_molecules = [
        ("Ethanol", "CCO"),
        ("Benzene", "c1ccccc1"),
        ("Acetic Acid", "CC(=O)O"),
        ("Triethylamine", "CCN(CC)CC"),
        ("Naphthalene", "c1ccc2ccccc2c1"),
        ("tert-Butanol", "CC(C)(C)O"),
        ("Palmitic Acid", "CCCCCCCCCCCCCCCC(=O)O"),
        ("Decalin", "C1CCC2CCCCC2C1"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ]

    print(f"\nAnalyzing {len(sample_molecules)} diverse molecules...")

    # Parse molecules
    parser = RDKitMolecularParser()
    molecules = []

    for name, smiles in sample_molecules:
        try:
            mol_data = parser.parse_smiles(smiles, mol_id=name)
            molecules.append(mol_data)
            print(f"✓ Parsed {name}: {smiles}")
        except Exception as e:
            print(f"✗ Failed to parse {name}: {e}")

    print(f"\nSuccessfully parsed {len(molecules)} molecules.")

    # Initialize constitutional descriptor calculator
    print("\n" + "=" * 60)
    print("Constitutional Descriptor Calculator")
    print("=" * 60)

    calc = ConstitutionalDescriptorCalculator(use_cuda=False)
    print(
        f"Initialized calculator with {len(calc.get_available_descriptors())} descriptors"
    )

    # Show descriptor categories
    categories = calc.get_descriptor_categories()
    print(f"\nAvailable descriptor categories: {', '.join(categories)}")

    # Calculate basic constitutional descriptors
    print("\n" + "-" * 40)
    print("Basic Constitutional Descriptors")
    print("-" * 40)

    basic_descriptors = [
        "molecular_weight",
        "num_atoms",
        "num_heavy_atoms",
        "num_bonds",
        "num_carbons",
        "num_nitrogens",
        "num_oxygens",
        "ring_count",
    ]

    basic_df = calc.calculate_descriptors_batch(molecules, basic_descriptors)
    print(basic_df.round(2))

    # Calculate complexity descriptors
    print("\n" + "-" * 40)
    print("Molecular Complexity Descriptors")
    print("-" * 40)

    complexity_descriptors = [
        "bertz_complexity",
        "molecular_complexity",
        "information_content",
        "bond_information_content",
        "structural_information_content",
    ]

    complexity_df = calc.calculate_descriptors_batch(molecules, complexity_descriptors)
    print(complexity_df.round(3))

    # Calculate all descriptors
    print("\n" + "=" * 60)
    print("Comprehensive Constitutional Analysis")
    print("=" * 60)

    all_df = calc.calculate_descriptors_batch(molecules)
    print(
        f"Calculated {len(all_df.columns) - 2} constitutional descriptors for {len(molecules)} molecules"
    )

    # Performance analysis
    print("\n" + "=" * 60)
    print("Performance Analysis")
    print("=" * 60)

    import time

    # Time descriptor calculation
    start_time = time.time()
    perf_df = calc.calculate_descriptors_batch(molecules)
    end_time = time.time()

    calculation_time = end_time - start_time
    descriptors_per_second = (len(molecules) * len(perf_df.columns)) / calculation_time

    print("Performance metrics:")
    print(f"  Molecules processed: {len(molecules)}")
    print(f"  Descriptors calculated: {len(perf_df.columns) - 2}")
    print(f"  Total calculations: {len(molecules) * (len(perf_df.columns) - 2)}")
    print(f"  Time taken: {calculation_time:.3f} seconds")
    print(f"  Rate: {descriptors_per_second:.0f} descriptor calculations/second")
    print(f"  Average per molecule: {calculation_time/len(molecules)*1000:.1f} ms")

    print("\n" + "=" * 80)
    print("Constitutional Descriptors Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
