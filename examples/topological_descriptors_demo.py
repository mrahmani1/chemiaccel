#!/usr/bin/env python3
"""
Topological Descriptors Demo for ChemiAccel

This script demonstrates the comprehensive topological descriptor calculations
implemented in ChemiAccel, including:
- Wiener, Zagreb, and Randic connectivity indices
- Topological charge and electrotopological state indices
- Kappa shape indices and molecular path counts
- Autocorrelation descriptors for spatial property correlations
- Graph-based calculations with optional GPU acceleration
"""

import os
import sys
import time

import numpy as np

# Add the src directory to the path so we can import chemi_accel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from rdkit import Chem

    RDKIT_AVAILABLE = True
except ImportError:
    print("RDKit not available. Please install RDKit to run this demo.")
    sys.exit(1)

from chemi_accel.core.molecular import MolecularData
from chemi_accel.descriptors.topological import (
    TopologicalDescriptorCalculator,
    calculate_topological_descriptors,
    get_topological_descriptor_info,
)


def create_sample_molecules():
    """Create a diverse set of sample molecules for testing."""
    molecules = []

    # Define test molecules with different structural features
    test_smiles = [
        ("benzene", "c1ccccc1"),
        ("toluene", "Cc1ccccc1"),
        ("phenol", "Oc1ccccc1"),
        ("aniline", "Nc1ccccc1"),
        ("naphthalene", "c1ccc2ccccc2c1"),
        ("ethane", "CC"),
        ("propane", "CCC"),
        ("butane", "CCCC"),
        ("isobutane", "CC(C)C"),
        ("cyclohexane", "C1CCCCC1"),
        ("methanol", "CO"),
        ("ethanol", "CCO"),
        ("acetone", "CC(=O)C"),
        ("acetic_acid", "CC(=O)O"),
        ("pyridine", "c1ccncc1"),
        ("furan", "c1ccoc1"),
        ("thiophene", "c1ccsc1"),
        ("imidazole", "c1c[nH]cn1"),
        ("caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ]

    for name, smiles in test_smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol_data = MolecularData(
                    mol_id=name, smiles=smiles, backend_data={"rdkit_mol": mol}
                )
                molecules.append(mol_data)
            else:
                print(f"Warning: Could not parse SMILES for {name}: {smiles}")
        except Exception as e:
            print(f"Error creating molecule {name}: {e}")

    return molecules


def demonstrate_descriptor_categories():
    """Demonstrate different categories of topological descriptors."""
    print("=" * 80)
    print("TOPOLOGICAL DESCRIPTOR CATEGORIES")
    print("=" * 80)

    # Get descriptor information
    descriptor_info = get_topological_descriptor_info()

    # Group descriptors by category
    categories = {}
    for desc_name, info in descriptor_info.items():
        category = info["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append((desc_name, info["description"]))

    # Display categories
    for category, descriptors in categories.items():
        print(f"\n{category.upper().replace('_', ' ')} DESCRIPTORS:")
        print("-" * 50)
        for desc_name, description in descriptors:
            print(f"  • {desc_name}: {description}")

    print(f"\nTotal descriptors available: {len(descriptor_info)}")


def demonstrate_basic_calculations():
    """Demonstrate basic topological descriptor calculations."""
    print("\n" + "=" * 80)
    print("BASIC TOPOLOGICAL DESCRIPTOR CALCULATIONS")
    print("=" * 80)

    # Create calculator
    calculator = TopologicalDescriptorCalculator(use_gpu=False)

    # Test with simple molecules
    simple_molecules = [
        ("ethane", "CC"),
        ("benzene", "c1ccccc1"),
        ("naphthalene", "c1ccc2ccccc2c1"),
    ]

    for name, smiles in simple_molecules:
        print(f"\n{name.upper()} ({smiles}):")
        print("-" * 40)

        mol = Chem.MolFromSmiles(smiles)
        mol_data = MolecularData(
            mol_id=name, smiles=smiles, backend_data={"rdkit_mol": mol}
        )

        # Calculate descriptors
        start_time = time.time()
        descriptors = calculator.calculate_topological_descriptors(mol_data)
        calc_time = time.time() - start_time

        # Display key descriptors
        key_descriptors = [
            "wiener_index",
            "zagreb_index_1",
            "zagreb_index_2",
            "randic_index",
            "kappa_1",
            "kappa_2",
            "path_count_1",
            "path_count_2",
            "path_count_3",
        ]

        for desc_name in key_descriptors:
            if desc_name in descriptors:
                value = descriptors[desc_name]
                print(f"  {desc_name:25}: {value:8.3f}")

        print(f"  {'calculation_time':25}: {calc_time:8.3f} seconds")


def demonstrate_connectivity_indices():
    """Demonstrate connectivity index calculations."""
    print("\n" + "=" * 80)
    print("CONNECTIVITY INDICES COMPARISON")
    print("=" * 80)

    # Test molecules with different connectivity patterns
    test_molecules = [
        ("linear_alkane", "CCCCCC"),
        ("branched_alkane", "CC(C)C(C)C"),
        ("cyclic", "C1CCCCC1"),
        ("aromatic", "c1ccccc1"),
        ("polycyclic", "c1ccc2ccccc2c1"),
    ]

    molecules = []
    for name, smiles in test_molecules:
        mol = Chem.MolFromSmiles(smiles)
        mol_data = MolecularData(
            mol_id=name, smiles=smiles, backend_data={"rdkit_mol": mol}
        )
        molecules.append(mol_data)

    # Calculate descriptors
    df = calculate_topological_descriptors(molecules, use_gpu=False)

    # Display connectivity indices
    connectivity_cols = [
        "wiener_index",
        "zagreb_index_1",
        "zagreb_index_2",
        "randic_index",
        "balaban_j",
        "harary_index",
    ]

    print(
        f"{'Molecule':<15} {'Wiener':<8} {'Zagreb1':<8} {'Zagreb2':<8} {'Randic':<8} {'Balaban':<8} {'Harary':<8}"
    )
    print("-" * 75)

    for _, row in df.iterrows():
        print(f"{row['mol_id']:<15} ", end="")
        for col in connectivity_cols:
            if col in df.columns:
                print(f"{row[col]:<8.2f} ", end="")
        print()


def demonstrate_shape_indices():
    """Demonstrate Kappa shape index calculations."""
    print("\n" + "=" * 80)
    print("KAPPA SHAPE INDICES")
    print("=" * 80)

    # Test molecules with different shapes
    shape_molecules = [
        ("linear", "CCCCCCCC"),
        ("branched", "CC(C)(C)CCCC"),
        ("cyclic", "C1CCCCCCC1"),
        ("spherical", "C1(C2(C3(C4(C1)C4)C3)C2)C5C5"),
        ("aromatic", "c1ccc2ccccc2c1"),
    ]

    molecules = []
    for name, smiles in shape_molecules:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol_data = MolecularData(
                mol_id=name, smiles=smiles, backend_data={"rdkit_mol": mol}
            )
            molecules.append(mol_data)

    # Calculate descriptors
    df = calculate_topological_descriptors(molecules, use_gpu=False)

    # Display shape indices
    shape_cols = ["kappa_1", "kappa_2", "kappa_3", "flexibility_index"]

    print(
        f"{'Molecule':<12} {'Kappa1':<10} {'Kappa2':<10} {'Kappa3':<10} {'Flexibility':<12}"
    )
    print("-" * 60)

    for _, row in df.iterrows():
        print(f"{row['mol_id']:<12} ", end="")
        for col in shape_cols:
            if col in df.columns:
                print(f"{row[col]:<10.3f} ", end="")
        print()


def demonstrate_autocorrelation():
    """Demonstrate autocorrelation descriptors."""
    print("\n" + "=" * 80)
    print("AUTOCORRELATION DESCRIPTORS")
    print("=" * 80)

    # Test molecules with different atomic compositions
    hetero_molecules = [
        ("hydrocarbon", "CCCCCC"),
        ("alcohol", "CCCCCO"),
        ("amine", "CCCCN"),
        ("halogenated", "CCCCCl"),
        ("mixed", "CC(O)C(N)C(Cl)C"),
    ]

    molecules = []
    for name, smiles in hetero_molecules:
        mol = Chem.MolFromSmiles(smiles)
        mol_data = MolecularData(
            mol_id=name, smiles=smiles, backend_data={"rdkit_mol": mol}
        )
        molecules.append(mol_data)

    # Calculate descriptors
    df = calculate_topological_descriptors(molecules, use_gpu=False)

    # Display autocorrelation descriptors
    autocorr_cols = [
        "moran_autocorr_mass",
        "moran_autocorr_electronegativity",
        "geary_autocorr_mass",
        "geary_autocorr_electronegativity",
    ]

    print(
        f"{'Molecule':<12} {'Moran_Mass':<12} {'Moran_EN':<12} {'Geary_Mass':<12} {'Geary_EN':<12}"
    )
    print("-" * 65)

    for _, row in df.iterrows():
        print(f"{row['mol_id']:<12} ", end="")
        for col in autocorr_cols:
            if col in df.columns:
                print(f"{row[col]:<12.3f} ", end="")
        print()


def demonstrate_performance():
    """Demonstrate performance characteristics."""
    print("\n" + "=" * 80)
    print("PERFORMANCE DEMONSTRATION")
    print("=" * 80)

    # Create molecules of different sizes
    molecules = create_sample_molecules()

    print(f"Testing with {len(molecules)} diverse molecules...")

    # Test CPU performance
    print("\nCPU Performance:")
    start_time = time.time()
    df_cpu = calculate_topological_descriptors(molecules, use_gpu=False)
    cpu_time = time.time() - start_time

    print(f"  Molecules processed: {len(df_cpu)}")
    print(
        f"  Descriptors calculated: {len(df_cpu.columns) - 2}"
    )  # Subtract mol_id and smiles
    print(f"  Total time: {cpu_time:.3f} seconds")
    print(f"  Time per molecule: {cpu_time/len(molecules):.3f} seconds")
    print(
        f"  Descriptors per second: {(len(df_cpu.columns) - 2) * len(molecules) / cpu_time:.0f}"
    )

    # Display descriptor statistics
    print("\nDescriptor Statistics:")
    numeric_cols = df_cpu.select_dtypes(include=[np.number]).columns
    stats = df_cpu[numeric_cols].describe()

    print(f"  Non-zero descriptors: {(df_cpu[numeric_cols] != 0).sum().sum()}")
    print(f"  NaN values: {df_cpu[numeric_cols].isna().sum().sum()}")
    print(f"  Mean descriptor value: {df_cpu[numeric_cols].mean().mean():.3f}")
    print(f"  Std descriptor value: {df_cpu[numeric_cols].std().mean():.3f}")


def demonstrate_descriptor_relationships():
    """Demonstrate relationships between different descriptor types."""
    print("\n" + "=" * 80)
    print("DESCRIPTOR RELATIONSHIPS")
    print("=" * 80)

    # Create molecules with systematic structural changes
    molecules = create_sample_molecules()
    df = calculate_topological_descriptors(molecules, use_gpu=False)

    # Analyze correlations between descriptor categories
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Group descriptors by type
    connectivity_desc = [
        col
        for col in numeric_cols
        if any(x in col for x in ["wiener", "zagreb", "randic"])
    ]
    shape_desc = [col for col in numeric_cols if "kappa" in col]
    path_desc = [col for col in numeric_cols if "path_count" in col]

    print("Connectivity Descriptors Correlation:")
    if len(connectivity_desc) > 1:
        conn_corr = df[connectivity_desc].corr()
        print(conn_corr.round(3))

    print("\nShape Descriptors Correlation:")
    if len(shape_desc) > 1:
        shape_corr = df[shape_desc].corr()
        print(shape_corr.round(3))

    # Find most discriminating descriptors
    print("\nMost Variable Descriptors (highest coefficient of variation):")
    cv_values = []
    for col in numeric_cols:
        if df[col].std() > 0 and df[col].mean() != 0:
            cv = df[col].std() / abs(df[col].mean())
            cv_values.append((col, cv))

    cv_values.sort(key=lambda x: x[1], reverse=True)
    for desc_name, cv in cv_values[:10]:
        print(f"  {desc_name:<30}: {cv:.3f}")


def main():
    """Main demonstration function."""
    print("ChemiAccel Topological Descriptors Demo")
    print("=" * 80)
    print("This demo showcases the comprehensive topological descriptor")
    print("calculations implemented in ChemiAccel, including connectivity")
    print("indices, shape descriptors, and autocorrelation measures.")
    print("=" * 80)

    try:
        # Run demonstrations
        demonstrate_descriptor_categories()
        demonstrate_basic_calculations()
        demonstrate_connectivity_indices()
        demonstrate_shape_indices()
        demonstrate_autocorrelation()
        demonstrate_performance()
        demonstrate_descriptor_relationships()

        print("\n" + "=" * 80)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("The topological descriptors module provides:")
        print("• 66+ comprehensive topological descriptors")
        print("• GPU acceleration support (when available)")
        print("• Efficient batch processing")
        print("• Robust error handling")
        print("• Integration with pandas DataFrames")
        print("• Extensive validation and testing")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
