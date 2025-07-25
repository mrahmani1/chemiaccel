#!/usr/bin/env python3
"""
ChemiAccel RDKit Integration Demo

This script demonstrates the comprehensive RDKit integration capabilities
of ChemiAccel, including SMILES parsing, validation, and descriptor calculation.
"""


import pandas as pd
from chemi_accel import Calculator
from chemi_accel.utils.rdkit_integration import (
    RDKitDescriptorCalculator,
    RDKitMolecularParser,
    calculate_basic_descriptors,
    parse_smiles,
    parse_smiles_batch,
)


def demo_basic_usage():
    """Demonstrate basic ChemiAccel usage with RDKit integration."""
    print("=" * 60)
    print("ChemiAccel RDKit Integration Demo")
    print("=" * 60)

    # Sample molecules for testing
    molecules = {
        "ethanol": "CCO",
        "benzene": "c1ccccc1",
        "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "water": "O",
        "methane": "C",
        "glucose": "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O",
    }

    print(f"\n1. Testing with {len(molecules)} molecules:")
    for name, smiles in molecules.items():
        print(f"   {name:12} : {smiles}")

    # Initialize Calculator
    print("\n2. Initializing ChemiAccel Calculator...")
    calc = Calculator(backend="cpu")
    print(f"   ✓ Calculator initialized with {calc.backend_type.value} backend")
    print(f"   ✓ Available descriptors: {len(calc.available_descriptors)}")

    # Calculate descriptors using Calculator
    print("\n3. Calculating descriptors using Calculator.pandas()...")
    smiles_list = list(molecules.values())
    mol_names = list(molecules.keys())

    # Create pandas Series with molecule names as index
    smiles_series = pd.Series(smiles_list, index=mol_names)

    result = calc.pandas(smiles_series)
    print(f"   ✓ Processed {len(result)} molecules")
    print(
        f"   ✓ Calculated {len(result.columns)-2} descriptors"
    )  # -2 for mol_id and smiles

    # Display results
    print("\n4. Results Summary:")
    display_cols = [
        "smiles",
        "molecular_weight",
        "num_atoms",
        "logp",
        "tpsa",
        "num_rotatable_bonds",
    ]
    print(result[display_cols].round(3))

    return result


def demo_advanced_features():
    """Demonstrate advanced RDKit integration features."""
    print("\n" + "=" * 60)
    print("Advanced RDKit Integration Features")
    print("=" * 60)

    # Test error handling with mixed valid/invalid SMILES
    print("\n1. Testing error handling with mixed SMILES...")
    mixed_smiles = [
        "CCO",  # Valid: ethanol
        "invalid_smiles",  # Invalid
        "CCC",  # Valid: propane
        "",  # Invalid: empty
        "c1ccccc1",  # Valid: benzene
        "C[C",  # Invalid: incomplete
        "O",  # Valid: water
    ]

    parser = RDKitMolecularParser()
    molecules = parser.parse_smiles_batch(mixed_smiles, error_handling="skip")
    print(
        f"   ✓ Successfully parsed {len(molecules)} out of {len(mixed_smiles)} SMILES"
    )

    # Test descriptor configuration
    print("\n2. Testing descriptor configuration...")
    calc = Calculator()

    # Configure specific descriptors
    calc.configure_descriptors(["molecular_weight", "logp", "tpsa", "num_atoms"])
    print(f"   ✓ Configured {len(calc.descriptor_sets)} specific descriptors")

    result = calc.pandas(["CCO", "CCC", "c1ccccc1"])
    print(f"   ✓ Result has {len(result.columns)} columns: {list(result.columns)}")

    # Test with all available descriptors
    print("\n3. Testing with all available descriptors...")
    calc.configure_descriptors("all")
    result_all = calc.pandas(["CCO", "CCC"])
    print(f"   ✓ Calculated {len(result_all.columns)-2} descriptors per molecule")

    return result_all


def demo_direct_rdkit_usage():
    """Demonstrate direct RDKit integration usage."""
    print("\n" + "=" * 60)
    print("Direct RDKit Integration Usage")
    print("=" * 60)

    # Direct parsing
    print("\n1. Direct SMILES parsing...")
    mol_data = parse_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "caffeine")
    print(f"   ✓ Parsed: {mol_data}")
    print(
        f"   ✓ Properties: MW={mol_data.molecular_weight:.2f}, "
        f"Atoms={mol_data.num_atoms}, Bonds={mol_data.num_bonds}"
    )

    # Batch parsing
    print("\n2. Batch parsing with convenience function...")
    drug_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    ]
    drug_names = ["aspirin", "ibuprofen", "caffeine"]

    molecules = parse_smiles_batch(drug_smiles, drug_names)
    print(f"   ✓ Parsed {len(molecules)} drug molecules")

    # Direct descriptor calculation
    print("\n3. Direct descriptor calculation...")
    df = calculate_basic_descriptors(molecules)
    print(f"   ✓ Calculated descriptors for {len(df)} molecules")

    # Show drug properties
    print("\n4. Drug Properties:")
    drug_props = df[
        ["mol_id", "molecular_weight", "logp", "tpsa", "num_hbd", "num_hba"]
    ].round(3)
    print(drug_props)

    return df


def demo_validation_features():
    """Demonstrate molecular validation features."""
    print("\n" + "=" * 60)
    print("Molecular Validation Features")
    print("=" * 60)

    parser = RDKitMolecularParser()

    # Test various edge cases
    test_cases = [
        ("valid_simple", "CCO"),
        ("valid_aromatic", "c1ccccc1"),
        ("valid_complex", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("empty_string", ""),
        ("whitespace", "   "),
        ("invalid_syntax", "C[C"),
        ("unknown_atom", "X"),
    ]

    print("\n1. Testing various SMILES validation cases...")
    for name, smiles in test_cases:
        try:
            mol_data = parser.parse_smiles(smiles, name)
            is_valid, issues = parser.validate_molecule(mol_data)
            status = "✓ VALID" if is_valid else f"⚠ ISSUES: {', '.join(issues)}"
            print(f"   {name:15} : {smiles:20} -> {status}")
        except Exception as e:
            print(f"   {name:15} : {smiles:20} -> ❌ ERROR: {str(e)[:50]}...")

    # Test large molecule
    print("\n2. Testing large molecule processing...")
    cholesterol = "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C"
    try:
        mol_data = parser.parse_smiles(cholesterol, "cholesterol")
        print(
            f"   ✓ Cholesterol: {mol_data.num_atoms} atoms, MW={mol_data.molecular_weight:.2f}"
        )

        # Calculate descriptors for large molecule
        calculator = RDKitDescriptorCalculator()
        descriptors = calculator.calculate_descriptors(
            mol_data, ["molecular_weight", "logp", "tpsa"]
        )
        print(
            f"   ✓ Descriptors: MW={descriptors['molecular_weight']:.2f}, "
            f"LogP={descriptors['logp']:.2f}, TPSA={descriptors['tpsa']:.2f}"
        )
    except Exception as e:
        print(f"   ❌ Failed to process cholesterol: {e}")


def demo_performance_comparison():
    """Demonstrate performance with larger datasets."""
    print("\n" + "=" * 60)
    print("Performance Demonstration")
    print("=" * 60)

    # Generate test dataset
    import time

    # Common drug-like molecules
    drug_smiles = [
        "CCO",  # Ethanol
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "c1ccccc1",  # Benzene
        "CCC",  # Propane
        "O",  # Water
        "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O",  # Glucose
    ]

    # Replicate to create larger dataset
    test_sizes = [10, 50, 100]

    for size in test_sizes:
        print(f"\n1. Testing with {size} molecules...")

        # Create test dataset
        test_smiles = (drug_smiles * (size // len(drug_smiles) + 1))[:size]

        # Time the calculation
        calc = Calculator()
        start_time = time.time()
        result = calc.pandas(test_smiles)
        end_time = time.time()

        processing_time = end_time - start_time
        molecules_per_second = (
            size / processing_time if processing_time > 0 else float("inf")
        )

        print(
            f"   ✓ Processed {len(result)} molecules in {processing_time:.3f} seconds"
        )
        print(f"   ✓ Rate: {molecules_per_second:.1f} molecules/second")
        print(f"   ✓ Calculated {len(result.columns)-2} descriptors per molecule")


def main():
    """Run all demonstration functions."""
    try:
        # Basic usage demo
        basic_result = demo_basic_usage()

        # Advanced features demo
        advanced_result = demo_advanced_features()

        # Direct RDKit usage demo
        direct_result = demo_direct_rdkit_usage()

        # Validation features demo
        demo_validation_features()

        # Performance demonstration
        demo_performance_comparison()

        print("\n" + "=" * 60)
        print("🎉 All demonstrations completed successfully!")
        print("=" * 60)
        print("\nChemiAccel RDKit integration is working perfectly!")
        print("You can now:")
        print("  • Parse SMILES strings with comprehensive error handling")
        print("  • Calculate 18+ molecular descriptors with GPU acceleration")
        print("  • Process batches of molecules efficiently")
        print("  • Integrate seamlessly with pandas workflows")
        print("  • Handle edge cases and invalid molecules gracefully")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
