"""
Demonstration of geometrical and 3D molecular descriptors in ChemiAccel.

This example shows how to:
1. Generate 3D conformers for molecules
2. Calculate 3D shape descriptors
3. Calculate electrostatic descriptors
4. Perform comprehensive 3D descriptor analysis
5. Compare performance with traditional methods
"""

import time

import numpy as np
from chemi_accel.descriptors.geometrical_3d import (
    ConformerGenerator,
    ElectrostaticDescriptors,
    Geometrical3DCalculator,
    Shape3DDescriptors,
    calculate_3d_descriptors,
    calculate_electrostatic_descriptors,
)

# Import ChemiAccel modules


def main():
    """Main demonstration function."""
    print("ChemiAccel 3D Geometrical Descriptors Demo")
    print("=" * 50)

    # Sample molecules with diverse 3D properties
    test_molecules = [
        "CCO",  # ethanol - simple molecule
        "c1ccccc1",  # benzene - planar aromatic
        "CC(C)C",  # isobutane - branched alkane
        "C1CCCCC1",  # cyclohexane - ring structure
        "CCN(CC)CC",  # triethylamine - tertiary amine
        "CC(=O)O",  # acetic acid - carboxylic acid
        "c1ccc2c(c1)cccn2",  # quinoline - bicyclic aromatic
        "CC(C)(C)c1ccc(O)cc1",  # 4-tert-butylphenol - substituted aromatic
        "CN1CCC[C@H]1c2cccnc2",  # nicotine - complex heterocycle
        "CC(=O)Nc1ccc(O)cc1",  # acetaminophen - drug molecule
    ]

    print(f"Processing {len(test_molecules)} test molecules...")

    # Parse SMILES into molecular data
    from chemi_accel.utils.rdkit_integration import RDKitMolecularParser

    parser = RDKitMolecularParser()
    molecules = parser.parse_smiles_batch(test_molecules, error_handling="skip")
    print(f"Successfully parsed {len(molecules)} molecules")

    # Demo 1: Conformer Generation
    print("\n1. Conformer Generation")
    print("-" * 30)

    conformer_generator = ConformerGenerator(
        num_conformers=5, optimize_conformers=True, force_field="UFF"
    )

    start_time = time.time()
    molecules_with_conformers = conformer_generator.generate_conformers_batch(molecules)
    conformer_time = time.time() - start_time

    print(
        f"Generated conformers for {len(molecules_with_conformers)} molecules in {conformer_time:.2f}s"
    )

    # Show conformer statistics
    conformer_counts = []
    for mol in molecules_with_conformers:
        rdkit_mol = mol.backend_data.get("rdkit_mol")
        if rdkit_mol:
            count = rdkit_mol.GetNumConformers()
            conformer_counts.append(count)
            print(f"  {mol.smiles}: {count} conformers")

    if conformer_counts:
        print(f"Average conformers per molecule: {np.mean(conformer_counts):.1f}")

    # Demo 2: 3D Shape Descriptors
    print("\n2. 3D Shape Descriptors")
    print("-" * 30)

    shape_calculator = Shape3DDescriptors()

    # Calculate shape descriptors for molecules with conformers
    valid_molecules = [
        mol
        for mol in molecules_with_conformers
        if mol.backend_data.get("rdkit_mol")
        and mol.backend_data.get("rdkit_mol").GetNumConformers() > 0
    ]

    if valid_molecules:
        start_time = time.time()
        shape_results = shape_calculator.calculate_shape_descriptors_batch(
            valid_molecules[:5]
        )
        shape_time = time.time() - start_time

        print(
            f"Calculated shape descriptors for {len(shape_results)} molecules in {shape_time:.2f}s"
        )

        if len(shape_results) > 0:
            print("\nShape Descriptor Results:")
            shape_cols = [
                col for col in shape_results.columns if col not in ["mol_id", "smiles"]
            ]
            print(shape_results[["smiles"] + shape_cols[:5]].to_string(index=False))

            # Show descriptor statistics
            print("\nShape Descriptor Statistics:")
            for col in shape_cols[:5]:
                if col in shape_results.columns:
                    values = shape_results[col].dropna()
                    if len(values) > 0:
                        print(
                            f"  {col}: mean={values.mean():.3f}, std={values.std():.3f}"
                        )

    # Demo 3: Electrostatic Descriptors
    print("\n3. Electrostatic Descriptors")
    print("-" * 30)

    electrostatic_calculator = ElectrostaticDescriptors(charge_model="gasteiger")

    if valid_molecules:
        start_time = time.time()
        electrostatic_results = calculate_electrostatic_descriptors(valid_molecules[:5])
        electrostatic_time = time.time() - start_time

        print(
            f"Calculated electrostatic descriptors for {len(electrostatic_results)} molecules in {electrostatic_time:.2f}s"
        )

        if len(electrostatic_results) > 0:
            print("\nElectrostatic Descriptor Results:")
            electrostatic_cols = [
                "total_charge",
                "dipole_moment",
                "charge_range",
                "positive_charge_sum",
            ]
            available_cols = [
                col
                for col in electrostatic_cols
                if col in electrostatic_results.columns
            ]
            print(
                electrostatic_results[["smiles"] + available_cols].to_string(
                    index=False
                )
            )

    # Demo 4: Comprehensive 3D Analysis
    print("\n4. Comprehensive 3D Analysis")
    print("-" * 30)

    calculator = Geometrical3DCalculator(
        generate_conformers=False, charge_model="gasteiger"  # Already have conformers
    )

    if valid_molecules:
        start_time = time.time()
        comprehensive_results = calculator.calculate_3d_descriptors_batch(
            valid_molecules[:5], include_shape=True, include_electrostatic=True
        )
        comprehensive_time = time.time() - start_time

        print(
            f"Comprehensive 3D analysis for {len(comprehensive_results)} molecules in {comprehensive_time:.2f}s"
        )

        if len(comprehensive_results) > 0:
            print(
                f"\nCalculated {len(comprehensive_results.columns) - 2} descriptors per molecule"
            )

            # Show available descriptor categories
            available_descriptors = calculator.get_available_descriptors()
            print(
                f"Available shape descriptors: {len(available_descriptors['shape_descriptors'])}"
            )
            print(
                f"Available electrostatic descriptors: {len(available_descriptors['electrostatic_descriptors'])}"
            )

    # Demo 5: Convenience Functions
    print("\n5. Convenience Functions")
    print("-" * 30)

    # Use convenience function for end-to-end processing
    start_time = time.time()
    convenience_results = calculate_3d_descriptors(
        molecules[:3],  # Use original molecules without conformers
        include_shape=True,
        include_electrostatic=True,
        generate_conformers=True,
        num_conformers=3,
    )
    convenience_time = time.time() - start_time

    print(
        f"End-to-end 3D descriptor calculation for {len(convenience_results)} molecules in {convenience_time:.2f}s"
    )

    if len(convenience_results) > 0:
        print("Sample results from convenience function:")
        sample_cols = [
            "smiles",
            "molecular_volume",
            "dipole_moment",
            "radius_of_gyration",
        ]
        available_sample_cols = [
            col for col in sample_cols if col in convenience_results.columns
        ]
        print(convenience_results[available_sample_cols].head().to_string(index=False))

    # Demo 6: Performance Comparison
    print("\n6. Performance Analysis")
    print("-" * 30)

    if valid_molecules:
        # Time individual components
        single_mol = valid_molecules[0]

        # Shape descriptors timing
        start_time = time.time()
        shape_desc = shape_calculator.calculate_shape_descriptors(single_mol)
        single_shape_time = time.time() - start_time

        # Electrostatic descriptors timing
        start_time = time.time()
        electrostatic_desc = (
            electrostatic_calculator.calculate_electrostatic_descriptors(single_mol)
        )
        single_electrostatic_time = time.time() - start_time

        print("Single molecule timing:")
        print(f"  Shape descriptors: {single_shape_time*1000:.1f}ms")
        print(f"  Electrostatic descriptors: {single_electrostatic_time*1000:.1f}ms")
        print(
            f"  Total per molecule: {(single_shape_time + single_electrostatic_time)*1000:.1f}ms"
        )

        # Estimate throughput
        total_time_per_mol = single_shape_time + single_electrostatic_time
        if total_time_per_mol > 0:
            throughput = 1.0 / total_time_per_mol
            print(f"  Estimated throughput: {throughput:.1f} molecules/second")

    # Demo 7: Descriptor Interpretation
    print("\n7. Descriptor Interpretation")
    print("-" * 30)

    if len(comprehensive_results) > 0:
        print("Descriptor interpretation examples:")

        for idx, row in comprehensive_results.head(3).iterrows():
            smiles = row["smiles"]
            print(f"\nMolecule: {smiles}")

            # Shape interpretation
            if "molecular_volume" in row:
                volume = row["molecular_volume"]
                print(f"  Volume: {volume:.1f} Ų (molecular size)")

            if "asphericity" in row:
                asphericity = row["asphericity"]
                shape_desc = (
                    "spherical"
                    if asphericity < 0.1
                    else "elongated"
                    if asphericity > 0.5
                    else "intermediate"
                )
                print(f"  Asphericity: {asphericity:.3f} ({shape_desc} shape)")

            if "geometric_diameter" in row:
                diameter = row["geometric_diameter"]
                print(f"  Diameter: {diameter:.1f} Å (maximum span)")

            # Electrostatic interpretation
            if "dipole_moment" in row:
                dipole = row["dipole_moment"]
                polarity = (
                    "nonpolar"
                    if dipole < 1.0
                    else "polar"
                    if dipole < 3.0
                    else "highly polar"
                )
                print(f"  Dipole moment: {dipole:.2f} D ({polarity})")

            if "total_charge" in row:
                charge = row["total_charge"]
                charge_desc = "neutral" if abs(charge) < 0.1 else "charged"
                print(f"  Total charge: {charge:.3f} ({charge_desc})")

    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("\nKey Features Demonstrated:")
    print("✓ 3D conformer generation with optimization")
    print("✓ Comprehensive shape descriptor calculations")
    print("✓ Electrostatic property analysis")
    print("✓ Batch processing capabilities")
    print("✓ Performance optimization")
    print("✓ Convenient high-level API")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error running demo: {e}")
        import traceback

        traceback.print_exc()
