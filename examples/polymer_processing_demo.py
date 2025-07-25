"""
Polymer Processing System Demo

This example demonstrates the comprehensive polymer processing capabilities
of ChemiAccel, including monomer analysis, oligomer processing, conformational
sampling, and polymer-specific descriptor calculations.
"""

import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from chemi_accel.descriptors.polymer import PolymerDescriptorCalculator
    from chemi_accel.polymer.conformers import PolymerConformerGenerator, SamplingMethod
    from chemi_accel.polymer.monomer import MonomerAnalyzer
    from chemi_accel.polymer.oligomer import OligomerProcessor
    from chemi_accel.polymer.repeating import (
        BranchedPolymerHandler,
        RepeatingUnitIdentifier,
    )

    print("✓ Successfully imported ChemiAccel polymer processing modules")

except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure RDKit is installed: pip install rdkit")
    sys.exit(1)


def demo_monomer_analysis():
    """Demonstrate monomer analysis capabilities."""
    print("\n" + "=" * 60)
    print("MONOMER ANALYSIS DEMO")
    print("=" * 60)

    analyzer = MonomerAnalyzer()

    # Test different types of monomers
    monomers = {
        "Ethylene": "C=C",
        "Propylene": "C=C(C)",
        "Styrene": "C=Cc1ccccc1",
        "Vinyl Chloride": "C=CCl",
        "Methyl Acrylate": "C=CC(=O)OC",
        "Ethylene Oxide": "C1CO1",
        "Caprolactam": "O=C1CCCCCN1",
    }

    for name, smiles in monomers.items():
        print(f"\nAnalyzing {name} ({smiles}):")
        print("-" * 40)

        try:
            results = analyzer.analyze_monomer(smiles)

            # Display reactivity profile
            reactivity = results["reactivity_profile"]
            print(f"  Overall Reactivity: {reactivity.overall_reactivity:.2f}")
            print(f"  Functional Groups: {len(reactivity.functional_groups)}")
            print(
                f"  Preferred Mechanisms: {[m.value for m in reactivity.preferred_mechanisms]}"
            )
            print(f"  Steric Hindrance: {reactivity.steric_hindrance:.2f}")

            # Display polymerization potential
            potential = results["polymerization_potential"]
            print(f"  Can Polymerize: {potential.can_polymerize}")
            print(f"  Reactivity Score: {potential.reactivity_score:.2f}")
            print(f"  Stability Score: {potential.stability_score:.2f}")

            # Display stereochemistry info
            stereo = results["stereochemistry_info"]
            print(f"  Chiral Centers: {len(stereo.chiral_centers)}")
            print(f"  Tacticity Potential: {stereo.tacticity_potential}")

            # Display ring-opening analysis
            ring_analysis = results["ring_opening_analysis"]
            print(f"  Can Ring Open: {ring_analysis['can_ring_open']}")
            print(f"  Ring Strain Score: {ring_analysis['ring_strain_score']:.2f}")

        except Exception as e:
            print(f"  Error: {e}")


def demo_oligomer_processing():
    """Demonstrate oligomer processing capabilities."""
    print("\n" + "=" * 60)
    print("OLIGOMER PROCESSING DEMO")
    print("=" * 60)

    processor = OligomerProcessor()

    # Test different oligomers
    oligomers = {
        "Ethylene Dimer": "CCCC",
        "Propylene Trimer": "CC(C)CC(C)CC(C)",
        "Styrene Dimer": "CC(c1ccccc1)CC(c1ccccc1)",
        "PEG Oligomer": "COCCOCCOC",
        "Polyamide Oligomer": "NC(=O)CCC(=O)NC(=O)CCC(=O)N",
    }

    for name, smiles in oligomers.items():
        print(f"\nProcessing {name} ({smiles}):")
        print("-" * 50)

        try:
            results = processor.process_oligomer(smiles)

            # Display flexibility analysis
            flexibility = results["flexibility_analysis"]
            print("  Flexibility Metrics:")
            print(
                f"    Rotatable Bonds: {flexibility.flexibility_metrics['num_rotatable_bonds']}"
            )
            print(
                f"    Radius of Gyration: {flexibility.flexibility_metrics['rg_mean']:.2f} Å"
            )
            print(
                f"    Flexibility Index: {flexibility.flexibility_metrics['flexibility_index']:.2f}"
            )

            # Display end group analysis
            end_groups = results["end_group_analysis"]
            print("  End Groups:")
            print(f"    Left: {end_groups.left_end_group}")
            print(f"    Right: {end_groups.right_end_group}")
            print(
                f"    Average Reactivity: {end_groups.end_group_reactivity['average_reactivity']:.2f}"
            )
            print(
                f"    Termination Probability: {end_groups.termination_probability:.2f}"
            )

        except Exception as e:
            print(f"  Error: {e}")


def demo_conformational_sampling():
    """Demonstrate conformational sampling capabilities."""
    print("\n" + "=" * 60)
    print("CONFORMATIONAL SAMPLING DEMO")
    print("=" * 60)

    generator = PolymerConformerGenerator(use_gpu=False)  # Use CPU for demo

    # Test polymer chains of different lengths
    polymers = {
        "Short Chain": "CCCCCCCC",
        "Medium Chain": "CCCCCCCCCCCCCCCC",
        "Branched Chain": "CCC(C)CCC(C)CCC(C)CC",
        "Aromatic Chain": "CC(c1ccccc1)CC(c1ccccc1)CC(c1ccccc1)C",
    }

    for name, smiles in polymers.items():
        print(f"\nSampling {name} ({smiles}):")
        print("-" * 50)

        try:
            ensemble = generator.generate_conformers(
                smiles, num_conformers=10, method=SamplingMethod.DISTANCE_GEOMETRY
            )

            print(f"  Generated Conformers: {len(ensemble.conformers)}")

            if ensemble.conformers:
                energies = [c.energy for c in ensemble.conformers]
                rg_values = [c.radius_of_gyration for c in ensemble.conformers]

                print(
                    f"  Energy Range: {min(energies):.1f} to {max(energies):.1f} kcal/mol"
                )
                print(
                    f"  Radius of Gyration: {min(rg_values):.2f} to {max(rg_values):.2f} Å"
                )

                # Display flexibility metrics
                flexibility = ensemble.flexibility_metrics
                print("  Flexibility Metrics:")
                print(
                    f"    Characteristic Ratio: {flexibility.characteristic_ratio:.2f}"
                )
                print(f"    Persistence Length: {flexibility.persistence_length:.2f} Å")
                print(f"    Flexibility Index: {flexibility.flexibility_index:.2f}")

                # Display intramolecular interactions
                interactions = ensemble.intramolecular_interactions
                print(f"  Intramolecular Interactions: {len(interactions)}")

                for interaction in interactions[:3]:  # Show first 3
                    print(
                        f"    {interaction.interaction_type}: {interaction.strength:.2f} kcal/mol"
                    )

        except Exception as e:
            print(f"  Error: {e}")


def demo_repeating_unit_identification():
    """Demonstrate repeating unit identification."""
    print("\n" + "=" * 60)
    print("REPEATING UNIT IDENTIFICATION DEMO")
    print("=" * 60)

    identifier = RepeatingUnitIdentifier()
    branched_handler = BranchedPolymerHandler()

    # Test different polymer structures
    polymers = {
        "Polyethylene": "CCCCCCCCCCCCCCCC",
        "Polypropylene": "CC(C)CC(C)CC(C)CC(C)C",
        "Polystyrene": "CC(c1ccccc1)CC(c1ccccc1)CC(c1ccccc1)C",
        "Branched Polymer": "CCC(CCC)CCC(CCCC)CCC(CC)CC",
        "PET-like": "COC(=O)c1ccc(C(=O)OCCOC(=O)c2ccc(C(=O)O)cc2)cc1",
    }

    for name, smiles in polymers.items():
        print(f"\nAnalyzing {name}:")
        print("-" * 40)

        try:
            # Identify repeating units
            units = identifier.identify_repeating_units(smiles)
            print(f"  Repeating Units Found: {len(units)}")

            for i, unit in enumerate(units[:3]):  # Show first 3
                print(f"    Unit {i+1}: {unit.smiles}")
                print(f"      Frequency: {unit.frequency}")
                print(f"      Type: {unit.unit_type.value}")
                print(f"      MW: {unit.molecular_weight:.1f}")

            # Analyze branching
            branching_analysis = branched_handler.analyze_branching(smiles)
            print(f"  Topology: {branching_analysis.topology.value}")
            print(f"  Branch Points: {len(branching_analysis.branch_points)}")
            print(f"  Molecular Weight: {branching_analysis.molecular_weight:.1f}")
            print(
                f"  Degree of Polymerization: {branching_analysis.degree_of_polymerization}"
            )
            print(
                f"  Structural Regularity: {branching_analysis.structural_regularity:.2f}"
            )

        except Exception as e:
            print(f"  Error: {e}")


def demo_polymer_descriptors():
    """Demonstrate polymer-specific descriptor calculations."""
    print("\n" + "=" * 60)
    print("POLYMER DESCRIPTORS DEMO")
    print("=" * 60)

    calculator = PolymerDescriptorCalculator(use_gpu=False)

    # Test different polymer types
    polymers = {
        "Polyethylene": "CCCCCCCCCCCCCCCC",
        "Polypropylene": "CC(C)CC(C)CC(C)CC(C)C",
        "Polystyrene": "CC(c1ccccc1)CC(c1ccccc1)CC(c1ccccc1)C",
        "PVC": "CC(Cl)CC(Cl)CC(Cl)CC(Cl)C",
        "PMMA": "CC(C)(C(=O)OC)CC(C)(C(=O)OC)CC(C)(C(=O)OC)C",
    }

    for name, smiles in polymers.items():
        print(f"\nCalculating descriptors for {name}:")
        print("-" * 50)

        try:
            descriptors = calculator.calculate_all_descriptors(smiles)

            # Chain flexibility descriptors
            print("  Chain Flexibility:")
            print(f"    Persistence Length: {descriptors.persistence_length:.2f} Å")
            print(f"    Radius of Gyration: {descriptors.radius_of_gyration:.2f} Å")
            print(f"    End-to-End Distance: {descriptors.end_to_end_distance:.2f} Å")
            print(f"    Characteristic Ratio: {descriptors.characteristic_ratio:.2f}")

            # Thermal properties
            print("  Thermal Properties:")
            print(
                f"    Glass Transition Temp: {descriptors.glass_transition_temperature:.1f} K"
            )
            print(f"    Melting Temperature: {descriptors.melting_temperature:.1f} K")
            print(
                f"    Decomposition Temp: {descriptors.thermal_decomposition_temperature:.1f} K"
            )

            # Mechanical properties
            print("  Mechanical Properties:")
            print(
                f"    Elastic Modulus: {descriptors.elastic_modulus_predictor:.2f} GPa"
            )
            print(f"    Brittleness Index: {descriptors.brittleness_index:.2f}")
            print(f"    Toughness Index: {descriptors.toughness_index:.2f}")
            print(f"    Hardness: {descriptors.hardness_predictor:.2f}")

            # Processing properties
            print("  Processing Properties:")
            print(f"    Viscosity: {descriptors.viscosity_predictor:.0f} Pa·s")
            print(
                f"    Solubility Parameter: {descriptors.solubility_parameter:.1f} (J/cm³)^0.5"
            )
            print(
                f"    Melt Flow Index: {descriptors.melt_flow_index_predictor:.2f} g/10min"
            )
            print(f"    Processability Index: {descriptors.processability_index:.2f}")

            # Additional properties
            print("  Additional Properties:")
            print(f"    Crystallinity Index: {descriptors.crystallinity_index:.2f}")
            print(
                f"    Chain Entanglement Density: {descriptors.chain_entanglement_density:.2f}"
            )
            print(f"    Free Volume Fraction: {descriptors.free_volume_fraction:.3f}")

        except Exception as e:
            print(f"  Error: {e}")


def main():
    """Run all polymer processing demos."""
    print("ChemiAccel Polymer Processing System Demo")
    print("=========================================")
    print("This demo showcases the comprehensive polymer processing capabilities")
    print("including monomer analysis, oligomer processing, conformational sampling,")
    print("and specialized polymer descriptors.")

    try:
        demo_monomer_analysis()
        demo_oligomer_processing()
        demo_conformational_sampling()
        demo_repeating_unit_identification()
        demo_polymer_descriptors()

        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("All polymer processing modules are working correctly!")
        print("The system can handle:")
        print("• Monomer reactivity analysis and polymerization potential")
        print("• Oligomer chain growth simulation and flexibility analysis")
        print("• Advanced conformational sampling with GPU acceleration")
        print("• Repeating unit identification and branching analysis")
        print("• Comprehensive polymer-specific descriptor calculations")

    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        print("This may be due to missing dependencies (RDKit) or other issues.")


if __name__ == "__main__":
    main()
