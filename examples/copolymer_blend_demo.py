"""
Demonstration of copolymer and blend property modeling capabilities.

This example shows how to:
1. Create copolymer systems with different architectures
2. Predict copolymer properties using various mixing rules
3. Analyze microphase separation in block copolymers
4. Study polymer blend miscibility
5. Generate composition studies
6. Visualize phase diagrams
"""

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from chemi_accel.polymer.copolymer import (
    CopolymerBlendModeler,
    CopolymerType,
    MixingRule,
)


def demo_styrene_butadiene_copolymer():
    """Demonstrate styrene-butadiene copolymer analysis."""
    print("=" * 60)
    print("STYRENE-BUTADIENE COPOLYMER ANALYSIS")
    print("=" * 60)

    modeler = CopolymerBlendModeler()

    # Define monomer data
    styrene_data = {
        "smiles": "C=Cc1ccccc1",
        "name": "Styrene",
        "mole_fraction": 0.7,
        "weight_fraction": 0.75,
        "molecular_weight": 104.15,
        "glass_transition_temp": 373.0,
        "density": 1.05,
        "solubility_parameter": 18.5,
        "properties": {
            "youngs_modulus": 3200.0,
            "tensile_strength": 40.0,
            "thermal_conductivity": 0.13,
        },
    }

    butadiene_data = {
        "smiles": "C=CC=C",
        "name": "Butadiene",
        "mole_fraction": 0.3,
        "weight_fraction": 0.25,
        "molecular_weight": 54.09,
        "glass_transition_temp": 165.0,
        "density": 0.90,
        "solubility_parameter": 17.0,
        "properties": {
            "youngs_modulus": 1.5,
            "tensile_strength": 2.0,
            "thermal_conductivity": 0.25,
        },
    }

    # Create different copolymer architectures
    architectures = [
        (CopolymerType.RANDOM, "Random Copolymer (SBR)"),
        (CopolymerType.ALTERNATING, "Alternating Copolymer"),
        (CopolymerType.BLOCK, "Block Copolymer (SBS)"),
        (CopolymerType.GRAFT, "Graft Copolymer"),
    ]

    monomer_data = [styrene_data, butadiene_data]

    print("Comparing different copolymer architectures:")
    print("-" * 50)

    for arch_type, arch_name in architectures:
        # Create copolymer system
        system = modeler.create_copolymer_system(
            monomer_data,
            arch_type,
            sequence_length=1000 if arch_type == CopolymerType.BLOCK else None,
        )

        # Predict properties
        results = modeler.predict_copolymer_properties(system)
        properties = results["predicted_properties"]

        print(f"\n{arch_name}:")
        print(f"  Glass Transition Temp: {properties.glass_transition_temp:.1f} K")
        print(f"  Density: {properties.density:.3f} g/cm³")
        print(f"  Young's Modulus: {properties.youngs_modulus:.1f} MPa")

        # Show microphase analysis for block copolymer
        if arch_type == CopolymerType.BLOCK and "microphase_info" in results:
            microphase = results["microphase_info"]
            print(f"  Microphase Separated: {microphase.is_microphase_separated}")
            if microphase.is_microphase_separated:
                print(f"  Morphology: {microphase.morphology}")
                print(f"  Domain Size: {microphase.domain_size:.1f} nm")
                print(f"  Chi Parameter: {microphase.chi_parameter:.3f}")


def demo_mixing_rules_comparison():
    """Demonstrate different mixing rules for property prediction."""
    print("\n" + "=" * 60)
    print("MIXING RULES COMPARISON")
    print("=" * 60)

    modeler = CopolymerBlendModeler()

    # Create a binary copolymer system
    monomer_data = [
        {
            "smiles": "C=Cc1ccccc1",
            "name": "Styrene",
            "mole_fraction": 0.5,
            "weight_fraction": 0.55,
            "molecular_weight": 104.15,
            "glass_transition_temp": 373.0,
        },
        {
            "smiles": "C=C(C)C(=O)OC",
            "name": "MMA",
            "mole_fraction": 0.5,
            "weight_fraction": 0.45,
            "molecular_weight": 100.12,
            "glass_transition_temp": 378.0,
        },
    ]

    system = modeler.create_copolymer_system(monomer_data, CopolymerType.RANDOM)

    # Test different mixing rules for Tg
    mixing_rules = [
        (MixingRule.LINEAR, "Linear Mixing"),
        (MixingRule.GORDON_TAYLOR, "Gordon-Taylor"),
        (MixingRule.FOX, "Fox Equation"),
        (MixingRule.LOGARITHMIC, "Logarithmic Mixing"),
    ]

    print("Glass transition temperature predictions:")
    print("-" * 40)

    for rule, rule_name in mixing_rules:
        try:
            custom_rules = {"glass_transition_temp": rule}
            properties = modeler.property_predictor.predict_properties(
                system, custom_rules
            )

            print(f"{rule_name:20}: {properties.glass_transition_temp:.1f} K")

        except Exception as e:
            print(f"{rule_name:20}: Error - {str(e)}")


def demo_composition_study():
    """Demonstrate composition study for binary copolymer."""
    print("\n" + "=" * 60)
    print("COMPOSITION STUDY")
    print("=" * 60)

    modeler = CopolymerBlendModeler()

    # Define monomers
    styrene_data = {
        "smiles": "C=Cc1ccccc1",
        "name": "Styrene",
        "molecular_weight": 104.15,
        "glass_transition_temp": 373.0,
        "density": 1.05,
        "solubility_parameter": 18.5,
    }

    mma_data = {
        "smiles": "C=C(C)C(=O)OC",
        "name": "MMA",
        "molecular_weight": 100.12,
        "glass_transition_temp": 378.0,
        "density": 1.19,
        "solubility_parameter": 19.3,
    }

    # Generate composition study
    comp_study = modeler.generate_composition_study(
        styrene_data, mma_data, composition_range=(0.0, 1.0), num_points=11
    )

    compositions = comp_study["compositions"]
    properties = comp_study["properties"]
    microphase_info = comp_study["microphase_info"]

    print("Composition study results:")
    print("-" * 50)
    print(f"{'Styrene %':>10} {'Tg (K)':>8} {'Density':>8} {'Microphase':>12}")
    print("-" * 50)

    for i, comp in enumerate(compositions):
        prop = properties[i]
        micro = microphase_info[i]

        print(
            f"{comp*100:>8.0f}% {prop.glass_transition_temp:>8.1f} "
            f"{prop.density:>8.3f} {micro.is_microphase_separated:>12}"
        )


def demo_polymer_blend_analysis():
    """Demonstrate polymer blend miscibility analysis."""
    print("\n" + "=" * 60)
    print("POLYMER BLEND MISCIBILITY ANALYSIS")
    print("=" * 60)

    modeler = CopolymerBlendModeler()

    # Define polymer systems
    pe_data = [
        {
            "smiles": "CC",
            "name": "Polyethylene",
            "mole_fraction": 1.0,
            "weight_fraction": 1.0,
            "molecular_weight": 28.05,
            "glass_transition_temp": 200.0,
            "density": 0.92,
            "solubility_parameter": 16.0,
        }
    ]

    ps_data = [
        {
            "smiles": "CCc1ccccc1",
            "name": "Polystyrene",
            "mole_fraction": 1.0,
            "weight_fraction": 1.0,
            "molecular_weight": 104.15,
            "glass_transition_temp": 373.0,
            "density": 1.05,
            "solubility_parameter": 18.5,
        }
    ]

    # Analyze blend at different compositions
    blend_ratios = [0.2, 0.5, 0.8]

    print("PE/PS blend analysis:")
    print("-" * 40)
    print(f"{'PE %':>6} {'Miscible':>10} {'Chi':>8} {'Phase State':>15}")
    print("-" * 40)

    for ratio in blend_ratios:
        blend_results = modeler.analyze_polymer_blend(
            pe_data, ps_data, blend_ratio=ratio, temperature=298.15
        )

        miscibility = blend_results["miscibility_info"]

        print(
            f"{ratio*100:>4.0f}% {miscibility.is_miscible:>10} "
            f"{miscibility.interaction_parameter:>8.3f} "
            f"{miscibility.phase_state.value:>15}"
        )

    # Analyze temperature effects
    print("\nTemperature effects on miscibility (50/50 blend):")
    print("-" * 40)
    print(f"{'Temp (K)':>8} {'Miscible':>10} {'Chi':>8}")
    print("-" * 40)

    temperatures = [250, 300, 350, 400, 450]

    for temp in temperatures:
        blend_results = modeler.analyze_polymer_blend(
            pe_data, ps_data, blend_ratio=0.5, temperature=temp
        )

        miscibility = blend_results["miscibility_info"]

        print(
            f"{temp:>6.0f}K {miscibility.is_miscible:>10} "
            f"{miscibility.interaction_parameter:>8.3f}"
        )


def demo_microphase_separation_analysis():
    """Demonstrate microphase separation analysis for block copolymers."""
    print("\n" + "=" * 60)
    print("MICROPHASE SEPARATION ANALYSIS")
    print("=" * 60)

    modeler = CopolymerBlendModeler()

    # Create block copolymer systems with different compositions
    compositions = [0.2, 0.3, 0.5, 0.7, 0.8]

    print("Block copolymer microphase separation:")
    print("-" * 60)
    print(
        f"{'f_A':>5} {'Separated':>10} {'Morphology':>12} {'Domain (nm)':>12} {'Chi*N':>8}"
    )
    print("-" * 60)

    for f_A in compositions:
        f_B = 1.0 - f_A

        # Create high molecular weight block copolymer
        monomer_data = [
            {
                "smiles": "C=Cc1ccccc1",
                "name": "Styrene",
                "mole_fraction": f_A,
                "weight_fraction": f_A,
                "molecular_weight": 104.15,
                "solubility_parameter": 18.5,
            },
            {
                "smiles": "C=CC=C",
                "name": "Butadiene",
                "mole_fraction": f_B,
                "weight_fraction": f_B,
                "molecular_weight": 54.09,
                "solubility_parameter": 17.0,
            },
        ]

        system = modeler.create_copolymer_system(
            monomer_data,
            CopolymerType.BLOCK,
            sequence_length=2000,  # High MW for strong segregation
        )

        results = modeler.predict_copolymer_properties(system)
        microphase = results["microphase_info"]

        print(
            f"{f_A:>4.1f} {microphase.is_microphase_separated:>10} "
            f"{microphase.morphology:>12} {microphase.domain_size:>10.1f} "
            f"{microphase.segregation_strength:>8.1f}"
        )


def plot_composition_properties():
    """Plot property variation with composition."""
    try:
        import matplotlib.pyplot as plt

        print("\n" + "=" * 60)
        print("PLOTTING COMPOSITION-PROPERTY RELATIONSHIPS")
        print("=" * 60)

        modeler = CopolymerBlendModeler()

        # Define monomers
        styrene_data = {
            "smiles": "C=Cc1ccccc1",
            "name": "Styrene",
            "molecular_weight": 104.15,
            "glass_transition_temp": 373.0,
            "density": 1.05,
        }

        butadiene_data = {
            "smiles": "C=CC=C",
            "name": "Butadiene",
            "molecular_weight": 54.09,
            "glass_transition_temp": 165.0,
            "density": 0.90,
        }

        # Generate detailed composition study
        comp_study = modeler.generate_composition_study(
            styrene_data, butadiene_data, composition_range=(0.0, 1.0), num_points=21
        )

        compositions = np.array(comp_study["compositions"])
        properties = comp_study["properties"]

        # Extract properties
        tg_values = [prop.glass_transition_temp for prop in properties]
        density_values = [prop.density for prop in properties]

        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot Tg vs composition
        ax1.plot(compositions * 100, tg_values, "bo-", linewidth=2, markersize=6)
        ax1.set_xlabel("Styrene Content (%)")
        ax1.set_ylabel("Glass Transition Temperature (K)")
        ax1.set_title("Tg vs Composition")
        ax1.grid(True, alpha=0.3)

        # Plot density vs composition
        ax2.plot(compositions * 100, density_values, "ro-", linewidth=2, markersize=6)
        ax2.set_xlabel("Styrene Content (%)")
        ax2.set_ylabel("Density (g/cm³)")
        ax2.set_title("Density vs Composition")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            "copolymer_properties_vs_composition.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        print("Plot saved as 'copolymer_properties_vs_composition.png'")

    except ImportError:
        print("Matplotlib not available for plotting")


def main():
    """Run all copolymer demonstration examples."""
    print("CHEMI-ACCEL COPOLYMER AND BLEND MODELING DEMONSTRATION")
    print("=" * 80)

    # Run demonstrations
    demo_styrene_butadiene_copolymer()
    demo_mixing_rules_comparison()
    demo_composition_study()
    demo_polymer_blend_analysis()
    demo_microphase_separation_analysis()
    plot_composition_properties()

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETED")
    print("=" * 80)
    print("\nThis demonstration showed:")
    print("• Copolymer system creation with different architectures")
    print("• Property prediction using various mixing rules")
    print("• Microphase separation analysis for block copolymers")
    print("• Polymer blend miscibility assessment")
    print("• Composition studies and property trends")
    print("• Visualization of composition-property relationships")
    print("\nThe copolymer modeling module provides comprehensive")
    print("tools for analyzing complex polymer systems!")


if __name__ == "__main__":
    main()
