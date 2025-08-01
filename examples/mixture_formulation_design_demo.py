"""
ChemiAccel Mixture & Formulation Design Demo

Comprehensive demonstration of AI-driven complex mixture and formulation design
capabilities including phase behavior prediction, interaction modeling, inverse
design, dynamic processes, and interactive visualization.

This example showcases the complete workflow from mixture creation to
optimization and visualization.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Import ChemiAccel mixture components
try:
    from chemi_accel.mixture import (
        ComponentData,
        DynamicProcessIntegrator,
        InteractionModeler,
        MixtureCalculator,
        MixtureData,
        MixtureInverseDesigner,
        MixtureVisualizer,
        PhaseBehaviorPredictor,
    )
    from chemi_accel.mixture.dynamic_processes import ProcessConditions
    from chemi_accel.mixture.inverse_design import (
        CompositionConstraint,
        CostObjective,
        PropertyConstraint,
        PropertyTargetObjective,
        StabilityObjective,
    )
except ImportError as e:
    print(f"Error importing ChemiAccel mixture components: {e}")
    print("Please ensure ChemiAccel is properly installed with mixture dependencies.")
    exit(1)


def demo_basic_mixture_analysis():
    """Demonstrate basic mixture analysis workflow."""
    print("=== Basic Mixture Analysis Demo ===")

    # Create mixture calculator with GPU acceleration
    calc = MixtureCalculator(backend="cuda", use_gpu=True)
    print("✓ Created MixtureCalculator with CUDA backend")

    # Create a ternary solvent mixture: water, ethanol, acetone
    components = [
        ("O", 0.5),  # Water - 50%
        ("CCO", 0.3),  # Ethanol - 30%
        ("CC(=O)C", 0.2),  # Acetone - 20%
    ]

    mixture = calc.create_mixture(
        components,
        temperature=298.15,  # Room temperature
        pressure=101325.0,  # Atmospheric pressure
    )

    print(f"✓ Created mixture with {mixture.n_components} components")
    print(f"  Components: {mixture.smiles_list}")
    print(f"  Composition: {mixture.composition_vector}")

    # Calculate basic mixture properties
    properties = calc.calculate_mixture_properties(mixture)
    print("✓ Calculated mixture properties:")
    for prop, value in properties.items():
        print(f"  {prop}: {value:.3f}")

    return calc, mixture


def demo_phase_behavior_prediction(calc: MixtureCalculator, mixture: MixtureData):
    """Demonstrate phase behavior prediction."""
    print("\n=== Phase Behavior Prediction Demo ===")

    # Predict phase equilibrium
    phase_result = calc.predict_phase_behavior(
        mixture,
        temperature_range=(250, 400),  # Temperature sweep
        pressure_range=(1e5, 5e5),  # Pressure sweep
        model="nrtl",  # Use NRTL thermodynamic model
    )

    print("✓ Predicted phase behavior:")
    print(f"  Temperature: {phase_result.temperature:.1f} K")
    print(f"  Pressure: {phase_result.pressure:.0f} Pa")
    print(f"  Number of phases: {phase_result.n_phases}")
    print(f"  System stability: {phase_result.is_stable}")

    if phase_result.stability:
        for metric, value in phase_result.stability.items():
            print(f"  {metric} stability: {value:.3f}")

    # Generate phase diagram data
    if phase_result.phase_diagram_data:
        data_points = len(phase_result.phase_diagram_data["data_points"])
        print(f"✓ Generated phase diagram with {data_points} data points")

    return phase_result


def demo_interaction_modeling(calc: MixtureCalculator, mixture: MixtureData):
    """Demonstrate molecular interaction modeling."""
    print("\n=== Molecular Interaction Modeling Demo ===")

    # Calculate all types of interactions
    interactions = calc.model_interactions(
        mixture,
        interaction_types=[
            "hydrogen_bonding",
            "van_der_waals",
            "electrostatic",
            "pi_stacking",
        ],
        include_self_interactions=True,
    )

    print(f"✓ Calculated {len(interactions)} molecular interactions")

    # Analyze interaction patterns
    interaction_summary = {}
    for interaction in interactions:
        itype = interaction.interaction_type
        if itype not in interaction_summary:
            interaction_summary[itype] = {"count": 0, "total_energy": 0.0}
        interaction_summary[itype]["count"] += 1
        interaction_summary[itype]["total_energy"] += interaction.energy

    print("  Interaction summary:")
    for itype, data in interaction_summary.items():
        avg_energy = data["total_energy"] / data["count"] if data["count"] > 0 else 0
        print(
            f"    {itype}: {data['count']} interactions, "
            f"avg energy: {avg_energy:.2f} kJ/mol"
        )

    # Predict mixing behavior
    mixing_prediction = calc.interaction_modeler.predict_mixing_behavior(
        mixture, interactions
    )

    print("✓ Mixing behavior prediction:")
    print(
        f"  Total interaction energy: {mixing_prediction['total_interaction_energy']:.2f} kJ/mol"
    )
    print(f"  Miscibility score: {mixing_prediction['miscibility_score']:.3f}")
    print(
        f"  Phase separation tendency: {mixing_prediction['phase_separation_tendency']:.3f}"
    )
    print(f"  Dominant interactions: {mixing_prediction['dominant_interactions']}")

    return interactions, mixing_prediction


def demo_inverse_design():
    """Demonstrate inverse design for optimal mixture composition."""
    print("\n=== Inverse Design Demo ===")

    calc = MixtureCalculator(backend="cuda")

    # Define target properties for an ideal solvent mixture
    target_properties = {
        "molecular_weight_avg": 35.0,  # Target average molecular weight
        "density_mixture": 0.85,  # Target density g/cm³
        "viscosity_mixture": 0.5,  # Target viscosity cP
    }

    # Available components for optimization
    component_library = [
        "O",  # Water
        "CCO",  # Ethanol
        "CC(=O)C",  # Acetone
        "CCCCCC",  # Hexane
        "c1ccccc1",  # Benzene
    ]

    # Define optimization constraints
    constraints = {
        "composition_bounds": {
            "O": (0.1, 0.7),  # Water: 10-70%
            "CCO": (0.0, 0.5),  # Ethanol: 0-50%
            "CCCCCC": (0.0, 0.3),  # Hexane: 0-30% (limit hydrocarbon)
        },
        "property_bounds": {"molecular_weight_avg": (20.0, 50.0)},  # MW constraint
    }

    print("✓ Defined optimization problem:")
    print(f"  Target properties: {target_properties}")
    print(f"  Component library: {len(component_library)} components")
    print("  Constraints: composition and property bounds")

    # Run inverse design optimization
    try:
        optimal_composition = calc.design_optimal_mixture(
            target_properties=target_properties,
            component_library=component_library,
            constraints=constraints,
            optimization_method="differential_evolution",
            max_iterations=50,
        )

        print("✓ Found optimal composition:")
        print(f"  Feasibility score: {optimal_composition.feasibility_score:.3f}")
        print(f"  Objective value: {optimal_composition.objective_values[0]:.4f}")

        for i, (component, fraction) in enumerate(
            zip(optimal_composition.components, optimal_composition.composition)
        ):
            if fraction > 0.01:  # Only show significant components
                print(f"    {component}: {fraction:.3f} ({fraction*100:.1f}%)")

        print("  Predicted properties:")
        for prop, value in optimal_composition.properties.items():
            print(f"    {prop}: {value:.3f}")

        return optimal_composition

    except Exception as e:
        print(f"⚠ Optimization failed: {e}")
        print("  This is expected in demo mode without full optimization dependencies")
        return None


def demo_multi_objective_optimization():
    """Demonstrate multi-objective optimization for competing goals."""
    print("\n=== Multi-Objective Optimization Demo ===")

    calc = MixtureCalculator(backend="cuda")

    # Define multiple competing objectives
    objectives = [
        PropertyTargetObjective({"molecular_weight_avg": 30.0}),  # Property target
        StabilityObjective(["thermal", "mechanical"]),  # Maximize stability
        CostObjective(
            {  # Minimize cost
                "O": 0.1,  # Water - cheap
                "CCO": 1.0,  # Ethanol - moderate
                "CC(=O)C": 1.5,  # Acetone - expensive
                "c1ccccc1": 3.0,  # Benzene - very expensive
            }
        ),
    ]

    component_library = ["O", "CCO", "CC(=O)C", "c1ccccc1"]

    print("✓ Defined multi-objective problem:")
    print(f"  Objectives: {len(objectives)} competing goals")
    print("    1. Property matching (minimize deviation)")
    print("    2. Stability maximization")
    print("    3. Cost minimization")

    try:
        # Find Pareto-optimal solutions
        pareto_solutions = calc.inverse_designer.design_pareto_optimal_mixtures(
            objectives=objectives,
            component_library=component_library,
            n_solutions=20,
            n_generations=30,
        )

        print(f"✓ Found {len(pareto_solutions)} Pareto-optimal solutions")

        # Display best solutions for each objective
        if pareto_solutions:
            best_property = min(pareto_solutions, key=lambda x: x.objective_values[0])
            best_stability = max(pareto_solutions, key=lambda x: x.objective_values[1])
            best_cost = min(pareto_solutions, key=lambda x: x.objective_values[2])

            print("  Trade-off analysis:")
            print(
                f"    Best property match: objectives = {best_property.objective_values}"
            )
            print(f"    Best stability: objectives = {best_stability.objective_values}")
            print(f"    Best cost: objectives = {best_cost.objective_values}")

        return pareto_solutions

    except Exception as e:
        print(f"⚠ Multi-objective optimization failed: {e}")
        print("  This is expected in demo mode without full optimization dependencies")
        return []


def demo_dynamic_process_simulation(calc: MixtureCalculator, mixture: MixtureData):
    """Demonstrate dynamic process simulation."""
    print("\n=== Dynamic Process Simulation Demo ===")

    # Define dynamic process conditions
    conditions = ProcessConditions()

    # Temperature ramp from 298K to 350K over 1 hour
    conditions.temperature_profile = lambda t: 298.15 + 52 * (t / 3600.0)

    # Constant pressure
    conditions.pressure_profile = lambda t: 101325.0

    # Increasing shear rate for mixing
    conditions.shear_rate_profile = lambda t: 10.0 * (1 + t / 1800.0)

    print("✓ Defined dynamic process conditions:")
    print("  Temperature: 298K → 350K (linear ramp)")
    print("  Pressure: constant 1 atm")
    print("  Shear rate: 10 → 20 s⁻¹ (mixing)")

    # Simulate crystallization and mixing processes
    try:
        process_result = calc.integrate_dynamic_processes(
            mixture=mixture,
            process_conditions=conditions,
            process_types=["crystallization", "mixing"],
            simulation_time=3600.0,  # 1 hour
            dt=60.0,  # 1 minute time steps
        )

        print("✓ Completed dynamic simulation:")
        print(f"  Process type: {process_result.process_type}")
        print(f"  Time points: {len(process_result.time_points)}")
        print(f"  Final time: {process_result.time_points[-1]:.0f} seconds")

        # Analyze final state
        final_properties = process_result.final_properties
        print("  Final properties:")
        for prop, value in final_properties.items():
            if isinstance(value, (int, float)):
                print(f"    {prop}: {value:.3f}")

        return process_result

    except Exception as e:
        print(f"⚠ Process simulation failed: {e}")
        print("  This is expected in demo mode")
        return None


def demo_interactive_visualization(
    calc: MixtureCalculator,
    mixture: MixtureData,
    phase_result=None,
    interactions=None,
    process_result=None,
):
    """Demonstrate interactive visualization capabilities."""
    print("\n=== Interactive Visualization Demo ===")

    visualizer = MixtureVisualizer()

    # Create comprehensive dashboard
    dashboard_config = visualizer.create_dashboard(
        mixture=mixture,
        phase_equilibrium=phase_result,
        interactions=interactions,
        process_results=process_result,
    )

    print("✓ Created interactive dashboard:")
    print(f"  Dashboard type: {dashboard_config['type']}")
    print(f"  Number of panels: {len(dashboard_config['panels'])}")
    print(f"  Layout: {dashboard_config['layout']}")

    # List dashboard panels
    print("  Panels:")
    for i, panel in enumerate(dashboard_config["panels"], 1):
        print(f"    {i}. {panel['title']} ({panel['visualization']})")

    # Generate phase diagram visualization
    if phase_result:
        try:
            phase_viz = visualizer.create_phase_diagram(
                phase_result, diagram_type="2d", interactive=True
            )
            print("✓ Generated phase diagram visualization:")
            print(f"  Type: {phase_viz['type']}")
            print(f"  Interactive: {phase_viz['interactive']}")
            if "widgets" in phase_viz:
                print(f"  Interactive widgets: {len(phase_viz['widgets'])}")
        except Exception as e:
            print(f"⚠ Phase diagram visualization failed: {e}")

    # Generate interaction network visualization
    if interactions:
        try:
            interaction_viz = visualizer.visualize_interactions(
                interactions, mixture, visualization_type="network", interactive=True
            )
            print("✓ Generated interaction network:")
            print(f"  Type: {interaction_viz['type']}")
            print(f"  Nodes: {len(interaction_viz['data']['nodes'])}")
            print(f"  Edges: {len(interaction_viz['data']['edges'])}")
        except Exception as e:
            print(f"⚠ Interaction visualization failed: {e}")

    # Generate process animation
    if process_result:
        try:
            process_animation = visualizer.create_process_animation(
                process_result,
                property_names=["temperature", "mixing_homogeneity"],
                animation_type="time_series",
                interactive=True,
            )
            print("✓ Generated process animation:")
            print(f"  Type: {process_animation['type']}")
            if "widgets" in process_animation:
                print(f"  Controls: {list(process_animation['widgets'].keys())}")
        except Exception as e:
            print(f"⚠ Process animation failed: {e}")

    return dashboard_config


def demo_property_landscape_generation():
    """Demonstrate property landscape generation for visualization."""
    print("\n=== Property Landscape Generation Demo ===")

    calc = MixtureCalculator(backend="cpu")  # Use CPU for broader compatibility
    visualizer = MixtureVisualizer()

    # Generate mixture data across composition space
    print("✓ Generating mixture data across composition space...")

    mixture_data = []
    n_points = 15

    # Binary system: water-ethanol
    for i in range(n_points):
        x_water = i / (n_points - 1)
        x_ethanol = 1.0 - x_water

        components = [("O", x_water), ("CCO", x_ethanol)]
        mixture = calc.create_mixture(components)

        # Calculate mock properties (in real use, these would be predicted)
        mixture.mixture_properties["density"] = 1.0 - 0.2 * x_ethanol
        mixture.mixture_properties["viscosity"] = 1.0 + 2.0 * x_water * x_ethanol
        mixture.mixture_properties["surface_tension"] = 72.0 - 20.0 * x_ethanol

        mixture_data.append(mixture)

    print(f"  Generated {len(mixture_data)} mixture compositions")

    # Create property landscapes
    properties_to_map = ["density", "viscosity", "surface_tension"]

    for prop_name in properties_to_map:
        try:
            landscape = visualizer.create_property_landscape(
                mixture_data=mixture_data,
                property_name=prop_name,
                resolution=20,
                interactive=True,
            )

            print(f"✓ Created {prop_name} landscape:")
            print(f"  Components: {landscape.components}")
            print(f"  Grid shape: {landscape.shape}")
            print(f"  Property range: {landscape.metadata['property_range']}")

            # Generate visualization
            viz_config = visualizer.visualize_property_landscape(
                landscape, plot_type="contour", interactive=True
            )

            print(f"  Visualization: {viz_config['type']}")

        except Exception as e:
            print(f"⚠ Property landscape generation failed for {prop_name}: {e}")

    return mixture_data


def main():
    """Run the complete mixture & formulation design demo."""
    print("ChemiAccel Mixture & Formulation Design Demo")
    print("=" * 50)

    try:
        # Basic mixture analysis
        calc, mixture = demo_basic_mixture_analysis()

        # Phase behavior prediction
        phase_result = demo_phase_behavior_prediction(calc, mixture)

        # Molecular interaction modeling
        interactions, mixing_prediction = demo_interaction_modeling(calc, mixture)

        # Inverse design optimization
        optimal_composition = demo_inverse_design()

        # Multi-objective optimization
        pareto_solutions = demo_multi_objective_optimization()

        # Dynamic process simulation
        process_result = demo_dynamic_process_simulation(calc, mixture)

        # Interactive visualization
        dashboard_config = demo_interactive_visualization(
            calc, mixture, phase_result, interactions, process_result
        )

        # Property landscape generation
        mixture_landscape_data = demo_property_landscape_generation()

        print("\n" + "=" * 50)
        print("✅ Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("• GPU-accelerated phase behavior prediction")
        print("• Comprehensive interaction modeling")
        print("• AI-driven inverse design optimization")
        print("• Multi-objective Pareto optimization")
        print("• Dynamic process integration")
        print("• Interactive visualization dashboards")
        print("• Property landscape mapping")

        module_classes = [
            "MixtureCalculator",
            "PhaseBehaviorPredictor",
            "InteractionModeler",
            "MixtureInverseDesigner",
            "DynamicProcessIntegrator",
            "MixtureVisualizer",
        ]
        print(
            f"\nMixture module provides {len(module_classes)} main classes for comprehensive formulation design."
        )

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("This may be due to missing optional dependencies.")
        print("The mixture module is designed to work with:")
        print("• CUDA/CuPy for GPU acceleration")
        print("• JAX for TPU support")
        print("• SciPy for optimization")
        print("• PyMOO for multi-objective optimization")
        print("• Scikit-learn for ML models")


if __name__ == "__main__":
    main()
