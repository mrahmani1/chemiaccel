#!/usr/bin/env python3
"""
Reaction Intelligence Module Demo

Demonstrates the capabilities of ChemiAccel's reaction intelligence system:
- Catalyst and reaction condition suggestions
- What-if scenario simulation
- Experimental feedback integration
- Knowledge graph-enhanced optimization
"""

import pandas as pd

from src.chemi_accel.reaction_networks import (
    CatalystSuggestionEngine,
    ExperimentalFeedbackProcessor,
    ExperimentalResult,
    ReactionConditions,
    ReactionIntelligenceEngine,
    ReactionScenario,
    WhatIfSimulator,
)


def demo_basic_reaction_analysis():
    """Demonstrate basic reaction analysis and catalyst suggestion."""
    print("=" * 60)
    print("Demo 1: Basic Reaction Analysis and Catalyst Suggestion")
    print("=" * 60)

    # Initialize the intelligence engine
    intelligence_engine = ReactionIntelligenceEngine()
    catalyst_engine = CatalystSuggestionEngine(intelligence_engine)

    # Define a cross-coupling reaction
    reactants = ["CCBr", "CC(=O)O"]  # Ethyl bromide + Acetic acid
    products = ["CC(=O)CC"]  # Butanone

    print(f"Reactants: {reactants}")
    print(f"Products: {products}")

    # Analyze reaction requirements
    print("\n--- Reaction Analysis ---")
    analysis = intelligence_engine.analyze_reaction_requirements(reactants, products)
    print(f"Reaction type: {analysis['reaction_type']}")
    print(f"Complexity score: {analysis['complexity_score']}")
    print(
        f"Thermodynamics: ΔG = {analysis['thermodynamics']['delta_g_298k']:.1f} kJ/mol"
    )

    # Get catalyst suggestions
    print("\n--- Catalyst Suggestions ---")
    suggestions = catalyst_engine.suggest_catalysts(
        reactants, products, max_suggestions=3
    )

    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n{i}. {suggestion.catalyst_name}")
        print(f"   Confidence: {suggestion.confidence_score:.2f}")
        print(f"   Literature precedents: {suggestion.literature_precedents}")
        print(f"   Compatible solvents: {', '.join(suggestion.compatible_solvents)}")
        if suggestion.notes:
            print(f"   Notes: {suggestion.notes}")

    # Get condition suggestions
    print("\n--- Optimal Conditions ---")
    conditions = catalyst_engine.suggest_conditions(reactants, products)
    print(
        f"Temperature: {conditions.temperature:.1f} K ({conditions.temperature - 273.15:.1f}°C)"
    )
    print(f"Solvent: {conditions.solvent}")
    print(f"Pressure: {conditions.pressure} bar")
    print(f"Reaction time: {conditions.reaction_time} hours")
    print(f"Concentration: {conditions.concentration} M")

    return intelligence_engine, catalyst_engine, reactants, products, conditions


def demo_what_if_simulation(intelligence_engine, reactants, products, base_conditions):
    """Demonstrate what-if scenario simulation."""
    print("\n" + "=" * 60)
    print("Demo 2: What-If Scenario Simulation")
    print("=" * 60)

    # Initialize simulator
    simulator = WhatIfSimulator(intelligence_engine)

    # Create base scenario
    base_scenario = ReactionScenario(
        reactants=reactants, products=products, conditions=base_conditions
    )
    simulator.set_base_scenario(base_scenario)

    print("Base scenario set up successfully!")

    # Simulate temperature effects
    print("\n--- Temperature Sweep Analysis ---")
    temp_scenarios = simulator.simulate_temperature_sweep(
        (298.15, 398.15), num_points=6
    )

    # Create comparison table
    comparison_df = simulator.compare_scenarios(temp_scenarios)
    print("\nTemperature vs. Predicted Performance:")
    print(comparison_df[["temperature", "predicted_yield", "reaction_rate"]].round(3))

    # Simulate catalyst changes
    print("\n--- Catalyst Comparison ---")
    catalysts_to_test = ["Pd(PPh3)4", "CuI", "TFA"]
    catalyst_scenarios = []

    for catalyst in catalysts_to_test:
        scenario = simulator.simulate_catalyst_change(catalyst)
        catalyst_scenarios.append(scenario)

    catalyst_df = simulator.compare_scenarios(catalyst_scenarios)
    print("\nCatalyst Performance Comparison:")
    print(
        catalyst_df[
            ["catalyst", "predicted_yield", "predicted_selectivity", "reaction_rate"]
        ].round(3)
    )

    # Simulate solvent effects
    print("\n--- Solvent Effects ---")
    solvents_to_test = ["DMF", "THF", "acetonitrile", "toluene"]
    solvent_scenarios = []

    for solvent in solvents_to_test:
        scenario = simulator.simulate_condition_change("solvent", solvent)
        solvent_scenarios.append(scenario)

    solvent_df = simulator.compare_scenarios(solvent_scenarios)
    print("\nSolvent Performance Comparison:")
    print(solvent_df[["solvent", "predicted_yield", "predicted_selectivity"]].round(3))

    return temp_scenarios[3]  # Return a mid-range temperature scenario for next demo


def demo_experimental_feedback(intelligence_engine, reactants, products, test_scenario):
    """Demonstrate experimental feedback integration."""
    print("\n" + "=" * 60)
    print("Demo 3: Experimental Feedback Integration")
    print("=" * 60)

    # Initialize feedback processor
    feedback_processor = ExperimentalFeedbackProcessor(intelligence_engine)

    # Simulate experimental results
    print("Simulating experimental results...")

    experimental_results = [
        ExperimentalResult(
            scenario=test_scenario,
            actual_yield=0.68,  # Lower than predicted
            actual_selectivity=0.82,
            notes="Reaction completed in 8 hours",
        ),
        ExperimentalResult(
            scenario=ReactionScenario(
                reactants=reactants,
                products=products,
                conditions=ReactionConditions(
                    temperature=373.15, catalyst="CuI", solvent="DMF"
                ),
                predicted_yield=0.60,
                predicted_selectivity=0.75,
            ),
            actual_yield=0.71,  # Higher than predicted
            actual_selectivity=0.80,
            notes="Faster reaction with CuI catalyst",
        ),
        ExperimentalResult(
            scenario=ReactionScenario(
                reactants=reactants,
                products=products,
                conditions=ReactionConditions(
                    temperature=298.15, catalyst="Pd(PPh3)4", solvent="THF"
                ),
                predicted_yield=0.75,
                predicted_selectivity=0.85,
            ),
            actual_yield=0.78,  # Close to predicted
            actual_selectivity=0.87,
            notes="Room temperature reaction worked well",
        ),
    ]

    # Ingest experimental data
    for result in experimental_results:
        feedback_processor.ingest_experimental_result(result)
        print(
            f"Ingested result: {result.actual_yield:.2f} yield with {result.scenario.conditions.catalyst}"
        )

    # Analyze prediction accuracy
    print("\n--- Prediction Accuracy Analysis ---")
    accuracy = feedback_processor.analyze_prediction_accuracy()
    print(f"Number of experiments: {accuracy['num_experiments']}")
    print(f"Mean yield error: {accuracy['mean_yield_error']:.3f}")
    print(f"Std yield error: {accuracy['std_yield_error']:.3f}")
    if "mean_selectivity_error" in accuracy:
        print(f"Mean selectivity error: {accuracy['mean_selectivity_error']:.3f}")

    # Refine model weights
    print("\n--- Model Weight Refinement ---")
    print("Initial weights:", feedback_processor.model_weights)
    feedback_processor.refine_model_weights()
    print("Updated weights:", feedback_processor.model_weights)

    # Suggest next experiments
    print("\n--- Next Experiment Suggestions ---")
    next_experiments = feedback_processor.suggest_next_experiments(
        (reactants, products), num_suggestions=3
    )

    for i, experiment in enumerate(next_experiments, 1):
        print(f"\n{i}. Temperature: {experiment.conditions.temperature:.1f} K")
        if experiment.conditions.catalyst:
            print(f"   Catalyst: {experiment.conditions.catalyst}")
        print(f"   Solvent: {experiment.conditions.solvent}")
        print("   Rationale: Explore underexplored parameter space")


def demo_knowledge_graph_integration():
    """Demonstrate knowledge graph integration (mock implementation)."""
    print("\n" + "=" * 60)
    print("Demo 4: Knowledge Graph Integration (Conceptual)")
    print("=" * 60)

    print(
        "Note: This demo shows the conceptual workflow for knowledge graph integration."
    )
    print(
        "In a real implementation, this would connect to a populated chemical knowledge graph."
    )

    # Mock knowledge graph queries
    print("\n--- Literature-Based Catalyst Ranking ---")
    mock_literature_data = {
        "Pd(PPh3)4": {"avg_yield": 0.82, "precedents": 1247, "success_rate": 0.89},
        "Pd(dppf)Cl2": {"avg_yield": 0.85, "precedents": 856, "success_rate": 0.91},
        "CuI": {"avg_yield": 0.68, "precedents": 432, "success_rate": 0.76},
        "Ni(cod)2": {"avg_yield": 0.71, "precedents": 234, "success_rate": 0.78},
    }

    print("Literature-derived catalyst performance:")
    for catalyst, data in mock_literature_data.items():
        print(f"  {catalyst}:")
        print(f"    Average yield: {data['avg_yield']:.2f}")
        print(f"    Literature precedents: {data['precedents']}")
        print(f"    Success rate: {data['success_rate']:.2f}")

    print("\n--- Recommended Conditions from Similar Reactions ---")
    print("Based on 15 similar coupling reactions in the literature:")
    print("  Optimal temperature: 85°C (±10°C)")
    print("  Preferred solvents: DMF (67%), dioxane (23%), toluene (10%)")
    print("  Typical catalyst loading: 5-10 mol%")
    print("  Success factors: Inert atmosphere, dry solvents, base additives")

    print("\n--- Validation Against Literature ---")
    print("Proposed reaction conditions validation:")
    print("  ✓ Temperature (80°C) matches literature optimum")
    print("  ✓ Solvent (DMF) is most successful in literature")
    print("  ✓ Catalyst (Pd-based) has strong precedent")
    print("  → Confidence score: 0.87/1.00")


def main():
    """Run complete reaction intelligence demonstration."""
    print("ChemiAccel Reaction Intelligence System Demo")
    print("=" * 60)
    print("This demo showcases AI-driven reaction optimization capabilities")
    print("for catalyst selection, condition optimization, and experimental feedback.")

    # Run all demonstration modules
    (
        intelligence_engine,
        catalyst_engine,
        reactants,
        products,
        conditions,
    ) = demo_basic_reaction_analysis()

    test_scenario = demo_what_if_simulation(
        intelligence_engine, reactants, products, conditions
    )

    demo_experimental_feedback(intelligence_engine, reactants, products, test_scenario)

    demo_knowledge_graph_integration()

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    print("\nKey capabilities demonstrated:")
    print("✓ Automated reaction analysis and classification")
    print("✓ Literature-informed catalyst suggestions")
    print("✓ Optimal reaction condition prediction")
    print("✓ Interactive what-if scenario simulation")
    print("✓ Experimental feedback integration")
    print("✓ Knowledge graph-enhanced optimization")
    print("\nThese tools can significantly accelerate reaction development")
    print("by providing data-driven insights and reducing experimental trials.")


if __name__ == "__main__":
    main()
