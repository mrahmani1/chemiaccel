"""
Comprehensive example demonstrating ChemiAccel's AI-driven predictive toxicology
and environmental fate modeling capabilities.

This example showcases the complete workflow from multi-endpoint toxicity prediction
through environmental fate modeling, metabolic pathway prediction, holistic risk
assessment, and safer-by-design chemical optimization.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

# Import ChemiAccel toxicology modules
from chemi_accel.toxicology import (
    ExposureRoute,
    ExposureScenario,
    OptimizationObjective,
    RiskCategory,
    SpeciesType,
    ToxicityConstraint,
    ToxicityEndpointType,
    ToxicologyCalculator,
)


def main():
    """Main example function demonstrating the complete toxicology workflow."""

    print("=== ChemiAccel AI-Driven Predictive Toxicology Example ===\n")

    # Initialize the toxicology calculator with GPU acceleration
    print("1. Initializing Toxicology Calculator...")
    calc = ToxicologyCalculator(
        backend="cuda",  # Use GPU acceleration if available
        use_gpu=True,
        uncertainty_quantification=True,
    )
    print(
        "   ✓ Calculator initialized with GPU acceleration and uncertainty quantification\n"
    )

    # Define test compounds
    test_compounds = {
        "Benzene": "c1ccccc1",
        "Ethanol": "CCO",
        "Phenol": "c1ccc(cc1)O",
        "Toluene": "Cc1ccccc1",
        "Acetone": "CC(=O)C",
    }

    print("2. Test Compounds:")
    for name, smiles in test_compounds.items():
        print(f"   - {name}: {smiles}")
    print()

    # Step 1: Multi-endpoint toxicity prediction
    print("3. Performing Multi-Endpoint Toxicity Prediction...")

    toxicity_results = {}
    for name, smiles in test_compounds.items():
        print(f"   Predicting toxicity for {name}...")

        # Predict comprehensive toxicity profile
        toxicity_data = calc.predict_toxicity(
            smiles,
            endpoints=[
                ToxicityEndpointType.ACUTE_ORAL,
                ToxicityEndpointType.ACUTE_DERMAL,
                ToxicityEndpointType.CARCINOGENICITY,
                ToxicityEndpointType.GENOTOXICITY,
                ToxicityEndpointType.AQUATIC_ACUTE,
                ToxicityEndpointType.SKIN_SENSITIZATION,
            ],
            species=[
                SpeciesType.RAT,
                SpeciesType.MOUSE,
                SpeciesType.FISH,
                SpeciesType.DAPHNIA,
            ],
            include_uncertainty=True,
        )

        toxicity_results[name] = toxicity_data

        print(f"     ✓ {len(toxicity_data.predicted_endpoints)} endpoint predictions")
        print(f"     ✓ Overall risk score: {toxicity_data.overall_risk_score:.3f}")

    print()

    # Step 2: Environmental fate modeling
    print("4. Predicting Environmental Fate and Degradation...")

    environmental_fate_results = {}
    for name, smiles in test_compounds.items():
        print(f"   Analyzing environmental fate for {name}...")

        # Predict environmental fate across multiple compartments
        env_fate = calc.predict_environmental_fate(
            smiles, environmental_conditions={"pH": 7.0, "temperature": 25.0}
        )

        environmental_fate_results[name] = env_fate

        print(
            f"     ✓ Fate predicted across {len(env_fate)} environmental compartments"
        )

    print()

    # Step 3: Metabolic pathway prediction
    print("5. Predicting Metabolic Pathways and Metabolite Toxicity...")

    metabolic_results = {}
    for name, smiles in test_compounds.items():
        print(f"   Predicting metabolism for {name}...")

        # Predict metabolic pathways
        metabolism = calc.predict_metabolic_pathways(
            smiles,
            species=SpeciesType.HUMAN,
            include_metabolite_toxicity=True,
            n_generations=2,
        )

        metabolic_results[name] = metabolism

        print(f"     ✓ {len(metabolism.metabolites)} metabolites identified")
        print(
            f"     ✓ {len(metabolism.toxic_metabolites)} potentially toxic metabolites"
        )

    print()

    # Step 4: Holistic risk assessment
    print("6. Performing Holistic Risk Assessment...")

    risk_assessments = {}
    for name, toxicity_data in toxicity_results.items():
        print(f"   Assessing risk for {name}...")

        # Define exposure scenarios
        exposure_scenarios = [
            ExposureScenario(
                scenario_name="consumer_oral",
                exposure_route=ExposureRoute.ORAL,
                exposure_level=0.01,  # mg/kg/day
                exposure_duration=365,
                exposure_frequency=1.0,
                population="adults",
            ),
            ExposureScenario(
                scenario_name="occupational_inhalation",
                exposure_route=ExposureRoute.INHALATION,
                exposure_level=10.0,  # mg/m3
                exposure_duration=8760,  # 1 year
                exposure_frequency=8 / 24,  # 8 hours per day
                population="workers",
            ),
        ]

        # Perform comprehensive risk assessment
        risk_score = calc.assess_holistic_risk(
            toxicity_data,
            environmental_fate_data=environmental_fate_results[name],
            exposure_scenarios=exposure_scenarios,
            include_uncertainty=True,
            monte_carlo_samples=1000,
        )

        risk_assessments[name] = risk_score

        print(f"     ✓ Risk category: {risk_score.risk_category.value}")
        print(f"     ✓ Overall risk: {risk_score.overall_risk:.3f}")
        print(f"     ✓ Confidence: {risk_score.confidence.value}")

    print()

    # Step 5: Identify high-risk compounds for optimization
    print("7. Identifying High-Risk Compounds for Safer Design...")

    high_risk_compounds = []
    for name, risk_score in risk_assessments.items():
        if risk_score.risk_category in [RiskCategory.HIGH, RiskCategory.VERY_HIGH]:
            high_risk_compounds.append((name, test_compounds[name], risk_score))
            print(f"   ⚠️  {name}: {risk_score.risk_category.value} risk")

    if not high_risk_compounds:
        print("   ✓ No high-risk compounds identified")
        # For demonstration, select benzene as it's typically high-risk
        high_risk_compounds = [
            ("Benzene", test_compounds["Benzene"], risk_assessments["Benzene"])
        ]

    print()

    # Step 6: Safer-by-design optimization
    print("8. Performing Safer-by-Design Chemical Optimization...")

    for name, smiles, risk_score in high_risk_compounds[
        :1
    ]:  # Optimize first high-risk compound
        print(f"   Optimizing {name} for improved safety...")

        # Define target properties
        target_properties = {
            "molecular_weight": 100.0,
            "logp": 2.0,
            "solubility": 0.5,
            "potency": 0.8,
        }

        # Define toxicity constraints
        toxicity_constraints = [
            ToxicityConstraint(
                endpoint_type=ToxicityEndpointType.ACUTE_ORAL,
                constraint_type="greater_than",
                target_value=2000.0,  # mg/kg (less toxic)
                priority="hard",
            ),
            ToxicityConstraint(
                endpoint_type=ToxicityEndpointType.CARCINOGENICITY,
                constraint_type="less_than",
                target_value=0.1,  # Low carcinogenicity probability
                priority="hard",
            ),
            ToxicityConstraint(
                endpoint_type=ToxicityEndpointType.AQUATIC_ACUTE,
                constraint_type="greater_than",
                target_value=100.0,  # mg/L (less aquatic toxicity)
                priority="soft",
            ),
        ]

        # Define optimization objectives
        optimization_objectives = [
            OptimizationObjective.MINIMIZE_TOXICITY,
            OptimizationObjective.MAXIMIZE_EFFICACY,
            OptimizationObjective.MINIMIZE_PERSISTENCE,
            OptimizationObjective.MAXIMIZE_BIODEGRADABILITY,
        ]

        # Perform safer design optimization
        safer_candidates = calc.design_safer_chemicals(
            smiles,
            target_properties=target_properties,
            toxicity_constraints=toxicity_constraints,
            optimization_objectives=optimization_objectives,
            n_candidates=10,
            max_iterations=50,
        )

        print(f"     ✓ Generated {len(safer_candidates)} optimized candidates")

        # Analyze optimization results
        viable_candidates = [c for c in safer_candidates if c.is_viable]
        print(f"     ✓ {len(viable_candidates)} candidates meet all constraints")

        if viable_candidates:
            best_candidate = viable_candidates[0]
            print(f"     ✓ Best candidate: {best_candidate.smiles}")
            print(
                f"     ✓ Overall desirability: {best_candidate.overall_desirability:.3f}"
            )
            print(
                f"     ✓ Similarity to parent: {best_candidate.similarity_to_parent:.3f}"
            )

    print()

    # Step 7: Generate comprehensive reports
    print("9. Generating Comprehensive Toxicology Reports...")

    for name, toxicity_data in list(toxicity_results.items())[
        :2
    ]:  # Generate reports for first 2 compounds
        print(f"   Generating report for {name}...")

        # Generate HTML report
        html_report = calc.create_toxicity_report(
            toxicity_data,
            include_environmental_fate=True,
            include_metabolic_pathways=True,
            include_risk_assessment=True,
            output_format="html",
        )

        # Save report to file
        report_filename = f"toxicology_report_{name.lower()}.html"
        with open(report_filename, "w") as f:
            f.write(html_report)

        print(f"     ✓ HTML report saved: {report_filename}")

    print()

    # Step 8: Comparative risk analysis
    print("10. Performing Comparative Risk Analysis...")

    # Create risk comparison table
    risk_comparison = pd.DataFrame(
        [
            {
                "Compound": name,
                "SMILES": test_compounds[name],
                "Overall_Risk": risk_score.overall_risk,
                "Risk_Category": risk_score.risk_category.value,
                "Confidence": risk_score.confidence.value,
                "Key_Drivers": "; ".join(risk_score.key_drivers[:2]),
            }
            for name, risk_score in risk_assessments.items()
        ]
    )

    # Sort by risk level
    risk_comparison = risk_comparison.sort_values("Overall_Risk", ascending=False)

    print("   Risk Ranking:")
    print(risk_comparison.to_string(index=False))

    print()

    # Step 9: Summary and recommendations
    print("11. Summary and Recommendations:")
    print("=" * 50)

    highest_risk = risk_comparison.iloc[0]
    lowest_risk = risk_comparison.iloc[-1]

    print(
        f"   Highest Risk Compound: {highest_risk['Compound']} (Risk: {highest_risk['Overall_Risk']:.3f})"
    )
    print(
        f"   Lowest Risk Compound: {lowest_risk['Compound']} (Risk: {lowest_risk['Overall_Risk']:.3f})"
    )

    high_risk_count = len(
        risk_comparison[risk_comparison["Risk_Category"].isin(["high", "very_high"])]
    )
    print(f"   High-Risk Compounds: {high_risk_count}/{len(risk_comparison)}")

    print("\n   Recommendations:")
    if high_risk_count > 0:
        print("   - Consider safer alternatives for high-risk compounds")
        print("   - Implement strict exposure controls for high-risk substances")
        print("   - Generate additional experimental data to validate predictions")
    else:
        print("   - Continue standard risk management practices")
        print("   - Monitor for new toxicity information")

    print("   - Use safer-by-design optimization for new chemical development")
    print("   - Integrate environmental fate considerations in risk management")

    print("\n=== Analysis Complete ===")


def demonstrate_advanced_features():
    """Demonstrate advanced features of the toxicology module."""

    print("\n=== Advanced Features Demonstration ===\n")

    calc = ToxicologyCalculator(backend="cpu")

    # Advanced uncertainty quantification
    print("1. Advanced Uncertainty Quantification...")

    toxicity_data = calc.predict_toxicity(
        "c1ccccc1",  # Benzene
        endpoints=[ToxicityEndpointType.CARCINOGENICITY],
        species=[SpeciesType.RAT],
        include_uncertainty=True,
    )

    # Perform Monte Carlo uncertainty analysis
    risk_score = calc.assess_holistic_risk(
        toxicity_data, include_uncertainty=True, monte_carlo_samples=5000
    )

    print(f"   Risk estimate: {risk_score.overall_risk:.3f}")
    print(
        f"   95% Uncertainty interval: [{risk_score.uncertainty_range[0]:.3f}, {risk_score.uncertainty_range[1]:.3f}]"
    )
    print(f"   Uncertainty factor: {risk_score.uncertainty_factor:.2f}x")

    # Drug-drug interaction prediction
    print("\n2. Drug-Drug Interaction Prediction...")

    interaction_results = calc.metabolic_predictor.predict_drug_drug_interactions(
        substrate_smiles="CCO", inhibitor_smiles="c1ccccc1"  # Ethanol  # Benzene
    )

    print(f"   Overall interaction risk: {interaction_results['overall_risk']}")
    for enzyme, data in interaction_results["enzyme_interactions"].items():
        print(f"   {enzyme}: {data['interaction_risk']} risk")

    # Pharmacokinetic simulation
    print("\n3. Pharmacokinetic Simulation...")

    pk_results = calc.metabolic_predictor.simulate_pharmacokinetics(
        smiles="CCO", dose=100.0, simulation_time=24.0  # mg  # hours
    )

    print(f"   Simulation time: {pk_results['simulation_time_hours']} hours")
    print(f"   Number of metabolites tracked: {len(pk_results['metabolite_profiles'])}")

    # Environmental multimedia fate modeling
    print("\n4. Environmental Multimedia Fate Modeling...")

    fate_results = calc.environmental_fate_predictor.predict_multimedia_fate(
        smiles="c1ccccc1",
        release_scenario={"atmosphere": 0.1, "water": 0.3, "soil": 0.6},
        simulation_time=8760,  # 1 year
    )

    print("   Final environmental distribution:")
    for compartment, fraction in fate_results["final_distribution"].items():
        print(f"   {compartment}: {fraction:.1%}")
    print(f"   Total degraded: {fate_results['total_degraded']:.1%}")

    # Bioaccumulation potential assessment
    print("\n5. Bioaccumulation Potential Assessment...")

    bioaccumulation = (
        calc.environmental_fate_predictor.estimate_bioaccumulation_potential(
            smiles="c1ccc(cc1)Cl", trophic_levels=4  # Chlorobenzene
        )
    )

    print(
        f"   Bioconcentration factor: {bioaccumulation['bioconcentration_factor']:.0f}"
    )
    print(
        f"   Biomagnification factor: {bioaccumulation['biomagnification_factor']:.1f}"
    )
    print(f"   Category: {bioaccumulation['bioaccumulation_category']}")

    print("\n=== Advanced Features Complete ===")


if __name__ == "__main__":
    # Run main example
    main()

    # Demonstrate advanced features
    demonstrate_advanced_features()

    print(f"\n{'='*60}")
    print("Example execution complete!")
    print("Check the generated HTML reports for detailed results.")
    print(f"{'='*60}")
