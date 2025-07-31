"""
Example: QML Uncertainty Quantification for ChemiAccel

This example demonstrates how to use the QML uncertainty quantification framework
to analyze and visualize the reliability of quantum machine learning predictions.
"""

import numpy as np

# ChemiAccel imports
from src.chemi_accel.qml import (
    BaseQMLModel,
    QMLPrediction,
    QMLUncertaintyConfig,
    QMLUncertaintyQuantifier,
    QMLUncertaintyVisualizer,
    QMLUQIntegration,
    QuantumProperty,
)


# Mock QML Model for demonstration
class MockQMLModel(BaseQMLModel):
    """Mock QML model that supports ensemble and MC dropout predictions."""

    def __init__(self):
        super().__init__()
        self._is_loaded = True
        np.random.seed(42)  # For reproducible results

    def load_model(self):
        """Mock model loading."""
        pass

    def predict(self, molecules, properties=None, **kwargs):
        """Mock prediction method."""
        if isinstance(molecules, str):
            molecules = [molecules]

        results = []
        for mol in molecules:
            # Generate mock predictions based on molecule hash
            mol_hash = hash(mol) % 1000
            np.random.seed(mol_hash)

            props = {}
            uncertainties = {}

            if properties is None:
                properties = [QuantumProperty.HOMO_ENERGY, QuantumProperty.LUMO_ENERGY]

            for prop in properties:
                if prop == QuantumProperty.HOMO_ENERGY:
                    base_value = -5.0 + np.random.normal(0, 0.5)
                    uncertainty = 0.1 + np.random.exponential(0.05)
                elif prop == QuantumProperty.LUMO_ENERGY:
                    base_value = -2.0 + np.random.normal(0, 0.3)
                    uncertainty = 0.08 + np.random.exponential(0.03)
                else:
                    base_value = np.random.normal(0, 1)
                    uncertainty = 0.1 + np.random.exponential(0.02)

                props[prop] = base_value
                uncertainties[prop] = uncertainty

            prediction = QMLPrediction(
                properties=props,
                uncertainty=uncertainties,
                model_name="MockQMLModel",
                confidence=0.8 + np.random.uniform(0, 0.2),
            )
            results.append(prediction)

        return results[0] if len(results) == 1 else results

    def predict_ensemble(self, molecules, properties, n_models=5):
        """Mock ensemble prediction."""
        ensemble_results = []

        for i in range(n_models):
            # Use different seeds for each ensemble member
            original_seed = np.random.get_state()
            np.random.seed(42 + i * 10)

            results = self.predict(molecules, properties)
            if not isinstance(results, list):
                results = [results]

            # Add noise to simulate ensemble variance
            for result in results:
                for prop in result.properties:
                    noise = np.random.normal(0, 0.05)  # Small ensemble variance
                    result.properties[prop] += noise

            ensemble_results.extend(results)
            np.random.set_state(original_seed)

        return ensemble_results

    def predict_mc_dropout(self, molecules, properties, n_samples=10):
        """Mock Monte Carlo Dropout prediction."""
        mc_results = []

        for i in range(n_samples):
            # Use different seeds for each MC sample
            original_seed = np.random.get_state()
            np.random.seed(42 + i * 5)

            results = self.predict(molecules, properties)
            if not isinstance(results, list):
                results = [results]

            # Add dropout noise
            for result in results:
                for prop in result.properties:
                    dropout_noise = np.random.normal(0, 0.08)  # MC dropout variance
                    result.properties[prop] += dropout_noise

            mc_results.extend(results)
            np.random.set_state(original_seed)

        return mc_results

    def get_supported_properties(self):
        """Return supported properties."""
        return [
            QuantumProperty.HOMO_ENERGY,
            QuantumProperty.LUMO_ENERGY,
            QuantumProperty.DIPOLE_MOMENT,
        ]

    def validate_molecule(self, molecule):
        """Mock validation - accept all molecules."""
        return True


def demonstrate_basic_uncertainty_quantification():
    """Demonstrate basic uncertainty quantification workflow."""
    print("=== Basic QML Uncertainty Quantification ===\n")

    # Create mock model and molecules
    model = MockQMLModel()
    molecules = ["CCO", "CCC", "CCCC", "C1CCCCC1", "c1ccccc1"]
    properties = [QuantumProperty.HOMO_ENERGY, QuantumProperty.LUMO_ENERGY]

    # Configure uncertainty quantification
    config = QMLUncertaintyConfig(
        uncertainty_methods=["ensemble", "mc_dropout"],
        ensemble_size=5,
        mc_samples=10,
        confidence_levels=[0.68, 0.95],
        enable_reliability_checking=True,
        min_confidence=0.7,
    )

    # Create uncertainty quantifier
    quantifier = QMLUncertaintyQuantifier(config)

    print(f"Analyzing {len(molecules)} molecules for {len(properties)} properties")
    print(f"Uncertainty methods: {config.uncertainty_methods}")
    print()

    # Quantify uncertainty using ensemble method
    print("1. Ensemble Uncertainty Quantification:")
    ensemble_result = quantifier.quantify_uncertainty(
        model, molecules, properties, method="ensemble"
    )

    for prop in properties:
        if prop in ensemble_result.predictions:
            pred = ensemble_result.predictions[prop]
            epistemic = ensemble_result.epistemic_uncertainty[prop]
            aleatoric = ensemble_result.aleatoric_uncertainty[prop]
            total = ensemble_result.total_uncertainty[prop]
            reliability = ensemble_result.reliability_scores[prop]

            print(f"\n{prop.name}:")
            print(f"  Predictions: {pred}")
            print(f"  Epistemic uncertainty: {epistemic}")
            print(f"  Aleatoric uncertainty: {aleatoric}")
            print(f"  Total uncertainty: {total}")
            print(f"  Reliability scores: {reliability}")

            # Show prediction intervals
            if prop in ensemble_result.prediction_intervals:
                intervals_95 = ensemble_result.prediction_intervals[prop].get(0.95)
                if intervals_95:
                    lower, upper = intervals_95
                    print(f"  95% Prediction intervals: [{lower}, {upper}]")

            # Show reliable predictions
            reliable_indices = ensemble_result.get_reliable_predictions(prop, 0.8)
            print(f"  Reliable predictions (>80% confidence): {reliable_indices}")

    print(f"\nCalibration metrics: {ensemble_result.calibration_metrics}")
    print(f"Method info: {ensemble_result.method_info}")

    return ensemble_result


def demonstrate_uncertainty_comparison():
    """Compare uncertainty from different methods."""
    print("\n=== Uncertainty Method Comparison ===\n")

    model = MockQMLModel()
    molecules = ["CCO", "CCC", "CCCC"]
    properties = [QuantumProperty.HOMO_ENERGY]

    config = QMLUncertaintyConfig()
    quantifier = QMLUncertaintyQuantifier(config)

    # Get uncertainty from both methods
    ensemble_result = quantifier.quantify_uncertainty(
        model, molecules, properties, method="ensemble"
    )

    mc_dropout_result = quantifier.quantify_uncertainty(
        model, molecules, properties, method="mc_dropout"
    )

    # Compare results
    prop = QuantumProperty.HOMO_ENERGY

    print("Uncertainty Comparison for HOMO Energy:")
    print("\nMethod          | Predictions | Total Uncertainty | Reliability")
    print("----------------|-------------|-------------------|------------")

    ens_pred = ensemble_result.predictions[prop]
    ens_unc = ensemble_result.total_uncertainty[prop]
    ens_rel = ensemble_result.reliability_scores[prop]

    mc_pred = mc_dropout_result.predictions[prop]
    mc_unc = mc_dropout_result.total_uncertainty[prop]
    mc_rel = mc_dropout_result.reliability_scores[prop]

    for i in range(len(molecules)):
        print(f"Molecule {i+1}:")
        print(
            f"  Ensemble      | {ens_pred[i]:10.3f} | {ens_unc[i]:16.3f} | {ens_rel[i]:10.3f}"
        )
        print(
            f"  MC Dropout    | {mc_pred[i]:10.3f} | {mc_unc[i]:16.3f} | {mc_rel[i]:10.3f}"
        )
        print()

    # Combine uncertainties
    print("Combining uncertainty estimates...")
    combined_result = quantifier.combine_uncertainties(
        [ensemble_result, mc_dropout_result], combination_method="average"
    )

    combined_pred = combined_result.predictions[prop]
    combined_unc = combined_result.total_uncertainty[prop]
    combined_rel = combined_result.reliability_scores[prop]

    print("\nCombined Results:")
    for i in range(len(molecules)):
        print(
            f"Molecule {i+1}: pred={combined_pred[i]:.3f}, "
            f"uncertainty={combined_unc[i]:.3f}, reliability={combined_rel[i]:.3f}"
        )

    print(f"\nCombined method info: {combined_result.method_info}")

    return ensemble_result, mc_dropout_result, combined_result


def demonstrate_integration_with_standard_uq():
    """Demonstrate integration with standard UQ framework."""
    print("\n=== Integration with Standard UQ Framework ===\n")

    model = MockQMLModel()
    molecules = ["CCO", "CCC"]
    properties = [QuantumProperty.HOMO_ENERGY]

    # Get QML uncertainty result
    config = QMLUncertaintyConfig()
    quantifier = QMLUncertaintyQuantifier(config)

    qml_result = quantifier.quantify_uncertainty(
        model, molecules, properties, method="ensemble"
    )

    # Convert to standard UQ format
    print("Converting QML uncertainty to standard UQ format...")
    standard_result = QMLUQIntegration.qml_to_standard_uq(
        qml_result, QuantumProperty.HOMO_ENERGY
    )

    print("Standard UQ Result:")
    print(f"  Predictions: {standard_result.predictions}")
    print(f"  Uncertainties: {standard_result.uncertainties}")
    print(f"  Method: {standard_result.method}")
    print(f"  Confidence intervals: {standard_result.confidence_intervals}")

    # Enhance QML prediction with uncertainty
    print("\nEnhancing QML prediction with uncertainty info...")
    original_prediction = model.predict(molecules[0], properties)

    enhanced_prediction = QMLUQIntegration.enhance_qml_prediction(
        original_prediction, qml_result
    )

    print("Enhanced prediction:")
    print(f"  Original uncertainty: {original_prediction.uncertainty}")
    print(f"  Enhanced total uncertainty: {enhanced_prediction.uncertainty}")
    print(f"  Epistemic uncertainty: {enhanced_prediction.epistemic_uncertainty}")
    print(f"  Aleatoric uncertainty: {enhanced_prediction.aleatoric_uncertainty}")
    print(f"  Prediction intervals: {enhanced_prediction.prediction_intervals}")
    print(f"  Uncertainty method: {enhanced_prediction.uncertainty_method}")

    return standard_result, enhanced_prediction


def demonstrate_visualization():
    """Demonstrate uncertainty visualization capabilities."""
    print("\n=== Uncertainty Visualization ===\n")

    # Check if plotting is available
    try:
        from src.chemi_accel.qml.uncertainty_quantification import PLOTTING_AVAILABLE

        if not PLOTTING_AVAILABLE:
            print("Plotting libraries not available. Skipping visualization demo.")
            return

        visualizer = QMLUncertaintyVisualizer()
        print("Visualization tools initialized successfully.")

        # Get some uncertainty results
        model = MockQMLModel()
        molecules = [
            "CCO",
            "CCC",
            "CCCC",
            "C1CCCCC1",
            "c1ccccc1",
        ] * 4  # More data for better plots
        properties = [QuantumProperty.HOMO_ENERGY]

        config = QMLUncertaintyConfig()
        quantifier = QMLUncertaintyQuantifier(config)

        result = quantifier.quantify_uncertainty(
            model, molecules, properties, method="ensemble"
        )

        print(f"Generated uncertainty data for {len(molecules)} molecules")

        # Note: In a real environment, you would call the plotting methods:
        # fig1 = visualizer.plot_uncertainty_distribution(result, QuantumProperty.HOMO_ENERGY)
        # fig2 = visualizer.plot_prediction_intervals(result, QuantumProperty.HOMO_ENERGY)
        #
        # For this example, we just demonstrate the interface exists
        print("Available visualization methods:")
        print("- plot_uncertainty_distribution()")
        print("- plot_prediction_intervals()")
        print("- plot_reliability_diagram()")
        print("- plot_uncertainty_comparison()")
        print("\nVisualization example completed successfully.")

    except Exception as e:
        print(f"Visualization demo failed: {e}")
        print("This is expected if matplotlib/seaborn are not installed.")


def demonstrate_reliability_analysis():
    """Demonstrate reliability analysis of QML predictions."""
    print("\n=== QML Prediction Reliability Analysis ===\n")

    model = MockQMLModel()
    molecules = ["CCO", "CCC", "CCCC", "C1CCCCC1", "c1ccccc1", "CCN", "CCC(C)C"]
    properties = [QuantumProperty.HOMO_ENERGY, QuantumProperty.LUMO_ENERGY]

    # Configure with strict reliability checking
    config = QMLUncertaintyConfig(
        uncertainty_methods=["ensemble"],
        enable_reliability_checking=True,
        min_confidence=0.75,  # Higher threshold
        uncertainty_threshold=0.12,
    )

    quantifier = QMLUncertaintyQuantifier(config)

    result = quantifier.quantify_uncertainty(model, molecules, properties)

    print(f"Reliability analysis for {len(molecules)} molecules:")
    print(f"Minimum confidence threshold: {config.min_confidence}")
    print(f"Uncertainty threshold: {config.uncertainty_threshold}")
    print()

    for prop in properties:
        if prop not in result.reliability_scores:
            continue

        reliability = result.reliability_scores[prop]
        predictions = result.predictions[prop]
        uncertainties = result.total_uncertainty[prop]

        print(f"{prop.name}:")
        print("  Molecule | Prediction | Uncertainty | Reliability | Status")
        print("  ---------|------------|-------------|-------------|--------")

        for i, mol in enumerate(molecules):
            if i < len(predictions):
                pred = predictions[i]
                unc = uncertainties[i]
                rel = reliability[i]

                status = "RELIABLE" if rel >= config.min_confidence else "UNRELIABLE"

                print(
                    f"  {mol:8s} | {pred:10.3f} | {unc:11.3f} | {rel:11.3f} | {status}"
                )

        # Calculate statistics
        reliable_count = np.sum(reliability >= config.min_confidence)
        total_count = len(reliability)
        reliability_rate = reliable_count / total_count if total_count > 0 else 0

        print(
            f"\n  Summary: {reliable_count}/{total_count} predictions reliable "
            f"({reliability_rate:.1%})"
        )
        print()


def main():
    """Run all demonstration examples."""
    print("ChemiAccel QML Uncertainty Quantification Demo")
    print("=" * 50)

    try:
        # Basic uncertainty quantification
        ensemble_result = demonstrate_basic_uncertainty_quantification()

        # Method comparison
        ens_result, mc_result, combined_result = demonstrate_uncertainty_comparison()

        # Integration with standard UQ
        standard_result, enhanced_pred = demonstrate_integration_with_standard_uq()

        # Visualization (may be skipped if libraries not available)
        demonstrate_visualization()

        # Reliability analysis
        demonstrate_reliability_analysis()

        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✓ Ensemble uncertainty quantification")
        print("✓ Monte Carlo Dropout uncertainty")
        print("✓ Uncertainty method comparison and combination")
        print("✓ Integration with standard UQ framework")
        print("✓ Reliability analysis and filtering")
        print("✓ Visualization capabilities (interface)")
        print("\nThe QML uncertainty quantification framework is ready for use!")

    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        print("This may be due to missing dependencies or mock model limitations.")
        raise


if __name__ == "__main__":
    main()
