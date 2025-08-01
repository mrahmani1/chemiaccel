"""
Integration tests for toxicology module.

Tests the complete workflow from toxicity prediction through risk assessment
to safer design optimization.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from chemi_accel.core.backend import get_backend
from chemi_accel.toxicology import (
    HolisticRiskAssessment,
    MultiEndpointToxicologyPredictor,
    OptimizationObjective,
    SaferByDesignOptimizer,
    SpeciesType,
    ToxicityConstraint,
    ToxicityData,
    ToxicityEndpointType,
    ToxicologyCalculator,
)


class TestToxicologyWorkflow:
    """Test complete toxicology workflow integration."""

    def test_complete_toxicology_workflow(self):
        """Test complete workflow from prediction to optimization."""
        # Initialize calculator
        calc = ToxicologyCalculator(backend="cpu", use_gpu=False)

        # Test compound (ethanol)
        test_smiles = "CCO"

        # Step 1: Predict toxicity
        with patch.object(
            calc.multi_endpoint_predictor, "predict_endpoints"
        ) as mock_predict:
            # Mock toxicity predictions
            mock_predictions = self._create_mock_predictions()
            mock_predict.return_value = mock_predictions

            toxicity_data = calc.predict_toxicity(
                test_smiles,
                endpoints=[
                    ToxicityEndpointType.ACUTE_ORAL,
                    ToxicityEndpointType.CARCINOGENICITY,
                    ToxicityEndpointType.AQUATIC_ACUTE,
                ],
                species=[SpeciesType.RAT, SpeciesType.FISH],
            )

            assert isinstance(toxicity_data, ToxicityData)
            assert toxicity_data.smiles == test_smiles
            assert len(toxicity_data.predicted_endpoints) == len(mock_predictions)

        # Step 2: Environmental fate prediction
        with patch.object(
            calc.environmental_fate_predictor, "predict_fate"
        ) as mock_fate:
            mock_fate.return_value = {"surface_water": []}

            env_fate = calc.predict_environmental_fate(test_smiles)
            assert env_fate is not None

        # Step 3: Metabolic pathway prediction
        with patch.object(
            calc.metabolic_predictor, "predict_metabolism"
        ) as mock_metabolism:
            mock_metabolism.return_value = MagicMock()

            metabolism = calc.predict_metabolic_pathways(test_smiles)
            assert metabolism is not None

        # Step 4: Risk assessment
        with patch.object(calc.risk_assessor, "assess_risk") as mock_risk:
            from chemi_accel.toxicology.core import ConfidenceLevel
            from chemi_accel.toxicology.risk_assessment import RiskCategory, RiskScore

            mock_risk_score = RiskScore(
                overall_risk=0.3,
                risk_category=RiskCategory.MODERATE,
                endpoint_contributions={"acute_oral": 0.6, "carcinogenicity": 0.4},
                exposure_contributions={"oral": 0.8, "dermal": 0.2},
                uncertainty_range=(0.2, 0.4),
                confidence=ConfidenceLevel.HIGH,
                key_drivers=["Moderate toxicity"],
            )
            mock_risk.return_value = mock_risk_score

            risk_assessment = calc.assess_holistic_risk(toxicity_data)
            assert risk_assessment == mock_risk_score

        # Step 5: Safer design optimization
        with patch.object(calc.safer_designer, "optimize_for_safety") as mock_optimize:
            from chemi_accel.toxicology.safer_design import SafeDesignCandidate

            mock_candidates = [
                SafeDesignCandidate(
                    smiles="CCCO",
                    predicted_properties={"molecular_weight": 60.0},
                    optimization_score=0.8,
                    similarity_to_parent=0.9,
                )
            ]
            mock_optimize.return_value = mock_candidates

            target_properties = {"molecular_weight": 50.0}
            toxicity_constraints = [
                ToxicityConstraint(
                    endpoint_type=ToxicityEndpointType.ACUTE_ORAL,
                    constraint_type="greater_than",
                    target_value=2000.0,
                )
            ]

            safer_candidates = calc.design_safer_chemicals(
                test_smiles, target_properties, toxicity_constraints
            )

            assert len(safer_candidates) == 1
            assert safer_candidates[0].smiles == "CCCO"

        # Step 6: Generate comprehensive report
        report = calc.create_toxicity_report(
            toxicity_data,
            include_environmental_fate=True,
            include_metabolic_pathways=True,
            include_risk_assessment=True,
            output_format="html",
        )

        assert isinstance(report, str)
        assert "<html>" in report
        assert test_smiles in report

    def test_batch_toxicity_prediction(self):
        """Test batch processing of multiple compounds."""
        calc = ToxicologyCalculator(backend="cpu")

        test_compounds = ["CCO", "CCC", "CCCO"]

        with patch.object(
            calc.multi_endpoint_predictor, "predict_endpoints"
        ) as mock_predict:
            mock_predict.return_value = self._create_mock_predictions()

            results = calc.predict_toxicity(test_compounds)

            assert isinstance(results, list)
            assert len(results) == 3
            assert all(isinstance(r, ToxicityData) for r in results)
            assert [r.smiles for r in results] == test_compounds

            # Should have called predictor for each compound
            assert mock_predict.call_count == 3

    def test_error_handling_in_workflow(self):
        """Test error handling throughout the workflow."""
        calc = ToxicologyCalculator(backend="cpu")

        # Test with invalid SMILES
        invalid_smiles = "INVALID_SMILES"

        with patch.object(
            calc.multi_endpoint_predictor, "predict_endpoints"
        ) as mock_predict:
            # Mock predictor to handle invalid SMILES gracefully
            mock_predict.return_value = []

            result = calc.predict_toxicity(invalid_smiles)

            assert isinstance(result, ToxicityData)
            assert result.smiles == invalid_smiles
            assert len(result.predicted_endpoints) == 0

    def test_uncertainty_propagation(self):
        """Test uncertainty propagation through workflow."""
        calc = ToxicologyCalculator(backend="cpu", uncertainty_quantification=True)

        with patch.object(
            calc.multi_endpoint_predictor, "predict_endpoints"
        ) as mock_predict:
            # Mock predictions with uncertainty
            mock_predictions = self._create_mock_predictions_with_uncertainty()
            mock_predict.return_value = mock_predictions

            toxicity_data = calc.predict_toxicity("CCO", include_uncertainty=True)

            assert all(
                pred.uncertainty > 0 for pred in toxicity_data.predicted_endpoints
            )

        # Test risk assessment with uncertainty
        with patch.object(calc.risk_assessor, "assess_risk") as mock_risk:
            from chemi_accel.toxicology.core import ConfidenceLevel
            from chemi_accel.toxicology.risk_assessment import RiskCategory, RiskScore

            mock_risk_score = RiskScore(
                overall_risk=0.3,
                risk_category=RiskCategory.MODERATE,
                endpoint_contributions={},
                exposure_contributions={},
                uncertainty_range=(0.1, 0.5),
                confidence=ConfidenceLevel.MODERATE,
                monte_carlo_samples=np.random.rand(100),
            )
            mock_risk.return_value = mock_risk_score

            risk_assessment = calc.assess_holistic_risk(
                toxicity_data, include_uncertainty=True
            )

            assert risk_assessment.monte_carlo_samples is not None
            assert len(risk_assessment.monte_carlo_samples) == 100

    def test_multi_species_multi_endpoint_prediction(self):
        """Test prediction across multiple species and endpoints."""
        calc = ToxicologyCalculator(backend="cpu")

        endpoints = [
            ToxicityEndpointType.ACUTE_ORAL,
            ToxicityEndpointType.CARCINOGENICITY,
            ToxicityEndpointType.AQUATIC_ACUTE,
        ]
        species = [SpeciesType.RAT, SpeciesType.MOUSE, SpeciesType.FISH]

        with patch.object(
            calc.multi_endpoint_predictor, "predict_endpoints"
        ) as mock_predict:
            # Mock predictions for all endpoint-species combinations
            mock_predictions = []
            for endpoint in endpoints:
                for species_type in species:
                    mock_predictions.extend(
                        self._create_mock_predictions(endpoint, species_type)
                    )
            mock_predict.return_value = mock_predictions

            result = calc.predict_toxicity("CCO", endpoints=endpoints, species=species)

            assert len(result.predicted_endpoints) == len(mock_predictions)

            # Check that all endpoint-species combinations are covered
            endpoint_species_pairs = {
                (pred.endpoint.endpoint_type, pred.endpoint.species)
                for pred in result.predicted_endpoints
            }

            expected_pairs = {(ep, sp) for ep in endpoints for sp in species}
            assert endpoint_species_pairs == expected_pairs

    def test_performance_with_large_dataset(self):
        """Test performance with larger dataset."""
        calc = ToxicologyCalculator(backend="cpu")

        # Generate test dataset
        test_compounds = [f"C{'C' * i}O" for i in range(10)]  # 10 alcohols

        with patch.object(
            calc.multi_endpoint_predictor, "predict_endpoints"
        ) as mock_predict:
            mock_predict.return_value = self._create_mock_predictions()

            import time

            start_time = time.time()

            results = calc.predict_toxicity(test_compounds)

            end_time = time.time()
            processing_time = end_time - start_time

            assert len(results) == 10
            assert processing_time < 5.0  # Should complete within 5 seconds
            print(
                f"Processed {len(test_compounds)} compounds in {processing_time:.2f} seconds"
            )

    def _create_mock_predictions(self, endpoint_type=None, species_type=None):
        """Create mock toxicity predictions."""
        from chemi_accel.toxicology.core import (
            ConfidenceLevel,
            ToxicityEndpoint,
            ToxicityPrediction,
        )

        if endpoint_type is None:
            endpoint_type = ToxicityEndpointType.ACUTE_ORAL
        if species_type is None:
            species_type = SpeciesType.RAT

        endpoint = ToxicityEndpoint(endpoint_type=endpoint_type, species=species_type)

        prediction = ToxicityPrediction(
            endpoint=endpoint,
            predicted_value=1500.0,  # mg/kg
            confidence=ConfidenceLevel.HIGH,
            uncertainty=0.15,
            model_type="mock_model",
        )

        return [prediction]

    def _create_mock_predictions_with_uncertainty(self):
        """Create mock predictions with varying uncertainty."""
        from chemi_accel.toxicology.core import (
            ConfidenceLevel,
            ToxicityEndpoint,
            ToxicityPrediction,
        )

        predictions = []

        endpoints = [
            ToxicityEndpointType.ACUTE_ORAL,
            ToxicityEndpointType.CARCINOGENICITY,
            ToxicityEndpointType.AQUATIC_ACUTE,
        ]

        uncertainties = [0.1, 0.3, 0.2]  # Different uncertainty levels

        for endpoint_type, uncertainty in zip(endpoints, uncertainties):
            endpoint = ToxicityEndpoint(
                endpoint_type=endpoint_type, species=SpeciesType.RAT
            )

            prediction = ToxicityPrediction(
                endpoint=endpoint,
                predicted_value=np.random.uniform(100, 2000),
                confidence=ConfidenceLevel.MODERATE,
                uncertainty=uncertainty,
                model_type="mock_model",
            )

            predictions.append(prediction)

        return predictions


class TestToxicologyModuleIntegration:
    """Test integration between different toxicology submodules."""

    def test_multi_endpoint_to_risk_assessment_integration(self):
        """Test integration from multi-endpoint prediction to risk assessment."""
        backend = get_backend("cpu")

        # Initialize components
        predictor = MultiEndpointToxicologyPredictor(backend)
        risk_assessor = HolisticRiskAssessment(backend)

        # Mock predictions
        with patch.object(predictor, "predict_endpoints") as mock_predict:
            mock_predict.return_value = self._create_mock_predictions_for_risk()

            # Get predictions
            predictions = predictor.predict_endpoints(
                "CCO",
                [ToxicityEndpointType.ACUTE_ORAL, ToxicityEndpointType.CARCINOGENICITY],
                [SpeciesType.RAT],
            )

            # Create toxicity data
            toxicity_data = ToxicityData(smiles="CCO", predicted_endpoints=predictions)

            # Assess risk
            risk_score = risk_assessor.assess_risk(
                toxicity_data, include_uncertainty=False
            )

            assert risk_score.overall_risk > 0
            assert len(risk_score.endpoint_contributions) > 0
            assert len(risk_score.key_drivers) > 0

    def test_risk_assessment_to_safer_design_integration(self):
        """Test integration from risk assessment to safer design."""
        backend = get_backend("cpu")

        # Initialize components
        risk_assessor = HolisticRiskAssessment(backend)
        optimizer = SaferByDesignOptimizer(backend)

        # Create high-risk toxicity data
        toxicity_data = self._create_high_risk_toxicity_data()

        # Assess risk
        risk_score = risk_assessor.assess_risk(toxicity_data, include_uncertainty=False)

        # Use risk assessment results to define optimization constraints
        constraints = []
        for endpoint_type, contribution in risk_score.endpoint_contributions.items():
            if contribution > 0.3:  # High contribution endpoints
                # Convert string back to enum (simplified)
                if endpoint_type == "acute_oral":
                    constraints.append(
                        ToxicityConstraint(
                            endpoint_type=ToxicityEndpointType.ACUTE_ORAL,
                            constraint_type="greater_than",
                            target_value=2000.0,
                        )
                    )

        # Mock optimization
        with patch.object(optimizer, "optimize_for_safety") as mock_optimize:
            from chemi_accel.toxicology.safer_design import SafeDesignCandidate

            mock_optimize.return_value = [
                SafeDesignCandidate(
                    smiles="CCCO",
                    predicted_properties={"molecular_weight": 60.0},
                    optimization_score=0.8,
                    constraint_violations=[],
                    similarity_to_parent=0.9,
                )
            ]

            # Optimize for safety
            candidates = optimizer.optimize_for_safety(
                toxicity_data.smiles,
                target_properties={"molecular_weight": 50.0},
                toxicity_constraints=constraints,
                optimization_objectives=[OptimizationObjective.MINIMIZE_TOXICITY],
            )

            assert len(candidates) > 0
            assert candidates[0].is_viable  # Should meet constraints

    def test_environmental_fate_integration(self):
        """Test integration with environmental fate prediction."""
        backend = get_backend("cpu")
        calc = ToxicologyCalculator(backend)

        # Test environmental fate prediction
        with patch.object(
            calc.environmental_fate_predictor, "predict_fate"
        ) as mock_fate:
            from chemi_accel.toxicology.core import ConfidenceLevel
            from chemi_accel.toxicology.environmental_fate import (
                DegradationConditions,
                DegradationMechanism,
                DegradationPathway,
                EnvironmentalCompartment,
            )

            mock_pathway = DegradationPathway(
                parent_compound="CCO",
                mechanism=DegradationMechanism.BIODEGRADATION,
                compartment=EnvironmentalCompartment.SOIL,
                conditions=[DegradationConditions.AEROBIC],
                products=[],
                overall_half_life=168.0,  # 1 week
                mass_balance=0.9,
                pathway_confidence=ConfidenceLevel.MODERATE,
            )

            mock_fate.return_value = {"soil": [mock_pathway]}

            env_fate = calc.predict_environmental_fate("CCO")

            assert "soil" in env_fate
            assert len(env_fate["soil"]) == 1
            assert env_fate["soil"][0].overall_half_life == 168.0

    def test_metabolic_prediction_integration(self):
        """Test integration with metabolic pathway prediction."""
        backend = get_backend("cpu")
        calc = ToxicologyCalculator(backend)

        # Test metabolic pathway prediction
        with patch.object(
            calc.metabolic_predictor, "predict_metabolism"
        ) as mock_metabolism:
            from chemi_accel.toxicology.metabolic_prediction import (
                MetabolicNetwork,
                Metabolite,
                MetaboliteType,
            )

            mock_metabolite = Metabolite(
                smiles="CC(=O)O",  # Acetic acid
                metabolite_type=MetaboliteType.PHASE_II,
                formation_pathway=[],
                abundance=0.3,
            )

            mock_network = MetabolicNetwork(
                parent_compound="CCO",
                species=SpeciesType.HUMAN,
                metabolites=[mock_metabolite],
                enzymatic_pathways={},
                network_confidence=ConfidenceLevel.MODERATE,
            )

            mock_metabolism.return_value = mock_network

            metabolism = calc.predict_metabolic_pathways("CCO")

            assert metabolism.parent_compound == "CCO"
            assert len(metabolism.metabolites) == 1
            assert metabolism.metabolites[0].smiles == "CC(=O)O"

    def _create_mock_predictions_for_risk(self):
        """Create mock predictions suitable for risk assessment."""
        from chemi_accel.toxicology.core import (
            ConfidenceLevel,
            ToxicityEndpoint,
            ToxicityPrediction,
        )

        predictions = []

        # High toxicity acute oral
        endpoint1 = ToxicityEndpoint(
            endpoint_type=ToxicityEndpointType.ACUTE_ORAL, species=SpeciesType.RAT
        )
        pred1 = ToxicityPrediction(
            endpoint=endpoint1,
            predicted_value=500.0,  # Toxic level
            confidence=ConfidenceLevel.HIGH,
            uncertainty=0.1,
            model_type="mock",
        )
        predictions.append(pred1)

        # Moderate carcinogenicity
        endpoint2 = ToxicityEndpoint(
            endpoint_type=ToxicityEndpointType.CARCINOGENICITY, species=SpeciesType.RAT
        )
        pred2 = ToxicityPrediction(
            endpoint=endpoint2,
            predicted_value=0.6,  # Moderate probability
            confidence=ConfidenceLevel.MODERATE,
            uncertainty=0.2,
            model_type="mock",
        )
        predictions.append(pred2)

        return predictions

    def _create_high_risk_toxicity_data(self):
        """Create toxicity data representing a high-risk compound."""
        from chemi_accel.toxicology.core import (
            ConfidenceLevel,
            ToxicityData,
            ToxicityEndpoint,
            ToxicityPrediction,
        )

        # Create high-risk predictions
        endpoint = ToxicityEndpoint(
            endpoint_type=ToxicityEndpointType.ACUTE_ORAL, species=SpeciesType.RAT
        )

        prediction = ToxicityPrediction(
            endpoint=endpoint,
            predicted_value=100.0,  # Very toxic
            confidence=ConfidenceLevel.HIGH,
            uncertainty=0.1,
            model_type="mock",
        )

        return ToxicityData(
            smiles="C1=CC=CC=C1",  # Benzene (known carcinogen)
            predicted_endpoints=[prediction],
        )


if __name__ == "__main__":
    pytest.main([__file__])
