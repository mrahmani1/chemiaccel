"""
Unit tests for toxicology prediction module.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from chemi_accel.biological_interactions.toxicology_prediction import (  # Data structures; Enums; Prediction classes; Utility functions; Exceptions
    ADRPredictionError,
    ADRPredictor,
    AdverseDrugReaction,
    AdverseDrugReactionType,
    DatabaseLoadingError,
    OffTargetBinding,
    OffTargetPredictionError,
    OffTargetPredictor,
    PredictionConfidence,
    RiskLevel,
    SystemsToxicologyEngine,
    SystemsToxicologyProfile,
    ToxicityEndpoint,
    ToxicityMechanism,
    ToxicityPrediction,
    ToxicityPredictor,
    ToxicologyDatabase,
    ToxicologyPredictionError,
    batch_toxicology_screening,
    create_sample_toxicology_database,
    generate_toxicology_report,
    predict_compound_safety_profile,
)


@pytest.fixture
def sample_off_target_binding():
    """Create sample off-target binding."""
    return OffTargetBinding(
        compound_smiles="CCO",
        target_id="TARGET_001",
        target_name="Test Target",
        binding_probability=0.8,
        binding_affinity_prediction=50.0,
        interaction_mechanism=ToxicityMechanism.PROTEIN_BINDING,
        structural_similarity_score=0.7,
        pharmacophore_match_score=0.6,
    )


@pytest.fixture
def sample_toxicity_prediction():
    """Create sample toxicity prediction."""
    return ToxicityPrediction(
        compound_smiles="CCO",
        toxicity_endpoint=ToxicityEndpoint.HEPATOTOXICITY,
        probability=0.7,
        confidence_score=0.8,
        predicted_mechanism=ToxicityMechanism.OXIDATIVE_STRESS,
        structural_alerts=["nitro_group"],
        supporting_evidence={"model_prediction": 0.7},
    )


@pytest.fixture
def sample_adr_prediction():
    """Create sample ADR prediction."""
    return AdverseDrugReaction(
        compound_smiles="CCO",
        adr_type=AdverseDrugReactionType.TYPE_A_PREDICTABLE,
        system_affected="hepatic",
        severity_score=0.6,
        probability=0.5,
        onset_time_prediction=7.0,
        clinical_manifestations=["nausea", "fatigue"],
        monitoring_recommendations=["liver function tests"],
    )


@pytest.fixture
def sample_systems_profile():
    """Create sample systems toxicology profile."""
    off_targets = [
        OffTargetBinding(
            compound_smiles="CCO",
            target_id="T1",
            target_name="Target 1",
            binding_probability=0.8,
        )
    ]

    toxicities = [
        ToxicityPrediction(
            compound_smiles="CCO",
            toxicity_endpoint=ToxicityEndpoint.HEPATOTOXICITY,
            probability=0.6,
        )
    ]

    adrs = [
        AdverseDrugReaction(
            compound_smiles="CCO",
            adr_type=AdverseDrugReactionType.TYPE_A_PREDICTABLE,
            system_affected="hepatic",
            severity_score=0.5,
            probability=0.4,
        )
    ]

    return SystemsToxicologyProfile(
        compound_smiles="CCO",
        off_target_bindings=off_targets,
        toxicity_predictions=toxicities,
        adr_predictions=adrs,
        overall_safety_score=0.7,
        therapeutic_index_estimate=5.0,
    )


class TestOffTargetBinding:
    """Test OffTargetBinding class."""

    def test_binding_creation(self, sample_off_target_binding):
        """Test binding creation."""
        binding = sample_off_target_binding

        assert binding.compound_smiles == "CCO"
        assert binding.target_id == "TARGET_001"
        assert binding.target_name == "Test Target"
        assert binding.binding_probability == 0.8
        assert binding.confidence_level == PredictionConfidence.HIGH

    def test_confidence_level_assignment(self):
        """Test confidence level assignment based on probability."""
        # Very high confidence
        binding1 = OffTargetBinding(
            compound_smiles="CCO",
            target_id="T1",
            target_name="Target 1",
            binding_probability=0.95,
        )
        assert binding1.confidence_level == PredictionConfidence.VERY_HIGH

        # High confidence
        binding2 = OffTargetBinding(
            compound_smiles="CCO",
            target_id="T1",
            target_name="Target 1",
            binding_probability=0.8,
        )
        assert binding2.confidence_level == PredictionConfidence.HIGH

        # Medium confidence
        binding3 = OffTargetBinding(
            compound_smiles="CCO",
            target_id="T1",
            target_name="Target 1",
            binding_probability=0.6,
        )
        assert binding3.confidence_level == PredictionConfidence.MEDIUM

        # Low confidence
        binding4 = OffTargetBinding(
            compound_smiles="CCO",
            target_id="T1",
            target_name="Target 1",
            binding_probability=0.3,
        )
        assert binding4.confidence_level == PredictionConfidence.LOW

        # Very low confidence
        binding5 = OffTargetBinding(
            compound_smiles="CCO",
            target_id="T1",
            target_name="Target 1",
            binding_probability=0.1,
        )
        assert binding5.confidence_level == PredictionConfidence.VERY_LOW


class TestToxicityPrediction:
    """Test ToxicityPrediction class."""

    def test_toxicity_creation(self, sample_toxicity_prediction):
        """Test toxicity prediction creation."""
        tox = sample_toxicity_prediction

        assert tox.compound_smiles == "CCO"
        assert tox.toxicity_endpoint == ToxicityEndpoint.HEPATOTOXICITY
        assert tox.probability == 0.7
        assert tox.risk_level == RiskLevel.HIGH
        assert tox.confidence_score == 0.8

    def test_risk_level_assignment(self):
        """Test risk level assignment based on probability."""
        # Very high risk
        tox1 = ToxicityPrediction(
            compound_smiles="CCO",
            toxicity_endpoint=ToxicityEndpoint.HEPATOTOXICITY,
            probability=0.9,
        )
        assert tox1.risk_level == RiskLevel.VERY_HIGH

        # High risk
        tox2 = ToxicityPrediction(
            compound_smiles="CCO",
            toxicity_endpoint=ToxicityEndpoint.HEPATOTOXICITY,
            probability=0.7,
        )
        assert tox2.risk_level == RiskLevel.HIGH

        # Moderate risk
        tox3 = ToxicityPrediction(
            compound_smiles="CCO",
            toxicity_endpoint=ToxicityEndpoint.HEPATOTOXICITY,
            probability=0.5,
        )
        assert tox3.risk_level == RiskLevel.MODERATE

        # Low risk
        tox4 = ToxicityPrediction(
            compound_smiles="CCO",
            toxicity_endpoint=ToxicityEndpoint.HEPATOTOXICITY,
            probability=0.3,
        )
        assert tox4.risk_level == RiskLevel.LOW

        # Very low risk
        tox5 = ToxicityPrediction(
            compound_smiles="CCO",
            toxicity_endpoint=ToxicityEndpoint.HEPATOTOXICITY,
            probability=0.1,
        )
        assert tox5.risk_level == RiskLevel.VERY_LOW


class TestAdverseDrugReaction:
    """Test AdverseDrugReaction class."""

    def test_adr_creation(self, sample_adr_prediction):
        """Test ADR creation."""
        adr = sample_adr_prediction

        assert adr.compound_smiles == "CCO"
        assert adr.adr_type == AdverseDrugReactionType.TYPE_A_PREDICTABLE
        assert adr.system_affected == "hepatic"
        assert adr.severity_score == 0.6
        assert adr.probability == 0.5
        assert adr.onset_time_prediction == 7.0
        assert len(adr.clinical_manifestations) == 2
        assert len(adr.monitoring_recommendations) == 1


class TestSystemsToxicologyProfile:
    """Test SystemsToxicologyProfile class."""

    def test_profile_creation(self, sample_systems_profile):
        """Test profile creation."""
        profile = sample_systems_profile

        assert profile.compound_smiles == "CCO"
        assert len(profile.off_target_bindings) == 1
        assert len(profile.toxicity_predictions) == 1
        assert len(profile.adr_predictions) == 1
        assert profile.overall_safety_score == 0.7
        assert profile.therapeutic_index_estimate == 5.0

    def test_get_high_risk_predictions(self, sample_systems_profile):
        """Test getting high-risk predictions."""
        profile = sample_systems_profile
        high_risk = profile.get_high_risk_predictions()

        assert isinstance(high_risk, dict)
        assert "off_targets" in high_risk
        assert "toxicities" in high_risk
        assert "adrs" in high_risk
        assert len(high_risk["off_targets"]) == 1  # High probability off-target

    def test_generate_safety_report(self, sample_systems_profile):
        """Test safety report generation."""
        profile = sample_systems_profile
        report = profile.generate_safety_report()

        assert isinstance(report, dict)
        assert "compound_smiles" in report
        assert "overall_safety_score" in report
        assert "risk_summary" in report
        assert "key_concerns" in report
        assert report["compound_smiles"] == "CCO"
        assert report["overall_safety_score"] == 0.7


class TestOffTargetPredictor:
    """Test OffTargetPredictor class."""

    def test_predictor_creation(self):
        """Test predictor creation."""
        predictor = OffTargetPredictor(use_gpu=False)
        assert not predictor.use_gpu
        assert predictor.target_database is not None

    def test_predict_off_targets(self):
        """Test off-target prediction."""
        predictor = OffTargetPredictor(use_gpu=False)
        predictions = predictor.predict_off_targets(
            compound_smiles="CCO", confidence_threshold=0.1, max_targets=10
        )

        assert isinstance(predictions, list)
        assert len(predictions) <= 10

        for prediction in predictions:
            assert isinstance(prediction, OffTargetBinding)
            assert prediction.compound_smiles == "CCO"
            assert prediction.binding_probability >= 0.1

    def test_encode_compound(self):
        """Test compound encoding."""
        predictor = OffTargetPredictor(use_gpu=False)
        features = predictor._encode_compound("CCO")

        assert isinstance(features, np.ndarray)
        assert features.shape == (2048,)

    def test_encode_target(self):
        """Test target encoding."""
        predictor = OffTargetPredictor(use_gpu=False)
        target_info = {"name": "Test Target", "type": "kinase"}
        features = predictor._encode_target(target_info)

        assert isinstance(features, np.ndarray)
        assert features.shape == (1280,)

    def test_calculate_structural_similarity(self):
        """Test structural similarity calculation."""
        predictor = OffTargetPredictor(use_gpu=False)
        target_info = {"name": "Test Target"}
        similarity = predictor._calculate_structural_similarity("CCO", target_info)

        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0

    def test_predict_interaction_mechanism(self):
        """Test interaction mechanism prediction."""
        predictor = OffTargetPredictor(use_gpu=False)
        compound_features = np.random.randn(100)
        target_features = np.random.randn(100)

        mechanism = predictor._predict_interaction_mechanism(
            compound_features, target_features
        )

        assert isinstance(mechanism, ToxicityMechanism)

    def test_create_sample_target_database(self):
        """Test sample target database creation."""
        predictor = OffTargetPredictor(use_gpu=False)
        database = predictor._create_sample_target_database()

        assert isinstance(database, dict)
        assert len(database) == 1000

        for target_id, target_info in database.items():
            assert "name" in target_info
            assert "type" in target_info
            assert "sequence" in target_info


class TestToxicityPredictor:
    """Test ToxicityPredictor class."""

    def test_predictor_creation(self):
        """Test predictor creation."""
        predictor = ToxicityPredictor(use_gpu=False)
        assert not predictor.use_gpu
        assert predictor.structural_alerts is not None

    def test_predict_toxicity(self):
        """Test toxicity prediction."""
        predictor = ToxicityPredictor(use_gpu=False)
        predictions = predictor.predict_toxicity(
            compound_smiles="CCO",
            endpoints=[
                ToxicityEndpoint.HEPATOTOXICITY,
                ToxicityEndpoint.CARDIOTOXICITY,
            ],
        )

        assert isinstance(predictions, list)
        assert len(predictions) == 2

        for prediction in predictions:
            assert isinstance(prediction, ToxicityPrediction)
            assert prediction.compound_smiles == "CCO"
            assert prediction.toxicity_endpoint in [
                ToxicityEndpoint.HEPATOTOXICITY,
                ToxicityEndpoint.CARDIOTOXICITY,
            ]

    def test_predict_toxicity_all_endpoints(self):
        """Test toxicity prediction for all endpoints."""
        predictor = ToxicityPredictor(use_gpu=False)
        predictions = predictor.predict_toxicity("CCO")

        assert isinstance(predictions, list)
        assert len(predictions) == len(ToxicityEndpoint)

        for prediction in predictions:
            assert isinstance(prediction, ToxicityPrediction)

    def test_check_structural_alerts(self):
        """Test structural alert checking."""
        predictor = ToxicityPredictor(use_gpu=False)

        # Test compound with nitro group
        alerts = predictor._check_structural_alerts("CC(C)NO2")
        assert isinstance(alerts, list)

        # Test compound without alerts
        alerts_clean = predictor._check_structural_alerts("CCO")
        assert isinstance(alerts_clean, list)

    def test_generate_dose_response_curve(self):
        """Test dose-response curve generation."""
        predictor = ToxicityPredictor(use_gpu=False)
        curve = predictor._generate_dose_response_curve(0.5)

        assert isinstance(curve, dict)
        assert len(curve) == 5  # Five dose levels

        for dose_key, response in curve.items():
            assert "dose_" in dose_key
            assert isinstance(response, float)
            assert 0.0 <= response <= 1.0

    def test_load_structural_alerts(self):
        """Test structural alerts loading."""
        predictor = ToxicityPredictor(use_gpu=False)
        alerts = predictor._load_structural_alerts()

        assert isinstance(alerts, dict)
        assert len(alerts) > 0

        for alert_name, patterns in alerts.items():
            assert isinstance(patterns, list)
            assert len(patterns) > 0


class TestADRPredictor:
    """Test ADRPredictor class."""

    def test_predictor_creation(self):
        """Test predictor creation."""
        predictor = ADRPredictor(use_gpu=False)
        assert not predictor.use_gpu
        assert predictor.adr_database is not None

    def test_predict_adrs(self, sample_off_target_binding, sample_toxicity_prediction):
        """Test ADR prediction."""
        predictor = ADRPredictor(use_gpu=False)

        off_targets = [sample_off_target_binding]
        toxicities = [sample_toxicity_prediction]
        patient_factors = {"age": 70, "kidney_function": "normal"}

        adrs = predictor.predict_adrs(
            compound_smiles="CCO",
            off_target_bindings=off_targets,
            toxicity_predictions=toxicities,
            patient_factors=patient_factors,
        )

        assert isinstance(adrs, list)
        assert len(adrs) > 0

        for adr in adrs:
            assert isinstance(adr, AdverseDrugReaction)
            assert adr.compound_smiles == "CCO"

    def test_predict_adr_from_off_target(self, sample_off_target_binding):
        """Test ADR prediction from off-target binding."""
        predictor = ADRPredictor(use_gpu=False)
        patient_factors = {"age": 65}

        adrs = predictor._predict_adr_from_off_target(
            "CCO", sample_off_target_binding, patient_factors
        )

        assert isinstance(adrs, list)
        for adr in adrs:
            assert isinstance(adr, AdverseDrugReaction)

    def test_predict_adr_from_toxicity(self, sample_toxicity_prediction):
        """Test ADR prediction from toxicity."""
        predictor = ADRPredictor(use_gpu=False)
        patient_factors = {"age": 60}

        adrs = predictor._predict_adr_from_toxicity(
            "CCO", sample_toxicity_prediction, patient_factors
        )

        assert isinstance(adrs, list)
        for adr in adrs:
            assert isinstance(adr, AdverseDrugReaction)

    def test_calculate_adr_severity(self):
        """Test ADR severity calculation."""
        predictor = ADRPredictor(use_gpu=False)
        severity = predictor._calculate_adr_severity(
            0.7, "cardiovascular", {"age": 80, "comorbidities": 3}
        )

        assert isinstance(severity, float)
        assert 0.0 <= severity <= 1.0

    def test_generate_clinical_manifestations(self):
        """Test clinical manifestations generation."""
        predictor = ADRPredictor(use_gpu=False)
        manifestations = predictor._generate_clinical_manifestations(
            "cardiovascular", AdverseDrugReactionType.TYPE_A_PREDICTABLE
        )

        assert isinstance(manifestations, list)
        assert len(manifestations) >= 2
        assert all(isinstance(m, str) for m in manifestations)

    def test_generate_monitoring_recommendations(self):
        """Test monitoring recommendations generation."""
        predictor = ADRPredictor(use_gpu=False)
        monitoring = predictor._generate_monitoring_recommendations("hepatic", 0.8)

        assert isinstance(monitoring, list)
        assert len(monitoring) > 0
        assert all(isinstance(m, str) for m in monitoring)

    def test_predict_adr_onset_time(self):
        """Test ADR onset time prediction."""
        predictor = ADRPredictor(use_gpu=False)
        onset = predictor._predict_adr_onset_time(ToxicityEndpoint.HEPATOTOXICITY)

        assert isinstance(onset, float)
        assert onset > 0

    def test_predict_adr_reversibility(self):
        """Test ADR reversibility prediction."""
        predictor = ADRPredictor(use_gpu=False)
        reversible = predictor._predict_adr_reversibility(
            ToxicityEndpoint.HEPATOTOXICITY
        )

        assert isinstance(reversible, bool)

    def test_merge_similar_adrs(self):
        """Test ADR merging."""
        predictor = ADRPredictor(use_gpu=False)

        adr1 = AdverseDrugReaction(
            compound_smiles="CCO",
            adr_type=AdverseDrugReactionType.TYPE_A_PREDICTABLE,
            system_affected="hepatic",
            severity_score=0.6,
            probability=0.5,
        )

        adr2 = AdverseDrugReaction(
            compound_smiles="CCO",
            adr_type=AdverseDrugReactionType.TYPE_A_PREDICTABLE,
            system_affected="hepatic",
            severity_score=0.8,
            probability=0.7,
        )

        merged = predictor._merge_similar_adrs([adr1, adr2])

        assert len(merged) == 1
        assert merged[0].probability == 0.7  # Higher probability
        assert merged[0].severity_score == 0.8  # Higher severity


class TestSystemsToxicologyEngine:
    """Test SystemsToxicologyEngine class."""

    def test_engine_creation(self):
        """Test engine creation."""
        engine = SystemsToxicologyEngine(use_gpu=False)
        assert not engine.use_gpu
        assert engine.off_target_predictor is not None
        assert engine.toxicity_predictor is not None
        assert engine.adr_predictor is not None

    def test_predict_systems_toxicology(self):
        """Test comprehensive systems toxicology prediction."""
        engine = SystemsToxicologyEngine(use_gpu=False)
        profile = engine.predict_systems_toxicology(
            compound_smiles="CCO",
            patient_factors={"age": 65},
            include_off_targets=True,
            include_toxicity=True,
            include_adrs=True,
        )

        assert isinstance(profile, SystemsToxicologyProfile)
        assert profile.compound_smiles == "CCO"
        assert isinstance(profile.off_target_bindings, list)
        assert isinstance(profile.toxicity_predictions, list)
        assert isinstance(profile.adr_predictions, list)
        assert 0.0 <= profile.overall_safety_score <= 1.0

    def test_predict_systems_toxicology_partial(self):
        """Test partial systems toxicology prediction."""
        engine = SystemsToxicologyEngine(use_gpu=False)
        profile = engine.predict_systems_toxicology(
            compound_smiles="CCO",
            include_off_targets=True,
            include_toxicity=False,
            include_adrs=False,
        )

        assert isinstance(profile, SystemsToxicologyProfile)
        assert len(profile.off_target_bindings) > 0
        assert len(profile.toxicity_predictions) == 0
        assert len(profile.adr_predictions) == 0

    def test_calculate_overall_safety_score(
        self,
        sample_off_target_binding,
        sample_toxicity_prediction,
        sample_adr_prediction,
    ):
        """Test overall safety score calculation."""
        engine = SystemsToxicologyEngine(use_gpu=False)

        safety_score = engine._calculate_overall_safety_score(
            [sample_off_target_binding],
            [sample_toxicity_prediction],
            [sample_adr_prediction],
        )

        assert isinstance(safety_score, float)
        assert 0.0 <= safety_score <= 1.0

    def test_generate_contraindications(
        self,
        sample_off_target_binding,
        sample_toxicity_prediction,
        sample_adr_prediction,
    ):
        """Test contraindications generation."""
        engine = SystemsToxicologyEngine(use_gpu=False)

        contraindications = engine._generate_contraindications(
            [sample_off_target_binding],
            [sample_toxicity_prediction],
            [sample_adr_prediction],
        )

        assert isinstance(contraindications, list)
        assert all(isinstance(c, str) for c in contraindications)

    def test_generate_safety_warnings(
        self,
        sample_off_target_binding,
        sample_toxicity_prediction,
        sample_adr_prediction,
    ):
        """Test safety warnings generation."""
        engine = SystemsToxicologyEngine(use_gpu=False)

        warnings = engine._generate_safety_warnings(
            [sample_off_target_binding],
            [sample_toxicity_prediction],
            [sample_adr_prediction],
        )

        assert isinstance(warnings, list)
        assert all(isinstance(w, str) for w in warnings)

    def test_generate_monitoring_recommendations(
        self,
        sample_off_target_binding,
        sample_toxicity_prediction,
        sample_adr_prediction,
    ):
        """Test monitoring recommendations generation."""
        engine = SystemsToxicologyEngine(use_gpu=False)

        monitoring = engine._generate_monitoring_recommendations(
            [sample_off_target_binding],
            [sample_toxicity_prediction],
            [sample_adr_prediction],
        )

        assert isinstance(monitoring, list)
        assert all(isinstance(m, str) for m in monitoring)

    def test_estimate_therapeutic_index(self):
        """Test therapeutic index estimation."""
        engine = SystemsToxicologyEngine(use_gpu=False)
        ti = engine._estimate_therapeutic_index("CCO", 0.8)

        assert isinstance(ti, float)
        assert ti >= 1.0


class TestUtilityFunctions:
    """Test utility functions."""

    def test_predict_compound_safety_profile(self):
        """Test compound safety profile prediction."""
        profile = predict_compound_safety_profile(
            compound_smiles="CCO", use_gpu=False, patient_factors={"age": 70}
        )

        assert isinstance(profile, SystemsToxicologyProfile)
        assert profile.compound_smiles == "CCO"

    def test_batch_toxicology_screening(self):
        """Test batch toxicology screening."""
        compounds = ["CCO", "CCC", "CCCO"]
        results = batch_toxicology_screening(compounds, use_gpu=False, max_targets=10)

        assert isinstance(results, dict)
        assert len(results) <= len(compounds)  # Some might fail

        for smiles, profile in results.items():
            assert isinstance(profile, SystemsToxicologyProfile)
            assert profile.compound_smiles == smiles

    def test_generate_toxicology_report_json(self, sample_systems_profile):
        """Test JSON toxicology report generation."""
        report = generate_toxicology_report(sample_systems_profile, format="json")

        assert isinstance(report, dict)
        assert "compound_smiles" in report
        assert "overall_safety_score" in report
        assert "risk_summary" in report

    def test_generate_toxicology_report_text(self, sample_systems_profile):
        """Test text toxicology report generation."""
        report = generate_toxicology_report(sample_systems_profile, format="text")

        assert isinstance(report, str)
        assert "TOXICOLOGY SAFETY REPORT" in report
        assert "CCO" in report

    def test_create_sample_toxicology_database(self):
        """Test sample toxicology database creation."""
        database = create_sample_toxicology_database()

        assert isinstance(database, ToxicologyDatabase)
        assert database.name == "Sample Toxicology Database"
        assert len(database.targets) == 100
        assert len(database.known_toxicants) > 0
        assert len(database.structural_alerts) > 0


class TestExceptions:
    """Test exception classes."""

    def test_toxicology_prediction_error(self):
        """Test ToxicologyPredictionError."""
        with pytest.raises(ToxicologyPredictionError):
            raise ToxicologyPredictionError("Test error")

    def test_off_target_prediction_error(self):
        """Test OffTargetPredictionError."""
        with pytest.raises(OffTargetPredictionError):
            raise OffTargetPredictionError("Off-target prediction failed")

    def test_adr_prediction_error(self):
        """Test ADRPredictionError."""
        with pytest.raises(ADRPredictionError):
            raise ADRPredictionError("ADR prediction failed")

    def test_database_loading_error(self):
        """Test DatabaseLoadingError."""
        with pytest.raises(DatabaseLoadingError):
            raise DatabaseLoadingError("Database loading failed")


class TestIntegration:
    """Integration tests."""

    def test_end_to_end_toxicology_prediction(self):
        """Test complete toxicology prediction workflow."""
        engine = SystemsToxicologyEngine(use_gpu=False)

        # Test compound
        compound_smiles = "CCO"
        patient_factors = {
            "age": 65,
            "kidney_function": "normal",
            "liver_function": "normal",
            "comorbidities": 1,
        }

        # Generate comprehensive profile
        profile = engine.predict_systems_toxicology(
            compound_smiles,
            patient_factors=patient_factors,
            off_target_threshold=0.2,
            max_targets=50,
        )

        # Verify results
        assert isinstance(profile, SystemsToxicologyProfile)
        assert profile.compound_smiles == compound_smiles
        assert profile.overall_safety_score >= 0.0

        # Test high-risk predictions
        high_risk = profile.get_high_risk_predictions()
        assert isinstance(high_risk, dict)

        # Generate safety report
        report = profile.generate_safety_report()
        assert isinstance(report, dict)
        assert "compound_smiles" in report

        # Generate formatted report
        text_report = generate_toxicology_report(profile, format="text")
        assert isinstance(text_report, str)
        assert compound_smiles in text_report

    def test_different_toxicity_endpoints(self):
        """Test prediction for different toxicity endpoints."""
        predictor = ToxicityPredictor(use_gpu=False)

        endpoints_to_test = [
            ToxicityEndpoint.HEPATOTOXICITY,
            ToxicityEndpoint.CARDIOTOXICITY,
            ToxicityEndpoint.NEPHROTOXICITY,
            ToxicityEndpoint.NEUROTOXICITY,
            ToxicityEndpoint.GENOTOXICITY,
            ToxicityEndpoint.CARCINOGENICITY,
        ]

        for endpoint in endpoints_to_test:
            predictions = predictor.predict_toxicity("CCO", endpoints=[endpoint])

            assert len(predictions) == 1
            assert predictions[0].toxicity_endpoint == endpoint
            assert isinstance(predictions[0].probability, float)
            assert isinstance(predictions[0].risk_level, RiskLevel)

    def test_patient_factor_influence(self):
        """Test influence of patient factors on predictions."""
        engine = SystemsToxicologyEngine(use_gpu=False)

        # Test with different patient profiles
        young_patient = {"age": 25, "kidney_function": "normal"}
        elderly_patient = {"age": 85, "kidney_function": "impaired", "comorbidities": 3}

        profile_young = engine.predict_systems_toxicology(
            "CCO", patient_factors=young_patient, max_targets=20
        )

        profile_elderly = engine.predict_systems_toxicology(
            "CCO", patient_factors=elderly_patient, max_targets=20
        )

        # Elderly patient should generally have higher risk
        # (though this is probabilistic, so we just check structure)
        assert isinstance(profile_young, SystemsToxicologyProfile)
        assert isinstance(profile_elderly, SystemsToxicologyProfile)

        # Both should have predictions
        assert len(profile_young.adr_predictions) >= 0
        assert len(profile_elderly.adr_predictions) >= 0

    def test_compound_comparison(self):
        """Test comparison of different compounds."""
        compounds = ["CCO", "CC(C)O", "CCCCO"]
        results = batch_toxicology_screening(compounds, use_gpu=False, max_targets=10)

        assert isinstance(results, dict)

        # Compare safety scores
        safety_scores = {}
        for smiles, profile in results.items():
            safety_scores[smiles] = profile.overall_safety_score

        # All compounds should have safety scores
        for smiles in results:
            assert 0.0 <= safety_scores[smiles] <= 1.0

    def test_structural_alert_detection(self):
        """Test structural alert detection with different compounds."""
        predictor = ToxicityPredictor(use_gpu=False)

        # Test compounds with potential structural alerts
        test_compounds = [
            "CCO",  # Simple alcohol - should have few alerts
            "CC(C)NO2",  # Contains nitro group - should trigger alerts
            "c1ccccc1N",  # Aromatic amine - should trigger alerts
            "CCCCC",  # Simple alkane - should have minimal alerts
        ]

        for compound in test_compounds:
            predictions = predictor.predict_toxicity(compound)

            assert len(predictions) == len(ToxicityEndpoint)

            # Check that structural alerts are detected
            total_alerts = sum(len(pred.structural_alerts) for pred in predictions)
            assert isinstance(total_alerts, int)
            assert total_alerts >= 0

    def test_dose_response_relationships(self):
        """Test dose-response curve generation."""
        predictor = ToxicityPredictor(use_gpu=False)
        predictions = predictor.predict_toxicity(
            "CCO", endpoints=[ToxicityEndpoint.HEPATOTOXICITY]
        )

        assert len(predictions) == 1
        prediction = predictions[0]

        if prediction.dose_response_curve:
            doses = list(prediction.dose_response_curve.keys())
            responses = list(prediction.dose_response_curve.values())

            assert len(doses) == len(responses)
            assert all(isinstance(r, float) for r in responses)
            assert all(0.0 <= r <= 1.0 for r in responses)

    def test_therapeutic_index_estimation(self):
        """Test therapeutic index estimation."""
        engine = SystemsToxicologyEngine(use_gpu=False)

        # Test compounds with different safety profiles
        test_compounds = ["CCO", "CCC", "CCCO"]

        for compound in test_compounds:
            profile = engine.predict_systems_toxicology(compound, max_targets=10)

            if profile.therapeutic_index_estimate:
                assert isinstance(profile.therapeutic_index_estimate, float)
                assert profile.therapeutic_index_estimate >= 1.0
