"""
Test cases for holistic risk assessment and uncertainty quantification.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from chemi_accel.core.backend import get_backend
from chemi_accel.toxicology.core import (
    ConfidenceLevel,
    SpeciesType,
    ToxicityData,
    ToxicityEndpoint,
    ToxicityEndpointType,
    ToxicityPrediction,
)
from chemi_accel.toxicology.risk_assessment import (
    ExposureRoute,
    ExposureScenario,
    HolisticRiskAssessment,
    RiskCategory,
    RiskScore,
    UncertaintyDistribution,
    UncertaintyQuantification,
    UncertaintySource,
)


class TestExposureScenario:
    """Test ExposureScenario class."""

    def test_exposure_scenario_creation(self):
        """Test creating exposure scenario."""
        scenario = ExposureScenario(
            scenario_name="consumer_oral",
            exposure_route=ExposureRoute.ORAL,
            exposure_level=0.01,
            exposure_duration=365,
            exposure_frequency=1.0,
            population="adults",
        )

        assert scenario.scenario_name == "consumer_oral"
        assert scenario.exposure_route == ExposureRoute.ORAL
        assert scenario.exposure_level == 0.01
        assert scenario.exposure_duration == 365
        assert scenario.exposure_frequency == 1.0
        assert scenario.population == "adults"

    def test_total_dose_calculation(self):
        """Test total dose calculation."""
        scenario = ExposureScenario(
            scenario_name="test",
            exposure_route=ExposureRoute.ORAL,
            exposure_level=0.1,  # mg/kg/day
            exposure_duration=10,  # days
            exposure_frequency=2.0,  # twice per day
            population="test",
        )

        expected_total = 0.1 * 10 * 2.0  # 2.0 mg/kg total
        assert scenario.total_dose == expected_total

    def test_chronic_equivalent_dose(self):
        """Test chronic equivalent dose calculation."""
        scenario = ExposureScenario(
            scenario_name="test",
            exposure_route=ExposureRoute.ORAL,
            exposure_level=1.0,
            exposure_duration=365,  # 1 year
            exposure_frequency=1.0,
            population="test",
        )

        lifetime_days = 70 * 365  # 70 years
        expected_chronic = (1.0 * 365 * 1.0) / lifetime_days
        assert abs(scenario.chronic_equivalent_dose - expected_chronic) < 1e-6


class TestUncertaintyDistribution:
    """Test UncertaintyDistribution class."""

    def test_normal_distribution_sampling(self):
        """Test normal distribution sampling."""
        dist = UncertaintyDistribution(
            parameter_name="test_param", distribution_type="normal", mean=10.0, std=2.0
        )

        samples = dist.sample(1000)
        assert len(samples) == 1000
        assert abs(np.mean(samples) - 10.0) < 0.5  # Close to expected mean
        assert abs(np.std(samples) - 2.0) < 0.5  # Close to expected std

    def test_lognormal_distribution_sampling(self):
        """Test lognormal distribution sampling."""
        dist = UncertaintyDistribution(
            parameter_name="test_param",
            distribution_type="lognormal",
            mean=0.0,  # log-scale mean
            std=0.5,  # log-scale std
        )

        samples = dist.sample(1000)
        assert len(samples) == 1000
        assert np.all(samples > 0)  # Lognormal samples are positive

    def test_uniform_distribution_sampling(self):
        """Test uniform distribution sampling."""
        dist = UncertaintyDistribution(
            parameter_name="test_param",
            distribution_type="uniform",
            mean=5.0,  # Not used for uniform
            min_value=2.0,
            max_value=8.0,
        )

        samples = dist.sample(1000)
        assert len(samples) == 1000
        assert np.all(samples >= 2.0)
        assert np.all(samples <= 8.0)

    def test_triangular_distribution_sampling(self):
        """Test triangular distribution sampling."""
        dist = UncertaintyDistribution(
            parameter_name="test_param",
            distribution_type="triangular",
            mean=5.0,  # mode
            min_value=2.0,
            max_value=8.0,
        )

        samples = dist.sample(1000)
        assert len(samples) == 1000
        assert np.all(samples >= 2.0)
        assert np.all(samples <= 8.0)

    def test_unsupported_distribution(self):
        """Test error handling for unsupported distribution."""
        dist = UncertaintyDistribution(
            parameter_name="test_param", distribution_type="unsupported", mean=5.0
        )

        with pytest.raises(ValueError, match="Unsupported distribution type"):
            dist.sample(100)


class TestRiskScore:
    """Test RiskScore class."""

    def test_risk_score_creation(self):
        """Test creating risk score."""
        risk_score = RiskScore(
            overall_risk=0.3,
            risk_category=RiskCategory.MODERATE,
            endpoint_contributions={"acute_oral": 0.6, "carcinogenicity": 0.4},
            exposure_contributions={"oral": 0.8, "dermal": 0.2},
            uncertainty_range=(0.2, 0.5),
            confidence=ConfidenceLevel.HIGH,
        )

        assert risk_score.overall_risk == 0.3
        assert risk_score.risk_category == RiskCategory.MODERATE
        assert risk_score.confidence == ConfidenceLevel.HIGH
        assert risk_score.uncertainty_range == (0.2, 0.5)

    def test_uncertainty_factor_calculation(self):
        """Test uncertainty factor calculation."""
        # Test with Monte Carlo samples
        samples = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        risk_score = RiskScore(
            overall_risk=0.3,
            risk_category=RiskCategory.MODERATE,
            endpoint_contributions={},
            exposure_contributions={},
            uncertainty_range=(0.1, 0.9),
            confidence=ConfidenceLevel.MODERATE,
            monte_carlo_samples=samples,
        )

        p50 = np.percentile(samples, 50)  # 0.55
        p95 = np.percentile(samples, 95)  # 0.955
        expected_factor = p95 / p50

        assert abs(risk_score.uncertainty_factor - expected_factor) < 0.01

        # Test without Monte Carlo samples
        risk_score_no_mc = RiskScore(
            overall_risk=0.3,
            risk_category=RiskCategory.MODERATE,
            endpoint_contributions={},
            exposure_contributions={},
            uncertainty_range=(0.2, 0.6),
            confidence=ConfidenceLevel.MODERATE,
        )

        expected_factor_no_mc = 0.6 / 0.3  # 2.0
        assert abs(risk_score_no_mc.uncertainty_factor - expected_factor_no_mc) < 0.01


class TestUncertaintyQuantification:
    """Test UncertaintyQuantification class."""

    def test_uncertainty_quantifier_initialization(self):
        """Test uncertainty quantifier initialization."""
        backend = get_backend("cpu")
        uq = UncertaintyQuantification(backend)

        assert uq.backend == backend
        assert isinstance(uq.uncertainty_sources, dict)
        assert uq.correlation_matrix is None

    def test_add_uncertainty_source(self):
        """Test adding uncertainty source."""
        backend = get_backend("cpu")
        uq = UncertaintyQuantification(backend)

        dist = UncertaintyDistribution(
            parameter_name="test_param", distribution_type="normal", mean=1.0, std=0.1
        )

        uq.add_uncertainty_source(UncertaintySource.MODEL_UNCERTAINTY, dist)

        assert UncertaintySource.MODEL_UNCERTAINTY in uq.uncertainty_sources
        assert uq.uncertainty_sources[UncertaintySource.MODEL_UNCERTAINTY] == dist

    def test_propagate_uncertainty(self):
        """Test uncertainty propagation."""
        backend = get_backend("cpu")
        uq = UncertaintyQuantification(backend)

        # Add uncertainty sources
        dist1 = UncertaintyDistribution("param1", "normal", 1.0, 0.1)
        dist2 = UncertaintyDistribution("param2", "normal", 2.0, 0.2)

        uq.add_uncertainty_source(UncertaintySource.MODEL_UNCERTAINTY, dist1)
        uq.add_uncertainty_source(UncertaintySource.EXPOSURE_VARIABILITY, dist2)

        # Define test risk function
        def test_risk_function(
            model_uncertainty=1.0, exposure_variability=2.0, **kwargs
        ):
            return model_uncertainty * exposure_variability

        # Propagate uncertainty
        results = uq.propagate_uncertainty(test_risk_function, n_samples=100)

        assert "mean" in results
        assert "median" in results
        assert "std" in results
        assert "percentiles" in results
        assert "samples" in results
        assert len(results["samples"]) <= 100  # May be less due to failed samples

        # Check that mean is approximately 1.0 * 2.0 = 2.0
        assert abs(results["mean"] - 2.0) < 0.5

    def test_sensitivity_analysis(self):
        """Test sensitivity analysis."""
        backend = get_backend("cpu")
        uq = UncertaintyQuantification(backend)

        # Define test function
        def test_function(param1=1.0, param2=2.0, **kwargs):
            return param1**2 + param2

        base_params = {"param1": 1.0, "param2": 2.0}

        sensitivities = uq.sensitivity_analysis(test_function, base_params)

        assert "param1" in sensitivities
        assert "param2" in sensitivities
        assert isinstance(sensitivities["param1"], float)
        assert isinstance(sensitivities["param2"], float)

        # param1 should be more sensitive (quadratic term)
        assert sensitivities["param1"] > sensitivities["param2"]


class TestHolisticRiskAssessment:
    """Test HolisticRiskAssessment class."""

    def test_risk_assessor_initialization(self):
        """Test risk assessor initialization."""
        backend = get_backend("cpu")
        assessor = HolisticRiskAssessment(backend)

        assert assessor.backend == backend
        assert isinstance(assessor.uncertainty_quantifier, UncertaintyQuantification)
        assert isinstance(assessor.safety_factors, dict)
        assert isinstance(assessor.risk_thresholds, dict)

    def test_create_default_exposure_scenarios(self):
        """Test creating default exposure scenarios."""
        backend = get_backend("cpu")
        assessor = HolisticRiskAssessment(backend)

        scenarios = assessor._create_default_exposure_scenarios()

        assert len(scenarios) >= 3  # Should have multiple scenarios
        assert all(isinstance(s, ExposureScenario) for s in scenarios)

        # Check that different exposure routes are covered
        routes = [s.exposure_route for s in scenarios]
        assert ExposureRoute.ORAL in routes
        assert ExposureRoute.DERMAL in routes or ExposureRoute.INHALATION in routes

    def test_calculate_deterministic_risk(self):
        """Test deterministic risk calculation."""
        backend = get_backend("cpu")
        assessor = HolisticRiskAssessment(backend)

        # Create test toxicity data
        endpoint = ToxicityEndpoint(
            endpoint_type=ToxicityEndpointType.ACUTE_ORAL, species=SpeciesType.RAT
        )
        prediction = ToxicityPrediction(
            endpoint=endpoint,
            predicted_value=500.0,  # mg/kg
            confidence=ConfidenceLevel.HIGH,
            uncertainty=0.1,
            model_type="test",
        )
        toxicity_data = ToxicityData(smiles="CCO", predicted_endpoints=[prediction])

        # Create test exposure scenario
        scenario = ExposureScenario(
            scenario_name="test_oral",
            exposure_route=ExposureRoute.ORAL,
            exposure_level=1.0,  # mg/kg/day
            exposure_duration=1,
            exposure_frequency=1.0,
            population="test",
        )

        risk = assessor._calculate_deterministic_risk(toxicity_data, None, [scenario])

        assert isinstance(risk, float)
        assert risk >= 0.0
        # Risk should be > 0 since exposure (1.0) < toxicity threshold (500.0)
        # but safety factors are applied

    def test_get_relevant_endpoints(self):
        """Test getting relevant endpoints for exposure routes."""
        backend = get_backend("cpu")
        assessor = HolisticRiskAssessment(backend)

        # Test oral exposure
        oral_endpoints = assessor._get_relevant_endpoints(ExposureRoute.ORAL)
        assert ToxicityEndpointType.ACUTE_ORAL in oral_endpoints

        # Test dermal exposure
        dermal_endpoints = assessor._get_relevant_endpoints(ExposureRoute.DERMAL)
        assert ToxicityEndpointType.ACUTE_DERMAL in dermal_endpoints
        assert ToxicityEndpointType.SKIN_IRRITATION in dermal_endpoints

        # Test environmental exposure
        env_endpoints = assessor._get_relevant_endpoints(ExposureRoute.ENVIRONMENTAL)
        assert ToxicityEndpointType.AQUATIC_ACUTE in env_endpoints

    def test_categorize_risk(self):
        """Test risk categorization."""
        backend = get_backend("cpu")
        assessor = HolisticRiskAssessment(backend)

        assert assessor._categorize_risk(0.005) == RiskCategory.NEGLIGIBLE
        assert assessor._categorize_risk(0.05) == RiskCategory.LOW
        assert assessor._categorize_risk(0.3) == RiskCategory.MODERATE
        assert assessor._categorize_risk(0.7) == RiskCategory.HIGH
        assert assessor._categorize_risk(0.9) == RiskCategory.VERY_HIGH
        assert assessor._categorize_risk(1.5) == RiskCategory.UNACCEPTABLE

    def test_assess_risk_basic(self):
        """Test basic risk assessment."""
        backend = get_backend("cpu")
        assessor = HolisticRiskAssessment(backend)

        # Create test toxicity data
        endpoint = ToxicityEndpoint(
            endpoint_type=ToxicityEndpointType.ACUTE_ORAL, species=SpeciesType.RAT
        )
        prediction = ToxicityPrediction(
            endpoint=endpoint,
            predicted_value=1000.0,
            confidence=ConfidenceLevel.HIGH,
            uncertainty=0.1,
            model_type="test",
        )
        toxicity_data = ToxicityData(smiles="CCO", predicted_endpoints=[prediction])

        # Assess risk
        risk_score = assessor.assess_risk(
            toxicity_data,
            include_uncertainty=False,
            monte_carlo_samples=10,  # Small number for testing
        )

        assert isinstance(risk_score, RiskScore)
        assert 0.0 <= risk_score.overall_risk <= 1.0
        assert isinstance(risk_score.risk_category, RiskCategory)
        assert isinstance(risk_score.confidence, ConfidenceLevel)
        assert isinstance(risk_score.endpoint_contributions, dict)
        assert isinstance(risk_score.exposure_contributions, dict)
        assert isinstance(risk_score.key_drivers, list)

    def test_assess_risk_with_uncertainty(self):
        """Test risk assessment with uncertainty quantification."""
        backend = get_backend("cpu")
        assessor = HolisticRiskAssessment(backend)

        # Create test toxicity data
        endpoint = ToxicityEndpoint(
            endpoint_type=ToxicityEndpointType.CARCINOGENICITY, species=SpeciesType.RAT
        )
        prediction = ToxicityPrediction(
            endpoint=endpoint,
            predicted_value=0.8,  # High carcinogenicity
            confidence=ConfidenceLevel.MODERATE,
            uncertainty=0.3,
            model_type="test",
        )
        toxicity_data = ToxicityData(
            smiles="C1=CC=CC=C1", predicted_endpoints=[prediction]  # Benzene
        )

        # Assess risk with uncertainty
        risk_score = assessor.assess_risk(
            toxicity_data,
            include_uncertainty=True,
            monte_carlo_samples=50,  # Small number for testing
        )

        assert isinstance(risk_score, RiskScore)
        assert risk_score.monte_carlo_samples is not None
        assert len(risk_score.monte_carlo_samples) <= 50
        assert (
            risk_score.uncertainty_range[0]
            <= risk_score.overall_risk
            <= risk_score.uncertainty_range[1]
        )

    def test_compare_scenarios(self):
        """Test scenario comparison."""
        backend = get_backend("cpu")
        assessor = HolisticRiskAssessment(backend)

        # Create test toxicity data for two compounds
        endpoint1 = ToxicityEndpoint(
            endpoint_type=ToxicityEndpointType.ACUTE_ORAL, species=SpeciesType.RAT
        )
        prediction1 = ToxicityPrediction(
            endpoint=endpoint1,
            predicted_value=100.0,  # High toxicity
            confidence=ConfidenceLevel.HIGH,
            uncertainty=0.1,
            model_type="test",
        )
        toxicity_data1 = ToxicityData(smiles="CCC", predicted_endpoints=[prediction1])

        endpoint2 = ToxicityEndpoint(
            endpoint_type=ToxicityEndpointType.ACUTE_ORAL, species=SpeciesType.RAT
        )
        prediction2 = ToxicityPrediction(
            endpoint=endpoint2,
            predicted_value=2000.0,  # Low toxicity
            confidence=ConfidenceLevel.HIGH,
            uncertainty=0.1,
            model_type="test",
        )
        toxicity_data2 = ToxicityData(smiles="CCO", predicted_endpoints=[prediction2])

        # Compare scenarios
        comparison = assessor.compare_scenarios(
            [toxicity_data1, toxicity_data2],
            ["Compound A", "Compound B"],
            include_uncertainty=False,
        )

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert "scenario" in comparison.columns
        assert "compound" in comparison.columns
        assert "overall_risk" in comparison.columns
        assert "risk_category" in comparison.columns

        # High toxicity compound should have higher risk
        compound_a_risk = comparison[comparison["scenario"] == "Compound A"][
            "overall_risk"
        ].iloc[0]
        compound_b_risk = comparison[comparison["scenario"] == "Compound B"][
            "overall_risk"
        ].iloc[0]
        assert compound_a_risk > compound_b_risk

    def test_generate_risk_report(self):
        """Test risk report generation."""
        backend = get_backend("cpu")
        assessor = HolisticRiskAssessment(backend)

        # Create test data
        endpoint = ToxicityEndpoint(
            endpoint_type=ToxicityEndpointType.CARCINOGENICITY, species=SpeciesType.RAT
        )
        prediction = ToxicityPrediction(
            endpoint=endpoint,
            predicted_value=0.9,
            confidence=ConfidenceLevel.HIGH,
            uncertainty=0.1,
            model_type="test",
        )
        toxicity_data = ToxicityData(
            smiles="C1=CC=CC=C1", predicted_endpoints=[prediction]
        )

        risk_score = RiskScore(
            overall_risk=0.8,
            risk_category=RiskCategory.HIGH,
            endpoint_contributions={"carcinogenicity": 1.0},
            exposure_contributions={"oral": 0.6, "dermal": 0.4},
            uncertainty_range=(0.7, 0.9),
            confidence=ConfidenceLevel.HIGH,
            key_drivers=["High carcinogenicity toxicity", "High oral exposure"],
        )

        report = assessor.generate_risk_report(risk_score, toxicity_data)

        assert isinstance(report, dict)
        assert "executive_summary" in report
        assert "risk_breakdown" in report
        assert "uncertainty_analysis" in report
        assert "recommendations" in report

        # Check executive summary
        exec_summary = report["executive_summary"]
        assert exec_summary["compound"] == "C1=CC=CC=C1"
        assert exec_summary["overall_risk_score"] == 0.8
        assert exec_summary["risk_category"] == "high"

        # Check recommendations
        recommendations = report["recommendations"]
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("alternatives" in rec.lower() for rec in recommendations)


if __name__ == "__main__":
    pytest.main([__file__])
