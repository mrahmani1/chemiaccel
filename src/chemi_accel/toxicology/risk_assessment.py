"""
Holistic risk assessment and uncertainty quantification.

Integrates predicted toxicity, environmental fate, and exposure scenarios
into a holistic risk assessment score with comprehensive uncertainty quantification.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..core.backend import Backend
from ..core.exceptions import ChemiAccelError
from .core import ConfidenceLevel, ToxicityData


class RiskCategory(Enum):
    """Risk categories for chemical assessment."""

    NEGLIGIBLE = "negligible"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    UNACCEPTABLE = "unacceptable"


class ExposureRoute(Enum):
    """Routes of exposure for risk assessment."""

    ORAL = "oral"
    DERMAL = "dermal"
    INHALATION = "inhalation"
    ENVIRONMENTAL = "environmental"
    OCCUPATIONAL = "occupational"


class UncertaintySource(Enum):
    """Sources of uncertainty in risk assessment."""

    MODEL_UNCERTAINTY = "model_uncertainty"
    DATA_QUALITY = "data_quality"
    SPECIES_EXTRAPOLATION = "species_extrapolation"
    EXPOSURE_VARIABILITY = "exposure_variability"
    ENDPOINT_CORRELATION = "endpoint_correlation"
    TEMPORAL_VARIATION = "temporal_variation"


@dataclass
class ExposureScenario:
    """Exposure scenario for risk assessment."""

    scenario_name: str
    exposure_route: ExposureRoute
    exposure_level: float  # mg/kg/day or mg/m3
    exposure_duration: float  # days
    exposure_frequency: float  # exposures per day
    population: str  # e.g., "adults", "children", "workers"
    confidence: ConfidenceLevel = ConfidenceLevel.MODERATE
    variability_factor: float = 10.0  # Inter-individual variability

    @property
    def total_dose(self) -> float:
        """Calculate total cumulative dose."""
        return self.exposure_level * self.exposure_frequency * self.exposure_duration

    @property
    def chronic_equivalent_dose(self) -> float:
        """Calculate chronic equivalent dose (lifetime average)."""
        lifetime_days = 70 * 365  # 70 years
        return self.total_dose / lifetime_days


@dataclass
class UncertaintyDistribution:
    """Uncertainty distribution for Monte Carlo analysis."""

    parameter_name: str
    distribution_type: str  # "normal", "lognormal", "uniform", "triangular"
    mean: float
    std: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    confidence_interval: Tuple[float, float] = (0.025, 0.975)  # 95% CI

    def sample(self, n_samples: int) -> np.ndarray:
        """Sample from uncertainty distribution."""
        if self.distribution_type == "normal":
            return np.random.normal(self.mean, self.std, n_samples)
        elif self.distribution_type == "lognormal":
            return np.random.lognormal(self.mean, self.std, n_samples)
        elif self.distribution_type == "uniform":
            return np.random.uniform(self.min_value, self.max_value, n_samples)
        elif self.distribution_type == "triangular":
            mode = self.mean
            return np.random.triangular(self.min_value, mode, self.max_value, n_samples)
        else:
            raise ValueError(f"Unsupported distribution type: {self.distribution_type}")


@dataclass
class RiskScore:
    """Comprehensive risk score with uncertainty."""

    overall_risk: float  # 0-1 scale
    risk_category: RiskCategory
    endpoint_contributions: Dict[str, float]
    exposure_contributions: Dict[str, float]
    uncertainty_range: Tuple[float, float]  # (5th percentile, 95th percentile)
    confidence: ConfidenceLevel
    monte_carlo_samples: Optional[np.ndarray] = None
    key_drivers: List[str] = field(default_factory=list)

    @property
    def uncertainty_factor(self) -> float:
        """Calculate uncertainty factor (95th / 50th percentile)."""
        if self.monte_carlo_samples is not None:
            p50 = np.percentile(self.monte_carlo_samples, 50)
            p95 = np.percentile(self.monte_carlo_samples, 95)
            return p95 / max(p50, 1e-10)
        else:
            return self.uncertainty_range[1] / max(self.overall_risk, 1e-10)


class UncertaintyQuantification:
    """Comprehensive uncertainty quantification system."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.uncertainty_sources = {}
        self.correlation_matrix = None

    def add_uncertainty_source(
        self, source: UncertaintySource, distribution: UncertaintyDistribution
    ):
        """Add uncertainty source to the analysis."""
        self.uncertainty_sources[source] = distribution

    def set_correlation_matrix(self, correlation_matrix: np.ndarray):
        """Set correlation matrix between uncertainty sources."""
        self.correlation_matrix = correlation_matrix

    def propagate_uncertainty(
        self, risk_function: callable, n_samples: int = 10000, **kwargs
    ) -> Dict[str, Any]:
        """Propagate uncertainty through risk calculation using Monte Carlo."""

        # Sample from all uncertainty distributions
        samples = {}
        sample_matrix = np.zeros((n_samples, len(self.uncertainty_sources)))

        for i, (source, distribution) in enumerate(self.uncertainty_sources.items()):
            sample_matrix[:, i] = distribution.sample(n_samples)
            samples[source.value] = sample_matrix[:, i]

        # Apply correlations if specified
        if self.correlation_matrix is not None:
            # Cholesky decomposition for correlated sampling
            try:
                L = np.linalg.cholesky(self.correlation_matrix)
                uncorrelated = np.random.randn(n_samples, len(self.uncertainty_sources))
                correlated = uncorrelated @ L.T

                # Transform to original distributions
                for i, (source, distribution) in enumerate(
                    self.uncertainty_sources.items()
                ):
                    # Convert standard normal to original distribution
                    if distribution.distribution_type == "normal":
                        samples[source.value] = (
                            distribution.mean + distribution.std * correlated[:, i]
                        )
                    elif distribution.distribution_type == "lognormal":
                        samples[source.value] = np.exp(
                            distribution.mean + distribution.std * correlated[:, i]
                        )
                    # Add other distributions as needed

            except np.linalg.LinAlgError:
                print(
                    "Warning: Correlation matrix not positive definite, using independent sampling"
                )

        # Run Monte Carlo simulation
        risk_samples = np.zeros(n_samples)

        for i in range(n_samples):
            sample_params = {key: values[i] for key, values in samples.items()}
            try:
                risk_samples[i] = risk_function(**sample_params, **kwargs)
            except Exception:
                risk_samples[i] = np.nan

        # Remove invalid samples
        valid_samples = risk_samples[~np.isnan(risk_samples)]

        if len(valid_samples) == 0:
            raise ChemiAccelError("All Monte Carlo samples failed")

        # Calculate statistics
        uncertainty_stats = {
            "mean": np.mean(valid_samples),
            "median": np.median(valid_samples),
            "std": np.std(valid_samples),
            "percentiles": {
                "5th": np.percentile(valid_samples, 5),
                "25th": np.percentile(valid_samples, 25),
                "75th": np.percentile(valid_samples, 75),
                "95th": np.percentile(valid_samples, 95),
            },
            "samples": valid_samples,
            "n_valid_samples": len(valid_samples),
            "n_failed_samples": n_samples - len(valid_samples),
        }

        return uncertainty_stats

    def sensitivity_analysis(
        self,
        risk_function: callable,
        base_parameters: Dict[str, float],
        perturbation_fraction: float = 0.1,
        **kwargs,
    ) -> Dict[str, float]:
        """Perform sensitivity analysis using local gradients."""

        base_risk = risk_function(**base_parameters, **kwargs)
        sensitivities = {}

        for param_name, base_value in base_parameters.items():
            # Positive perturbation
            perturbed_params = base_parameters.copy()
            perturbation = abs(base_value) * perturbation_fraction
            if perturbation == 0:
                perturbation = 0.01

            perturbed_params[param_name] = base_value + perturbation
            risk_high = risk_function(**perturbed_params, **kwargs)

            # Negative perturbation
            perturbed_params[param_name] = base_value - perturbation
            risk_low = risk_function(**perturbed_params, **kwargs)

            # Calculate sensitivity (normalized derivative)
            if abs(risk_high - risk_low) > 1e-10:
                sensitivity = (
                    (risk_high - risk_low) / (2 * perturbation) * base_value / base_risk
                )
            else:
                sensitivity = 0.0

            sensitivities[param_name] = abs(sensitivity)

        return sensitivities


class HolisticRiskAssessment:
    """
    Comprehensive holistic risk assessment system.

    Integrates toxicity predictions, environmental fate, exposure scenarios,
    and uncertainty quantification for comprehensive chemical risk assessment.
    """

    def __init__(self, backend: Backend):
        self.backend = backend
        self.uncertainty_quantifier = UncertaintyQuantification(backend)

        # Risk assessment parameters
        self.safety_factors = {
            "interspecies": 10.0,  # Animal to human extrapolation
            "intraspecies": 10.0,  # Human variability
            "subchronic_to_chronic": 2.0,  # Duration extrapolation
            "loael_to_noael": 3.0,  # Effect level extrapolation
            "database_deficiency": 3.0,  # Missing data
        }

        self.risk_thresholds = {
            RiskCategory.NEGLIGIBLE: 0.01,
            RiskCategory.LOW: 0.1,
            RiskCategory.MODERATE: 0.5,
            RiskCategory.HIGH: 0.8,
            RiskCategory.VERY_HIGH: 0.95,
            RiskCategory.UNACCEPTABLE: 1.0,
        }

    def assess_risk(
        self,
        toxicity_data: ToxicityData,
        environmental_fate_data: Optional[Any] = None,
        exposure_scenarios: Optional[List[ExposureScenario]] = None,
        include_uncertainty: bool = True,
        monte_carlo_samples: int = 10000,
        **kwargs,
    ) -> RiskScore:
        """
        Perform comprehensive holistic risk assessment.

        Args:
            toxicity_data: Comprehensive toxicity data
            environmental_fate_data: Environmental fate modeling results
            exposure_scenarios: List of exposure scenarios
            include_uncertainty: Include uncertainty quantification
            monte_carlo_samples: Number of Monte Carlo samples

        Returns:
            Comprehensive risk score with uncertainty
        """

        # Default exposure scenarios if not provided
        if exposure_scenarios is None:
            exposure_scenarios = self._create_default_exposure_scenarios()

        # Calculate deterministic risk
        base_risk = self._calculate_deterministic_risk(
            toxicity_data, environmental_fate_data, exposure_scenarios
        )

        # Calculate uncertainty if requested
        uncertainty_range = (base_risk, base_risk)
        monte_carlo_results = None

        if include_uncertainty:
            uncertainty_results = self._quantify_uncertainty(
                toxicity_data,
                environmental_fate_data,
                exposure_scenarios,
                monte_carlo_samples,
            )
            uncertainty_range = (
                uncertainty_results["percentiles"]["5th"],
                uncertainty_results["percentiles"]["95th"],
            )
            monte_carlo_results = uncertainty_results["samples"]

        # Determine risk category
        risk_category = self._categorize_risk(base_risk)

        # Calculate endpoint contributions
        endpoint_contributions = self._calculate_endpoint_contributions(toxicity_data)

        # Calculate exposure contributions
        exposure_contributions = self._calculate_exposure_contributions(
            exposure_scenarios
        )

        # Identify key risk drivers
        key_drivers = self._identify_key_drivers(
            toxicity_data, exposure_scenarios, endpoint_contributions
        )

        # Determine overall confidence
        confidence = self._assess_confidence(toxicity_data, exposure_scenarios)

        return RiskScore(
            overall_risk=base_risk,
            risk_category=risk_category,
            endpoint_contributions=endpoint_contributions,
            exposure_contributions=exposure_contributions,
            uncertainty_range=uncertainty_range,
            confidence=confidence,
            monte_carlo_samples=monte_carlo_results,
            key_drivers=key_drivers,
        )

    def _create_default_exposure_scenarios(self) -> List[ExposureScenario]:
        """Create default exposure scenarios."""
        return [
            # Consumer exposure
            ExposureScenario(
                scenario_name="consumer_oral",
                exposure_route=ExposureRoute.ORAL,
                exposure_level=0.01,  # mg/kg/day
                exposure_duration=365,  # 1 year
                exposure_frequency=1.0,
                population="adults",
            ),
            ExposureScenario(
                scenario_name="consumer_dermal",
                exposure_route=ExposureRoute.DERMAL,
                exposure_level=0.1,  # mg/kg/day
                exposure_duration=365,
                exposure_frequency=1.0,
                population="adults",
            ),
            # Occupational exposure
            ExposureScenario(
                scenario_name="occupational_inhalation",
                exposure_route=ExposureRoute.INHALATION,
                exposure_level=10.0,  # mg/m3
                exposure_duration=250 * 40,  # 40 years working
                exposure_frequency=8 / 24,  # 8 hours per day
                population="workers",
                variability_factor=5.0,
            ),
            # Environmental exposure
            ExposureScenario(
                scenario_name="environmental_general",
                exposure_route=ExposureRoute.ENVIRONMENTAL,
                exposure_level=0.001,  # mg/kg/day
                exposure_duration=365 * 70,  # lifetime
                exposure_frequency=1.0,
                population="general",
            ),
        ]

    def _calculate_deterministic_risk(
        self,
        toxicity_data: ToxicityData,
        environmental_fate_data: Optional[Any],
        exposure_scenarios: List[ExposureScenario],
    ) -> float:
        """Calculate deterministic risk score."""

        total_risk = 0.0
        scenario_weights = {
            ExposureRoute.ORAL: 1.0,
            ExposureRoute.DERMAL: 0.8,
            ExposureRoute.INHALATION: 1.2,
            ExposureRoute.ENVIRONMENTAL: 0.6,
            ExposureRoute.OCCUPATIONAL: 1.5,
        }

        total_weight = 0.0

        for scenario in exposure_scenarios:
            # Calculate hazard quotient for this scenario
            hazard_quotient = self._calculate_hazard_quotient(
                toxicity_data, scenario, environmental_fate_data
            )

            weight = scenario_weights.get(scenario.exposure_route, 1.0)
            total_risk += hazard_quotient * weight
            total_weight += weight

        # Normalize and apply overall safety factors
        if total_weight > 0:
            avg_risk = total_risk / total_weight
            # Apply composite safety factor
            composite_sf = np.prod(list(self.safety_factors.values()))
            adjusted_risk = avg_risk * composite_sf
        else:
            adjusted_risk = 0.0

        # Convert to 0-1 risk scale
        risk_score = min(adjusted_risk / 100.0, 1.0)

        return risk_score

    def _calculate_hazard_quotient(
        self,
        toxicity_data: ToxicityData,
        scenario: ExposureScenario,
        environmental_fate_data: Optional[Any],
    ) -> float:
        """Calculate hazard quotient for exposure scenario."""

        # Get relevant toxicity endpoints for exposure route
        relevant_endpoints = self._get_relevant_endpoints(scenario.exposure_route)

        if not relevant_endpoints:
            return 0.0

        # Find most relevant toxicity value
        reference_dose = None
        for endpoint_type in relevant_endpoints:
            endpoint_predictions = toxicity_data.get_endpoint_predictions(endpoint_type)
            if endpoint_predictions:
                # Use most conservative (lowest) prediction
                min_prediction = min(
                    endpoint_predictions, key=lambda x: x.predicted_value
                )
                reference_dose = min_prediction.predicted_value
                break

        if reference_dose is None:
            return 0.0

        # Calculate exposure dose
        exposure_dose = scenario.exposure_level

        # Apply environmental fate adjustment if available
        if environmental_fate_data:
            exposure_dose *= self._get_environmental_adjustment_factor(
                environmental_fate_data
            )

        # Calculate hazard quotient
        hazard_quotient = exposure_dose / max(reference_dose, 1e-10)

        return hazard_quotient

    def _get_relevant_endpoints(self, exposure_route: ExposureRoute) -> List:
        """Get toxicity endpoints relevant to exposure route."""
        from .core import ToxicityEndpointType

        endpoint_mapping = {
            ExposureRoute.ORAL: [
                ToxicityEndpointType.ACUTE_ORAL,
                ToxicityEndpointType.CHRONIC_ORAL,
            ],
            ExposureRoute.DERMAL: [
                ToxicityEndpointType.ACUTE_DERMAL,
                ToxicityEndpointType.SKIN_IRRITATION,
                ToxicityEndpointType.SKIN_SENSITIZATION,
            ],
            ExposureRoute.INHALATION: [ToxicityEndpointType.ACUTE_INHALATION],
            ExposureRoute.ENVIRONMENTAL: [
                ToxicityEndpointType.AQUATIC_ACUTE,
                ToxicityEndpointType.AQUATIC_CHRONIC,
                ToxicityEndpointType.TERRESTRIAL_TOXICITY,
            ],
            ExposureRoute.OCCUPATIONAL: [
                ToxicityEndpointType.ACUTE_ORAL,
                ToxicityEndpointType.ACUTE_INHALATION,
                ToxicityEndpointType.ACUTE_DERMAL,
            ],
        }

        return endpoint_mapping.get(exposure_route, [])

    def _get_environmental_adjustment_factor(
        self, environmental_fate_data: Any
    ) -> float:
        """Calculate environmental adjustment factor from fate data."""
        # Simplified implementation - would use actual fate modeling results
        # Factors like persistence, bioaccumulation, transport

        persistence_factor = 1.0  # Default no adjustment
        bioaccumulation_factor = 1.0

        # This would be based on actual environmental fate data structure
        if hasattr(environmental_fate_data, "persistence_score"):
            if environmental_fate_data.persistence_score > 0.8:
                persistence_factor = 2.0  # Highly persistent
            elif environmental_fate_data.persistence_score > 0.5:
                persistence_factor = 1.5

        if hasattr(environmental_fate_data, "bioaccumulation_score"):
            if environmental_fate_data.bioaccumulation_score > 0.8:
                bioaccumulation_factor = 3.0  # High bioaccumulation
            elif environmental_fate_data.bioaccumulation_score > 0.5:
                bioaccumulation_factor = 1.5

        return persistence_factor * bioaccumulation_factor

    def _quantify_uncertainty(
        self,
        toxicity_data: ToxicityData,
        environmental_fate_data: Optional[Any],
        exposure_scenarios: List[ExposureScenario],
        n_samples: int,
    ) -> Dict[str, Any]:
        """Quantify uncertainty using Monte Carlo analysis."""

        # Setup uncertainty distributions
        self.uncertainty_quantifier.add_uncertainty_source(
            UncertaintySource.MODEL_UNCERTAINTY,
            UncertaintyDistribution(
                parameter_name="model_uncertainty",
                distribution_type="lognormal",
                mean=0.0,  # log scale
                std=0.5,  # Factor of ~1.6 uncertainty
            ),
        )

        self.uncertainty_quantifier.add_uncertainty_source(
            UncertaintySource.EXPOSURE_VARIABILITY,
            UncertaintyDistribution(
                parameter_name="exposure_variability",
                distribution_type="lognormal",
                mean=0.0,
                std=0.7,  # Factor of ~2 uncertainty
            ),
        )

        self.uncertainty_quantifier.add_uncertainty_source(
            UncertaintySource.SPECIES_EXTRAPOLATION,
            UncertaintyDistribution(
                parameter_name="species_extrapolation",
                distribution_type="lognormal",
                mean=0.0,
                std=0.3,  # Factor of ~1.3 uncertainty
            ),
        )

        # Define risk function for Monte Carlo
        def risk_with_uncertainty(**uncertainty_params):
            # Apply uncertainty factors to base calculation
            base_risk = self._calculate_deterministic_risk(
                toxicity_data, environmental_fate_data, exposure_scenarios
            )

            # Apply uncertainty multipliers
            total_uncertainty = 1.0
            for param_value in uncertainty_params.values():
                total_uncertainty *= param_value

            return base_risk * total_uncertainty

        # Run uncertainty propagation
        return self.uncertainty_quantifier.propagate_uncertainty(
            risk_with_uncertainty, n_samples
        )

    def _categorize_risk(self, risk_score: float) -> RiskCategory:
        """Categorize risk score into risk category."""
        for category, threshold in self.risk_thresholds.items():
            if risk_score <= threshold:
                return category
        return RiskCategory.UNACCEPTABLE

    def _calculate_endpoint_contributions(
        self, toxicity_data: ToxicityData
    ) -> Dict[str, float]:
        """Calculate contribution of each endpoint to overall risk."""
        contributions = {}

        for prediction in toxicity_data.predicted_endpoints:
            endpoint_name = prediction.endpoint.endpoint_type.value
            # Weight by toxicity and confidence
            confidence_weight = self._confidence_to_weight(prediction.confidence)
            risk_contribution = prediction.predicted_value * confidence_weight

            if endpoint_name in contributions:
                contributions[endpoint_name] += risk_contribution
            else:
                contributions[endpoint_name] = risk_contribution

        # Normalize contributions
        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}

        return contributions

    def _calculate_exposure_contributions(
        self, exposure_scenarios: List[ExposureScenario]
    ) -> Dict[str, float]:
        """Calculate contribution of each exposure scenario."""
        contributions = {}
        total_exposure = 0.0

        for scenario in exposure_scenarios:
            exposure_magnitude = scenario.total_dose
            contributions[scenario.scenario_name] = exposure_magnitude
            total_exposure += exposure_magnitude

        # Normalize
        if total_exposure > 0:
            contributions = {k: v / total_exposure for k, v in contributions.items()}

        return contributions

    def _identify_key_drivers(
        self,
        toxicity_data: ToxicityData,
        exposure_scenarios: List[ExposureScenario],
        endpoint_contributions: Dict[str, float],
    ) -> List[str]:
        """Identify key risk drivers."""
        drivers = []

        # Top toxic endpoints
        top_endpoints = sorted(
            endpoint_contributions.items(), key=lambda x: x[1], reverse=True
        )[:3]

        for endpoint, contribution in top_endpoints:
            if contribution > 0.1:  # >10% contribution
                drivers.append(f"High {endpoint} toxicity")

        # High exposure scenarios
        for scenario in exposure_scenarios:
            if scenario.total_dose > 1.0:  # High exposure
                drivers.append(f"High {scenario.scenario_name} exposure")

        # Uncertainty sources
        if any(pred.uncertainty > 0.5 for pred in toxicity_data.predicted_endpoints):
            drivers.append("High model uncertainty")

        return drivers

    def _assess_confidence(
        self, toxicity_data: ToxicityData, exposure_scenarios: List[ExposureScenario]
    ) -> ConfidenceLevel:
        """Assess overall confidence in risk assessment."""

        # Average confidence from toxicity predictions
        if toxicity_data.predicted_endpoints:
            confidence_scores = [
                self._confidence_to_score(pred.confidence)
                for pred in toxicity_data.predicted_endpoints
            ]
            avg_tox_confidence = np.mean(confidence_scores)
        else:
            avg_tox_confidence = 0.3  # Low confidence without data

        # Average confidence from exposure scenarios
        exposure_confidence_scores = [
            self._confidence_to_score(scenario.confidence)
            for scenario in exposure_scenarios
        ]
        avg_exp_confidence = np.mean(exposure_confidence_scores)

        # Overall confidence (geometric mean)
        overall_confidence_score = np.sqrt(avg_tox_confidence * avg_exp_confidence)

        return self._score_to_confidence(overall_confidence_score)

    def _confidence_to_weight(self, confidence: ConfidenceLevel) -> float:
        """Convert confidence level to weight factor."""
        mapping = {
            ConfidenceLevel.VERY_LOW: 0.2,
            ConfidenceLevel.LOW: 0.4,
            ConfidenceLevel.MODERATE: 0.6,
            ConfidenceLevel.HIGH: 0.8,
            ConfidenceLevel.VERY_HIGH: 1.0,
        }
        return mapping.get(confidence, 0.5)

    def _confidence_to_score(self, confidence: ConfidenceLevel) -> float:
        """Convert confidence level to numeric score."""
        mapping = {
            ConfidenceLevel.VERY_LOW: 0.1,
            ConfidenceLevel.LOW: 0.3,
            ConfidenceLevel.MODERATE: 0.5,
            ConfidenceLevel.HIGH: 0.7,
            ConfidenceLevel.VERY_HIGH: 0.9,
        }
        return mapping.get(confidence, 0.5)

    def _score_to_confidence(self, score: float) -> ConfidenceLevel:
        """Convert numeric score to confidence level."""
        if score >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.6:
            return ConfidenceLevel.HIGH
        elif score >= 0.4:
            return ConfidenceLevel.MODERATE
        elif score >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def compare_scenarios(
        self,
        toxicity_data_list: List[ToxicityData],
        scenario_names: List[str],
        exposure_scenarios: Optional[List[ExposureScenario]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Compare risk across multiple scenarios or compounds."""

        results = []

        for i, toxicity_data in enumerate(toxicity_data_list):
            scenario_name = (
                scenario_names[i] if i < len(scenario_names) else f"Scenario_{i+1}"
            )

            risk_score = self.assess_risk(
                toxicity_data, exposure_scenarios=exposure_scenarios, **kwargs
            )

            results.append(
                {
                    "scenario": scenario_name,
                    "compound": toxicity_data.smiles,
                    "overall_risk": risk_score.overall_risk,
                    "risk_category": risk_score.risk_category.value,
                    "uncertainty_low": risk_score.uncertainty_range[0],
                    "uncertainty_high": risk_score.uncertainty_range[1],
                    "confidence": risk_score.confidence.value,
                    "key_drivers": "; ".join(risk_score.key_drivers),
                }
            )

        return pd.DataFrame(results)

    def generate_risk_report(
        self,
        risk_score: RiskScore,
        toxicity_data: ToxicityData,
        include_uncertainty_analysis: bool = True,
    ) -> Dict[str, Any]:
        """Generate comprehensive risk assessment report."""

        report = {
            "executive_summary": {
                "compound": toxicity_data.smiles,
                "overall_risk_score": risk_score.overall_risk,
                "risk_category": risk_score.risk_category.value,
                "confidence_level": risk_score.confidence.value,
                "key_concerns": risk_score.key_drivers[:3],
            },
            "risk_breakdown": {
                "endpoint_contributions": risk_score.endpoint_contributions,
                "exposure_contributions": risk_score.exposure_contributions,
                "top_risk_endpoints": [
                    endpoint.endpoint.endpoint_type.value
                    for endpoint in toxicity_data.get_highest_risk_endpoints(3)
                ],
            },
            "uncertainty_analysis": {
                "uncertainty_range": risk_score.uncertainty_range,
                "uncertainty_factor": risk_score.uncertainty_factor,
                "confidence_assessment": risk_score.confidence.value,
            }
            if include_uncertainty_analysis
            else None,
            "recommendations": self._generate_recommendations(
                risk_score, toxicity_data
            ),
        }

        return report

    def _generate_recommendations(
        self, risk_score: RiskScore, toxicity_data: ToxicityData
    ) -> List[str]:
        """Generate risk management recommendations."""

        recommendations = []

        if risk_score.risk_category in [
            RiskCategory.HIGH,
            RiskCategory.VERY_HIGH,
            RiskCategory.UNACCEPTABLE,
        ]:
            recommendations.append(
                "Consider alternatives or significant risk mitigation measures"
            )
            recommendations.append("Implement strict exposure controls")

        if risk_score.confidence in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW]:
            recommendations.append(
                "Generate additional toxicity data to reduce uncertainty"
            )

        if "High model uncertainty" in risk_score.key_drivers:
            recommendations.append("Validate predictions with experimental data")

        # Endpoint-specific recommendations
        high_risk_endpoints = toxicity_data.get_highest_risk_endpoints(2)
        for endpoint in high_risk_endpoints:
            endpoint_type = endpoint.endpoint.endpoint_type.value
            if "carcinogen" in endpoint_type:
                recommendations.append("Implement cancer risk management protocols")
            elif "reproductive" in endpoint_type:
                recommendations.append("Protect reproductive-age populations")
            elif "aquatic" in endpoint_type:
                recommendations.append(
                    "Implement environmental monitoring and controls"
                )

        if not recommendations:
            recommendations.append("Continue standard risk management practices")
            recommendations.append("Monitor for new toxicity information")

        return recommendations
