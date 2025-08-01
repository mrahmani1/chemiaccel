"""
Integration with Predictive Toxicology & Lifecycle

This module integrates bio-interface design with predictive toxicology
and material lifecycle analysis to assess potential adverse biological
reactions and model material degradation in biological environments.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import cudf
    import cupy as cp

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from ..core.exceptions import ChemiAccelError
from ..sustainability.lifecycle_visualization import LifecycleAnalyzer
from ..toxicology.core import ToxicologyPredictor

logger = logging.getLogger(__name__)


class BiologicalEnvironment(Enum):
    """Types of biological environments for material exposure."""

    BLOOD = "blood"
    INTERSTITIAL_FLUID = "interstitial_fluid"
    INTRACELLULAR = "intracellular"
    CEREBROSPINAL_FLUID = "cerebrospinal_fluid"
    SYNOVIAL_FLUID = "synovial_fluid"
    GASTRIC_FLUID = "gastric_fluid"
    INTESTINAL_FLUID = "intestinal_fluid"
    PULMONARY_SURFACTANT = "pulmonary_surfactant"
    SALIVA = "saliva"
    TEARS = "tears"


class ToxicityEndpoint(Enum):
    """Toxicological endpoints to evaluate."""

    CYTOTOXICITY = "cytotoxicity"
    GENOTOXICITY = "genotoxicity"
    IMMUNOTOXICITY = "immunotoxicity"
    HEMATOTOXICITY = "hematotoxicity"
    HEPATOTOXICITY = "hepatotoxicity"
    NEPHROTOXICITY = "nephrotoxicity"
    NEUROTOXICITY = "neurotoxicity"
    CARDIOTOXICITY = "cardiotoxicity"
    REPRODUCTIVE_TOXICITY = "reproductive_toxicity"
    CARCINOGENICITY = "carcinogenicity"


@dataclass
class BiologicalExposureScenario:
    """Biological exposure scenario definition."""

    environment: BiologicalEnvironment
    exposure_duration: float  # hours
    material_dose: float  # mg/kg or mg/L
    exposure_route: str  # "implant", "injection", "inhalation", "dermal"
    patient_demographics: Dict[str, Any]
    physiological_conditions: Dict[str, float]


@dataclass
class ToxicityAssessment:
    """Comprehensive toxicity assessment results."""

    material_id: str
    exposure_scenario: BiologicalExposureScenario
    endpoint_predictions: Dict[ToxicityEndpoint, float]
    confidence_scores: Dict[ToxicityEndpoint, float]
    safety_margins: Dict[ToxicityEndpoint, float]
    adverse_outcome_pathways: List[Dict[str, Any]]
    dose_response_curves: Dict[ToxicityEndpoint, List[Tuple[float, float]]]
    no_observed_adverse_effect_level: Dict[ToxicityEndpoint, float]
    benchmark_dose: Dict[ToxicityEndpoint, float]
    risk_characterization: Dict[str, float]


@dataclass
class BiodegradationProfile:
    """Material biodegradation profile in biological environments."""

    material_id: str
    environment: BiologicalEnvironment
    degradation_kinetics: Dict[str, float]  # rate constants
    degradation_products: List[Dict[str, Any]]
    product_toxicity: Dict[str, float]
    mass_loss_profile: List[Tuple[float, float]]  # time, mass remaining
    mechanical_property_changes: Dict[str, List[Tuple[float, float]]]
    surface_property_evolution: Dict[str, List[Tuple[float, float]]]
    biocompatibility_over_time: List[Tuple[float, float]]


@dataclass
class IntegratedRiskAssessment:
    """Integrated risk assessment combining toxicology and lifecycle."""

    material_id: str
    exposure_scenarios: List[BiologicalExposureScenario]
    toxicity_assessments: List[ToxicityAssessment]
    biodegradation_profiles: List[BiodegradationProfile]
    cumulative_risk_score: float
    critical_exposure_pathways: List[str]
    risk_mitigation_strategies: List[str]
    regulatory_compliance: Dict[str, bool]
    recommended_safety_testing: List[str]


class ToxicologyLifecycleIntegrator:
    """
    Integrates bio-interface design with predictive toxicology and
    material lifecycle analysis for comprehensive safety assessment.
    """

    def __init__(
        self,
        toxicology_predictor: Optional[ToxicologyPredictor] = None,
        lifecycle_analyzer: Optional[LifecycleAnalyzer] = None,
        backend: str = "auto",
        use_gpu: bool = True,
    ):
        """
        Initialize toxicology-lifecycle integrator.

        Args:
            toxicology_predictor: Toxicology prediction engine
            lifecycle_analyzer: Lifecycle analysis engine
            backend: Computing backend ('cuda', 'jax', 'cpu', 'auto')
            use_gpu: Whether to use GPU acceleration
        """
        self.backend = self._select_backend(backend, use_gpu)

        # Initialize prediction engines
        self.toxicology_predictor = toxicology_predictor or ToxicologyPredictor(
            backend=self.backend
        )
        self.lifecycle_analyzer = lifecycle_analyzer or LifecycleAnalyzer(
            backend=self.backend
        )

        # Initialize biological environment models
        self._initialize_biological_environments()
        self._initialize_toxicity_models()
        self._initialize_degradation_models()

        logger.info(
            f"Initialized ToxicologyLifecycleIntegrator with {self.backend} backend"
        )

    def _select_backend(self, backend: str, use_gpu: bool) -> str:
        """Select appropriate backend."""
        if backend == "auto":
            if CUDA_AVAILABLE and use_gpu:
                return "cuda"
            elif JAX_AVAILABLE:
                return "jax"
            else:
                return "cpu"
        return backend

    def _initialize_biological_environments(self) -> None:
        """Initialize biological environment models."""
        self.biological_environments = {
            BiologicalEnvironment.BLOOD: {
                "ph": 7.4,
                "ionic_strength": 0.16,  # M
                "protein_concentration": 70,  # g/L
                "glucose": 5.5,  # mM
                "oxygen_concentration": 0.1,  # mM
                "temperature": 37.0,  # °C
                "flow_rate": 5.0,  # L/min (cardiac output)
                "major_proteins": ["albumin", "immunoglobulin", "fibrinogen"],
                "enzymes": ["plasmin", "thrombin", "complement"],
            },
            BiologicalEnvironment.INTERSTITIAL_FLUID: {
                "ph": 7.35,
                "ionic_strength": 0.15,
                "protein_concentration": 20,  # g/L
                "glucose": 4.5,  # mM
                "oxygen_concentration": 0.05,  # mM
                "temperature": 37.0,
                "flow_rate": 0.002,  # L/min (lymphatic flow)
                "major_proteins": ["albumin", "immunoglobulin"],
                "enzymes": ["hyaluronidase", "collagenase"],
            },
            BiologicalEnvironment.INTRACELLULAR: {
                "ph": 7.0,
                "ionic_strength": 0.14,
                "protein_concentration": 200,  # g/L
                "glucose": 1.0,  # mM
                "oxygen_concentration": 0.01,  # mM
                "temperature": 37.0,
                "flow_rate": 0.0,
                "major_proteins": ["actin", "myosin", "tubulin"],
                "enzymes": ["proteasome", "lysosomal_enzymes", "oxidases"],
            },
            BiologicalEnvironment.SYNOVIAL_FLUID: {
                "ph": 7.4,
                "ionic_strength": 0.15,
                "protein_concentration": 30,  # g/L
                "hyaluronic_acid": 2.5,  # g/L
                "temperature": 37.0,
                "flow_rate": 0.0001,
                "major_proteins": ["albumin", "hyaluronic_acid"],
                "enzymes": ["hyaluronidase", "matrix_metalloproteinases"],
            },
            BiologicalEnvironment.GASTRIC_FLUID: {
                "ph": 1.5,
                "ionic_strength": 0.10,
                "protein_concentration": 3,  # g/L
                "pepsin": 0.5,  # g/L
                "temperature": 37.0,
                "flow_rate": 0.05,  # L/min
                "major_proteins": ["pepsin", "mucin"],
                "enzymes": ["pepsin", "gastric_lipase"],
            },
        }

    def _initialize_toxicity_models(self) -> None:
        """Initialize toxicity prediction models."""
        self.toxicity_models = {
            ToxicityEndpoint.CYTOTOXICITY: self._cytotoxicity_model,
            ToxicityEndpoint.GENOTOXICITY: self._genotoxicity_model,
            ToxicityEndpoint.IMMUNOTOXICITY: self._immunotoxicity_model,
            ToxicityEndpoint.HEMATOTOXICITY: self._hematotoxicity_model,
            ToxicityEndpoint.HEPATOTOXICITY: self._hepatotoxicity_model,
            ToxicityEndpoint.NEPHROTOXICITY: self._nephrotoxicity_model,
            ToxicityEndpoint.NEUROTOXICITY: self._neurotoxicity_model,
            ToxicityEndpoint.CARDIOTOXICITY: self._cardiotoxicity_model,
            ToxicityEndpoint.REPRODUCTIVE_TOXICITY: self._reproductive_toxicity_model,
            ToxicityEndpoint.CARCINOGENICITY: self._carcinogenicity_model,
        }

    def _initialize_degradation_models(self) -> None:
        """Initialize biodegradation models."""
        self.degradation_models = {
            "hydrolysis": self._hydrolysis_model,
            "enzymatic": self._enzymatic_degradation_model,
            "oxidative": self._oxidative_degradation_model,
            "mechanical_wear": self._mechanical_wear_model,
            "corrosion": self._corrosion_model,
        }

    def assess_integrated_biocompatibility(
        self,
        material_composition: Dict[str, float],
        surface_properties: Dict[str, float],
        exposure_scenarios: List[BiologicalExposureScenario],
        assessment_duration: float = 168.0,  # hours (1 week)
    ) -> IntegratedRiskAssessment:
        """
        Perform integrated biocompatibility assessment.

        Args:
            material_composition: Chemical composition of material
            surface_properties: Physical/chemical surface properties
            exposure_scenarios: Biological exposure scenarios to evaluate
            assessment_duration: Duration for assessment

        Returns:
            IntegratedRiskAssessment with comprehensive evaluation
        """
        logger.info("Starting integrated biocompatibility assessment")

        material_id = f"material_{hash(str(material_composition))}"

        # Perform toxicity assessments for each scenario
        toxicity_assessments = []
        for scenario in exposure_scenarios:
            toxicity_assessment = self._assess_scenario_toxicity(
                material_composition, surface_properties, scenario
            )
            toxicity_assessments.append(toxicity_assessment)

        # Perform biodegradation analysis for each environment
        biodegradation_profiles = []
        unique_environments = set(
            scenario.environment for scenario in exposure_scenarios
        )

        for environment in unique_environments:
            biodegradation_profile = self._assess_biodegradation(
                material_composition,
                surface_properties,
                environment,
                assessment_duration,
            )
            biodegradation_profiles.append(biodegradation_profile)

        # Calculate cumulative risk
        cumulative_risk = self._calculate_cumulative_risk(
            toxicity_assessments, biodegradation_profiles
        )

        # Identify critical pathways
        critical_pathways = self._identify_critical_pathways(
            toxicity_assessments, biodegradation_profiles
        )

        # Generate risk mitigation strategies
        mitigation_strategies = self._generate_mitigation_strategies(
            toxicity_assessments, biodegradation_profiles
        )

        # Check regulatory compliance
        regulatory_compliance = self._check_regulatory_compliance(
            toxicity_assessments, biodegradation_profiles
        )

        # Recommend safety testing
        safety_testing = self._recommend_safety_testing(
            toxicity_assessments, biodegradation_profiles
        )

        result = IntegratedRiskAssessment(
            material_id=material_id,
            exposure_scenarios=exposure_scenarios,
            toxicity_assessments=toxicity_assessments,
            biodegradation_profiles=biodegradation_profiles,
            cumulative_risk_score=cumulative_risk,
            critical_exposure_pathways=critical_pathways,
            risk_mitigation_strategies=mitigation_strategies,
            regulatory_compliance=regulatory_compliance,
            recommended_safety_testing=safety_testing,
        )

        logger.info("Completed integrated biocompatibility assessment")
        return result

    def _assess_scenario_toxicity(
        self,
        material_composition: Dict[str, float],
        surface_properties: Dict[str, float],
        scenario: BiologicalExposureScenario,
    ) -> ToxicityAssessment:
        """Assess toxicity for a specific exposure scenario."""

        # Get biological environment parameters
        env_params = self.biological_environments[scenario.environment]

        # Predict toxicity for each endpoint
        endpoint_predictions = {}
        confidence_scores = {}
        safety_margins = {}
        dose_response_curves = {}
        noael = {}
        benchmark_doses = {}

        for endpoint in ToxicityEndpoint:
            # Predict toxicity using material properties and exposure conditions
            toxicity_score = self.toxicity_models[endpoint](
                material_composition, surface_properties, scenario, env_params
            )

            endpoint_predictions[endpoint] = toxicity_score
            confidence_scores[endpoint] = self._calculate_toxicity_confidence(
                material_composition, endpoint
            )
            safety_margins[endpoint] = self._calculate_safety_margin(
                toxicity_score, scenario.material_dose
            )

            # Generate dose-response curve
            dose_response_curves[endpoint] = self._generate_dose_response_curve(
                material_composition, surface_properties, scenario, endpoint
            )

            # Calculate NOAEL and benchmark dose
            noael[endpoint] = self._calculate_noael(dose_response_curves[endpoint])
            benchmark_doses[endpoint] = self._calculate_benchmark_dose(
                dose_response_curves[endpoint]
            )

        # Identify adverse outcome pathways
        adverse_pathways = self._identify_adverse_outcome_pathways(
            material_composition, endpoint_predictions
        )

        # Perform risk characterization
        risk_characterization = self._perform_risk_characterization(
            endpoint_predictions, safety_margins, scenario
        )

        return ToxicityAssessment(
            material_id=f"material_{hash(str(material_composition))}",
            exposure_scenario=scenario,
            endpoint_predictions=endpoint_predictions,
            confidence_scores=confidence_scores,
            safety_margins=safety_margins,
            adverse_outcome_pathways=adverse_pathways,
            dose_response_curves=dose_response_curves,
            no_observed_adverse_effect_level=noael,
            benchmark_dose=benchmark_doses,
            risk_characterization=risk_characterization,
        )

    def _assess_biodegradation(
        self,
        material_composition: Dict[str, float],
        surface_properties: Dict[str, float],
        environment: BiologicalEnvironment,
        duration: float,
    ) -> BiodegradationProfile:
        """Assess material biodegradation in biological environment."""

        env_params = self.biological_environments[environment]

        # Calculate degradation kinetics for different mechanisms
        degradation_kinetics = {}
        for mechanism in self.degradation_models:
            rate_constant = self.degradation_models[mechanism](
                material_composition, surface_properties, env_params
            )
            degradation_kinetics[mechanism] = rate_constant

        # Predict degradation products
        degradation_products = self._predict_degradation_products(
            material_composition, degradation_kinetics, env_params
        )

        # Assess toxicity of degradation products
        product_toxicity = {}
        for i, product in enumerate(degradation_products):
            toxicity = self._assess_product_toxicity(product, environment)
            product_toxicity[f"product_{i}"] = toxicity

        # Generate mass loss profile
        time_points = np.linspace(0, duration, 100)
        mass_loss_profile = []

        for t in time_points:
            remaining_mass = self._calculate_remaining_mass(
                t, degradation_kinetics, material_composition
            )
            mass_loss_profile.append((float(t), remaining_mass))

        # Predict mechanical property changes
        mechanical_changes = self._predict_mechanical_property_changes(
            time_points, degradation_kinetics, surface_properties
        )

        # Predict surface property evolution
        surface_evolution = self._predict_surface_property_evolution(
            time_points, degradation_kinetics, surface_properties
        )

        # Assess biocompatibility over time
        biocompatibility_profile = []
        for t in time_points:
            biocompat_score = self._calculate_time_dependent_biocompatibility(
                t, degradation_kinetics, product_toxicity
            )
            biocompatibility_profile.append((float(t), biocompat_score))

        return BiodegradationProfile(
            material_id=f"material_{hash(str(material_composition))}",
            environment=environment,
            degradation_kinetics=degradation_kinetics,
            degradation_products=degradation_products,
            product_toxicity=product_toxicity,
            mass_loss_profile=mass_loss_profile,
            mechanical_property_changes=mechanical_changes,
            surface_property_evolution=surface_evolution,
            biocompatibility_over_time=biocompatibility_profile,
        )

    # Toxicity model implementations (simplified)
    def _cytotoxicity_model(
        self,
        composition: Dict[str, float],
        surface_props: Dict[str, float],
        scenario: BiologicalExposureScenario,
        env_params: Dict[str, Any],
    ) -> float:
        """Predict cytotoxicity."""
        cytotoxicity = 0.1  # Base cytotoxicity

        # Material-specific contributions
        toxic_elements = {"lead": 0.9, "mercury": 0.95, "cadmium": 0.8, "chromium": 0.6}
        for element, fraction in composition.items():
            if element in toxic_elements:
                cytotoxicity += toxic_elements[element] * fraction

        # Dose-response relationship
        dose_factor = min(1.0, scenario.material_dose / 100.0)  # mg/kg
        cytotoxicity *= dose_factor

        # Surface area effect
        surface_area = surface_props.get("specific_surface_area", 1.0)  # m²/g
        cytotoxicity *= 1 + 0.1 * np.log10(surface_area)

        return np.clip(cytotoxicity, 0, 1)

    def _immunotoxicity_model(
        self,
        composition: Dict[str, float],
        surface_props: Dict[str, float],
        scenario: BiologicalExposureScenario,
        env_params: Dict[str, Any],
    ) -> float:
        """Predict immunotoxicity."""
        immunotoxicity = 0.05  # Base immunotoxicity

        # Material factors
        immunogenic_materials = {
            "nickel": 0.7,
            "cobalt": 0.6,
            "beryllium": 0.8,
            "silica": 0.4,
        }

        for material, fraction in composition.items():
            if material in immunogenic_materials:
                immunotoxicity += immunogenic_materials[material] * fraction

        # Surface charge effect (charged surfaces more immunogenic)
        zeta_potential = abs(surface_props.get("zeta_potential", 0))
        immunotoxicity += 0.2 * min(1.0, zeta_potential / 50)

        return np.clip(immunotoxicity, 0, 1)

    def _genotoxicity_model(
        self,
        composition: Dict[str, float],
        surface_props: Dict[str, float],
        scenario: BiologicalExposureScenario,
        env_params: Dict[str, Any],
    ) -> float:
        """Predict genotoxicity."""
        genotoxicity = 0.02  # Base genotoxicity

        # Genotoxic materials
        genotoxic_materials = {
            "asbestos": 0.9,
            "silica": 0.5,
            "chromium": 0.6,
            "nickel": 0.4,
        }

        for material, fraction in composition.items():
            if material in genotoxic_materials:
                genotoxicity += genotoxic_materials[material] * fraction

        # Particle size effect (smaller particles more genotoxic)
        particle_size = surface_props.get("particle_size", 1000)  # nm
        if particle_size < 100:  # Nanoparticles
            genotoxicity *= 2.0

        return np.clip(genotoxicity, 0, 1)

    # Simplified implementations for other toxicity endpoints
    def _hematotoxicity_model(self, composition, surface_props, scenario, env_params):
        return min(
            0.5,
            sum(
                composition.get(element, 0) * 0.3
                for element in ["lead", "benzene", "toluene"]
            ),
        )

    def _hepatotoxicity_model(self, composition, surface_props, scenario, env_params):
        return min(
            0.6,
            sum(
                composition.get(element, 0) * 0.4
                for element in ["carbon_tetrachloride", "acetaminophen"]
            ),
        )

    def _nephrotoxicity_model(self, composition, surface_props, scenario, env_params):
        return min(
            0.4,
            sum(
                composition.get(element, 0) * 0.5
                for element in ["mercury", "cadmium", "lead"]
            ),
        )

    def _neurotoxicity_model(self, composition, surface_props, scenario, env_params):
        return min(
            0.7,
            sum(
                composition.get(element, 0) * 0.6
                for element in ["lead", "mercury", "aluminum"]
            ),
        )

    def _cardiotoxicity_model(self, composition, surface_props, scenario, env_params):
        return min(
            0.3,
            sum(composition.get(element, 0) * 0.2 for element in ["cobalt", "arsenic"]),
        )

    def _reproductive_toxicity_model(
        self, composition, surface_props, scenario, env_params
    ):
        return min(
            0.5,
            sum(
                composition.get(element, 0) * 0.4
                for element in ["lead", "mercury", "cadmium"]
            ),
        )

    def _carcinogenicity_model(self, composition, surface_props, scenario, env_params):
        carcinogenic_materials = {"asbestos": 0.95, "benzene": 0.8, "formaldehyde": 0.6}
        return min(
            0.9,
            sum(
                composition.get(material, 0) * potency
                for material, potency in carcinogenic_materials.items()
            ),
        )

    # Biodegradation model implementations
    def _hydrolysis_model(
        self,
        composition: Dict[str, float],
        surface_props: Dict[str, float],
        env_params: Dict[str, Any],
    ) -> float:
        """Model hydrolytic degradation."""
        base_rate = 1e-6  # s⁻¹

        # pH effect
        ph = env_params.get("ph", 7.4)
        if ph < 7:  # Acid catalysis
            ph_factor = 10 ** (7 - ph)
        else:  # Base catalysis
            ph_factor = 10 ** (ph - 7)

        # Material susceptibility to hydrolysis
        hydrolyzable_bonds = {
            "polyester": 0.8,
            "polyamide": 0.6,
            "polyanhydride": 0.9,
            "polyurethane": 0.4,
        }

        susceptibility = 0.1  # Base susceptibility
        for material, fraction in composition.items():
            if material in hydrolyzable_bonds:
                susceptibility += hydrolyzable_bonds[material] * fraction

        return base_rate * ph_factor * susceptibility

    def _enzymatic_degradation_model(
        self,
        composition: Dict[str, float],
        surface_props: Dict[str, float],
        env_params: Dict[str, Any],
    ) -> float:
        """Model enzymatic degradation."""
        base_rate = 1e-7  # s⁻¹

        # Enzyme availability
        enzymes = env_params.get("enzymes", [])
        enzyme_concentration = len(enzymes) * 0.1  # Simplified

        # Material susceptibility to enzymes
        enzyme_susceptible = {
            "collagen": 0.9,
            "gelatin": 0.8,
            "hyaluronic_acid": 0.7,
            "chitosan": 0.6,
        }

        susceptibility = 0.0
        for material, fraction in composition.items():
            if material in enzyme_susceptible:
                susceptibility += enzyme_susceptible[material] * fraction

        return base_rate * enzyme_concentration * susceptibility

    def _oxidative_degradation_model(
        self,
        composition: Dict[str, float],
        surface_props: Dict[str, float],
        env_params: Dict[str, Any],
    ) -> float:
        """Model oxidative degradation."""
        base_rate = 1e-8  # s⁻¹

        # Oxygen concentration effect
        oxygen_conc = env_params.get("oxygen_concentration", 0.1)  # mM
        oxygen_factor = oxygen_conc / 0.1  # Normalized to blood oxygen

        # Material susceptibility to oxidation
        oxidizable_materials = {
            "polyethylene": 0.3,
            "polypropylene": 0.4,
            "natural_rubber": 0.7,
            "polyurethane": 0.5,
        }

        susceptibility = 0.1
        for material, fraction in composition.items():
            if material in oxidizable_materials:
                susceptibility += oxidizable_materials[material] * fraction

        return base_rate * oxygen_factor * susceptibility

    def _mechanical_wear_model(
        self,
        composition: Dict[str, float],
        surface_props: Dict[str, float],
        env_params: Dict[str, Any],
    ) -> float:
        """Model mechanical wear degradation."""
        base_rate = 1e-9  # s⁻¹

        # Flow rate effect
        flow_rate = env_params.get("flow_rate", 0.1)  # L/min
        flow_factor = flow_rate / 5.0  # Normalized to cardiac output

        # Material hardness effect
        hardness = surface_props.get("hardness", 100)  # HV
        hardness_factor = max(0.1, 200 / hardness)  # Softer materials wear more

        return base_rate * flow_factor * hardness_factor

    def _corrosion_model(
        self,
        composition: Dict[str, float],
        surface_props: Dict[str, float],
        env_params: Dict[str, Any],
    ) -> float:
        """Model corrosion degradation."""
        base_rate = 1e-10  # s⁻¹

        # Ionic strength effect
        ionic_strength = env_params.get("ionic_strength", 0.15)
        ionic_factor = ionic_strength / 0.15

        # Material corrosion resistance
        corrosion_susceptible = {
            "iron": 0.8,
            "steel": 0.6,
            "aluminum": 0.4,
            "magnesium": 0.9,
        }

        susceptibility = 0.0
        for material, fraction in composition.items():
            if material in corrosion_susceptible:
                susceptibility += corrosion_susceptible[material] * fraction

        # Corrosion inhibitors
        if "chromium" in composition and composition["chromium"] > 0.1:
            susceptibility *= 0.1  # Stainless steel effect

        return base_rate * ionic_factor * susceptibility

    # Helper methods (simplified implementations)
    def _calculate_toxicity_confidence(
        self, composition: Dict[str, float], endpoint: ToxicityEndpoint
    ) -> float:
        """Calculate confidence in toxicity prediction."""
        # Base confidence
        confidence = 0.7

        # Higher confidence for well-studied materials
        well_studied = ["titanium", "stainless_steel", "polyethylene"]
        for material in well_studied:
            if material in composition:
                confidence += 0.1 * composition[material]

        # Lower confidence for complex compositions
        n_components = len(composition)
        confidence -= 0.05 * max(0, n_components - 3)

        return np.clip(confidence, 0.1, 0.95)

    def _calculate_safety_margin(self, toxicity_score: float, dose: float) -> float:
        """Calculate safety margin."""
        if toxicity_score < 0.01:
            return 100.0  # High safety margin for very low toxicity

        # Simplified safety margin calculation
        threshold_dose = 100.0 * (1 - toxicity_score)  # mg/kg
        return threshold_dose / max(dose, 0.01)

    def _generate_dose_response_curve(
        self,
        composition: Dict[str, float],
        surface_props: Dict[str, float],
        scenario: BiologicalExposureScenario,
        endpoint: ToxicityEndpoint,
    ) -> List[Tuple[float, float]]:
        """Generate dose-response curve."""
        doses = np.logspace(-2, 3, 20)  # 0.01 to 1000 mg/kg
        responses = []

        for dose in doses:
            # Create modified scenario with new dose
            modified_scenario = BiologicalExposureScenario(
                environment=scenario.environment,
                exposure_duration=scenario.exposure_duration,
                material_dose=dose,
                exposure_route=scenario.exposure_route,
                patient_demographics=scenario.patient_demographics,
                physiological_conditions=scenario.physiological_conditions,
            )

            # Calculate response
            env_params = self.biological_environments[scenario.environment]
            response = self.toxicity_models[endpoint](
                composition, surface_props, modified_scenario, env_params
            )
            responses.append((float(dose), response))

        return responses

    def _calculate_noael(self, dose_response: List[Tuple[float, float]]) -> float:
        """Calculate No Observed Adverse Effect Level."""
        threshold = 0.1  # 10% effect level

        for dose, response in dose_response:
            if response > threshold:
                # Find previous dose (NOAEL)
                idx = dose_response.index((dose, response))
                if idx > 0:
                    return dose_response[idx - 1][0]
                else:
                    return dose * 0.1  # Conservative estimate

        return dose_response[-1][0]  # No adverse effects observed

    def _calculate_benchmark_dose(
        self, dose_response: List[Tuple[float, float]]
    ) -> float:
        """Calculate benchmark dose (dose causing 10% response)."""
        target_response = 0.1

        # Linear interpolation to find BMD10
        for i in range(len(dose_response) - 1):
            dose1, resp1 = dose_response[i]
            dose2, resp2 = dose_response[i + 1]

            if resp1 <= target_response <= resp2:
                # Interpolate
                if resp2 - resp1 > 1e-6:
                    fraction = (target_response - resp1) / (resp2 - resp1)
                    return dose1 + fraction * (dose2 - dose1)

        return dose_response[-1][0]  # Conservative fallback

    def _identify_adverse_outcome_pathways(
        self, composition: Dict[str, float], predictions: Dict[ToxicityEndpoint, float]
    ) -> List[Dict[str, Any]]:
        """Identify adverse outcome pathways."""
        pathways = []

        # Simplified pathway identification
        if predictions[ToxicityEndpoint.CYTOTOXICITY] > 0.3:
            pathways.append(
                {
                    "pathway": "cellular_stress_response",
                    "molecular_initiating_event": "membrane_damage",
                    "key_events": [
                        "mitochondrial_dysfunction",
                        "oxidative_stress",
                        "apoptosis",
                    ],
                    "adverse_outcome": "cell_death",
                }
            )

        if predictions[ToxicityEndpoint.IMMUNOTOXICITY] > 0.2:
            pathways.append(
                {
                    "pathway": "immune_activation",
                    "molecular_initiating_event": "protein_binding",
                    "key_events": [
                        "hapten_formation",
                        "antigen_presentation",
                        "T_cell_activation",
                    ],
                    "adverse_outcome": "hypersensitivity_reaction",
                }
            )

        return pathways

    def _perform_risk_characterization(
        self,
        predictions: Dict[ToxicityEndpoint, float],
        safety_margins: Dict[ToxicityEndpoint, float],
        scenario: BiologicalExposureScenario,
    ) -> Dict[str, float]:
        """Perform risk characterization."""
        # Calculate overall risk score
        weighted_risk = 0.0
        weights = {
            ToxicityEndpoint.CYTOTOXICITY: 0.2,
            ToxicityEndpoint.GENOTOXICITY: 0.3,
            ToxicityEndpoint.CARCINOGENICITY: 0.3,
            ToxicityEndpoint.IMMUNOTOXICITY: 0.2,
        }

        for endpoint, weight in weights.items():
            if endpoint in predictions:
                risk = predictions[endpoint] / max(safety_margins[endpoint], 0.1)
                weighted_risk += weight * risk

        return {
            "overall_risk_score": weighted_risk,
            "risk_category": "low"
            if weighted_risk < 0.1
            else "moderate"
            if weighted_risk < 0.5
            else "high",
            "acceptable_exposure_duration": max(
                1.0, 168.0 / (1 + 10 * weighted_risk)
            ),  # hours
        }

    def _predict_degradation_products(
        self,
        composition: Dict[str, float],
        kinetics: Dict[str, float],
        env_params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Predict degradation products."""
        products = []

        # Simplified product prediction based on composition
        for material, fraction in composition.items():
            if material == "polyester" and kinetics.get("hydrolysis", 0) > 1e-6:
                products.append(
                    {
                        "name": "carboxylic_acid",
                        "formation_rate": kinetics["hydrolysis"] * fraction,
                        "molecular_weight": 150.0,
                        "solubility": "high",
                    }
                )
                products.append(
                    {
                        "name": "alcohol",
                        "formation_rate": kinetics["hydrolysis"] * fraction,
                        "molecular_weight": 100.0,
                        "solubility": "high",
                    }
                )

            elif material == "titanium" and kinetics.get("corrosion", 0) > 1e-9:
                products.append(
                    {
                        "name": "titanium_dioxide",
                        "formation_rate": kinetics["corrosion"] * fraction,
                        "molecular_weight": 79.9,
                        "solubility": "low",
                    }
                )

        return products

    def _assess_product_toxicity(
        self, product: Dict[str, Any], environment: BiologicalEnvironment
    ) -> float:
        """Assess toxicity of degradation product."""
        # Simplified product toxicity database
        product_toxicity = {
            "carboxylic_acid": 0.1,
            "alcohol": 0.05,
            "titanium_dioxide": 0.02,
            "metal_ion": 0.3,
        }

        base_toxicity = product_toxicity.get(product["name"], 0.2)

        # Modify based on molecular weight and solubility
        mw_factor = min(1.0, product["molecular_weight"] / 300.0)
        solubility_factor = 1.5 if product["solubility"] == "high" else 0.8

        return base_toxicity * mw_factor * solubility_factor

    def _calculate_remaining_mass(
        self, time: float, kinetics: Dict[str, float], composition: Dict[str, float]
    ) -> float:
        """Calculate remaining mass fraction."""
        # Simple first-order kinetics
        total_rate = sum(kinetics.values())
        return np.exp(-total_rate * time)

    def _predict_mechanical_property_changes(
        self,
        time_points: np.ndarray,
        kinetics: Dict[str, float],
        properties: Dict[str, float],
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Predict mechanical property changes over time."""
        changes = {}

        # Elastic modulus degradation
        initial_modulus = properties.get("elastic_modulus", 100.0)  # GPa
        modulus_profile = []

        for t in time_points:
            # Degradation reduces mechanical properties
            degradation_factor = self._calculate_remaining_mass(t, kinetics, {})
            current_modulus = initial_modulus * degradation_factor
            modulus_profile.append((float(t), current_modulus))

        changes["elastic_modulus"] = modulus_profile

        # Tensile strength degradation
        initial_strength = properties.get("tensile_strength", 500.0)  # MPa
        strength_profile = []

        for t in time_points:
            degradation_factor = self._calculate_remaining_mass(t, kinetics, {})
            current_strength = initial_strength * (
                degradation_factor**0.5
            )  # Faster degradation
            strength_profile.append((float(t), current_strength))

        changes["tensile_strength"] = strength_profile

        return changes

    def _predict_surface_property_evolution(
        self,
        time_points: np.ndarray,
        kinetics: Dict[str, float],
        properties: Dict[str, float],
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Predict surface property evolution over time."""
        evolution = {}

        # Surface roughness increase
        initial_roughness = properties.get("roughness", 0.1)  # μm
        roughness_profile = []

        for t in time_points:
            # Degradation increases surface roughness
            degradation_extent = 1 - self._calculate_remaining_mass(t, kinetics, {})
            current_roughness = initial_roughness * (1 + 5 * degradation_extent)
            roughness_profile.append((float(t), current_roughness))

        evolution["roughness"] = roughness_profile

        # Contact angle changes
        initial_angle = properties.get("contact_angle", 90.0)  # degrees
        angle_profile = []

        for t in time_points:
            degradation_extent = 1 - self._calculate_remaining_mass(t, kinetics, {})
            # Degradation typically makes surfaces more hydrophilic
            current_angle = initial_angle * (1 - 0.3 * degradation_extent)
            angle_profile.append((float(t), current_angle))

        evolution["contact_angle"] = angle_profile

        return evolution

    def _calculate_time_dependent_biocompatibility(
        self,
        time: float,
        kinetics: Dict[str, float],
        product_toxicity: Dict[str, float],
    ) -> float:
        """Calculate biocompatibility as a function of time."""
        # Initial biocompatibility
        initial_biocompat = 0.8

        # Degradation reduces biocompatibility due to product formation
        degradation_extent = 1 - self._calculate_remaining_mass(time, kinetics, {})

        # Average toxicity of products
        avg_product_toxicity = (
            np.mean(list(product_toxicity.values())) if product_toxicity else 0.1
        )

        # Time-dependent biocompatibility
        current_biocompat = initial_biocompat * (
            1 - 0.5 * degradation_extent * avg_product_toxicity
        )

        return max(0.1, current_biocompat)

    def _calculate_cumulative_risk(
        self,
        toxicity_assessments: List[ToxicityAssessment],
        biodegradation_profiles: List[BiodegradationProfile],
    ) -> float:
        """Calculate cumulative risk score."""
        # Combine risks from all assessments
        toxicity_risks = []
        for assessment in toxicity_assessments:
            risk_score = assessment.risk_characterization["overall_risk_score"]
            toxicity_risks.append(risk_score)

        # Consider degradation product toxicity
        degradation_risks = []
        for profile in biodegradation_profiles:
            avg_product_toxicity = (
                np.mean(list(profile.product_toxicity.values()))
                if profile.product_toxicity
                else 0.1
            )
            degradation_risks.append(avg_product_toxicity)

        # Combine risks
        avg_toxicity_risk = np.mean(toxicity_risks) if toxicity_risks else 0.1
        avg_degradation_risk = np.mean(degradation_risks) if degradation_risks else 0.1

        cumulative_risk = 0.7 * avg_toxicity_risk + 0.3 * avg_degradation_risk

        return min(1.0, cumulative_risk)

    def _identify_critical_pathways(
        self,
        toxicity_assessments: List[ToxicityAssessment],
        biodegradation_profiles: List[BiodegradationProfile],
    ) -> List[str]:
        """Identify critical exposure pathways."""
        critical_pathways = []

        # Identify high-risk scenarios
        for assessment in toxicity_assessments:
            if assessment.risk_characterization["overall_risk_score"] > 0.5:
                pathway = f"{assessment.exposure_scenario.environment.value}_{assessment.exposure_scenario.exposure_route}"
                critical_pathways.append(pathway)

        # Identify problematic degradation
        for profile in biodegradation_profiles:
            if any(toxicity > 0.5 for toxicity in profile.product_toxicity.values()):
                critical_pathways.append(f"degradation_{profile.environment.value}")

        return list(set(critical_pathways))  # Remove duplicates

    def _generate_mitigation_strategies(
        self,
        toxicity_assessments: List[ToxicityAssessment],
        biodegradation_profiles: List[BiodegradationProfile],
    ) -> List[str]:
        """Generate risk mitigation strategies."""
        strategies = []

        # General strategies
        strategies.append("Implement comprehensive pre-clinical safety testing")
        strategies.append("Establish patient monitoring protocols")
        strategies.append("Develop material degradation tracking methods")

        # Specific strategies based on risks
        high_cytotoxicity = any(
            assessment.endpoint_predictions[ToxicityEndpoint.CYTOTOXICITY] > 0.3
            for assessment in toxicity_assessments
            if ToxicityEndpoint.CYTOTOXICITY in assessment.endpoint_predictions
        )

        if high_cytotoxicity:
            strategies.append(
                "Consider surface coating to reduce direct cellular contact"
            )
            strategies.append("Implement controlled release mechanisms")

        high_immunogenicity = any(
            assessment.endpoint_predictions[ToxicityEndpoint.IMMUNOTOXICITY] > 0.2
            for assessment in toxicity_assessments
            if ToxicityEndpoint.IMMUNOTOXICITY in assessment.endpoint_predictions
        )

        if high_immunogenicity:
            strategies.append("Apply immunosuppressive surface treatments")
            strategies.append("Consider patient immunomodulation protocols")

        return strategies

    def _check_regulatory_compliance(
        self,
        toxicity_assessments: List[ToxicityAssessment],
        biodegradation_profiles: List[BiodegradationProfile],
    ) -> Dict[str, bool]:
        """Check regulatory compliance."""
        compliance = {}

        # ISO 10993 biocompatibility standards
        max_cytotoxicity = max(
            assessment.endpoint_predictions.get(ToxicityEndpoint.CYTOTOXICITY, 0)
            for assessment in toxicity_assessments
        )
        compliance["ISO_10993_cytotoxicity"] = max_cytotoxicity < 0.3

        # FDA guidance compliance
        max_genotoxicity = max(
            assessment.endpoint_predictions.get(ToxicityEndpoint.GENOTOXICITY, 0)
            for assessment in toxicity_assessments
        )
        compliance["FDA_genotoxicity"] = max_genotoxicity < 0.1

        # USP Class VI compliance (simplified)
        overall_risk = max(
            assessment.risk_characterization["overall_risk_score"]
            for assessment in toxicity_assessments
        )
        compliance["USP_Class_VI"] = overall_risk < 0.2

        return compliance

    def _recommend_safety_testing(
        self,
        toxicity_assessments: List[ToxicityAssessment],
        biodegradation_profiles: List[BiodegradationProfile],
    ) -> List[str]:
        """Recommend safety testing protocols."""
        tests = []

        # Standard tests
        tests.append("ISO 10993-5: Cytotoxicity testing")
        tests.append("ISO 10993-10: Irritation and sensitization")
        tests.append("ISO 10993-11: Systemic toxicity")

        # Endpoint-specific tests
        for assessment in toxicity_assessments:
            if (
                assessment.endpoint_predictions.get(ToxicityEndpoint.GENOTOXICITY, 0)
                > 0.05
            ):
                tests.append("ISO 10993-3: Genotoxicity testing")

            if (
                assessment.endpoint_predictions.get(ToxicityEndpoint.HEMATOTOXICITY, 0)
                > 0.1
            ):
                tests.append("ISO 10993-4: Hemolysis testing")

            if (
                assessment.endpoint_predictions.get(ToxicityEndpoint.CARCINOGENICITY, 0)
                > 0.1
            ):
                tests.append("ISO 10993-7: Carcinogenicity testing")

        # Degradation-specific tests
        if any(profile.product_toxicity for profile in biodegradation_profiles):
            tests.append("ISO 10993-9: Degradation product analysis")
            tests.append("ISO 10993-13: Polymer identification")

        return list(set(tests))  # Remove duplicates
