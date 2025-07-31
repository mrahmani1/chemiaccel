"""
AI-Robot Integration Layer

This module provides the integration layer between ChemiAccel's AI decision-making
systems (Active Learning, Inverse Design) and robotic platforms, enabling seamless
communication and coordination for autonomous experimentation.
"""

import asyncio
import logging
import threading
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# Import AI modules
try:
    from ..ai.active_learning import ActiveLearningEngine, QueryStrategy
    from ..ai.inverse_design import (
        DesignTarget,
        InverseDesignEngine,
        MolecularCandidate,
    )

    CHEMI_ACCEL_AI_AVAILABLE = True
except ImportError:
    CHEMI_ACCEL_AI_AVAILABLE = False

from .adaptive_learning import AdaptiveLearningManager
from .closed_loop_optimization import ClosedLoopOptimizer
from .robotic_control import ExperimentalAction, ExperimentPlan, RobotManager

logger = logging.getLogger(__name__)


class AIDecisionType(Enum):
    """Types of AI decisions that can drive robotic experiments."""

    ACTIVE_LEARNING_QUERY = "active_learning_query"
    INVERSE_DESIGN_CANDIDATE = "inverse_design_candidate"
    OPTIMIZATION_SUGGESTION = "optimization_suggestion"
    EXPLORATORY_EXPERIMENT = "exploratory_experiment"
    VALIDATION_EXPERIMENT = "validation_experiment"


class SynthesisComplexity(Enum):
    """Complexity levels for synthesis tasks."""

    SIMPLE = "simple"  # Single step, well-known reactions
    MODERATE = "moderate"  # Multi-step, standard conditions
    COMPLEX = "complex"  # Multi-step, challenging conditions
    NOVEL = "novel"  # Unexplored reaction space


@dataclass
class AIDecision:
    """Represents a decision made by an AI system."""

    decision_id: str
    decision_type: AIDecisionType
    ai_system: str  # Name of the AI system making the decision
    confidence: float  # 0.0 to 1.0
    reasoning: Dict[str, Any]  # Explanation of the decision
    parameters: Dict[str, Any]  # Parameters for the decision
    priority: int = 1  # 1=low, 5=high
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SynthesisTask:
    """Represents a synthesis task derived from AI decisions."""

    task_id: str
    target_molecule: Dict[str, Any]  # Molecular representation
    synthesis_route: List[Dict[str, Any]]  # Step-by-step synthesis plan
    complexity: SynthesisComplexity
    success_probability: float
    estimated_duration: float  # seconds
    resource_requirements: Dict[str, Any]
    safety_considerations: List[str]
    quality_metrics: Dict[str, float]
    ai_decision_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ExperimentalDesign:
    """Complete experimental design for robotic execution."""

    design_id: str
    synthesis_tasks: List[SynthesisTask]
    analysis_protocols: List[Dict[str, Any]]
    quality_controls: List[Dict[str, Any]]
    success_criteria: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    ai_decisions: List[AIDecision]
    total_estimated_time: float
    resource_utilization: Dict[str, float]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AIDecisionEngine(ABC):
    """Abstract base class for AI decision engines."""

    @abstractmethod
    async def make_decision(
        self, context: Dict[str, Any], constraints: Optional[Dict[str, Any]] = None
    ) -> AIDecision:
        """Make an AI-driven decision based on context."""
        pass

    @abstractmethod
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about this AI engine."""
        pass


class ActiveLearningDecisionEngine(AIDecisionEngine):
    """AI decision engine based on active learning strategies."""

    def __init__(self, adaptive_learning_manager: AdaptiveLearningManager):
        self.adaptive_manager = adaptive_learning_manager
        self.query_history: List[Dict[str, Any]] = []
        self.performance_metrics = {
            "queries_generated": 0,
            "successful_queries": 0,
            "average_information_gain": 0.0,
        }

    async def make_decision(
        self, context: Dict[str, Any], constraints: Optional[Dict[str, Any]] = None
    ) -> AIDecision:
        """Make active learning-based decision."""
        try:
            # Get current model uncertainty
            model_uncertainty = await self._assess_model_uncertainty(context)

            # Identify high-uncertainty regions
            query_candidates = await self._generate_query_candidates(
                context, constraints or {}
            )

            # Select best query based on information gain
            best_candidate = self._select_best_candidate(
                query_candidates, model_uncertainty
            )

            # Calculate confidence based on expected information gain
            confidence = self._calculate_query_confidence(
                best_candidate, model_uncertainty
            )

            decision = AIDecision(
                decision_id=str(uuid.uuid4()),
                decision_type=AIDecisionType.ACTIVE_LEARNING_QUERY,
                ai_system="active_learning_engine",
                confidence=confidence,
                reasoning={
                    "model_uncertainty": model_uncertainty,
                    "expected_information_gain": best_candidate.get(
                        "information_gain", 0.0
                    ),
                    "query_diversity": best_candidate.get("diversity_score", 0.0),
                    "exploration_balance": best_candidate.get("exploration_score", 0.0),
                },
                parameters=best_candidate.get("parameters", {}),
                priority=self._calculate_priority(confidence, model_uncertainty),
                constraints=constraints or {},
                metadata={
                    "query_strategy": "uncertainty_sampling",
                    "model_version": (
                        self.adaptive_manager.get_model_version()
                        if hasattr(self.adaptive_manager, "get_model_version")
                        else "unknown"
                    ),
                },
            )

            # Update metrics
            self.performance_metrics["queries_generated"] += 1
            self.query_history.append(
                {
                    "decision_id": decision.decision_id,
                    "timestamp": decision.created_at,
                    "confidence": confidence,
                    "parameters": best_candidate.get("parameters", {}),
                }
            )

            logger.info(
                f"Generated active learning decision {decision.decision_id} with confidence {confidence:.3f}"
            )
            return decision

        except Exception as e:
            logger.error(f"Error making active learning decision: {e}")
            # Return a fallback random exploration decision
            return self._create_fallback_decision(context, constraints)

    async def _assess_model_uncertainty(self, context: Dict[str, Any]) -> float:
        """Assess current model uncertainty."""
        try:
            # Get recent performance metrics
            if hasattr(self.adaptive_manager, "performance_monitor"):
                performance_summary = (
                    self.adaptive_manager.performance_monitor.get_performance_summary()
                )
                recent_performance = performance_summary.get("recent_performance", [])

                if recent_performance:
                    # Calculate uncertainty based on performance variance
                    performance_values = [
                        p.get("confidence", 0.5) for p in recent_performance[-10:]
                    ]
                    uncertainty = 1.0 - np.std(
                        performance_values
                    )  # Higher std = higher uncertainty
                    return max(0.1, min(1.0, uncertainty))

            return 0.5  # Default medium uncertainty

        except Exception as e:
            logger.warning(f"Error assessing model uncertainty: {e}")
            return 0.5

    async def _generate_query_candidates(
        self, context: Dict[str, Any], constraints: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate candidate queries for active learning."""
        candidates = []

        # Get parameter space from context
        parameter_space = context.get("parameter_space", {})
        if not parameter_space:
            # Create default chemical parameter space
            parameter_space = {
                "temperature": (20, 200),  # Celsius
                "reaction_time": (0.5, 24),  # hours
                "concentration": (0.01, 1.0),  # M
                "catalyst_loading": (0.001, 0.1),  # mol%
            }

        # Generate diverse candidates
        for i in range(20):  # Generate 20 candidates
            candidate_params = {}

            for param_name, (min_val, max_val) in parameter_space.items():
                # Add some intelligent sampling strategies
                if i < 5:
                    # Exploit: near current best
                    best_params = context.get("best_parameters", {})
                    if param_name in best_params:
                        current_best = best_params[param_name]
                        # Sample near best with some noise
                        noise_scale = (max_val - min_val) * 0.1
                        value = np.random.normal(current_best, noise_scale)
                        value = np.clip(value, min_val, max_val)
                    else:
                        value = np.random.uniform(min_val, max_val)
                elif i < 10:
                    # Explore: boundary regions
                    if np.random.random() < 0.5:
                        value = np.random.uniform(
                            min_val, min_val + (max_val - min_val) * 0.2
                        )
                    else:
                        value = np.random.uniform(
                            max_val - (max_val - min_val) * 0.2, max_val
                        )
                else:
                    # Random exploration
                    value = np.random.uniform(min_val, max_val)

                candidate_params[param_name] = value

            # Calculate scores for this candidate
            information_gain = self._estimate_information_gain(
                candidate_params, context
            )
            diversity_score = self._calculate_diversity_score(candidate_params, context)
            exploration_score = self._calculate_exploration_score(
                candidate_params, parameter_space
            )
            feasibility_score = self._assess_feasibility(candidate_params, constraints)

            candidate = {
                "parameters": candidate_params,
                "information_gain": information_gain,
                "diversity_score": diversity_score,
                "exploration_score": exploration_score,
                "feasibility_score": feasibility_score,
                "combined_score": (
                    0.4 * information_gain
                    + 0.3 * diversity_score
                    + 0.2 * exploration_score
                    + 0.1 * feasibility_score
                ),
            }

            candidates.append(candidate)

        return candidates

    def _select_best_candidate(
        self, candidates: List[Dict[str, Any]], model_uncertainty: float
    ) -> Dict[str, Any]:
        """Select the best candidate query."""
        # Adjust selection strategy based on model uncertainty
        if model_uncertainty > 0.7:
            # High uncertainty: prioritize information gain
            candidates = sorted(
                candidates, key=lambda x: x["information_gain"], reverse=True
            )
        elif model_uncertainty < 0.3:
            # Low uncertainty: prioritize exploration
            candidates = sorted(
                candidates, key=lambda x: x["exploration_score"], reverse=True
            )
        else:
            # Medium uncertainty: use combined score
            candidates = sorted(
                candidates, key=lambda x: x["combined_score"], reverse=True
            )

        return candidates[0] if candidates else {}

    def _calculate_query_confidence(
        self, candidate: Dict[str, Any], model_uncertainty: float
    ) -> float:
        """Calculate confidence in the query decision."""
        base_confidence = candidate.get("combined_score", 0.5)

        # Adjust confidence based on model uncertainty
        uncertainty_factor = 1.0 - model_uncertainty * 0.3

        # Consider feasibility
        feasibility = candidate.get("feasibility_score", 1.0)

        final_confidence = base_confidence * uncertainty_factor * feasibility
        return max(0.1, min(1.0, final_confidence))

    def _calculate_priority(self, confidence: float, model_uncertainty: float) -> int:
        """Calculate priority for the query."""
        if confidence > 0.8 and model_uncertainty > 0.6:
            return 5  # High priority: high confidence, high uncertainty
        elif confidence > 0.6:
            return 3  # Medium priority
        else:
            return 1  # Low priority

    def _estimate_information_gain(
        self, parameters: Dict[str, Any], context: Dict[str, Any]
    ) -> float:
        """Estimate expected information gain from query."""
        # Simplified information gain estimation
        # In practice, this would use the model's predictive uncertainty

        # Calculate distance from previous queries
        previous_queries = context.get("previous_experiments", [])
        if not previous_queries:
            return 1.0  # High gain if no previous data

        min_distance = float("inf")
        for prev_query in previous_queries:
            prev_params = prev_query.get("parameters", {})
            distance = 0.0
            param_count = 0

            for param_name, value in parameters.items():
                if param_name in prev_params:
                    # Normalized distance
                    param_space = context.get("parameter_space", {})
                    if param_name in param_space:
                        min_val, max_val = param_space[param_name]
                        norm_distance = abs(value - prev_params[param_name]) / (
                            max_val - min_val
                        )
                        distance += norm_distance**2
                        param_count += 1

            if param_count > 0:
                distance = np.sqrt(distance / param_count)
                min_distance = min(min_distance, distance)

        # Information gain inversely related to minimum distance
        information_gain = min(1.0, min_distance * 2.0)  # Scale to 0-1
        return information_gain

    def _calculate_diversity_score(
        self, parameters: Dict[str, Any], context: Dict[str, Any]
    ) -> float:
        """Calculate diversity score for parameter set."""
        # Similar to information gain but focuses on coverage
        return self._estimate_information_gain(parameters, context)

    def _calculate_exploration_score(
        self, parameters: Dict[str, Any], parameter_space: Dict[str, Any]
    ) -> float:
        """Calculate exploration score (preference for boundary regions)."""
        boundary_scores = []

        for param_name, value in parameters.items():
            if param_name in parameter_space:
                min_val, max_val = parameter_space[param_name]
                # Distance from center, normalized
                center = (min_val + max_val) / 2.0
                range_size = max_val - min_val
                distance_from_center = abs(value - center) / (range_size / 2.0)
                boundary_scores.append(distance_from_center)

        return np.mean(boundary_scores) if boundary_scores else 0.5

    def _assess_feasibility(
        self, parameters: Dict[str, Any], constraints: Dict[str, Any]
    ) -> float:
        """Assess feasibility of parameters given constraints."""
        # Check basic parameter bounds
        feasibility = 1.0

        # Check resource constraints
        resource_limits = constraints.get("resource_limits", {})
        for resource, limit in resource_limits.items():
            if resource in parameters:
                if parameters[resource] > limit:
                    feasibility *= 0.5  # Reduce feasibility for constraint violations

        # Check safety constraints
        safety_limits = constraints.get("safety_limits", {})
        for param, max_safe in safety_limits.items():
            if param in parameters and parameters[param] > max_safe:
                feasibility *= 0.1  # Heavily penalize safety violations

        return feasibility

    def _create_fallback_decision(
        self, context: Dict[str, Any], constraints: Optional[Dict[str, Any]]
    ) -> AIDecision:
        """Create a fallback decision when the main algorithm fails."""
        # Generate random parameters
        parameter_space = context.get(
            "parameter_space",
            {
                "temperature": (20, 200),
                "reaction_time": (0.5, 24),
                "concentration": (0.01, 1.0),
            },
        )

        fallback_params = {}
        for param_name, (min_val, max_val) in parameter_space.items():
            fallback_params[param_name] = np.random.uniform(min_val, max_val)

        return AIDecision(
            decision_id=str(uuid.uuid4()),
            decision_type=AIDecisionType.EXPLORATORY_EXPERIMENT,
            ai_system="active_learning_engine",
            confidence=0.3,
            reasoning={"fallback": True, "reason": "main algorithm failed"},
            parameters=fallback_params,
            priority=1,
            constraints=constraints or {},
        )

    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about this AI engine."""
        return {
            "engine_type": "active_learning",
            "description": "Active learning-based decision engine for experimental design",
            "performance_metrics": self.performance_metrics,
            "query_history_size": len(self.query_history),
            "supported_strategies": [
                "uncertainty_sampling",
                "diversity_sampling",
                "hybrid",
            ],
        }


class InverseDesignDecisionEngine(AIDecisionEngine):
    """AI decision engine based on inverse molecular design."""

    def __init__(self):
        self.design_history: List[Dict[str, Any]] = []
        self.performance_metrics = {
            "designs_generated": 0,
            "successful_syntheses": 0,
            "average_success_rate": 0.0,
        }
        self.molecular_database = {}  # Cache of designed molecules

    async def make_decision(
        self, context: Dict[str, Any], constraints: Optional[Dict[str, Any]] = None
    ) -> AIDecision:
        """Make inverse design-based decision."""
        try:
            # Extract design target from context
            target_properties = context.get("target_properties", {})
            if not target_properties:
                # Default target for optimization
                target_properties = {
                    "molecular_weight": (200, 400),
                    "logp": (2, 4),
                    "tpsa": (40, 90),
                    "activity": "maximize",
                }

            # Generate molecular candidates
            candidates = await self._generate_molecular_candidates(
                target_properties, constraints or {}
            )

            # Select best candidate based on design criteria
            best_candidate = self._select_best_design(candidates, target_properties)

            # Generate synthesis route
            synthesis_route = await self._generate_synthesis_route(best_candidate)

            # Calculate confidence based on synthetic accessibility
            confidence = self._calculate_design_confidence(
                best_candidate, synthesis_route
            )

            decision = AIDecision(
                decision_id=str(uuid.uuid4()),
                decision_type=AIDecisionType.INVERSE_DESIGN_CANDIDATE,
                ai_system="inverse_design_engine",
                confidence=confidence,
                reasoning={
                    "target_properties": target_properties,
                    "predicted_properties": best_candidate.get(
                        "predicted_properties", {}
                    ),
                    "synthetic_accessibility": best_candidate.get("sa_score", 0.5),
                    "novelty_score": best_candidate.get("novelty", 0.5),
                    "synthesis_steps": len(synthesis_route),
                },
                parameters={
                    "target_molecule": best_candidate,
                    "synthesis_route": synthesis_route,
                    "synthesis_parameters": self._extract_synthesis_parameters(
                        synthesis_route
                    ),
                },
                priority=self._calculate_design_priority(confidence, best_candidate),
                constraints=constraints or {},
                metadata={
                    "design_method": "property_optimization",
                    "molecular_representation": best_candidate.get(
                        "representation", "smiles"
                    ),
                },
            )

            # Update metrics and history
            self.performance_metrics["designs_generated"] += 1
            self.design_history.append(
                {
                    "decision_id": decision.decision_id,
                    "timestamp": decision.created_at,
                    "target_properties": target_properties,
                    "molecular_candidate": best_candidate,
                }
            )

            logger.info(
                f"Generated inverse design decision {decision.decision_id} with confidence {confidence:.3f}"
            )
            return decision

        except Exception as e:
            logger.error(f"Error making inverse design decision: {e}")
            return self._create_fallback_design_decision(context, constraints)

    async def _generate_molecular_candidates(
        self, target_properties: Dict[str, Any], constraints: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate molecular candidates based on target properties."""
        candidates = []

        # Simple molecular generation (in practice, would use advanced ML models)
        for i in range(10):  # Generate 10 candidates
            # Generate random molecular properties within target ranges
            candidate = {
                "candidate_id": f"mol_{i:03d}",
                "smiles": self._generate_random_smiles(),
                "predicted_properties": {},
                "sa_score": np.random.uniform(0.3, 0.9),  # Synthetic accessibility
                "novelty": np.random.uniform(0.2, 0.8),
                "representation": "smiles",
            }

            # Predict properties for each target
            for prop_name, prop_target in target_properties.items():
                if isinstance(prop_target, tuple) and len(prop_target) == 2:
                    # Range target
                    min_val, max_val = prop_target
                    predicted_value = np.random.uniform(min_val, max_val)
                    candidate["predicted_properties"][prop_name] = predicted_value
                elif prop_target == "maximize":
                    candidate["predicted_properties"][prop_name] = np.random.uniform(
                        0.7, 1.0
                    )
                elif prop_target == "minimize":
                    candidate["predicted_properties"][prop_name] = np.random.uniform(
                        0.0, 0.3
                    )

            # Apply constraints
            if self._satisfies_constraints(candidate, constraints):
                candidates.append(candidate)

        return candidates

    def _generate_random_smiles(self) -> str:
        """Generate a random SMILES string (simplified)."""
        # This is a very simplified example - in practice would use proper molecular generation
        fragments = [
            "CCc1ccc(cc1)",  # ethylbenzene
            "c1ccc(cc1)C",  # toluene
            "CCO",  # ethanol
            "c1ccc(cc1)O",  # phenol
            "CC(=O)O",  # acetic acid
            "c1ccc2c(c1)ccc(c2)",  # naphthalene derivative
        ]

        base = np.random.choice(fragments)

        # Add some random modifications
        modifications = ["[Cl]", "[F]", "[Br]", "N", "S", "O"]
        if np.random.random() < 0.3:
            base += np.random.choice(modifications)

        return base

    def _satisfies_constraints(
        self, candidate: Dict[str, Any], constraints: Dict[str, Any]
    ) -> bool:
        """Check if candidate satisfies constraints."""
        # Check molecular weight constraints
        mw_constraint = constraints.get("max_molecular_weight")
        if mw_constraint:
            predicted_mw = candidate["predicted_properties"].get("molecular_weight", 0)
            if predicted_mw > mw_constraint:
                return False

        # Check synthetic accessibility constraints
        min_sa = constraints.get("min_synthetic_accessibility", 0.0)
        if candidate["sa_score"] < min_sa:
            return False

        # Check chemical safety constraints
        safety_constraints = constraints.get("safety_constraints", [])
        for constraint in safety_constraints:
            # In practice, would check for dangerous functional groups, etc.
            pass

        return True

    def _select_best_design(
        self, candidates: List[Dict[str, Any]], target_properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Select the best molecular design from candidates."""
        if not candidates:
            return {}

        scored_candidates = []

        for candidate in candidates:
            score = 0.0
            property_scores = []

            # Score based on target property matching
            for prop_name, prop_target in target_properties.items():
                predicted_value = candidate["predicted_properties"].get(prop_name, 0)

                if isinstance(prop_target, tuple):
                    # Range target
                    min_val, max_val = prop_target
                    if min_val <= predicted_value <= max_val:
                        property_scores.append(1.0)
                    else:
                        # Distance from range
                        if predicted_value < min_val:
                            distance = (min_val - predicted_value) / (max_val - min_val)
                        else:
                            distance = (predicted_value - max_val) / (max_val - min_val)
                        property_scores.append(max(0.0, 1.0 - distance))
                elif prop_target == "maximize":
                    property_scores.append(predicted_value)
                elif prop_target == "minimize":
                    property_scores.append(1.0 - predicted_value)

            # Combine property scores
            property_score = np.mean(property_scores) if property_scores else 0.5

            # Add synthetic accessibility and novelty
            sa_score = candidate["sa_score"]
            novelty_score = candidate["novelty"]

            # Weighted combination
            total_score = 0.6 * property_score + 0.3 * sa_score + 0.1 * novelty_score

            scored_candidates.append((total_score, candidate))

        # Return best candidate
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return scored_candidates[0][1]

    async def _generate_synthesis_route(
        self, molecular_candidate: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate synthesis route for molecular candidate."""
        # Simplified synthesis route generation
        # In practice, would use retrosynthesis algorithms

        smiles = molecular_candidate.get("smiles", "")
        sa_score = molecular_candidate.get("sa_score", 0.5)

        # Estimate number of steps based on synthetic accessibility
        if sa_score > 0.8:
            num_steps = 1  # Simple synthesis
        elif sa_score > 0.6:
            num_steps = 2  # Moderate synthesis
        else:
            num_steps = 3  # Complex synthesis

        synthesis_route = []

        for step in range(num_steps):
            step_data = {
                "step_number": step + 1,
                "reaction_type": self._select_reaction_type(step, num_steps),
                "reactants": self._generate_reactants(step),
                "conditions": self._generate_reaction_conditions(),
                "estimated_yield": np.random.uniform(0.6, 0.9),
                "estimated_time": np.random.uniform(2, 12),  # hours
                "safety_rating": np.random.choice(["low", "medium", "high"]),
            }
            synthesis_route.append(step_data)

        return synthesis_route

    def _select_reaction_type(self, step: int, total_steps: int) -> str:
        """Select appropriate reaction type for synthesis step."""
        if step == 0:  # First step
            return np.random.choice(
                [
                    "suzuki_coupling",
                    "heck_reaction",
                    "aldol_condensation",
                    "grignard_reaction",
                    "diels_alder",
                ]
            )
        elif step == total_steps - 1:  # Last step
            return np.random.choice(
                ["reduction", "oxidation", "deprotection", "cyclization"]
            )
        else:  # Middle steps
            return np.random.choice(
                ["substitution", "addition", "elimination", "rearrangement"]
            )

    def _generate_reactants(self, step: int) -> List[str]:
        """Generate reactants for synthesis step."""
        # Simplified reactant generation
        common_reactants = [
            "benzene",
            "toluene",
            "acetone",
            "ethanol",
            "acetic_acid",
            "sodium_hydroxide",
            "hydrochloric_acid",
            "magnesium",
            "palladium_catalyst",
            "copper_catalyst",
        ]

        num_reactants = np.random.choice([2, 3])
        return np.random.choice(
            common_reactants, size=num_reactants, replace=False
        ).tolist()

    def _generate_reaction_conditions(self) -> Dict[str, Any]:
        """Generate reaction conditions."""
        return {
            "temperature": np.random.uniform(20, 150),  # Celsius
            "pressure": np.random.uniform(1, 5),  # atm
            "solvent": np.random.choice(["THF", "DMF", "acetone", "water", "ethanol"]),
            "atmosphere": np.random.choice(["air", "nitrogen", "argon"]),
            "catalyst_loading": np.random.uniform(0.01, 0.1),  # mol%
        }

    def _extract_synthesis_parameters(
        self, synthesis_route: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract synthesis parameters for robotic execution."""
        parameters = {}

        for i, step in enumerate(synthesis_route):
            step_prefix = f"step_{i+1}_"
            conditions = step["conditions"]

            parameters[f"{step_prefix}temperature"] = conditions["temperature"]
            parameters[f"{step_prefix}time"] = step["estimated_time"]
            parameters[f"{step_prefix}catalyst_loading"] = conditions.get(
                "catalyst_loading", 0.05
            )

            # Add reactant concentrations
            for j, reactant in enumerate(step["reactants"]):
                parameters[f"{step_prefix}reactant_{j+1}_conc"] = np.random.uniform(
                    0.1, 1.0
                )

        return parameters

    def _calculate_design_confidence(
        self, candidate: Dict[str, Any], synthesis_route: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence in the design decision."""
        # Base confidence from synthetic accessibility
        sa_confidence = candidate.get("sa_score", 0.5)

        # Adjust based on synthesis complexity
        complexity_penalty = 0.1 * len(synthesis_route)  # Penalty for complex routes

        # Consider predicted yields
        overall_yield = 1.0
        for step in synthesis_route:
            overall_yield *= step.get("estimated_yield", 0.8)

        yield_confidence = overall_yield

        # Combine factors
        confidence = (
            0.5 * sa_confidence
            + 0.3 * yield_confidence
            + 0.2 * (1.0 - complexity_penalty)
        )

        return max(0.1, min(1.0, confidence))

    def _calculate_design_priority(
        self, confidence: float, candidate: Dict[str, Any]
    ) -> int:
        """Calculate priority for the design."""
        novelty = candidate.get("novelty", 0.5)

        if confidence > 0.8 and novelty > 0.6:
            return 5  # High priority: confident and novel
        elif confidence > 0.6:
            return 3  # Medium priority
        else:
            return 1  # Low priority

    def _create_fallback_design_decision(
        self, context: Dict[str, Any], constraints: Optional[Dict[str, Any]]
    ) -> AIDecision:
        """Create fallback design decision."""
        fallback_molecule = {
            "candidate_id": "fallback_001",
            "smiles": "c1ccc(cc1)C",  # Simple toluene
            "predicted_properties": {"molecular_weight": 92.14},
            "sa_score": 0.9,
            "novelty": 0.1,
        }

        fallback_route = [
            {
                "step_number": 1,
                "reaction_type": "methylation",
                "reactants": ["benzene", "methyl_iodide"],
                "conditions": {"temperature": 80, "time": 4},
                "estimated_yield": 0.8,
            }
        ]

        return AIDecision(
            decision_id=str(uuid.uuid4()),
            decision_type=AIDecisionType.INVERSE_DESIGN_CANDIDATE,
            ai_system="inverse_design_engine",
            confidence=0.4,
            reasoning={"fallback": True, "reason": "main algorithm failed"},
            parameters={
                "target_molecule": fallback_molecule,
                "synthesis_route": fallback_route,
                "synthesis_parameters": {"step_1_temperature": 80, "step_1_time": 4},
            },
            priority=1,
            constraints=constraints or {},
        )

    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about this AI engine."""
        return {
            "engine_type": "inverse_design",
            "description": "Inverse design engine for molecular discovery and synthesis planning",
            "performance_metrics": self.performance_metrics,
            "design_history_size": len(self.design_history),
            "supported_representations": ["smiles", "mol2", "sdf"],
            "supported_properties": ["molecular_weight", "logp", "tpsa", "activity"],
        }


class AIRobotIntegrationManager:
    """
    Main manager for integrating AI decision-making with robotic platforms.

    Coordinates between AI engines, robotic systems, and optimization workflows
    to enable autonomous experimental campaigns.
    """

    def __init__(
        self,
        robot_manager: RobotManager,
        adaptive_learning_manager: AdaptiveLearningManager,
        closed_loop_optimizer: Optional[ClosedLoopOptimizer] = None,
    ):
        self.robot_manager = robot_manager
        self.adaptive_manager = adaptive_learning_manager
        self.optimizer = closed_loop_optimizer

        # AI Decision Engines
        self.ai_engines: Dict[str, AIDecisionEngine] = {}
        self._initialize_ai_engines()

        # Integration state
        self.active_campaigns: Dict[str, Dict[str, Any]] = {}
        self.decision_history: List[AIDecision] = []
        self.synthesis_tasks: Dict[str, SynthesisTask] = {}
        self.experimental_designs: Dict[str, ExperimentalDesign] = {}

        # Performance tracking
        self.integration_metrics = {
            "total_ai_decisions": 0,
            "successful_syntheses": 0,
            "failed_syntheses": 0,
            "average_synthesis_time": 0.0,
            "ai_robot_communication_latency": 0.0,
        }

        # Threading and async support
        self._is_running = False
        self._integration_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()

        # Callbacks
        self.decision_callbacks: List[Callable[[AIDecision], None]] = []
        self.synthesis_callbacks: List[Callable[[SynthesisTask], None]] = []
        self.design_callbacks: List[Callable[[ExperimentalDesign], None]] = []

    def _initialize_ai_engines(self) -> None:
        """Initialize AI decision engines."""
        # Active Learning Engine
        active_learning_engine = ActiveLearningDecisionEngine(self.adaptive_manager)
        self.ai_engines["active_learning"] = active_learning_engine

        # Inverse Design Engine
        inverse_design_engine = InverseDesignDecisionEngine()
        self.ai_engines["inverse_design"] = inverse_design_engine

        logger.info(f"Initialized {len(self.ai_engines)} AI decision engines")

    async def start_integration(self) -> None:
        """Start the AI-robot integration system."""
        if self._is_running:
            logger.warning("AI-robot integration already running")
            return

        self._is_running = True

        # Start integration loop
        self._integration_task = asyncio.create_task(self._integration_loop())

        logger.info("Started AI-robot integration system")

    async def stop_integration(self) -> None:
        """Stop the integration system."""
        self._is_running = False

        if self._integration_task:
            self._integration_task.cancel()
            try:
                await self._integration_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped AI-robot integration system")

    async def _integration_loop(self) -> None:
        """Main integration loop coordinating AI decisions with robot execution."""
        while self._is_running:
            try:
                # Check for new AI decisions needed
                decision_requests = await self._identify_decision_needs()

                # Process each decision request
                for request in decision_requests:
                    await self._process_decision_request(request)

                # Monitor active campaigns
                await self._monitor_active_campaigns()

                # Update integration metrics
                self._update_integration_metrics()

                # Brief pause
                await asyncio.sleep(5.0)

            except Exception as e:
                logger.error(f"Error in integration loop: {e}")
                await asyncio.sleep(10.0)

    async def _identify_decision_needs(self) -> List[Dict[str, Any]]:
        """Identify situations where AI decisions are needed."""
        decision_requests = []

        # Check if optimization needs new experiments
        if self.optimizer and hasattr(self.optimizer, "optimization_state"):
            opt_state = self.optimizer.optimization_state

            # Request active learning decision if model uncertainty is high
            recent_results = getattr(self.optimizer, "experiment_results", [])[-5:]
            if recent_results:
                success_rate = sum(1 for r in recent_results if r.success) / len(
                    recent_results
                )
                if success_rate < 0.7:  # Low success rate indicates need for learning
                    decision_requests.append(
                        {
                            "type": "active_learning",
                            "context": {
                                "recent_results": recent_results,
                                "optimization_state": opt_state,
                                "parameter_space": getattr(
                                    self.optimizer.config, "parameter_space", {}
                                ),
                            },
                            "priority": 3,
                        }
                    )

        # Check if new molecular designs are needed
        recent_designs = [
            d
            for d in self.decision_history
            if d.decision_type == AIDecisionType.INVERSE_DESIGN_CANDIDATE
            and (datetime.now(timezone.utc) - d.created_at).total_seconds() < 3600
        ]

        if len(recent_designs) < 2:  # Need more design diversity
            decision_requests.append(
                {
                    "type": "inverse_design",
                    "context": {
                        "target_properties": {
                            "activity": "maximize",
                            "molecular_weight": (200, 500),
                        },
                        "recent_designs": recent_designs,
                    },
                    "priority": 2,
                }
            )

        return decision_requests

    async def _process_decision_request(self, request: Dict[str, Any]) -> None:
        """Process a single decision request."""
        try:
            engine_type = request["type"]
            context = request["context"]

            if engine_type in self.ai_engines:
                engine = self.ai_engines[engine_type]

                # Make AI decision
                start_time = datetime.now(timezone.utc)
                decision = await engine.make_decision(context)
                decision_time = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds()

                # Update communication latency metric
                self.integration_metrics["ai_robot_communication_latency"] = (
                    self.integration_metrics["ai_robot_communication_latency"] * 0.9
                    + decision_time * 0.1
                )

                # Process the decision
                await self._execute_ai_decision(decision)

                # Update history and metrics
                with self._lock:
                    self.decision_history.append(decision)
                    self.integration_metrics["total_ai_decisions"] += 1

                # Notify callbacks
                for callback in self.decision_callbacks:
                    try:
                        callback(decision)
                    except Exception as e:
                        logger.error(f"Error in decision callback: {e}")

                logger.info(f"Processed {engine_type} decision {decision.decision_id}")

        except Exception as e:
            logger.error(f"Error processing decision request: {e}")

    async def _execute_ai_decision(self, decision: AIDecision) -> None:
        """Execute an AI decision by coordinating with robotic systems."""
        try:
            if decision.decision_type == AIDecisionType.ACTIVE_LEARNING_QUERY:
                await self._execute_active_learning_query(decision)
            elif decision.decision_type == AIDecisionType.INVERSE_DESIGN_CANDIDATE:
                await self._execute_inverse_design_synthesis(decision)
            elif decision.decision_type in [
                AIDecisionType.EXPLORATORY_EXPERIMENT,
                AIDecisionType.VALIDATION_EXPERIMENT,
            ]:
                await self._execute_general_experiment(decision)

        except Exception as e:
            logger.error(f"Error executing AI decision {decision.decision_id}: {e}")

    async def _execute_active_learning_query(self, decision: AIDecision) -> None:
        """Execute an active learning query as robotic experiments."""
        parameters = decision.parameters

        # Create experimental design
        design = ExperimentalDesign(
            design_id=f"al_design_{decision.decision_id}",
            synthesis_tasks=[],  # Will be populated
            analysis_protocols=[
                {
                    "protocol": "comprehensive_analysis",
                    "techniques": ["NMR", "HPLC", "MS"],
                    "quality_metrics": ["yield", "purity", "selectivity"],
                }
            ],
            quality_controls=[{"type": "blank_control", "frequency": 0.1}],
            success_criteria={
                "min_yield": 0.3,
                "min_purity": 0.8,
                "max_execution_time": 24 * 3600,  # 24 hours
            },
            risk_assessment={"safety_level": "standard", "environmental_impact": "low"},
            ai_decisions=[decision],
            total_estimated_time=0.0,
            resource_utilization={},
        )

        # Create synthesis task from parameters
        synthesis_task = SynthesisTask(
            task_id=f"al_synthesis_{decision.decision_id}",
            target_molecule={"parameters": parameters},
            synthesis_route=[
                {
                    "step_number": 1,
                    "reaction_type": "parameter_optimization",
                    "conditions": parameters,
                    "estimated_yield": 0.7,
                    "estimated_time": 4.0,
                }
            ],
            complexity=SynthesisComplexity.SIMPLE,
            success_probability=decision.confidence,
            estimated_duration=14400.0,  # 4 hours
            resource_requirements={"reactor": 1, "analytical_instruments": 2},
            safety_considerations=["standard_precautions"],
            quality_metrics={"expected_yield": 0.7, "expected_purity": 0.8},
            ai_decision_id=decision.decision_id,
        )

        design.synthesis_tasks = [synthesis_task]
        design.total_estimated_time = synthesis_task.estimated_duration

        # Execute the design
        await self._execute_experimental_design(design)

    async def _execute_inverse_design_synthesis(self, decision: AIDecision) -> None:
        """Execute inverse design synthesis through robotic platforms."""
        parameters = decision.parameters
        target_molecule = parameters.get("target_molecule", {})
        synthesis_route = parameters.get("synthesis_route", [])

        # Create synthesis task
        synthesis_task = SynthesisTask(
            task_id=f"id_synthesis_{decision.decision_id}",
            target_molecule=target_molecule,
            synthesis_route=synthesis_route,
            complexity=self._assess_synthesis_complexity(synthesis_route),
            success_probability=decision.confidence,
            estimated_duration=sum(
                step.get("estimated_time", 2) * 3600 for step in synthesis_route
            ),
            resource_requirements=self._calculate_resource_requirements(
                synthesis_route
            ),
            safety_considerations=self._assess_safety_considerations(synthesis_route),
            quality_metrics={"target_yield": 0.6, "target_purity": 0.9},
            ai_decision_id=decision.decision_id,
        )

        # Create experimental design
        design = ExperimentalDesign(
            design_id=f"id_design_{decision.decision_id}",
            synthesis_tasks=[synthesis_task],
            analysis_protocols=[
                {
                    "protocol": "structure_confirmation",
                    "techniques": ["NMR", "MS", "IR"],
                    "target_structure": target_molecule.get("smiles", ""),
                }
            ],
            quality_controls=[{"type": "reference_standard", "frequency": 1.0}],
            success_criteria={
                "structure_match": True,
                "min_yield": 0.3,
                "min_purity": 0.7,
            },
            risk_assessment=self._assess_design_risk(synthesis_route),
            ai_decisions=[decision],
            total_estimated_time=synthesis_task.estimated_duration,
            resource_utilization=synthesis_task.resource_requirements,
        )

        # Execute the design
        await self._execute_experimental_design(design)

    async def _execute_general_experiment(self, decision: AIDecision) -> None:
        """Execute general experimental decision."""
        # Create simple experimental task
        parameters = decision.parameters

        synthesis_task = SynthesisTask(
            task_id=f"gen_synthesis_{decision.decision_id}",
            target_molecule={"type": "exploratory"},
            synthesis_route=[
                {
                    "step_number": 1,
                    "reaction_type": "general_synthesis",
                    "conditions": parameters,
                    "estimated_yield": 0.5,
                    "estimated_time": 6.0,
                }
            ],
            complexity=SynthesisComplexity.MODERATE,
            success_probability=decision.confidence,
            estimated_duration=21600.0,  # 6 hours
            resource_requirements={"reactor": 1, "workup": 1},
            safety_considerations=["standard_precautions"],
            quality_metrics={"expected_outcome": "exploratory_data"},
            ai_decision_id=decision.decision_id,
        )

        design = ExperimentalDesign(
            design_id=f"gen_design_{decision.decision_id}",
            synthesis_tasks=[synthesis_task],
            analysis_protocols=[{"protocol": "basic_analysis"}],
            quality_controls=[],
            success_criteria={"data_quality": "acceptable"},
            risk_assessment={"safety_level": "standard"},
            ai_decisions=[decision],
            total_estimated_time=synthesis_task.estimated_duration,
            resource_utilization=synthesis_task.resource_requirements,
        )

        await self._execute_experimental_design(design)

    async def _execute_experimental_design(self, design: ExperimentalDesign) -> None:
        """Execute an experimental design through the robot manager."""
        try:
            # Convert design to robot experiment plans
            experiment_plans = await self._convert_design_to_robot_plans(design)

            # Execute plans
            for plan in experiment_plans:
                plan_id = await self.robot_manager.execute_experiment_plan(plan)
                logger.info(
                    f"Executed experiment plan {plan_id} for design {design.design_id}"
                )

            # Track active design
            with self._lock:
                self.experimental_designs[design.design_id] = design
                for task in design.synthesis_tasks:
                    self.synthesis_tasks[task.task_id] = task

            # Notify callbacks
            for callback in self.design_callbacks:
                try:
                    callback(design)
                except Exception as e:
                    logger.error(f"Error in design callback: {e}")

        except Exception as e:
            logger.error(f"Error executing experimental design {design.design_id}: {e}")

    async def _convert_design_to_robot_plans(
        self, design: ExperimentalDesign
    ) -> List[ExperimentPlan]:
        """Convert experimental design to robot experiment plans."""
        plans = []

        for i, synthesis_task in enumerate(design.synthesis_tasks):
            plan_id = f"{design.design_id}_plan_{i}"
            experiment_id = f"{design.design_id}_exp_{i}"

            # Create actions for synthesis steps
            actions = []
            for j, step in enumerate(synthesis_task.synthesis_route):
                # Find suitable robot for this step
                suitable_robots = self.robot_manager.find_robots_with_capability(
                    "chemical_synthesis"
                )

                if suitable_robots:
                    action = ExperimentalAction(
                        action_id=f"{experiment_id}_step_{j}",
                        robot_id=suitable_robots[0],
                        capability_id="chemical_synthesis",
                        parameters={
                            "reaction_type": step.get("reaction_type", "general"),
                            "conditions": step.get("conditions", {}),
                            "reactants": step.get("reactants", []),
                            "expected_yield": step.get("estimated_yield", 0.5),
                        },
                        priority=3,
                        expected_duration=step.get("estimated_time", 2) * 3600,
                    )
                    actions.append(action)

            # Add analysis actions
            analysis_robots = self.robot_manager.find_robots_with_capability("analysis")
            if analysis_robots:
                for k, protocol in enumerate(design.analysis_protocols):
                    analysis_action = ExperimentalAction(
                        action_id=f"{experiment_id}_analysis_{k}",
                        robot_id=analysis_robots[0],
                        capability_id="analysis",
                        parameters={
                            "protocol": protocol.get("protocol", "standard"),
                            "techniques": protocol.get("techniques", []),
                            "target": synthesis_task.target_molecule,
                        },
                        priority=2,
                        dependencies=[
                            f"{experiment_id}_step_{j}"
                            for j in range(len(synthesis_task.synthesis_route))
                        ],
                        expected_duration=1800.0,  # 30 minutes
                    )
                    actions.append(analysis_action)

            # Create experiment plan
            plan = ExperimentPlan(
                plan_id=plan_id,
                experiment_id=experiment_id,
                actions=actions,
                optimization_objective=(
                    getattr(self.optimizer.config, "objective", None)
                    if self.optimizer
                    else None
                ),
                target_parameters=synthesis_task.target_molecule,
                constraints=design.success_criteria,
                metadata={
                    "design_id": design.design_id,
                    "synthesis_task_id": synthesis_task.task_id,
                    "ai_driven": True,
                },
            )

            plans.append(plan)

        return plans

    def _assess_synthesis_complexity(
        self, synthesis_route: List[Dict[str, Any]]
    ) -> SynthesisComplexity:
        """Assess complexity of synthesis route."""
        num_steps = len(synthesis_route)

        if num_steps == 1:
            return SynthesisComplexity.SIMPLE
        elif num_steps == 2:
            return SynthesisComplexity.MODERATE
        elif num_steps <= 4:
            return SynthesisComplexity.COMPLEX
        else:
            return SynthesisComplexity.NOVEL

    def _calculate_resource_requirements(
        self, synthesis_route: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate resource requirements for synthesis route."""
        requirements = defaultdict(int)

        for step in synthesis_route:
            requirements["reactor"] += 1
            requirements["analytical_time"] += 0.5

            # Special requirements based on reaction type
            reaction_type = step.get("reaction_type", "")
            if "coupling" in reaction_type.lower():
                requirements["inert_atmosphere"] += 1
            if "reduction" in reaction_type.lower():
                requirements["hydrogen_source"] += 1

        return dict(requirements)

    def _assess_safety_considerations(
        self, synthesis_route: List[Dict[str, Any]]
    ) -> List[str]:
        """Assess safety considerations for synthesis route."""
        safety_considerations = []

        for step in synthesis_route:
            conditions = step.get("conditions", {})

            # Temperature-based safety
            temp = conditions.get("temperature", 25)
            if temp > 100:
                safety_considerations.append("high_temperature_handling")

            # Pressure-based safety
            pressure = conditions.get("pressure", 1)
            if pressure > 2:
                safety_considerations.append("pressure_vessel_required")

            # Reagent-based safety
            reactants = step.get("reactants", [])
            for reactant in reactants:
                if any(
                    hazard in reactant.lower()
                    for hazard in ["lithium", "sodium", "potassium"]
                ):
                    safety_considerations.append("handle_alkali_metals")

        return list(set(safety_considerations))  # Remove duplicates

    def _assess_design_risk(
        self, synthesis_route: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess risk for experimental design."""
        num_steps = len(synthesis_route)

        if num_steps <= 2:
            risk_level = "low"
        elif num_steps <= 4:
            risk_level = "medium"
        else:
            risk_level = "high"

        return {
            "safety_level": risk_level,
            "complexity_risk": risk_level,
            "resource_intensity": "medium" if num_steps > 2 else "low",
        }

    async def _monitor_active_campaigns(self) -> None:
        """Monitor active experimental campaigns."""
        # Check for completed designs
        completed_designs = []

        with self._lock:
            for design_id, design in self.experimental_designs.items():
                # Check if all synthesis tasks are complete (simplified check)
                all_complete = True  # In practice, would check robot feedback

                if all_complete:
                    completed_designs.append(design_id)

        # Process completed designs
        for design_id in completed_designs:
            await self._process_completed_design(design_id)

    async def _process_completed_design(self, design_id: str) -> None:
        """Process a completed experimental design."""
        try:
            with self._lock:
                design = self.experimental_designs.pop(design_id, None)

            if design:
                # Update metrics
                for task in design.synthesis_tasks:
                    if (
                        task.success_probability > 0.5
                    ):  # Assume success if high probability
                        self.integration_metrics["successful_syntheses"] += 1
                    else:
                        self.integration_metrics["failed_syntheses"] += 1

                # Update average synthesis time
                total_time = design.total_estimated_time
                current_avg = self.integration_metrics["average_synthesis_time"]
                total_syntheses = (
                    self.integration_metrics["successful_syntheses"]
                    + self.integration_metrics["failed_syntheses"]
                )

                if total_syntheses > 0:
                    self.integration_metrics["average_synthesis_time"] = (
                        current_avg * (total_syntheses - len(design.synthesis_tasks))
                        + total_time
                    ) / total_syntheses

                logger.info(f"Completed experimental design {design_id}")

        except Exception as e:
            logger.error(f"Error processing completed design {design_id}: {e}")

    def _update_integration_metrics(self) -> None:
        """Update integration performance metrics."""
        # Metrics are updated in real-time throughout the system
        pass

    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status."""
        with self._lock:
            return {
                "is_running": self._is_running,
                "ai_engines": {
                    name: engine.get_engine_info()
                    for name, engine in self.ai_engines.items()
                },
                "performance_metrics": self.integration_metrics.copy(),
                "active_designs": len(self.experimental_designs),
                "active_synthesis_tasks": len(self.synthesis_tasks),
                "decision_history_size": len(self.decision_history),
                "recent_decisions": [
                    {
                        "decision_id": d.decision_id,
                        "type": d.decision_type.value,
                        "confidence": d.confidence,
                        "timestamp": d.created_at.isoformat(),
                    }
                    for d in self.decision_history[-5:]
                ],
            }

    def add_decision_callback(self, callback: Callable[[AIDecision], None]) -> None:
        """Add callback for AI decisions."""
        self.decision_callbacks.append(callback)

    def add_synthesis_callback(self, callback: Callable[[SynthesisTask], None]) -> None:
        """Add callback for synthesis tasks."""
        self.synthesis_callbacks.append(callback)

    def add_design_callback(
        self, callback: Callable[[ExperimentalDesign], None]
    ) -> None:
        """Add callback for experimental designs."""
        self.design_callbacks.append(callback)


# Factory functions


def create_comprehensive_ai_robot_system(
    robot_manager: RobotManager,
    adaptive_learning_manager: AdaptiveLearningManager,
    closed_loop_optimizer: Optional[ClosedLoopOptimizer] = None,
) -> AIRobotIntegrationManager:
    """Create comprehensive AI-robot integration system."""
    integration_manager = AIRobotIntegrationManager(
        robot_manager, adaptive_learning_manager, closed_loop_optimizer
    )

    return integration_manager


def create_active_learning_robot_system(
    robot_manager: RobotManager, adaptive_learning_manager: AdaptiveLearningManager
) -> AIRobotIntegrationManager:
    """Create AI-robot system focused on active learning."""
    integration_manager = AIRobotIntegrationManager(
        robot_manager, adaptive_learning_manager, None
    )

    # Configure for active learning focus
    al_engine = integration_manager.ai_engines["active_learning"]
    al_engine.performance_metrics["query_frequency"] = "high"

    return integration_manager


def create_inverse_design_robot_system(
    robot_manager: RobotManager, adaptive_learning_manager: AdaptiveLearningManager
) -> AIRobotIntegrationManager:
    """Create AI-robot system focused on inverse design."""
    integration_manager = AIRobotIntegrationManager(
        robot_manager, adaptive_learning_manager, None
    )

    # Configure for inverse design focus
    id_engine = integration_manager.ai_engines["inverse_design"]
    id_engine.performance_metrics["design_novelty_weight"] = 0.8

    return integration_manager
