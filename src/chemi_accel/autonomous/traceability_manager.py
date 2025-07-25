"""
Traceability Manager for Autonomous Laboratory Operations

This module coordinates comprehensive traceability across all autonomous lab
operations, providing end-to-end tracking from molecule selection through
experimental results and AI decision feedback loops.
"""

import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from .adaptive_learning import AdaptiveLearningManager
from .closed_loop_optimization import ClosedLoopOptimizer, ExperimentResult
from .decision_logging import (
    DecisionContext,
    DecisionLogger,
    DecisionType,
    ExperimentalDesignDecision,
    ExperimentLifecycleRecord,
    ExperimentPhase,
    InfluenceType,
    MoleculeSelectionDecision,
    SynthesisRouteDecision,
)
from .robotic_control import ExperimentStatus, RobotFeedback, RobotManager


@dataclass
class TraceabilityConfig:
    """Configuration for traceability system."""

    enable_real_time_logging: bool = True
    enable_influence_tracking: bool = True
    enable_performance_monitoring: bool = True
    storage_backend: str = "sqlite"
    storage_config: Dict[str, Any] = field(default_factory=dict)
    audit_retention_days: int = 365
    automatic_correlation_threshold: float = 0.7
    decision_context_depth: int = 3


@dataclass
class TraceabilityMetrics:
    """Metrics for traceability system performance."""

    total_decisions_logged: int = 0
    total_experiments_tracked: int = 0
    total_influence_relationships: int = 0
    average_decision_chain_length: float = 0.0
    decision_types_distribution: Dict[str, int] = field(default_factory=dict)
    experiment_success_rate: float = 0.0
    logging_latency_ms: float = 0.0
    storage_utilization_mb: float = 0.0


class TraceabilityManager:
    """Comprehensive traceability management system."""

    def __init__(
        self,
        config: TraceabilityConfig,
        robot_manager: Optional[RobotManager] = None,
        optimizer: Optional[ClosedLoopOptimizer] = None,
        adaptive_manager: Optional[AdaptiveLearningManager] = None,
    ):
        self.config = config
        self.robot_manager = robot_manager
        self.optimizer = optimizer
        self.adaptive_manager = adaptive_manager

        # Initialize decision logger
        self.decision_logger = DecisionLogger(
            storage_backend=config.storage_backend, storage_config=config.storage_config
        )

        # Internal state
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.molecule_histories: Dict[str, List[str]] = defaultdict(
            list
        )  # SMILES -> decision_ids
        self.decision_correlations: Dict[str, Set[str]] = defaultdict(set)
        self.performance_metrics = TraceabilityMetrics()
        self._correlation_cache: Dict[str, Dict[str, float]] = {}

        # Setup system integrations
        self._setup_integrations()

    def _setup_integrations(self):
        """Setup integrations with other autonomous lab systems."""
        if self.robot_manager:
            self.robot_manager.add_feedback_callback(self._handle_robot_feedback)

        if self.optimizer:
            self.optimizer.add_experiment_callback(self._handle_optimization_result)
            self.optimizer.add_optimization_callback(
                self._handle_optimization_state_change
            )

        if self.adaptive_manager:
            # Add callback for model retraining events
            self.adaptive_manager.add_retraining_callback(self._handle_model_retraining)

    async def start_experiment_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        objectives: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new experiment session with full traceability."""
        session = {
            "session_id": session_id,
            "user_id": user_id,
            "start_time": datetime.now(timezone.utc),
            "objectives": objectives or {},
            "constraints": constraints or {},
            "experiments": [],
            "decisions": [],
            "active": True,
        }

        self.active_sessions[session_id] = session

        # Log session start as a decision
        context = DecisionContext(
            model_version="session_manager_v1.0",
            model_type="session_management",
            confidence_score=1.0,
            input_data_hash=self._hash_data(session),
            computational_resources={"cpu": "low", "memory": "low"},
            hyperparameters={},
        )

        decision_id = await self.decision_logger.log_decision(
            decision_type=DecisionType.EXPERIMENTAL_DESIGN,
            context=context,
            decision_data={"session_start": session},
            user_id=user_id,
            session_id=session_id,
            metadata={"event_type": "session_start"},
        )

        session["decisions"].append(decision_id)
        return decision_id

    async def log_molecule_selection(
        self,
        selected_molecules: List[str],
        selection_criteria: Dict[str, Any],
        candidate_pool: List[str],
        predicted_properties: Dict[str, Dict[str, float]],
        model_info: Dict[str, Any],
        session_id: str,
        user_id: Optional[str] = None,
        parent_decision_id: Optional[str] = None,
    ) -> str:
        """Log a molecule selection decision with full context."""
        # Create decision context
        context = DecisionContext(
            model_version=model_info.get("version", "unknown"),
            model_type=model_info.get("type", "molecule_selector"),
            confidence_score=model_info.get("confidence", 0.0),
            input_data_hash=self._hash_data(candidate_pool),
            computational_resources=model_info.get("resources", {}),
            hyperparameters=model_info.get("hyperparameters", {}),
            feature_importance=model_info.get("feature_importance"),
            uncertainty_metrics=model_info.get("uncertainty_metrics"),
        )

        # Create detailed decision data
        decision_data = MoleculeSelectionDecision(
            selected_molecules=selected_molecules,
            candidate_pool_size=len(candidate_pool),
            selection_criteria=selection_criteria,
            diversity_metrics=self._calculate_diversity_metrics(
                selected_molecules, candidate_pool
            ),
            predicted_properties=predicted_properties,
            exclusion_reasons=self._analyze_exclusions(
                candidate_pool, selected_molecules, selection_criteria
            ),
            similarity_analysis=self._analyze_molecular_similarity(selected_molecules),
        )

        # Log the decision
        decision_id = await self.decision_logger.log_decision(
            decision_type=DecisionType.MOLECULE_SELECTION,
            context=context,
            decision_data=decision_data,
            user_id=user_id,
            session_id=session_id,
            parent_decision_id=parent_decision_id,
            metadata={"candidate_pool_size": len(candidate_pool)},
        )

        # Update molecule histories
        for molecule in selected_molecules:
            self.molecule_histories[molecule].append(decision_id)

        # Update session
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["decisions"].append(decision_id)

        # Update metrics
        self.performance_metrics.total_decisions_logged += 1
        self.performance_metrics.decision_types_distribution["molecule_selection"] = (
            self.performance_metrics.decision_types_distribution.get(
                "molecule_selection", 0
            )
            + 1
        )

        return decision_id

    async def log_synthesis_planning(
        self,
        target_molecule: str,
        selected_route: Dict[str, Any],
        alternative_routes: List[Dict[str, Any]],
        model_info: Dict[str, Any],
        session_id: str,
        experiment_id: Optional[str] = None,
        user_id: Optional[str] = None,
        parent_decision_id: Optional[str] = None,
    ) -> str:
        """Log synthesis route planning decision."""
        context = DecisionContext(
            model_version=model_info.get("version", "unknown"),
            model_type=model_info.get("type", "synthesis_planner"),
            confidence_score=model_info.get("confidence", 0.0),
            input_data_hash=self._hash_data(target_molecule),
            computational_resources=model_info.get("resources", {}),
            hyperparameters=model_info.get("hyperparameters", {}),
        )

        decision_data = SynthesisRouteDecision(
            target_molecule=target_molecule,
            selected_route=selected_route,
            alternative_routes=alternative_routes,
            route_scoring=model_info.get("route_scores", {}),
            reagent_availability=self._check_reagent_availability(selected_route),
            predicted_yield=selected_route.get("predicted_yield", 0.0),
            predicted_difficulty=selected_route.get("predicted_difficulty", 0.0),
            cost_analysis=self._estimate_synthesis_cost(selected_route),
        )

        decision_id = await self.decision_logger.log_decision(
            decision_type=DecisionType.SYNTHESIS_ROUTE,
            context=context,
            decision_data=decision_data,
            user_id=user_id,
            session_id=session_id,
            experiment_id=experiment_id,
            parent_decision_id=parent_decision_id,
            metadata={"num_alternative_routes": len(alternative_routes)},
        )

        # Update molecule history
        self.molecule_histories[target_molecule].append(decision_id)

        # Create influence relationship with molecule selection if applicable
        if parent_decision_id:
            await self.decision_logger.log_influence_relationship(
                source_decision_id=parent_decision_id,
                target_decision_id=decision_id,
                influence_type=InfluenceType.DIRECT_CONSEQUENCE,
                influence_strength=1.0,
                explanation="Synthesis route planning directly follows molecule selection",
                quantitative_metrics={
                    "confidence_correlation": model_info.get("confidence", 0.0)
                },
            )

        return decision_id

    async def log_experimental_design(
        self,
        design_parameters: Dict[str, Any],
        experimental_conditions: Dict[str, Any],
        resource_allocation: Dict[str, Any],
        model_info: Dict[str, Any],
        session_id: str,
        experiment_id: Optional[str] = None,
        user_id: Optional[str] = None,
        parent_decision_id: Optional[str] = None,
    ) -> str:
        """Log experimental design decisions."""
        context = DecisionContext(
            model_version=model_info.get("version", "unknown"),
            model_type=model_info.get("type", "experimental_designer"),
            confidence_score=model_info.get("confidence", 0.0),
            input_data_hash=self._hash_data(design_parameters),
            computational_resources=model_info.get("resources", {}),
            hyperparameters=model_info.get("hyperparameters", {}),
        )

        decision_data = ExperimentalDesignDecision(
            design_type=design_parameters.get("design_type", "unknown"),
            parameter_space=design_parameters.get("parameter_space", {}),
            selected_conditions=experimental_conditions,
            doe_strategy=design_parameters.get("doe_strategy", "random"),
            expected_outcomes=design_parameters.get("expected_outcomes", {}),
            resource_requirements=resource_allocation,
            time_estimation=design_parameters.get("estimated_time", 0.0),
            risk_assessment=design_parameters.get("risk_assessment"),
        )

        decision_id = await self.decision_logger.log_decision(
            decision_type=DecisionType.EXPERIMENTAL_DESIGN,
            context=context,
            decision_data=decision_data,
            user_id=user_id,
            session_id=session_id,
            experiment_id=experiment_id,
            parent_decision_id=parent_decision_id,
        )

        return decision_id

    async def create_experiment_trace(
        self,
        molecule_smiles: str,
        target_properties: Dict[str, float],
        synthesis_route: Dict[str, Any],
        experimental_conditions: Dict[str, Any],
        session_id: str,
        related_decision_ids: List[str],
        experiment_id: Optional[str] = None,
    ) -> str:
        """Create a comprehensive experiment trace."""
        # Create experiment record
        experiment_id = await self.decision_logger.create_experiment_record(
            molecule_smiles=molecule_smiles,
            target_properties=target_properties,
            synthesis_route=synthesis_route,
            experimental_conditions=experimental_conditions,
            experiment_id=experiment_id,
        )

        # Add decisions to experiment chain
        for decision_id in related_decision_ids:
            await self.decision_logger.add_decision_to_experiment(
                experiment_id, decision_id
            )

        # Update session
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["experiments"].append(experiment_id)

        # Update metrics
        self.performance_metrics.total_experiments_tracked += 1

        return experiment_id

    async def update_experiment_results(
        self,
        experiment_id: str,
        results: Dict[str, float],
        quality_metrics: Dict[str, float],
        success: bool,
        phase: ExperimentPhase = ExperimentPhase.COMPLETED,
        robot_feedback: Optional[List[RobotFeedback]] = None,
    ):
        """Update experiment with results and analyze influence on future decisions."""
        # Update experiment record
        await self.decision_logger.update_experiment_phase(
            experiment_id=experiment_id,
            new_phase=phase,
            results=results,
            quality_metrics=quality_metrics,
            description=f"Experiment {'completed successfully' if success else 'failed'}",
        )

        # Analyze and log influences on future decisions
        await self._analyze_result_influences(
            experiment_id, results, quality_metrics, success
        )

        # Update success rate metrics
        total_experiments = self.performance_metrics.total_experiments_tracked
        current_success_rate = self.performance_metrics.experiment_success_rate
        new_success_rate = (
            (current_success_rate * (total_experiments - 1)) + (1.0 if success else 0.0)
        ) / total_experiments
        self.performance_metrics.experiment_success_rate = new_success_rate

    async def _analyze_result_influences(
        self,
        experiment_id: str,
        results: Dict[str, float],
        quality_metrics: Dict[str, float],
        success: bool,
    ):
        """Analyze how experiment results might influence future decisions."""
        experiment = self.decision_logger.experiment_records.get(experiment_id)
        if not experiment:
            return

        # Find decisions that might be influenced by these results
        molecule_smiles = experiment.molecule_smiles
        similar_experiments = []

        for other_exp_id, other_exp in self.decision_logger.experiment_records.items():
            if (
                other_exp_id != experiment_id
                and other_exp.molecule_smiles == molecule_smiles
            ):
                similarity = self._calculate_experimental_similarity(
                    experiment, other_exp
                )
                if similarity > self.config.automatic_correlation_threshold:
                    similar_experiments.append((other_exp_id, similarity))

        # Log influence relationships for similar experiments
        for similar_exp_id, similarity in similar_experiments:
            similar_exp = self.decision_logger.experiment_records[similar_exp_id]

            # Find most recent decision in similar experiment
            if similar_exp.decision_chain:
                recent_decision_id = similar_exp.decision_chain[-1]

                await self.decision_logger.log_influence_relationship(
                    source_decision_id=experiment.decision_chain[-1]
                    if experiment.decision_chain
                    else "",
                    target_decision_id=recent_decision_id,
                    influence_type=InfluenceType.FEEDBACK_LOOP,
                    influence_strength=similarity,
                    explanation=f"Results from experiment {experiment_id} inform similar experiment {similar_exp_id}",
                    quantitative_metrics={
                        "similarity_score": similarity,
                        "success_influence": 1.0 if success else -0.5,
                        "result_correlation": self._calculate_result_correlation(
                            results, similar_exp.actual_results
                        ),
                    },
                )

    def _handle_robot_feedback(self, feedback: RobotFeedback):
        """Handle robot feedback for traceability."""
        asyncio.create_task(self._process_robot_feedback(feedback))

    async def _process_robot_feedback(self, feedback: RobotFeedback):
        """Process robot feedback asynchronously."""
        # Find associated experiment
        experiment_id = None
        for exp_id, exp in self.decision_logger.experiment_records.items():
            if any(
                decision.get("robot_id") == feedback.robot_id
                for decision in exp.metadata.get("robot_decisions", [])
            ):
                experiment_id = exp_id
                break

        if experiment_id and feedback.status == ExperimentStatus.COMPLETED:
            # Extract results from feedback
            results = feedback.data.get("objective_values", {})
            quality_metrics = feedback.data.get("quality_metrics", {})

            await self.update_experiment_results(
                experiment_id=experiment_id,
                results=results,
                quality_metrics=quality_metrics,
                success=True,
                robot_feedback=[feedback],
            )

    def _handle_optimization_result(self, result: ExperimentResult):
        """Handle optimization results for traceability."""
        asyncio.create_task(self._process_optimization_result(result))

    async def _process_optimization_result(self, result: ExperimentResult):
        """Process optimization results asynchronously."""
        # Log optimization decision
        context = DecisionContext(
            model_version="closed_loop_optimizer_v1.0",
            model_type="bayesian_optimization",
            confidence_score=0.8,  # Default confidence
            input_data_hash=self._hash_data(result.parameters),
            computational_resources={"optimization": "high"},
            hyperparameters={},
        )

        decision_id = await self.decision_logger.log_decision(
            decision_type=DecisionType.PARAMETER_OPTIMIZATION,
            context=context,
            decision_data={
                "optimization_result": {
                    "parameters": result.parameters,
                    "objective_values": result.objective_values,
                    "success": result.success,
                    "execution_time": result.execution_time,
                }
            },
            experiment_id=result.experiment_id,
            metadata={"optimization_iteration": True},
        )

        # Update experiment with optimization decision
        if result.experiment_id:
            await self.decision_logger.add_decision_to_experiment(
                result.experiment_id, decision_id
            )

    def _handle_optimization_state_change(self, state):
        """Handle optimization state changes."""
        # Log state change as metadata
        pass

    def _handle_model_retraining(self, retraining_info: Dict[str, Any]):
        """Handle model retraining events for traceability."""
        asyncio.create_task(self._process_model_retraining(retraining_info))

    async def _process_model_retraining(self, retraining_info: Dict[str, Any]):
        """Process model retraining events."""
        context = DecisionContext(
            model_version=retraining_info.get("new_version", "unknown"),
            model_type="adaptive_learning",
            confidence_score=retraining_info.get("improvement_score", 0.0),
            input_data_hash=self._hash_data(retraining_info),
            computational_resources=retraining_info.get("resources", {}),
            hyperparameters=retraining_info.get("hyperparameters", {}),
        )

        await self.decision_logger.log_decision(
            decision_type=DecisionType.QUALITY_ASSESSMENT,
            context=context,
            decision_data={
                "retraining_event": retraining_info,
                "model_performance": retraining_info.get("performance_metrics", {}),
                "data_used": retraining_info.get("training_data_summary", {}),
            },
            metadata={"event_type": "model_retraining"},
        )

    # Helper methods
    def _hash_data(self, data: Any) -> str:
        """Create hash of data for consistency checking."""
        import hashlib

        return hashlib.md5(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()

    def _calculate_diversity_metrics(
        self, selected: List[str], candidate_pool: List[str]
    ) -> Dict[str, float]:
        """Calculate diversity metrics for molecule selection."""
        # Simplified diversity calculation
        return {
            "selection_diversity": len(set(selected)) / len(selected)
            if selected
            else 0.0,
            "coverage_ratio": len(selected) / len(candidate_pool)
            if candidate_pool
            else 0.0,
            "uniqueness_score": 1.0,  # Placeholder
        }

    def _analyze_exclusions(
        self, candidate_pool: List[str], selected: List[str], criteria: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Analyze why molecules were excluded."""
        excluded = [mol for mol in candidate_pool if mol not in selected]
        return {
            "filtered_out": excluded[:10],  # First 10 excluded
            "criteria_failed": ["placeholder_criterion"] * min(10, len(excluded)),
        }

    def _analyze_molecular_similarity(self, molecules: List[str]) -> Dict[str, Any]:
        """Analyze molecular similarity within selection."""
        return {
            "avg_similarity": 0.7,  # Placeholder
            "similarity_matrix": "computed",
            "cluster_count": min(3, len(molecules)),
        }

    def _check_reagent_availability(self, route: Dict[str, Any]) -> Dict[str, bool]:
        """Check availability of reagents for synthesis route."""
        reagents = route.get("reagents", [])
        return dict.fromkeys(reagents, True)  # Placeholder

    def _estimate_synthesis_cost(self, route: Dict[str, Any]) -> Dict[str, float]:
        """Estimate cost of synthesis route."""
        return {
            "material_cost": 100.0,  # Placeholder
            "time_cost": 50.0,
            "total_estimated_cost": 150.0,
        }

    def _calculate_experimental_similarity(
        self, exp1: ExperimentLifecycleRecord, exp2: ExperimentLifecycleRecord
    ) -> float:
        """Calculate similarity between experiments."""
        # Simplified similarity based on conditions
        conditions1 = exp1.experimental_conditions
        conditions2 = exp2.experimental_conditions

        common_keys = set(conditions1.keys()) & set(conditions2.keys())
        if not common_keys:
            return 0.0

        similarity_sum = 0.0
        for key in common_keys:
            try:
                val1, val2 = float(conditions1[key]), float(conditions2[key])
                similarity_sum += 1.0 - abs(val1 - val2) / max(
                    abs(val1), abs(val2), 1.0
                )
            except (ValueError, TypeError):
                similarity_sum += 1.0 if conditions1[key] == conditions2[key] else 0.0

        return similarity_sum / len(common_keys)

    def _calculate_result_correlation(
        self, results1: Dict[str, float], results2: Dict[str, float]
    ) -> float:
        """Calculate correlation between experimental results."""
        common_keys = set(results1.keys()) & set(results2.keys())
        if not common_keys:
            return 0.0

        # Simple correlation calculation
        correlation_sum = 0.0
        for key in common_keys:
            try:
                val1, val2 = results1[key], results2[key]
                correlation_sum += 1.0 - abs(val1 - val2) / max(
                    abs(val1), abs(val2), 1.0
                )
            except (ValueError, TypeError, ZeroDivisionError):
                correlation_sum += 0.0

        return correlation_sum / len(common_keys)

    async def get_traceability_summary(self) -> Dict[str, Any]:
        """Get comprehensive traceability summary."""
        total_decisions = len(self.decision_logger.decision_records)
        total_experiments = len(self.decision_logger.experiment_records)
        total_relationships = len(self.decision_logger.influence_relationships)

        # Calculate average decision chain length
        chain_lengths = [
            len(exp.decision_chain)
            for exp in self.decision_logger.experiment_records.values()
        ]
        avg_chain_length = (
            sum(chain_lengths) / len(chain_lengths) if chain_lengths else 0.0
        )

        return {
            "system_status": {
                "total_decisions_logged": total_decisions,
                "total_experiments_tracked": total_experiments,
                "total_influence_relationships": total_relationships,
                "average_decision_chain_length": avg_chain_length,
                "active_sessions": len(self.active_sessions),
            },
            "performance_metrics": {
                "decision_types_distribution": dict(
                    self.performance_metrics.decision_types_distribution
                ),
                "experiment_success_rate": self.performance_metrics.experiment_success_rate,
                "logging_performance": {
                    "average_latency_ms": self.performance_metrics.logging_latency_ms,
                    "storage_utilization_mb": self.performance_metrics.storage_utilization_mb,
                },
            },
            "traceability_coverage": {
                "molecules_with_history": len(self.molecule_histories),
                "decision_correlations": len(self.decision_correlations),
                "influence_networks": self._analyze_influence_networks(),
            },
        }

    def _analyze_influence_networks(self) -> Dict[str, Any]:
        """Analyze influence networks for summary."""
        relationships = self.decision_logger.influence_relationships.values()

        # Count influence types
        influence_types = defaultdict(int)
        for rel in relationships:
            influence_types[rel.influence_type.value] += 1

        # Calculate network density
        unique_decisions = set()
        for rel in relationships:
            unique_decisions.add(rel.source_decision_id)
            unique_decisions.add(rel.target_decision_id)

        network_density = (
            len(relationships) / (len(unique_decisions) ** 2)
            if unique_decisions
            else 0.0
        )

        return {
            "influence_type_distribution": dict(influence_types),
            "network_density": network_density,
            "connected_decisions": len(unique_decisions),
        }
