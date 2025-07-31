"""
Influence Analyzer for AI Decision Correlations

This module provides advanced analysis of how experimental results influence
subsequent AI decisions, enabling comprehensive understanding of the feedback
loops in autonomous laboratory operations.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .decision_logging import (
    DecisionLogger,
    DecisionRecord,
    DecisionType,
    ExperimentLifecycleRecord,
)


class InfluenceStrength(Enum):
    """Quantified influence strength levels."""

    NEGLIGIBLE = "negligible"  # < 0.1
    WEAK = "weak"  # 0.1 - 0.3
    MODERATE = "moderate"  # 0.3 - 0.6
    STRONG = "strong"  # 0.6 - 0.8
    VERY_STRONG = "very_strong"  # > 0.8


class CorrelationType(Enum):
    """Types of correlations between results and decisions."""

    POSITIVE_REINFORCEMENT = "positive_reinforcement"
    NEGATIVE_FEEDBACK = "negative_feedback"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    STRATEGY_SHIFT = "strategy_shift"
    EXPLORATION_EXPANSION = "exploration_expansion"
    EXPLOITATION_FOCUS = "exploitation_focus"


@dataclass
class InfluenceMetric:
    """Quantified influence metrics."""

    influence_strength: float
    confidence_level: float
    temporal_proximity: float
    semantic_similarity: float
    outcome_correlation: float
    frequency_weight: float
    statistical_significance: float


@dataclass
class ResultCorrelation:
    """Correlation between experimental results and subsequent decisions."""

    correlation_id: str
    source_experiment_id: str
    target_decision_id: str
    correlation_type: CorrelationType
    influence_metrics: InfluenceMetric
    result_properties: Dict[str, float]
    decision_changes: Dict[str, Any]
    temporal_gap: float  # hours
    explanation: str
    supporting_evidence: List[Dict[str, Any]]
    timestamp: datetime


@dataclass
class InfluenceNetwork:
    """Network representation of influence relationships."""

    network_id: str
    nodes: Dict[str, Dict[str, Any]]  # decision/experiment nodes
    edges: List[Dict[str, Any]]  # influence relationships
    centrality_metrics: Dict[str, float]
    clusters: List[List[str]]
    influence_paths: List[List[str]]
    network_metrics: Dict[str, float]


@dataclass
class FeedbackLoop:
    """Identified feedback loop in the decision-making process."""

    loop_id: str
    participating_decisions: List[str]
    participating_experiments: List[str]
    loop_type: str  # "reinforcing" or "balancing"
    loop_strength: float
    iteration_count: int
    convergence_behavior: str
    identified_patterns: List[str]
    optimization_impact: Dict[str, float]


class InfluenceAnalyzer:
    """Advanced influence analysis system."""

    def __init__(self, decision_logger: DecisionLogger):
        self.decision_logger = decision_logger
        self.influence_history: List[ResultCorrelation] = []
        self.influence_networks: Dict[str, InfluenceNetwork] = {}
        self.feedback_loops: Dict[str, FeedbackLoop] = {}

        # Analysis parameters
        self.correlation_window_hours = 168  # 1 week
        self.min_influence_threshold = 0.1
        self.similarity_threshold = 0.3
        self.significance_threshold = 0.05

        # Caching for performance
        self._correlation_cache: Dict[str, List[ResultCorrelation]] = {}
        self._network_cache: Dict[str, InfluenceNetwork] = {}
        self._analysis_timestamps: Dict[str, datetime] = {}

    async def analyze_result_influence(
        self, experiment_id: str, time_window_hours: Optional[float] = None
    ) -> List[ResultCorrelation]:
        """Analyze how an experiment's results influenced subsequent decisions."""
        if time_window_hours is None:
            time_window_hours = self.correlation_window_hours

        experiment = self.decision_logger.experiment_records.get(experiment_id)
        if not experiment or not experiment.actual_results:
            return []

        # Check cache
        cache_key = f"{experiment_id}_{time_window_hours}"
        if cache_key in self._correlation_cache:
            cache_time = self._analysis_timestamps.get(cache_key)
            if (
                cache_time
                and (datetime.now(timezone.utc) - cache_time).total_seconds() < 3600
            ):
                return self._correlation_cache[cache_key]

        correlations = []
        experiment_completion = experiment.updated_timestamp
        cutoff_time = experiment_completion + timedelta(hours=time_window_hours)

        # Find subsequent decisions within time window
        subsequent_decisions = [
            decision
            for decision in self.decision_logger.decision_records.values()
            if experiment_completion < decision.timestamp <= cutoff_time
        ]

        for decision in subsequent_decisions:
            correlation = await self._analyze_single_correlation(experiment, decision)
            if (
                correlation
                and correlation.influence_metrics.influence_strength
                >= self.min_influence_threshold
            ):
                correlations.append(correlation)

        # Cache results
        self._correlation_cache[cache_key] = correlations
        self._analysis_timestamps[cache_key] = datetime.now(timezone.utc)

        return correlations

    async def _analyze_single_correlation(
        self, experiment: ExperimentLifecycleRecord, decision: DecisionRecord
    ) -> Optional[ResultCorrelation]:
        """Analyze correlation between a single experiment and decision."""
        # Calculate influence metrics
        influence_metrics = await self._calculate_influence_metrics(
            experiment, decision
        )

        if influence_metrics.influence_strength < self.min_influence_threshold:
            return None

        # Determine correlation type
        correlation_type = self._determine_correlation_type(
            experiment, decision, influence_metrics
        )

        # Analyze decision changes
        decision_changes = await self._analyze_decision_changes(experiment, decision)

        # Calculate temporal gap
        temporal_gap = (
            decision.timestamp - experiment.updated_timestamp
        ).total_seconds() / 3600

        # Generate explanation
        explanation = self._generate_correlation_explanation(
            experiment, decision, correlation_type, influence_metrics
        )

        # Gather supporting evidence
        supporting_evidence = await self._gather_supporting_evidence(
            experiment, decision
        )

        correlation = ResultCorrelation(
            correlation_id=f"corr_{experiment.experiment_id}_{decision.decision_id}",
            source_experiment_id=experiment.experiment_id,
            target_decision_id=decision.decision_id,
            correlation_type=correlation_type,
            influence_metrics=influence_metrics,
            result_properties=experiment.actual_results,
            decision_changes=decision_changes,
            temporal_gap=temporal_gap,
            explanation=explanation,
            supporting_evidence=supporting_evidence,
            timestamp=datetime.now(timezone.utc),
        )

        return correlation

    async def _calculate_influence_metrics(
        self, experiment: ExperimentLifecycleRecord, decision: DecisionRecord
    ) -> InfluenceMetric:
        """Calculate comprehensive influence metrics."""
        # Temporal proximity (closer in time = higher influence)
        time_diff_hours = (
            decision.timestamp - experiment.updated_timestamp
        ).total_seconds() / 3600
        temporal_proximity = max(
            0, 1 - (time_diff_hours / self.correlation_window_hours)
        )

        # Semantic similarity (similar context = higher influence)
        semantic_similarity = await self._calculate_semantic_similarity(
            experiment, decision
        )

        # Outcome correlation (result quality vs decision confidence)
        outcome_correlation = self._calculate_outcome_correlation(experiment, decision)

        # Frequency weight (repeated patterns increase influence)
        frequency_weight = await self._calculate_frequency_weight(experiment, decision)

        # Statistical significance
        statistical_significance = await self._calculate_statistical_significance(
            experiment, decision
        )

        # Overall influence strength (weighted combination)
        influence_strength = (
            0.3 * temporal_proximity
            + 0.25 * semantic_similarity
            + 0.25 * outcome_correlation
            + 0.1 * frequency_weight
            + 0.1 * statistical_significance
        )

        # Confidence level in the analysis
        confidence_level = min(
            1.0,
            (semantic_similarity + outcome_correlation + statistical_significance) / 3.0
            + 0.2,
        )

        return InfluenceMetric(
            influence_strength=influence_strength,
            confidence_level=confidence_level,
            temporal_proximity=temporal_proximity,
            semantic_similarity=semantic_similarity,
            outcome_correlation=outcome_correlation,
            frequency_weight=frequency_weight,
            statistical_significance=statistical_significance,
        )

    async def _calculate_semantic_similarity(
        self, experiment: ExperimentLifecycleRecord, decision: DecisionRecord
    ) -> float:
        """Calculate semantic similarity between experiment and decision contexts."""
        # Compare molecules if decision involves molecule selection
        if decision.decision_type == DecisionType.MOLECULE_SELECTION:
            if hasattr(decision.decision_data, "selected_molecules"):
                selected_molecules = decision.decision_data.selected_molecules
                if experiment.molecule_smiles in selected_molecules:
                    return 1.0
                else:
                    # Calculate molecular similarity (placeholder)
                    return 0.3

        # Compare experimental conditions
        if hasattr(decision.decision_data, "selected_conditions"):
            decision_conditions = decision.decision_data.selected_conditions
            exp_conditions = experiment.experimental_conditions

            common_keys = set(decision_conditions.keys()) & set(exp_conditions.keys())
            if common_keys:
                similarity_sum = 0.0
                for key in common_keys:
                    try:
                        dec_val = float(decision_conditions[key])
                        exp_val = float(exp_conditions[key])
                        # Normalized similarity
                        similarity_sum += 1.0 - abs(dec_val - exp_val) / max(
                            abs(dec_val), abs(exp_val), 1.0
                        )
                    except (ValueError, TypeError):
                        similarity_sum += (
                            1.0
                            if decision_conditions[key] == exp_conditions[key]
                            else 0.0
                        )

                return similarity_sum / len(common_keys)

        # Default similarity based on experiment context
        if decision.experiment_id == experiment.experiment_id:
            return 0.8

        return 0.1

    def _calculate_outcome_correlation(
        self, experiment: ExperimentLifecycleRecord, decision: DecisionRecord
    ) -> float:
        """Calculate correlation between experiment outcomes and decision confidence."""
        # Get experiment success metrics
        success_score = 0.0
        if experiment.actual_results:
            # Simplified success score based on target achievement
            target_properties = experiment.target_properties
            actual_results = experiment.actual_results

            achievements = []
            for prop, target in target_properties.items():
                if prop in actual_results:
                    actual = actual_results[prop]
                    # Calculate achievement ratio (closer to target = better)
                    achievement = 1.0 - abs(actual - target) / max(abs(target), 1.0)
                    achievements.append(max(0, achievement))

            success_score = np.mean(achievements) if achievements else 0.0

        # Get decision confidence
        decision_confidence = decision.context.confidence_score

        # Calculate correlation
        correlation = abs(success_score - 0.5) * abs(decision_confidence - 0.5) * 4
        return min(1.0, correlation)

    async def _calculate_frequency_weight(
        self, experiment: ExperimentLifecycleRecord, decision: DecisionRecord
    ) -> float:
        """Calculate frequency weight based on pattern repetition."""
        # Count similar experiment-decision pairs
        similar_count = 0
        total_count = 0

        for other_exp in self.decision_logger.experiment_records.values():
            if other_exp.experiment_id == experiment.experiment_id:
                continue

            total_count += 1

            # Check if similar experiment led to similar decision
            exp_similarity = self._calculate_experimental_similarity(
                experiment, other_exp
            )
            if exp_similarity > self.similarity_threshold:
                # Find decisions made after this experiment
                for other_decision in self.decision_logger.decision_records.values():
                    if (
                        other_decision.timestamp > other_exp.updated_timestamp
                        and other_decision.decision_type == decision.decision_type
                    ):
                        decision_similarity = self._calculate_decision_similarity(
                            decision, other_decision
                        )
                        if decision_similarity > self.similarity_threshold:
                            similar_count += 1
                            break

        return min(1.0, similar_count / max(total_count, 1))

    async def _calculate_statistical_significance(
        self, experiment: ExperimentLifecycleRecord, decision: DecisionRecord
    ) -> float:
        """Calculate statistical significance of the correlation."""
        # Simplified statistical significance calculation
        # In practice, this would involve proper statistical tests

        # Look for sufficient sample size
        similar_experiments = [
            exp
            for exp in self.decision_logger.experiment_records.values()
            if self._calculate_experimental_similarity(experiment, exp) > 0.5
        ]

        sample_size = len(similar_experiments)
        if sample_size < 3:
            return 0.0
        elif sample_size < 10:
            return 0.3
        elif sample_size < 30:
            return 0.6
        else:
            return 0.9

    def _determine_correlation_type(
        self,
        experiment: ExperimentLifecycleRecord,
        decision: DecisionRecord,
        influence_metrics: InfluenceMetric,
    ) -> CorrelationType:
        """Determine the type of correlation between experiment and decision."""
        # Analyze experiment success
        experiment_success = self._calculate_experiment_success_score(experiment)

        # Analyze decision type and context
        decision_confidence = decision.context.confidence_score

        if experiment_success > 0.7:
            if decision_confidence > 0.7:
                return CorrelationType.POSITIVE_REINFORCEMENT
            else:
                return CorrelationType.EXPLOITATION_FOCUS
        elif experiment_success < 0.3:
            if decision.decision_type == DecisionType.PARAMETER_OPTIMIZATION:
                return CorrelationType.PARAMETER_ADJUSTMENT
            elif decision_confidence < 0.5:
                return CorrelationType.EXPLORATION_EXPANSION
            else:
                return CorrelationType.NEGATIVE_FEEDBACK
        else:
            # Moderate success
            if decision.decision_type == DecisionType.EXPERIMENTAL_DESIGN:
                return CorrelationType.STRATEGY_SHIFT
            else:
                return CorrelationType.PARAMETER_ADJUSTMENT

    async def _analyze_decision_changes(
        self, experiment: ExperimentLifecycleRecord, decision: DecisionRecord
    ) -> Dict[str, Any]:
        """Analyze what changed in the decision compared to previous decisions."""
        changes = {
            "parameter_changes": {},
            "strategy_changes": [],
            "confidence_change": 0.0,
            "approach_change": "none",
        }

        # Find previous similar decisions
        previous_decisions = [
            d
            for d in self.decision_logger.decision_records.values()
            if (
                d.decision_type == decision.decision_type
                and d.timestamp < decision.timestamp
                and d.timestamp > experiment.created_timestamp - timedelta(days=30)
            )
        ]

        if previous_decisions:
            # Sort by timestamp
            previous_decisions.sort(key=lambda x: x.timestamp, reverse=True)
            most_recent = previous_decisions[0]

            # Compare confidence
            changes["confidence_change"] = (
                decision.context.confidence_score - most_recent.context.confidence_score
            )

            # Compare parameters (simplified)
            if hasattr(decision.decision_data, "__dict__") and hasattr(
                most_recent.decision_data, "__dict__"
            ):
                decision_dict = decision.decision_data.__dict__
                previous_dict = most_recent.decision_data.__dict__

                for key in decision_dict:
                    if (
                        key in previous_dict
                        and decision_dict[key] != previous_dict[key]
                    ):
                        changes["parameter_changes"][key] = {
                            "previous": previous_dict[key],
                            "current": decision_dict[key],
                        }

        return changes

    def _generate_correlation_explanation(
        self,
        experiment: ExperimentLifecycleRecord,
        decision: DecisionRecord,
        correlation_type: CorrelationType,
        influence_metrics: InfluenceMetric,
    ) -> str:
        """Generate human-readable explanation of the correlation."""
        experiment_success = self._calculate_experiment_success_score(experiment)
        strength = self._quantify_influence_strength(
            influence_metrics.influence_strength
        )

        explanations = {
            CorrelationType.POSITIVE_REINFORCEMENT: f"The successful experiment (score: {experiment_success:.2f}) {strength.value} reinforced the AI's confidence in similar approaches, leading to continued exploration in this direction.",
            CorrelationType.NEGATIVE_FEEDBACK: f"The unsuccessful experiment (score: {experiment_success:.2f}) {strength.value} caused the AI to adjust its strategy and avoid similar approaches.",
            CorrelationType.PARAMETER_ADJUSTMENT: f"The moderate results (score: {experiment_success:.2f}) {strength.value} prompted fine-tuning of experimental parameters for optimization.",
            CorrelationType.STRATEGY_SHIFT: f"The experiment outcomes {strength.value} influenced a strategic shift in the AI's experimental design approach.",
            CorrelationType.EXPLORATION_EXPANSION: f"The mixed results {strength.value} encouraged broader exploration of the chemical space.",
            CorrelationType.EXPLOITATION_FOCUS: f"The promising results {strength.value} focused the AI's attention on exploiting this successful region.",
        }

        return explanations.get(correlation_type, "Unknown correlation type detected.")

    async def _gather_supporting_evidence(
        self, experiment: ExperimentLifecycleRecord, decision: DecisionRecord
    ) -> List[Dict[str, Any]]:
        """Gather supporting evidence for the correlation."""
        evidence = []

        # Temporal evidence
        time_gap = (
            decision.timestamp - experiment.updated_timestamp
        ).total_seconds() / 3600
        evidence.append(
            {
                "type": "temporal",
                "description": f"Decision made {time_gap:.1f} hours after experiment completion",
                "strength": (
                    "strong"
                    if time_gap < 24
                    else "moderate"
                    if time_gap < 168
                    else "weak"
                ),
            }
        )

        # Context similarity evidence
        semantic_sim = await self._calculate_semantic_similarity(experiment, decision)
        evidence.append(
            {
                "type": "contextual",
                "description": f"High contextual similarity (score: {semantic_sim:.2f})",
                "strength": (
                    "strong"
                    if semantic_sim > 0.7
                    else "moderate"
                    if semantic_sim > 0.4
                    else "weak"
                ),
            }
        )

        # Pattern evidence
        frequency_weight = await self._calculate_frequency_weight(experiment, decision)
        evidence.append(
            {
                "type": "pattern",
                "description": f"Similar patterns observed in {frequency_weight:.1%} of cases",
                "strength": (
                    "strong"
                    if frequency_weight > 0.7
                    else "moderate"
                    if frequency_weight > 0.3
                    else "weak"
                ),
            }
        )

        return evidence

    async def build_influence_network(
        self, time_window_days: int = 30, min_influence_threshold: float = 0.2
    ) -> InfluenceNetwork:
        """Build comprehensive influence network."""
        network_id = f"network_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        # Get recent decisions and experiments
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=time_window_days)

        recent_decisions = [
            d
            for d in self.decision_logger.decision_records.values()
            if d.timestamp >= cutoff_time
        ]

        recent_experiments = [
            e
            for e in self.decision_logger.experiment_records.values()
            if e.created_timestamp >= cutoff_time
        ]

        # Build nodes
        nodes = {}

        # Add decision nodes
        for decision in recent_decisions:
            nodes[decision.decision_id] = {
                "type": "decision",
                "decision_type": decision.decision_type.value,
                "timestamp": decision.timestamp.isoformat(),
                "confidence": decision.context.confidence_score,
                "metadata": {"experiment_id": decision.experiment_id},
            }

        # Add experiment nodes
        for experiment in recent_experiments:
            nodes[experiment.experiment_id] = {
                "type": "experiment",
                "molecule": experiment.molecule_smiles,
                "timestamp": experiment.created_timestamp.isoformat(),
                "success_score": self._calculate_experiment_success_score(experiment),
                "phase": experiment.current_phase.value,
            }

        # Build edges (influence relationships)
        edges = []

        for experiment in recent_experiments:
            correlations = await self.analyze_result_influence(experiment.experiment_id)
            for correlation in correlations:
                if (
                    correlation.influence_metrics.influence_strength
                    >= min_influence_threshold
                ):
                    edges.append(
                        {
                            "source": correlation.source_experiment_id,
                            "target": correlation.target_decision_id,
                            "weight": correlation.influence_metrics.influence_strength,
                            "type": correlation.correlation_type.value,
                            "temporal_gap": correlation.temporal_gap,
                        }
                    )

        # Calculate network metrics
        network_metrics = self._calculate_network_metrics(nodes, edges)

        # Calculate centrality metrics
        centrality_metrics = self._calculate_centrality_metrics(nodes, edges)

        # Identify clusters
        clusters = self._identify_clusters(nodes, edges)

        # Find influence paths
        influence_paths = self._find_influence_paths(nodes, edges)

        network = InfluenceNetwork(
            network_id=network_id,
            nodes=nodes,
            edges=edges,
            centrality_metrics=centrality_metrics,
            clusters=clusters,
            influence_paths=influence_paths,
            network_metrics=network_metrics,
        )

        self.influence_networks[network_id] = network
        return network

    async def identify_feedback_loops(
        self, network: InfluenceNetwork, max_loop_length: int = 5
    ) -> List[FeedbackLoop]:
        """Identify feedback loops in the influence network."""
        feedback_loops = []

        # Find cycles in the network
        cycles = self._find_cycles(network.nodes, network.edges, max_loop_length)

        for cycle in cycles:
            loop_id = f"loop_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{len(feedback_loops)}"

            # Classify participating nodes
            participating_decisions = [
                node_id
                for node_id in cycle
                if network.nodes[node_id]["type"] == "decision"
            ]
            participating_experiments = [
                node_id
                for node_id in cycle
                if network.nodes[node_id]["type"] == "experiment"
            ]

            # Determine loop type and strength
            loop_type, loop_strength = self._analyze_loop_characteristics(
                cycle, network
            )

            # Identify patterns
            patterns = self._identify_loop_patterns(cycle, network)

            # Calculate optimization impact
            optimization_impact = self._calculate_optimization_impact(cycle, network)

            feedback_loop = FeedbackLoop(
                loop_id=loop_id,
                participating_decisions=participating_decisions,
                participating_experiments=participating_experiments,
                loop_type=loop_type,
                loop_strength=loop_strength,
                iteration_count=len(cycle),
                convergence_behavior="stable",  # Simplified
                identified_patterns=patterns,
                optimization_impact=optimization_impact,
            )

            feedback_loops.append(feedback_loop)

        # Store feedback loops
        for loop in feedback_loops:
            self.feedback_loops[loop.loop_id] = loop

        return feedback_loops

    # Helper methods
    def _calculate_experimental_similarity(
        self, exp1: ExperimentLifecycleRecord, exp2: ExperimentLifecycleRecord
    ) -> float:
        """Calculate similarity between two experiments."""
        # Molecule similarity
        mol_similarity = 1.0 if exp1.molecule_smiles == exp2.molecule_smiles else 0.3

        # Conditions similarity
        conditions1 = exp1.experimental_conditions
        conditions2 = exp2.experimental_conditions

        common_keys = set(conditions1.keys()) & set(conditions2.keys())
        if not common_keys:
            condition_similarity = 0.0
        else:
            similarity_sum = 0.0
            for key in common_keys:
                try:
                    val1, val2 = float(conditions1[key]), float(conditions2[key])
                    similarity_sum += 1.0 - abs(val1 - val2) / max(
                        abs(val1), abs(val2), 1.0
                    )
                except (ValueError, TypeError):
                    similarity_sum += (
                        1.0 if conditions1[key] == conditions2[key] else 0.0
                    )
            condition_similarity = similarity_sum / len(common_keys)

        return 0.5 * mol_similarity + 0.5 * condition_similarity

    def _calculate_decision_similarity(
        self, dec1: DecisionRecord, dec2: DecisionRecord
    ) -> float:
        """Calculate similarity between two decisions."""
        if dec1.decision_type != dec2.decision_type:
            return 0.0

        # Context similarity
        confidence_similarity = 1.0 - abs(
            dec1.context.confidence_score - dec2.context.confidence_score
        )

        # Decision data similarity (simplified)
        data_similarity = 0.5  # Placeholder

        return 0.5 * confidence_similarity + 0.5 * data_similarity

    def _calculate_experiment_success_score(
        self, experiment: ExperimentLifecycleRecord
    ) -> float:
        """Calculate success score for an experiment."""
        if not experiment.actual_results or not experiment.target_properties:
            return 0.0

        achievements = []
        for prop, target in experiment.target_properties.items():
            if prop in experiment.actual_results:
                actual = experiment.actual_results[prop]
                # Normalized achievement (closer to target = better)
                achievement = max(0, 1.0 - abs(actual - target) / max(abs(target), 1.0))
                achievements.append(achievement)

        return np.mean(achievements) if achievements else 0.0

    def _quantify_influence_strength(self, strength: float) -> InfluenceStrength:
        """Convert numerical strength to categorical strength."""
        if strength < 0.1:
            return InfluenceStrength.NEGLIGIBLE
        elif strength < 0.3:
            return InfluenceStrength.WEAK
        elif strength < 0.6:
            return InfluenceStrength.MODERATE
        elif strength < 0.8:
            return InfluenceStrength.STRONG
        else:
            return InfluenceStrength.VERY_STRONG

    def _calculate_network_metrics(
        self, nodes: Dict[str, Dict[str, Any]], edges: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate network-level metrics."""
        num_nodes = len(nodes)
        num_edges = len(edges)

        if num_nodes <= 1:
            return {"density": 0.0, "connectivity": 0.0, "efficiency": 0.0}

        max_possible_edges = num_nodes * (num_nodes - 1)
        density = num_edges / max_possible_edges if max_possible_edges > 0 else 0.0

        # Simplified connectivity and efficiency metrics
        connectivity = min(1.0, num_edges / num_nodes)
        efficiency = density * connectivity

        return {
            "density": density,
            "connectivity": connectivity,
            "efficiency": efficiency,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
        }

    def _calculate_centrality_metrics(
        self, nodes: Dict[str, Dict[str, Any]], edges: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate centrality metrics for nodes."""
        centrality = {}

        # Degree centrality (simplified)
        degree_count = defaultdict(int)
        for edge in edges:
            degree_count[edge["source"]] += 1
            degree_count[edge["target"]] += 1

        max_degree = max(degree_count.values()) if degree_count else 1

        for node_id in nodes:
            centrality[node_id] = degree_count[node_id] / max_degree

        return centrality

    def _identify_clusters(
        self, nodes: Dict[str, Dict[str, Any]], edges: List[Dict[str, Any]]
    ) -> List[List[str]]:
        """Identify clusters in the network."""
        # Simplified clustering based on connected components
        visited = set()
        clusters = []

        # Build adjacency list
        adjacency = defaultdict(list)
        for edge in edges:
            adjacency[edge["source"]].append(edge["target"])
            adjacency[edge["target"]].append(edge["source"])

        def dfs(node, cluster):
            visited.add(node)
            cluster.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    dfs(neighbor, cluster)

        for node_id in nodes:
            if node_id not in visited:
                cluster = []
                dfs(node_id, cluster)
                if len(cluster) > 1:
                    clusters.append(cluster)

        return clusters

    def _find_influence_paths(
        self, nodes: Dict[str, Dict[str, Any]], edges: List[Dict[str, Any]]
    ) -> List[List[str]]:
        """Find significant influence paths."""
        # Simplified path finding
        paths = []

        # Build directed adjacency list with weights
        adjacency = defaultdict(list)
        for edge in edges:
            if edge["weight"] > 0.5:  # Only strong influences
                adjacency[edge["source"]].append((edge["target"], edge["weight"]))

        # Find paths of length 2-4
        for start_node in nodes:
            if nodes[start_node]["type"] == "experiment":
                paths.extend(
                    self._find_paths_from_node(start_node, adjacency, max_length=4)
                )

        return paths[:20]  # Limit to top 20 paths

    def _find_paths_from_node(
        self,
        start_node: str,
        adjacency: Dict[str, List[Tuple[str, float]]],
        max_length: int,
        current_path: Optional[List[str]] = None,
        visited: Optional[Set[str]] = None,
    ) -> List[List[str]]:
        """Find paths from a given node."""
        if current_path is None:
            current_path = [start_node]
        if visited is None:
            visited = {start_node}

        paths = []

        if len(current_path) >= max_length:
            return paths

        for neighbor, weight in adjacency[start_node]:
            if neighbor not in visited and weight > 0.3:
                new_path = current_path + [neighbor]
                new_visited = visited | {neighbor}

                if len(new_path) >= 2:
                    paths.append(new_path)

                paths.extend(
                    self._find_paths_from_node(
                        neighbor, adjacency, max_length, new_path, new_visited
                    )
                )

        return paths

    def _find_cycles(
        self,
        nodes: Dict[str, Dict[str, Any]],
        edges: List[Dict[str, Any]],
        max_length: int,
    ) -> List[List[str]]:
        """Find cycles in the network."""
        cycles = []

        # Build adjacency list
        adjacency = defaultdict(list)
        for edge in edges:
            adjacency[edge["source"]].append(edge["target"])

        def dfs_cycle(node, path, start_node):
            if len(path) > max_length:
                return

            if node == start_node and len(path) > 2:
                cycles.append(path[:])
                return

            if node in path[:-1]:  # Avoid infinite loops
                return

            for neighbor in adjacency[node]:
                dfs_cycle(neighbor, path + [neighbor], start_node)

        for start_node in nodes:
            dfs_cycle(start_node, [start_node], start_node)

        return cycles[:10]  # Limit cycles

    def _analyze_loop_characteristics(
        self, cycle: List[str], network: InfluenceNetwork
    ) -> Tuple[str, float]:
        """Analyze characteristics of a feedback loop."""
        # Determine loop type based on node types and edge weights
        total_weight = 0.0
        edge_count = 0

        for i in range(len(cycle)):
            source = cycle[i]
            target = cycle[(i + 1) % len(cycle)]

            # Find edge weight
            for edge in network.edges:
                if edge["source"] == source and edge["target"] == target:
                    total_weight += edge["weight"]
                    edge_count += 1
                    break

        loop_strength = total_weight / max(edge_count, 1)

        # Determine loop type (simplified)
        if loop_strength > 0.6:
            loop_type = "reinforcing"
        else:
            loop_type = "balancing"

        return loop_type, loop_strength

    def _identify_loop_patterns(
        self, cycle: List[str], network: InfluenceNetwork
    ) -> List[str]:
        """Identify patterns in feedback loops."""
        patterns = []

        # Check for alternating experiment-decision pattern
        node_types = [network.nodes[node_id]["type"] for node_id in cycle]
        if node_types == ["experiment", "decision"] * (len(cycle) // 2):
            patterns.append("alternating_exp_decision")

        # Check for rapid iteration pattern
        timestamps = [network.nodes[node_id]["timestamp"] for node_id in cycle]
        if len(timestamps) > 1:
            time_diffs = []
            for i in range(1, len(timestamps)):
                t1 = datetime.fromisoformat(timestamps[i - 1])
                t2 = datetime.fromisoformat(timestamps[i])
                time_diffs.append((t2 - t1).total_seconds() / 3600)

            avg_time_diff = np.mean(time_diffs)
            if avg_time_diff < 24:
                patterns.append("rapid_iteration")
            elif avg_time_diff > 168:
                patterns.append("slow_iteration")

        return patterns

    def _calculate_optimization_impact(
        self, cycle: List[str], network: InfluenceNetwork
    ) -> Dict[str, float]:
        """Calculate the optimization impact of a feedback loop."""
        # Simplified impact calculation
        experiment_nodes = [
            node_id
            for node_id in cycle
            if network.nodes[node_id]["type"] == "experiment"
        ]

        if not experiment_nodes:
            return {"performance_improvement": 0.0, "efficiency_gain": 0.0}

        # Get success scores
        success_scores = [
            network.nodes[node_id]["success_score"] for node_id in experiment_nodes
        ]

        # Calculate trend
        if len(success_scores) > 1:
            performance_improvement = success_scores[-1] - success_scores[0]
        else:
            performance_improvement = 0.0

        efficiency_gain = max(0, performance_improvement * 0.5)

        return {
            "performance_improvement": performance_improvement,
            "efficiency_gain": efficiency_gain,
            "loop_effectiveness": min(1.0, abs(performance_improvement)),
        }

    async def get_influence_summary(self) -> Dict[str, Any]:
        """Get comprehensive influence analysis summary."""
        return {
            "total_correlations_analyzed": len(self.influence_history),
            "active_influence_networks": len(self.influence_networks),
            "identified_feedback_loops": len(self.feedback_loops),
            "influence_strength_distribution": self._analyze_influence_distribution(),
            "correlation_type_distribution": self._analyze_correlation_types(),
            "feedback_loop_effectiveness": self._analyze_feedback_effectiveness(),
            "analysis_performance": {
                "cache_hit_rate": len(self._correlation_cache)
                / max(len(self.influence_history), 1),
                "average_analysis_time": "estimated_low",
            },
        }

    def _analyze_influence_distribution(self) -> Dict[str, int]:
        """Analyze distribution of influence strengths."""
        distribution = {strength.value: 0 for strength in InfluenceStrength}

        for correlation in self.influence_history:
            strength_category = self._quantify_influence_strength(
                correlation.influence_metrics.influence_strength
            )
            distribution[strength_category.value] += 1

        return distribution

    def _analyze_correlation_types(self) -> Dict[str, int]:
        """Analyze distribution of correlation types."""
        distribution = {cor_type.value: 0 for cor_type in CorrelationType}

        for correlation in self.influence_history:
            distribution[correlation.correlation_type.value] += 1

        return distribution

    def _analyze_feedback_effectiveness(self) -> Dict[str, float]:
        """Analyze effectiveness of feedback loops."""
        if not self.feedback_loops:
            return {
                "average_effectiveness": 0.0,
                "reinforcing_loops": 0,
                "balancing_loops": 0,
            }

        effectiveness_scores = []
        reinforcing_count = 0
        balancing_count = 0

        for loop in self.feedback_loops.values():
            effectiveness_scores.append(
                loop.optimization_impact.get("loop_effectiveness", 0.0)
            )

            if loop.loop_type == "reinforcing":
                reinforcing_count += 1
            else:
                balancing_count += 1

        return {
            "average_effectiveness": np.mean(effectiveness_scores),
            "reinforcing_loops": reinforcing_count,
            "balancing_loops": balancing_count,
            "total_loops": len(self.feedback_loops),
        }
