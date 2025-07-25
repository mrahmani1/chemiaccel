"""
Auditable Traceability and Decision Logging System

This module provides comprehensive logging and tracking for all AI-driven decisions,
experimental designs, and results within the autonomous laboratory loop, ensuring
full traceability and auditability of the entire discovery process.
"""

import asyncio
import json
import sqlite3
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class DecisionType(Enum):
    """Types of AI decisions that can be logged."""

    MOLECULE_SELECTION = "molecule_selection"
    SYNTHESIS_ROUTE = "synthesis_route"
    EXPERIMENTAL_DESIGN = "experimental_design"
    PARAMETER_OPTIMIZATION = "parameter_optimization"
    ACTIVE_LEARNING_QUERY = "active_learning_query"
    INVERSE_DESIGN = "inverse_design"
    QUALITY_ASSESSMENT = "quality_assessment"
    RESOURCE_ALLOCATION = "resource_allocation"


class ExperimentPhase(Enum):
    """Phases of experimental lifecycle."""

    PLANNING = "planning"
    SYNTHESIS = "synthesis"
    CHARACTERIZATION = "characterization"
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    COMPLETED = "completed"
    FAILED = "failed"


class InfluenceType(Enum):
    """Types of influence relationships between decisions."""

    DIRECT_CONSEQUENCE = "direct_consequence"
    INFORMED_BY = "informed_by"
    OPTIMIZATION_ITERATION = "optimization_iteration"
    FEEDBACK_LOOP = "feedback_loop"
    CONSTRAINT_DRIVEN = "constraint_driven"
    SERENDIPITOUS = "serendipitous"


@dataclass
class DecisionContext:
    """Context information for AI decisions."""

    model_version: str
    model_type: str
    confidence_score: float
    input_data_hash: str
    computational_resources: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    training_data_version: Optional[str] = None
    feature_importance: Optional[Dict[str, float]] = None
    uncertainty_metrics: Optional[Dict[str, float]] = None


@dataclass
class MoleculeSelectionDecision:
    """Detailed logging for molecule selection decisions."""

    selected_molecules: List[str]
    candidate_pool_size: int
    selection_criteria: Dict[str, Any]
    diversity_metrics: Dict[str, float]
    predicted_properties: Dict[str, Dict[str, float]]
    exclusion_reasons: Dict[str, List[str]]
    similarity_analysis: Optional[Dict[str, Any]] = None


@dataclass
class SynthesisRouteDecision:
    """Detailed logging for synthesis route planning."""

    target_molecule: str
    selected_route: Dict[str, Any]
    alternative_routes: List[Dict[str, Any]]
    route_scoring: Dict[str, float]
    reagent_availability: Dict[str, bool]
    predicted_yield: float
    predicted_difficulty: float
    cost_analysis: Optional[Dict[str, float]] = None


@dataclass
class ExperimentalDesignDecision:
    """Detailed logging for experimental design decisions."""

    design_type: str
    parameter_space: Dict[str, Any]
    selected_conditions: Dict[str, Any]
    doe_strategy: str
    expected_outcomes: Dict[str, float]
    resource_requirements: Dict[str, Any]
    time_estimation: float
    risk_assessment: Optional[Dict[str, float]] = None


@dataclass
class DecisionRecord:
    """Complete record of an AI decision."""

    decision_id: str
    timestamp: datetime
    decision_type: DecisionType
    context: DecisionContext
    decision_data: Union[
        MoleculeSelectionDecision,
        SynthesisRouteDecision,
        ExperimentalDesignDecision,
        Dict[str, Any],
    ]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    experiment_id: Optional[str] = None
    parent_decision_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentLifecycleRecord:
    """Complete lifecycle record of an experiment."""

    experiment_id: str
    molecule_smiles: str
    created_timestamp: datetime
    updated_timestamp: datetime
    current_phase: ExperimentPhase
    target_properties: Dict[str, float]
    actual_results: Dict[str, float]
    synthesis_route: Dict[str, Any]
    experimental_conditions: Dict[str, Any]
    decision_chain: List[str]  # List of decision_ids
    quality_metrics: Dict[str, float]
    resource_utilization: Dict[str, Any]
    timeline: List[Dict[str, Any]]
    success_indicators: Dict[str, bool]
    failure_reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InfluenceRelationship:
    """Record of how one decision influenced another."""

    relationship_id: str
    source_decision_id: str
    target_decision_id: str
    influence_type: InfluenceType
    influence_strength: float
    explanation: str
    quantitative_metrics: Dict[str, float]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditEvent:
    """Record of audit-relevant events."""

    event_id: str
    timestamp: datetime
    event_type: str
    user_id: Optional[str]
    session_id: Optional[str]
    affected_records: List[str]
    changes_made: Dict[str, Any]
    justification: str
    approval_status: Optional[str] = None
    reviewer_id: Optional[str] = None


class DecisionLogger:
    """Core decision logging system."""

    def __init__(
        self,
        storage_backend: str = "sqlite",
        storage_config: Optional[Dict[str, Any]] = None,
    ):
        self.storage_backend = storage_backend
        self.storage_config = storage_config or {}
        self.decision_records: Dict[str, DecisionRecord] = {}
        self.experiment_records: Dict[str, ExperimentLifecycleRecord] = {}
        self.influence_relationships: Dict[str, InfluenceRelationship] = {}
        self.audit_events: Dict[str, AuditEvent] = {}
        self._storage_initialized = False

        # Initialize storage backend
        asyncio.create_task(self._initialize_storage())

    async def _initialize_storage(self):
        """Initialize the storage backend."""
        if self.storage_backend == "sqlite":
            await self._initialize_sqlite()
        self._storage_initialized = True

    async def _initialize_sqlite(self):
        """Initialize SQLite database for logging."""
        db_path = self.storage_config.get("db_path", "autonomous_lab_audit.db")

        # Create tables
        async with self._get_db_connection() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decision_records (
                    decision_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    decision_type TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    decision_data_json TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    experiment_id TEXT,
                    parent_decision_id TEXT,
                    metadata_json TEXT
                )
            """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_lifecycle (
                    experiment_id TEXT PRIMARY KEY,
                    molecule_smiles TEXT NOT NULL,
                    created_timestamp TEXT NOT NULL,
                    updated_timestamp TEXT NOT NULL,
                    current_phase TEXT NOT NULL,
                    target_properties_json TEXT NOT NULL,
                    actual_results_json TEXT NOT NULL,
                    synthesis_route_json TEXT NOT NULL,
                    experimental_conditions_json TEXT NOT NULL,
                    decision_chain_json TEXT NOT NULL,
                    quality_metrics_json TEXT NOT NULL,
                    resource_utilization_json TEXT NOT NULL,
                    timeline_json TEXT NOT NULL,
                    success_indicators_json TEXT NOT NULL,
                    failure_reasons_json TEXT NOT NULL,
                    metadata_json TEXT
                )
            """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS influence_relationships (
                    relationship_id TEXT PRIMARY KEY,
                    source_decision_id TEXT NOT NULL,
                    target_decision_id TEXT NOT NULL,
                    influence_type TEXT NOT NULL,
                    influence_strength REAL NOT NULL,
                    explanation TEXT NOT NULL,
                    quantitative_metrics_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata_json TEXT,
                    FOREIGN KEY (source_decision_id) REFERENCES decision_records (decision_id),
                    FOREIGN KEY (target_decision_id) REFERENCES decision_records (decision_id)
                )
            """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    affected_records_json TEXT NOT NULL,
                    changes_made_json TEXT NOT NULL,
                    justification TEXT NOT NULL,
                    approval_status TEXT,
                    reviewer_id TEXT
                )
            """
            )

            # Create indexes for better performance
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_decision_timestamp ON decision_records (timestamp)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_decision_type ON decision_records (decision_type)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_experiment_id ON decision_records (experiment_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_experiment_phase ON experiment_lifecycle (current_phase)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_influence_source ON influence_relationships (source_decision_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_influence_target ON influence_relationships (target_decision_id)"
            )

            await conn.commit()

    @asynccontextmanager
    async def _get_db_connection(self):
        """Get database connection context manager."""
        if self.storage_backend == "sqlite":
            db_path = self.storage_config.get("db_path", "autonomous_lab_audit.db")
            conn = sqlite3.connect(db_path)
            try:
                yield conn
            finally:
                conn.close()

    async def log_decision(
        self,
        decision_type: DecisionType,
        context: DecisionContext,
        decision_data: Union[
            MoleculeSelectionDecision,
            SynthesisRouteDecision,
            ExperimentalDesignDecision,
            Dict[str, Any],
        ],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        parent_decision_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log an AI decision with full context."""
        decision_id = str(uuid.uuid4())

        record = DecisionRecord(
            decision_id=decision_id,
            timestamp=datetime.now(timezone.utc),
            decision_type=decision_type,
            context=context,
            decision_data=decision_data,
            user_id=user_id,
            session_id=session_id,
            experiment_id=experiment_id,
            parent_decision_id=parent_decision_id,
            metadata=metadata or {},
        )

        # Store in memory
        self.decision_records[decision_id] = record

        # Persist to storage
        await self._persist_decision_record(record)

        return decision_id

    async def _persist_decision_record(self, record: DecisionRecord):
        """Persist decision record to storage."""
        if not self._storage_initialized:
            await self._initialize_storage()

        if self.storage_backend == "sqlite":
            async with self._get_db_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO decision_records (
                        decision_id, timestamp, decision_type, context_json,
                        decision_data_json, user_id, session_id, experiment_id,
                        parent_decision_id, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        record.decision_id,
                        record.timestamp.isoformat(),
                        record.decision_type.value,
                        json.dumps(asdict(record.context)),
                        json.dumps(
                            asdict(record.decision_data)
                            if hasattr(record.decision_data, "__dict__")
                            else record.decision_data
                        ),
                        record.user_id,
                        record.session_id,
                        record.experiment_id,
                        record.parent_decision_id,
                        json.dumps(record.metadata),
                    ),
                )
                conn.commit()

    async def create_experiment_record(
        self,
        molecule_smiles: str,
        target_properties: Dict[str, float],
        synthesis_route: Dict[str, Any],
        experimental_conditions: Dict[str, Any],
        experiment_id: Optional[str] = None,
    ) -> str:
        """Create a new experiment lifecycle record."""
        if experiment_id is None:
            experiment_id = str(uuid.uuid4())

        record = ExperimentLifecycleRecord(
            experiment_id=experiment_id,
            molecule_smiles=molecule_smiles,
            created_timestamp=datetime.now(timezone.utc),
            updated_timestamp=datetime.now(timezone.utc),
            current_phase=ExperimentPhase.PLANNING,
            target_properties=target_properties,
            actual_results={},
            synthesis_route=synthesis_route,
            experimental_conditions=experimental_conditions,
            decision_chain=[],
            quality_metrics={},
            resource_utilization={},
            timeline=[
                {
                    "phase": ExperimentPhase.PLANNING.value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "description": "Experiment record created",
                }
            ],
            success_indicators={},
        )

        self.experiment_records[experiment_id] = record
        await self._persist_experiment_record(record)

        return experiment_id

    async def update_experiment_phase(
        self,
        experiment_id: str,
        new_phase: ExperimentPhase,
        results: Optional[Dict[str, float]] = None,
        quality_metrics: Optional[Dict[str, float]] = None,
        description: Optional[str] = None,
    ):
        """Update experiment phase and results."""
        if experiment_id not in self.experiment_records:
            raise ValueError(f"Experiment {experiment_id} not found")

        record = self.experiment_records[experiment_id]
        record.current_phase = new_phase
        record.updated_timestamp = datetime.now(timezone.utc)

        if results:
            record.actual_results.update(results)

        if quality_metrics:
            record.quality_metrics.update(quality_metrics)

        # Add timeline entry
        timeline_entry = {
            "phase": new_phase.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "description": description or f"Phase updated to {new_phase.value}",
        }
        if results:
            timeline_entry["results"] = results
        if quality_metrics:
            timeline_entry["quality_metrics"] = quality_metrics

        record.timeline.append(timeline_entry)

        await self._persist_experiment_record(record)

    async def _persist_experiment_record(self, record: ExperimentLifecycleRecord):
        """Persist experiment record to storage."""
        if not self._storage_initialized:
            await self._initialize_storage()

        if self.storage_backend == "sqlite":
            async with self._get_db_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO experiment_lifecycle (
                        experiment_id, molecule_smiles, created_timestamp,
                        updated_timestamp, current_phase, target_properties_json,
                        actual_results_json, synthesis_route_json,
                        experimental_conditions_json, decision_chain_json,
                        quality_metrics_json, resource_utilization_json,
                        timeline_json, success_indicators_json,
                        failure_reasons_json, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        record.experiment_id,
                        record.molecule_smiles,
                        record.created_timestamp.isoformat(),
                        record.updated_timestamp.isoformat(),
                        record.current_phase.value,
                        json.dumps(record.target_properties),
                        json.dumps(record.actual_results),
                        json.dumps(record.synthesis_route),
                        json.dumps(record.experimental_conditions),
                        json.dumps(record.decision_chain),
                        json.dumps(record.quality_metrics),
                        json.dumps(record.resource_utilization),
                        json.dumps(record.timeline),
                        json.dumps(record.success_indicators),
                        json.dumps(record.failure_reasons),
                        json.dumps(record.metadata),
                    ),
                )
                conn.commit()

    async def add_decision_to_experiment(self, experiment_id: str, decision_id: str):
        """Add a decision to an experiment's decision chain."""
        if experiment_id not in self.experiment_records:
            raise ValueError(f"Experiment {experiment_id} not found")

        record = self.experiment_records[experiment_id]
        if decision_id not in record.decision_chain:
            record.decision_chain.append(decision_id)
            record.updated_timestamp = datetime.now(timezone.utc)
            await self._persist_experiment_record(record)

    async def log_influence_relationship(
        self,
        source_decision_id: str,
        target_decision_id: str,
        influence_type: InfluenceType,
        influence_strength: float,
        explanation: str,
        quantitative_metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log how one decision influenced another."""
        relationship_id = str(uuid.uuid4())

        relationship = InfluenceRelationship(
            relationship_id=relationship_id,
            source_decision_id=source_decision_id,
            target_decision_id=target_decision_id,
            influence_type=influence_type,
            influence_strength=influence_strength,
            explanation=explanation,
            quantitative_metrics=quantitative_metrics or {},
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {},
        )

        self.influence_relationships[relationship_id] = relationship
        await self._persist_influence_relationship(relationship)

        return relationship_id

    async def _persist_influence_relationship(
        self, relationship: InfluenceRelationship
    ):
        """Persist influence relationship to storage."""
        if not self._storage_initialized:
            await self._initialize_storage()

        if self.storage_backend == "sqlite":
            async with self._get_db_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO influence_relationships (
                        relationship_id, source_decision_id, target_decision_id,
                        influence_type, influence_strength, explanation,
                        quantitative_metrics_json, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        relationship.relationship_id,
                        relationship.source_decision_id,
                        relationship.target_decision_id,
                        relationship.influence_type.value,
                        relationship.influence_strength,
                        relationship.explanation,
                        json.dumps(relationship.quantitative_metrics),
                        relationship.timestamp.isoformat(),
                        json.dumps(relationship.metadata),
                    ),
                )
                conn.commit()

    async def get_decision_lineage(self, decision_id: str) -> Dict[str, Any]:
        """Get the complete lineage of a decision."""
        decision = self.decision_records.get(decision_id)
        if not decision:
            return {}

        lineage = {
            "decision": asdict(decision),
            "ancestors": [],
            "descendants": [],
            "influences": {"influenced_by": [], "influences": []},
        }

        # Find ancestors
        current_id = decision.parent_decision_id
        while current_id:
            ancestor = self.decision_records.get(current_id)
            if ancestor:
                lineage["ancestors"].append(asdict(ancestor))
                current_id = ancestor.parent_decision_id
            else:
                break

        # Find descendants
        descendants = [
            d
            for d in self.decision_records.values()
            if d.parent_decision_id == decision_id
        ]
        lineage["descendants"] = [asdict(d) for d in descendants]

        # Find influence relationships
        for rel in self.influence_relationships.values():
            if rel.source_decision_id == decision_id:
                lineage["influences"]["influences"].append(asdict(rel))
            elif rel.target_decision_id == decision_id:
                lineage["influences"]["influenced_by"].append(asdict(rel))

        return lineage

    async def get_experiment_trace(self, experiment_id: str) -> Dict[str, Any]:
        """Get complete trace of an experiment."""
        experiment = self.experiment_records.get(experiment_id)
        if not experiment:
            return {}

        trace = {
            "experiment": asdict(experiment),
            "decisions": [],
            "influence_network": [],
        }

        # Get all decisions in the chain
        for decision_id in experiment.decision_chain:
            decision = self.decision_records.get(decision_id)
            if decision:
                trace["decisions"].append(asdict(decision))

        # Get influence relationships between decisions in this experiment
        experiment_decision_ids = set(experiment.decision_chain)
        for rel in self.influence_relationships.values():
            if (
                rel.source_decision_id in experiment_decision_ids
                or rel.target_decision_id in experiment_decision_ids
            ):
                trace["influence_network"].append(asdict(rel))

        return trace

    async def generate_audit_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        experiment_ids: Optional[List[str]] = None,
        decision_types: Optional[List[DecisionType]] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive audit report."""
        if start_time is None:
            start_time = datetime.min.replace(tzinfo=timezone.utc)
        if end_time is None:
            end_time = datetime.now(timezone.utc)

        # Filter records
        filtered_decisions = []
        for decision in self.decision_records.values():
            if start_time <= decision.timestamp <= end_time:
                if decision_types is None or decision.decision_type in decision_types:
                    if (
                        experiment_ids is None
                        or decision.experiment_id in experiment_ids
                    ):
                        filtered_decisions.append(decision)

        filtered_experiments = []
        for experiment in self.experiment_records.values():
            if start_time <= experiment.created_timestamp <= end_time:
                if experiment_ids is None or experiment.experiment_id in experiment_ids:
                    filtered_experiments.append(experiment)

        # Generate statistics
        decision_type_counts = {}
        for decision in filtered_decisions:
            decision_type_counts[decision.decision_type.value] = (
                decision_type_counts.get(decision.decision_type.value, 0) + 1
            )

        phase_counts = {}
        for experiment in filtered_experiments:
            phase_counts[experiment.current_phase.value] = (
                phase_counts.get(experiment.current_phase.value, 0) + 1
            )

        report = {
            "report_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
                "filters": {
                    "experiment_ids": experiment_ids,
                    "decision_types": [dt.value for dt in decision_types]
                    if decision_types
                    else None,
                },
            },
            "summary_statistics": {
                "total_decisions": len(filtered_decisions),
                "total_experiments": len(filtered_experiments),
                "decision_type_distribution": decision_type_counts,
                "experiment_phase_distribution": phase_counts,
                "total_influence_relationships": len(self.influence_relationships),
            },
            "decisions": [asdict(d) for d in filtered_decisions],
            "experiments": [asdict(e) for e in filtered_experiments],
            "influence_relationships": [
                asdict(r) for r in self.influence_relationships.values()
            ],
        }

        return report
