"""
Tests for Decision Logging System

This module contains comprehensive tests for the decision logging and traceability
system including decision records, experiment lifecycle tracking, and audit trails.
"""

import pytest
import asyncio
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from chemi_accel.autonomous.decision_logging import (
    DecisionType, ExperimentPhase, InfluenceType,
    DecisionContext, MoleculeSelectionDecision, SynthesisRouteDecision,
    ExperimentalDesignDecision, DecisionRecord, ExperimentLifecycleRecord,
    InfluenceRelationship, AuditEvent, DecisionLogger
)


class TestDecisionContext:
    """Test cases for DecisionContext dataclass."""
    
    def test_decision_context_creation(self):
        """Test basic decision context creation."""
        context = DecisionContext(
            model_version="test_model_v1.0",
            model_type="molecular_selector",
            confidence_score=0.85,
            input_data_hash="abc123",
            computational_resources={"cpu": "high", "memory": "8GB"},
            hyperparameters={"temperature": 0.7, "top_k": 10},
            training_data_version="training_v2.1",
            feature_importance={"molecular_weight": 0.3, "logp": 0.4},
            uncertainty_metrics={"epistemic": 0.1, "aleatoric": 0.05}
        )
        
        assert context.model_version == "test_model_v1.0"
        assert context.model_type == "molecular_selector"
        assert context.confidence_score == 0.85
        assert context.input_data_hash == "abc123"
        assert context.computational_resources["cpu"] == "high"
        assert context.hyperparameters["temperature"] == 0.7
        assert context.training_data_version == "training_v2.1"
        assert context.feature_importance["molecular_weight"] == 0.3
        assert context.uncertainty_metrics["epistemic"] == 0.1


class TestMoleculeSelectionDecision:
    """Test cases for MoleculeSelectionDecision dataclass."""
    
    def test_molecule_selection_decision_creation(self):
        """Test molecule selection decision creation."""
        decision = MoleculeSelectionDecision(
            selected_molecules=["CCO", "C1CCCCC1", "CC(=O)O"],
            candidate_pool_size=100,
            selection_criteria={"diversity": 0.8, "novelty": 0.6},
            diversity_metrics={"tanimoto_distance": 0.7, "scaffold_diversity": 0.5},
            predicted_properties={
                "CCO": {"logp": 1.2, "mw": 46.07},
                "C1CCCCC1": {"logp": 3.4, "mw": 84.16}
            },
            exclusion_reasons={
                "toxicity": ["CCCCCCCCCC", "C1CC1CCC"],
                "synthesis_difficulty": ["complex_molecule_1"]
            },
            similarity_analysis={"cluster_count": 3, "avg_similarity": 0.4}
        )
        
        assert len(decision.selected_molecules) == 3
        assert decision.candidate_pool_size == 100
        assert decision.selection_criteria["diversity"] == 0.8
        assert decision.diversity_metrics["tanimoto_distance"] == 0.7
        assert decision.predicted_properties["CCO"]["logp"] == 1.2
        assert "toxicity" in decision.exclusion_reasons
        assert decision.similarity_analysis["cluster_count"] == 3


class TestDecisionRecord:
    """Test cases for DecisionRecord dataclass."""
    
    def test_decision_record_creation(self):
        """Test decision record creation."""
        context = DecisionContext(
            model_version="v1.0",
            model_type="test",
            confidence_score=0.8,
            input_data_hash="hash123",
            computational_resources={},
            hyperparameters={}
        )
        
        decision_data = {"test_data": "value"}
        
        record = DecisionRecord(
            decision_id="decision_001",
            timestamp=datetime.now(timezone.utc),
            decision_type=DecisionType.MOLECULE_SELECTION,
            context=context,
            decision_data=decision_data,
            user_id="user_123",
            session_id="session_456",
            experiment_id="exp_789",
            parent_decision_id="parent_001",
            metadata={"custom_field": "custom_value"}
        )
        
        assert record.decision_id == "decision_001"
        assert record.decision_type == DecisionType.MOLECULE_SELECTION
        assert record.context.model_version == "v1.0"
        assert record.decision_data == {"test_data": "value"}
        assert record.user_id == "user_123"
        assert record.session_id == "session_456"
        assert record.experiment_id == "exp_789"
        assert record.parent_decision_id == "parent_001"
        assert record.metadata["custom_field"] == "custom_value"
        assert record.timestamp is not None


class TestExperimentLifecycleRecord:
    """Test cases for ExperimentLifecycleRecord dataclass."""
    
    def test_experiment_lifecycle_record_creation(self):
        """Test experiment lifecycle record creation."""
        record = ExperimentLifecycleRecord(
            experiment_id="exp_001",
            molecule_smiles="CCO",
            created_timestamp=datetime.now(timezone.utc),
            updated_timestamp=datetime.now(timezone.utc),
            current_phase=ExperimentPhase.SYNTHESIS,
            target_properties={"yield": 0.8, "purity": 0.95},
            actual_results={"yield": 0.75, "purity": 0.92},
            synthesis_route={"route_id": "route_001", "steps": 3},
            experimental_conditions={"temperature": 80, "time": 4},
            decision_chain=["decision_001", "decision_002"],
            quality_metrics={"r_squared": 0.95, "rmse": 0.05},
            resource_utilization={"cpu_hours": 2.5, "memory_gb": 8},
            timeline=[
                {"phase": "planning", "timestamp": "2024-01-01T10:00:00Z"},
                {"phase": "synthesis", "timestamp": "2024-01-01T14:00:00Z"}
            ],
            success_indicators={"yield_met": True, "purity_met": False},
            failure_reasons=[],
            metadata={"batch_id": "batch_001"}
        )
        
        assert record.experiment_id == "exp_001"
        assert record.molecule_smiles == "CCO"
        assert record.current_phase == ExperimentPhase.SYNTHESIS
        assert record.target_properties["yield"] == 0.8
        assert record.actual_results["yield"] == 0.75
        assert record.synthesis_route["steps"] == 3
        assert record.experimental_conditions["temperature"] == 80
        assert len(record.decision_chain) == 2
        assert record.quality_metrics["r_squared"] == 0.95
        assert record.resource_utilization["cpu_hours"] == 2.5
        assert len(record.timeline) == 2
        assert record.success_indicators["yield_met"] is True
        assert len(record.failure_reasons) == 0
        assert record.metadata["batch_id"] == "batch_001"


class TestInfluenceRelationship:
    """Test cases for InfluenceRelationship dataclass."""
    
    def test_influence_relationship_creation(self):
        """Test influence relationship creation."""
        relationship = InfluenceRelationship(
            relationship_id="rel_001",
            source_decision_id="decision_001",
            target_decision_id="decision_002",
            influence_type=InfluenceType.DIRECT_CONSEQUENCE,
            influence_strength=0.85,
            explanation="Strong positive influence based on successful results",
            quantitative_metrics={
                "correlation_coefficient": 0.75,
                "confidence_interval": [0.6, 0.9],
                "p_value": 0.001
            },
            timestamp=datetime.now(timezone.utc),
            metadata={"analysis_method": "pearson_correlation"}
        )
        
        assert relationship.relationship_id == "rel_001"
        assert relationship.source_decision_id == "decision_001"
        assert relationship.target_decision_id == "decision_002"
        assert relationship.influence_type == InfluenceType.DIRECT_CONSEQUENCE
        assert relationship.influence_strength == 0.85
        assert "successful results" in relationship.explanation
        assert relationship.quantitative_metrics["correlation_coefficient"] == 0.75
        assert relationship.quantitative_metrics["p_value"] == 0.001
        assert relationship.timestamp is not None
        assert relationship.metadata["analysis_method"] == "pearson_correlation"


class TestDecisionLogger:
    """Test cases for DecisionLogger class."""
    
    @pytest.fixture
    def decision_logger(self):
        """Create decision logger instance for testing."""
        return DecisionLogger(storage_backend="sqlite", storage_config={"db_path": ":memory:"})
    
    @pytest.fixture
    def sample_context(self):
        """Create sample decision context."""
        return DecisionContext(
            model_version="test_v1.0",
            model_type="test_model",
            confidence_score=0.7,
            input_data_hash="test_hash",
            computational_resources={"cpu": "medium"},
            hyperparameters={"param1": "value1"}
        )
    
    @pytest.fixture
    def sample_molecule_selection(self):
        """Create sample molecule selection decision."""
        return MoleculeSelectionDecision(
            selected_molecules=["CCO", "CCC"],
            candidate_pool_size=50,
            selection_criteria={"diversity": 0.7},
            diversity_metrics={"tanimoto": 0.6},
            predicted_properties={"CCO": {"mw": 46}},
            exclusion_reasons={"toxicity": ["toxic_mol"]}
        )
    
    @pytest.mark.asyncio
    async def test_decision_logger_initialization(self, decision_logger):
        """Test decision logger initialization."""
        assert decision_logger.storage_backend == "sqlite"
        assert len(decision_logger.decision_records) == 0
        assert len(decision_logger.experiment_records) == 0
        assert len(decision_logger.influence_relationships) == 0
    
    @pytest.mark.asyncio
    async def test_log_decision(self, decision_logger, sample_context, sample_molecule_selection):
        """Test logging a decision."""
        decision_id = await decision_logger.log_decision(
            decision_type=DecisionType.MOLECULE_SELECTION,
            context=sample_context,
            decision_data=sample_molecule_selection,
            user_id="test_user",
            session_id="test_session",
            experiment_id="test_experiment",
            metadata={"test_key": "test_value"}
        )
        
        assert decision_id is not None
        assert decision_id in decision_logger.decision_records
        
        record = decision_logger.decision_records[decision_id]
        assert record.decision_type == DecisionType.MOLECULE_SELECTION
        assert record.context.model_version == "test_v1.0"
        assert record.user_id == "test_user"
        assert record.session_id == "test_session"
        assert record.experiment_id == "test_experiment"
        assert record.metadata["test_key"] == "test_value"
    
    @pytest.mark.asyncio
    async def test_create_experiment_record(self, decision_logger):
        """Test creating experiment record."""
        experiment_id = await decision_logger.create_experiment_record(
            molecule_smiles="CCO",
            target_properties={"yield": 0.8},
            synthesis_route={"route_type": "suzuki"},
            experimental_conditions={"temp": 80, "time": 4}
        )
        
        assert experiment_id is not None
        assert experiment_id in decision_logger.experiment_records
        
        record = decision_logger.experiment_records[experiment_id]
        assert record.molecule_smiles == "CCO"
        assert record.target_properties["yield"] == 0.8
        assert record.synthesis_route["route_type"] == "suzuki"
        assert record.experimental_conditions["temp"] == 80
        assert record.current_phase == ExperimentPhase.PLANNING
        assert len(record.timeline) == 1
    
    @pytest.mark.asyncio
    async def test_update_experiment_phase(self, decision_logger):
        """Test updating experiment phase."""
        # Create experiment first
        experiment_id = await decision_logger.create_experiment_record(
            molecule_smiles="CCO",
            target_properties={"yield": 0.8},
            synthesis_route={},
            experimental_conditions={}
        )
        
        # Update phase
        await decision_logger.update_experiment_phase(
            experiment_id=experiment_id,
            new_phase=ExperimentPhase.SYNTHESIS,
            results={"yield": 0.75},
            quality_metrics={"purity": 0.9},
            description="Synthesis completed"
        )
        
        record = decision_logger.experiment_records[experiment_id]
        assert record.current_phase == ExperimentPhase.SYNTHESIS
        assert record.actual_results["yield"] == 0.75
        assert record.quality_metrics["purity"] == 0.9
        assert len(record.timeline) == 2
        assert record.timeline[-1]["description"] == "Synthesis completed"
    
    @pytest.mark.asyncio
    async def test_add_decision_to_experiment(self, decision_logger, sample_context):
        """Test adding decision to experiment chain."""
        # Create experiment
        experiment_id = await decision_logger.create_experiment_record(
            molecule_smiles="CCO",
            target_properties={},
            synthesis_route={},
            experimental_conditions={}
        )
        
        # Create decision
        decision_id = await decision_logger.log_decision(
            decision_type=DecisionType.EXPERIMENTAL_DESIGN,
            context=sample_context,
            decision_data={"design": "factorial"},
            experiment_id=experiment_id
        )
        
        # Add decision to experiment
        await decision_logger.add_decision_to_experiment(experiment_id, decision_id)
        
        record = decision_logger.experiment_records[experiment_id]
        assert decision_id in record.decision_chain
    
    @pytest.mark.asyncio
    async def test_log_influence_relationship(self, decision_logger, sample_context):
        """Test logging influence relationships."""
        # Create two decisions
        decision_id_1 = await decision_logger.log_decision(
            decision_type=DecisionType.MOLECULE_SELECTION,
            context=sample_context,
            decision_data={"molecules": ["CCO"]}
        )
        
        decision_id_2 = await decision_logger.log_decision(
            decision_type=DecisionType.SYNTHESIS_ROUTE,
            context=sample_context,
            decision_data={"route": "direct"},
            parent_decision_id=decision_id_1
        )
        
        # Log influence relationship
        relationship_id = await decision_logger.log_influence_relationship(
            source_decision_id=decision_id_1,
            target_decision_id=decision_id_2,
            influence_type=InfluenceType.DIRECT_CONSEQUENCE,
            influence_strength=0.9,
            explanation="Molecule selection directly influenced route planning",
            quantitative_metrics={"confidence": 0.95}
        )
        
        assert relationship_id is not None
        assert relationship_id in decision_logger.influence_relationships
        
        relationship = decision_logger.influence_relationships[relationship_id]
        assert relationship.source_decision_id == decision_id_1
        assert relationship.target_decision_id == decision_id_2
        assert relationship.influence_type == InfluenceType.DIRECT_CONSEQUENCE
        assert relationship.influence_strength == 0.9
        assert relationship.quantitative_metrics["confidence"] == 0.95
    
    @pytest.mark.asyncio
    async def test_get_decision_lineage(self, decision_logger, sample_context):
        """Test getting decision lineage."""
        # Create parent decision
        parent_id = await decision_logger.log_decision(
            decision_type=DecisionType.MOLECULE_SELECTION,
            context=sample_context,
            decision_data={"molecules": ["CCO"]}
        )
        
        # Create child decision
        child_id = await decision_logger.log_decision(
            decision_type=DecisionType.SYNTHESIS_ROUTE,
            context=sample_context,
            decision_data={"route": "direct"},
            parent_decision_id=parent_id
        )
        
        # Create grandchild decision
        grandchild_id = await decision_logger.log_decision(
            decision_type=DecisionType.EXPERIMENTAL_DESIGN,
            context=sample_context,
            decision_data={"design": "factorial"},
            parent_decision_id=child_id
        )
        
        # Add influence relationship
        await decision_logger.log_influence_relationship(
            source_decision_id=parent_id,
            target_decision_id=child_id,
            influence_type=InfluenceType.DIRECT_CONSEQUENCE,
            influence_strength=0.8,
            explanation="Test influence"
        )
        
        # Get lineage
        lineage = await decision_logger.get_decision_lineage(child_id)
        
        assert "decision" in lineage
        assert lineage["decision"]["decision_id"] == child_id
        assert len(lineage["ancestors"]) == 1
        assert lineage["ancestors"][0]["decision_id"] == parent_id
        assert len(lineage["descendants"]) == 1
        assert lineage["descendants"][0]["decision_id"] == grandchild_id
        assert len(lineage["influences"]["influenced_by"]) == 1
    
    @pytest.mark.asyncio
    async def test_get_experiment_trace(self, decision_logger, sample_context):
        """Test getting experiment trace."""
        # Create experiment
        experiment_id = await decision_logger.create_experiment_record(
            molecule_smiles="CCO",
            target_properties={"yield": 0.8},
            synthesis_route={},
            experimental_conditions={}
        )
        
        # Create decisions
        decision_id_1 = await decision_logger.log_decision(
            decision_type=DecisionType.MOLECULE_SELECTION,
            context=sample_context,
            decision_data={"molecules": ["CCO"]},
            experiment_id=experiment_id
        )
        
        decision_id_2 = await decision_logger.log_decision(
            decision_type=DecisionType.EXPERIMENTAL_DESIGN,
            context=sample_context,
            decision_data={"design": "factorial"},
            experiment_id=experiment_id
        )
        
        # Add decisions to experiment
        await decision_logger.add_decision_to_experiment(experiment_id, decision_id_1)
        await decision_logger.add_decision_to_experiment(experiment_id, decision_id_2)
        
        # Add influence relationship
        await decision_logger.log_influence_relationship(
            source_decision_id=decision_id_1,
            target_decision_id=decision_id_2,
            influence_type=InfluenceType.INFORMED_BY,
            influence_strength=0.6,
            explanation="Design informed by selection"
        )
        
        # Get trace
        trace = await decision_logger.get_experiment_trace(experiment_id)
        
        assert "experiment" in trace
        assert trace["experiment"]["experiment_id"] == experiment_id
        assert len(trace["decisions"]) == 2
        assert len(trace["influence_network"]) == 1
        
        decision_ids = [d["decision_id"] for d in trace["decisions"]]
        assert decision_id_1 in decision_ids
        assert decision_id_2 in decision_ids
    
    @pytest.mark.asyncio
    async def test_generate_audit_report(self, decision_logger, sample_context):
        """Test generating audit report."""
        # Create some test data
        experiment_id = await decision_logger.create_experiment_record(
            molecule_smiles="CCO",
            target_properties={"yield": 0.8},
            synthesis_route={},
            experimental_conditions={}
        )
        
        decision_id = await decision_logger.log_decision(
            decision_type=DecisionType.MOLECULE_SELECTION,
            context=sample_context,
            decision_data={"molecules": ["CCO"]},
            experiment_id=experiment_id
        )
        
        await decision_logger.log_influence_relationship(
            source_decision_id=decision_id,
            target_decision_id=decision_id,  # Self-reference for testing
            influence_type=InfluenceType.FEEDBACK_LOOP,
            influence_strength=0.5,
            explanation="Test feedback"
        )
        
        # Generate report
        report = await decision_logger.generate_audit_report()
        
        assert "report_metadata" in report
        assert "summary_statistics" in report
        assert "decisions" in report
        assert "experiments" in report
        assert "influence_relationships" in report
        
        assert report["summary_statistics"]["total_decisions"] == 1
        assert report["summary_statistics"]["total_experiments"] == 1
        assert report["summary_statistics"]["total_influence_relationships"] == 1
        assert DecisionType.MOLECULE_SELECTION.value in report["summary_statistics"]["decision_type_distribution"]
    
    @pytest.mark.asyncio
    async def test_generate_audit_report_with_filters(self, decision_logger, sample_context):
        """Test generating audit report with filters."""
        # Create test data
        experiment_id_1 = await decision_logger.create_experiment_record(
            molecule_smiles="CCO",
            target_properties={},
            synthesis_route={},
            experimental_conditions={}
        )
        
        experiment_id_2 = await decision_logger.create_experiment_record(
            molecule_smiles="CCC",
            target_properties={},
            synthesis_route={},
            experimental_conditions={}
        )
        
        await decision_logger.log_decision(
            decision_type=DecisionType.MOLECULE_SELECTION,
            context=sample_context,
            decision_data={"molecules": ["CCO"]},
            experiment_id=experiment_id_1
        )
        
        await decision_logger.log_decision(
            decision_type=DecisionType.SYNTHESIS_ROUTE,
            context=sample_context,
            decision_data={"route": "direct"},
            experiment_id=experiment_id_2
        )
        
        # Generate filtered report
        report = await decision_logger.generate_audit_report(
            experiment_ids=[experiment_id_1],
            decision_types=[DecisionType.MOLECULE_SELECTION]
        )
        
        assert report["summary_statistics"]["total_decisions"] == 1
        assert report["summary_statistics"]["total_experiments"] == 1
        assert DecisionType.MOLECULE_SELECTION.value in report["summary_statistics"]["decision_type_distribution"]
        assert DecisionType.SYNTHESIS_ROUTE.value not in report["summary_statistics"]["decision_type_distribution"]
    
    @pytest.mark.asyncio
    async def test_sqlite_persistence(self, decision_logger, sample_context):
        """Test SQLite persistence functionality."""
        # Wait for initialization
        await asyncio.sleep(0.1)
        
        # Create a decision
        decision_id = await decision_logger.log_decision(
            decision_type=DecisionType.MOLECULE_SELECTION,
            context=sample_context,
            decision_data={"test": "data"}
        )
        
        # Create experiment
        experiment_id = await decision_logger.create_experiment_record(
            molecule_smiles="CCO",
            target_properties={"yield": 0.8},
            synthesis_route={},
            experimental_conditions={}
        )
        
        # Verify data persisted in memory
        assert decision_id in decision_logger.decision_records
        assert experiment_id in decision_logger.experiment_records
        
        # The actual SQLite persistence testing would require
        # more complex setup with proper database file handling


class TestDecisionLoggingEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def decision_logger(self):
        """Create decision logger for edge case testing."""
        return DecisionLogger(storage_backend="sqlite", storage_config={"db_path": ":memory:"})
    
    @pytest.mark.asyncio
    async def test_update_nonexistent_experiment(self, decision_logger):
        """Test updating non-existent experiment."""
        with pytest.raises(ValueError, match="Experiment .* not found"):
            await decision_logger.update_experiment_phase(
                experiment_id="nonexistent",
                new_phase=ExperimentPhase.COMPLETED
            )
    
    @pytest.mark.asyncio
    async def test_add_decision_to_nonexistent_experiment(self, decision_logger):
        """Test adding decision to non-existent experiment."""
        with pytest.raises(ValueError, match="Experiment .* not found"):
            await decision_logger.add_decision_to_experiment("nonexistent", "decision_id")
    
    @pytest.mark.asyncio
    async def test_get_lineage_nonexistent_decision(self, decision_logger):
        """Test getting lineage for non-existent decision."""
        lineage = await decision_logger.get_decision_lineage("nonexistent")
        assert lineage == {}
    
    @pytest.mark.asyncio
    async def test_get_trace_nonexistent_experiment(self, decision_logger):
        """Test getting trace for non-existent experiment."""
        trace = await decision_logger.get_experiment_trace("nonexistent")
        assert trace == {}
    
    @pytest.mark.asyncio
    async def test_empty_audit_report(self, decision_logger):
        """Test generating audit report with no data."""
        report = await decision_logger.generate_audit_report()
        
        assert report["summary_statistics"]["total_decisions"] == 0
        assert report["summary_statistics"]["total_experiments"] == 0
        assert report["summary_statistics"]["total_influence_relationships"] == 0
        assert len(report["decisions"]) == 0
        assert len(report["experiments"]) == 0


class TestDecisionLoggingIntegration:
    """Integration tests for decision logging system."""
    
    @pytest.mark.asyncio
    async def test_complete_experimental_workflow(self):
        """Test complete experimental workflow logging."""
        logger = DecisionLogger(storage_backend="sqlite", storage_config={"db_path": ":memory:"})
        
        # Step 1: Molecule selection
        context = DecisionContext(
            model_version="v1.0",
            model_type="selector",
            confidence_score=0.8,
            input_data_hash="hash1",
            computational_resources={},
            hyperparameters={}
        )
        
        molecule_decision = MoleculeSelectionDecision(
            selected_molecules=["CCO"],
            candidate_pool_size=100,
            selection_criteria={"novelty": 0.7},
            diversity_metrics={"diversity": 0.6},
            predicted_properties={"CCO": {"mw": 46}},
            exclusion_reasons={}
        )
        
        molecule_decision_id = await logger.log_decision(
            decision_type=DecisionType.MOLECULE_SELECTION,
            context=context,
            decision_data=molecule_decision,
            user_id="researcher_1",
            session_id="session_1"
        )
        
        # Step 2: Synthesis route planning
        route_decision = SynthesisRouteDecision(
            target_molecule="CCO",
            selected_route={"route_id": "r1", "steps": 2},
            alternative_routes=[{"route_id": "r2", "steps": 3}],
            route_scoring={"r1": 0.9, "r2": 0.7},
            reagent_availability={"ethanol": True},
            predicted_yield=0.8,
            predicted_difficulty=0.3
        )
        
        route_decision_id = await logger.log_decision(
            decision_type=DecisionType.SYNTHESIS_ROUTE,
            context=context,
            decision_data=route_decision,
            user_id="researcher_1",
            session_id="session_1",
            parent_decision_id=molecule_decision_id
        )
        
        # Step 3: Create experiment
        experiment_id = await logger.create_experiment_record(
            molecule_smiles="CCO",
            target_properties={"yield": 0.8, "purity": 0.95},
            synthesis_route={"route_id": "r1"},
            experimental_conditions={"temp": 80, "time": 4}
        )
        
        # Step 4: Add decisions to experiment
        await logger.add_decision_to_experiment(experiment_id, molecule_decision_id)
        await logger.add_decision_to_experiment(experiment_id, route_decision_id)
        
        # Step 5: Log influence relationship
        influence_id = await logger.log_influence_relationship(
            source_decision_id=molecule_decision_id,
            target_decision_id=route_decision_id,
            influence_type=InfluenceType.DIRECT_CONSEQUENCE,
            influence_strength=0.9,
            explanation="Molecule selection directly determined synthesis route",
            quantitative_metrics={"correlation": 0.85}
        )
        
        # Step 6: Update experiment with results
        await logger.update_experiment_phase(
            experiment_id=experiment_id,
            new_phase=ExperimentPhase.COMPLETED,
            results={"yield": 0.75, "purity": 0.92},
            quality_metrics={"r_squared": 0.94},
            description="Experiment completed successfully"
        )
        
        # Verify complete workflow
        experiment_trace = await logger.get_experiment_trace(experiment_id)
        assert len(experiment_trace["decisions"]) == 2
        assert len(experiment_trace["influence_network"]) == 1
        
        decision_lineage = await logger.get_decision_lineage(route_decision_id)
        assert len(decision_lineage["ancestors"]) == 1
        assert decision_lineage["ancestors"][0]["decision_id"] == molecule_decision_id
        
        audit_report = await logger.generate_audit_report()
        assert audit_report["summary_statistics"]["total_decisions"] == 2
        assert audit_report["summary_statistics"]["total_experiments"] == 1
        assert audit_report["summary_statistics"]["total_influence_relationships"] == 1
    
    @pytest.mark.asyncio
    async def test_parallel_experiment_logging(self):
        """Test logging multiple experiments in parallel."""
        logger = DecisionLogger(storage_backend="sqlite", storage_config={"db_path": ":memory:"})
        
        context = DecisionContext(
            model_version="v1.0",
            model_type="test",
            confidence_score=0.7,
            input_data_hash="hash",
            computational_resources={},
            hyperparameters={}
        )
        
        # Create multiple experiments concurrently
        experiment_tasks = []
        for i in range(5):
            task = logger.create_experiment_record(
                molecule_smiles=f"molecule_{i}",
                target_properties={"prop": float(i)},
                synthesis_route={},
                experimental_conditions={}
            )
            experiment_tasks.append(task)
        
        experiment_ids = await asyncio.gather(*experiment_tasks)
        
        # Create decisions for each experiment
        decision_tasks = []
        for i, exp_id in enumerate(experiment_ids):
            task = logger.log_decision(
                decision_type=DecisionType.EXPERIMENTAL_DESIGN,
                context=context,
                decision_data={"design": f"design_{i}"},
                experiment_id=exp_id
            )
            decision_tasks.append(task)
        
        decision_ids = await asyncio.gather(*decision_tasks)
        
        # Verify all experiments and decisions were created
        assert len(experiment_ids) == 5
        assert len(decision_ids) == 5
        assert len(logger.experiment_records) == 5
        assert len(logger.decision_records) == 5
        
        # Verify unique IDs
        assert len(set(experiment_ids)) == 5
        assert len(set(decision_ids)) == 5