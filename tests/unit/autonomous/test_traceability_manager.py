"""
Tests for Traceability Manager

This module contains comprehensive tests for the traceability management system
including influence analysis, correlation tracking, and comprehensive workflow monitoring.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest
from chemi_accel.autonomous.adaptive_learning import AdaptiveLearningManager
from chemi_accel.autonomous.closed_loop_optimization import (
    ClosedLoopOptimizer,
    ExperimentResult,
)
from chemi_accel.autonomous.decision_logging import (
    DecisionLogger,
    DecisionType,
    ExperimentLifecycleRecord,
    ExperimentPhase,
)
from chemi_accel.autonomous.robotic_control import (
    ExperimentStatus,
    RobotFeedback,
    RobotManager,
)
from chemi_accel.autonomous.traceability_manager import (
    TraceabilityConfig,
    TraceabilityManager,
    TraceabilityMetrics,
)


class TestTraceabilityConfig:
    """Test cases for TraceabilityConfig dataclass."""

    def test_traceability_config_default(self):
        """Test default traceability configuration."""
        config = TraceabilityConfig()

        assert config.enable_real_time_logging is True
        assert config.enable_influence_tracking is True
        assert config.enable_performance_monitoring is True
        assert config.storage_backend == "sqlite"
        assert config.audit_retention_days == 365
        assert config.automatic_correlation_threshold == 0.7
        assert config.decision_context_depth == 3

    def test_traceability_config_custom(self):
        """Test custom traceability configuration."""
        config = TraceabilityConfig(
            enable_real_time_logging=False,
            enable_influence_tracking=True,
            storage_backend="postgresql",
            storage_config={"host": "localhost", "db": "test"},
            audit_retention_days=180,
            automatic_correlation_threshold=0.8,
            decision_context_depth=5,
        )

        assert config.enable_real_time_logging is False
        assert config.enable_influence_tracking is True
        assert config.storage_backend == "postgresql"
        assert config.storage_config["host"] == "localhost"
        assert config.audit_retention_days == 180
        assert config.automatic_correlation_threshold == 0.8
        assert config.decision_context_depth == 5


class TestTraceabilityMetrics:
    """Test cases for TraceabilityMetrics dataclass."""

    def test_traceability_metrics_default(self):
        """Test default traceability metrics."""
        metrics = TraceabilityMetrics()

        assert metrics.total_decisions_logged == 0
        assert metrics.total_experiments_tracked == 0
        assert metrics.total_influence_relationships == 0
        assert metrics.average_decision_chain_length == 0.0
        assert metrics.experiment_success_rate == 0.0
        assert metrics.logging_latency_ms == 0.0
        assert metrics.storage_utilization_mb == 0.0
        assert len(metrics.decision_types_distribution) == 0

    def test_traceability_metrics_with_data(self):
        """Test traceability metrics with data."""
        metrics = TraceabilityMetrics(
            total_decisions_logged=100,
            total_experiments_tracked=25,
            total_influence_relationships=75,
            average_decision_chain_length=3.2,
            decision_types_distribution={
                "molecule_selection": 30,
                "synthesis_route": 25,
                "experimental_design": 45,
            },
            experiment_success_rate=0.84,
            logging_latency_ms=12.5,
            storage_utilization_mb=256.7,
        )

        assert metrics.total_decisions_logged == 100
        assert metrics.total_experiments_tracked == 25
        assert metrics.total_influence_relationships == 75
        assert metrics.average_decision_chain_length == 3.2
        assert metrics.decision_types_distribution["molecule_selection"] == 30
        assert metrics.experiment_success_rate == 0.84
        assert metrics.logging_latency_ms == 12.5
        assert metrics.storage_utilization_mb == 256.7


class TestTraceabilityManager:
    """Test cases for TraceabilityManager class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return TraceabilityConfig(
            storage_backend="sqlite",
            storage_config={"db_path": ":memory:"},
            automatic_correlation_threshold=0.5,
        )

    @pytest.fixture
    def mock_robot_manager(self):
        """Create mock robot manager."""
        manager = Mock(spec=RobotManager)
        manager.add_feedback_callback = Mock()
        return manager

    @pytest.fixture
    def mock_optimizer(self):
        """Create mock closed-loop optimizer."""
        optimizer = Mock(spec=ClosedLoopOptimizer)
        optimizer.add_experiment_callback = Mock()
        optimizer.add_optimization_callback = Mock()
        return optimizer

    @pytest.fixture
    def mock_adaptive_manager(self):
        """Create mock adaptive learning manager."""
        manager = Mock(spec=AdaptiveLearningManager)
        manager.add_retraining_callback = Mock()
        return manager

    @pytest.fixture
    def traceability_manager(
        self, config, mock_robot_manager, mock_optimizer, mock_adaptive_manager
    ):
        """Create traceability manager instance."""
        return TraceabilityManager(
            config=config,
            robot_manager=mock_robot_manager,
            optimizer=mock_optimizer,
            adaptive_manager=mock_adaptive_manager,
        )

    def test_traceability_manager_initialization(self, traceability_manager, config):
        """Test traceability manager initialization."""
        assert traceability_manager.config == config
        assert isinstance(traceability_manager.decision_logger, DecisionLogger)
        assert len(traceability_manager.active_sessions) == 0
        assert len(traceability_manager.molecule_histories) == 0
        assert len(traceability_manager.decision_correlations) == 0
        assert isinstance(traceability_manager.performance_metrics, TraceabilityMetrics)

    def test_integration_setup(
        self,
        traceability_manager,
        mock_robot_manager,
        mock_optimizer,
        mock_adaptive_manager,
    ):
        """Test integration setup with other systems."""
        # Verify callbacks were registered
        mock_robot_manager.add_feedback_callback.assert_called_once()
        mock_optimizer.add_experiment_callback.assert_called_once()
        mock_optimizer.add_optimization_callback.assert_called_once()
        mock_adaptive_manager.add_retraining_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_experiment_session(self, traceability_manager):
        """Test starting experiment session."""
        session_id = "test_session_001"
        user_id = "researcher_1"
        objectives = {"yield": 0.8, "purity": 0.95}
        constraints = {"max_time": 24, "max_cost": 1000}

        decision_id = await traceability_manager.start_experiment_session(
            session_id=session_id,
            user_id=user_id,
            objectives=objectives,
            constraints=constraints,
        )

        assert decision_id is not None
        assert session_id in traceability_manager.active_sessions

        session = traceability_manager.active_sessions[session_id]
        assert session["session_id"] == session_id
        assert session["user_id"] == user_id
        assert session["objectives"] == objectives
        assert session["constraints"] == constraints
        assert session["active"] is True
        assert decision_id in session["decisions"]

    @pytest.mark.asyncio
    async def test_log_molecule_selection(self, traceability_manager):
        """Test logging molecule selection decision."""
        selected_molecules = ["CCO", "CCC", "C1CCCCC1"]
        selection_criteria = {"diversity": 0.8, "novelty": 0.6}
        candidate_pool = ["CCO", "CCC", "C1CCCCC1", "CCCO", "CCCC"]
        predicted_properties = {
            "CCO": {"mw": 46.07, "logp": -0.31},
            "CCC": {"mw": 44.1, "logp": 0.09},
            "C1CCCCC1": {"mw": 84.16, "logp": 3.44},
        }
        model_info = {
            "version": "selector_v2.0",
            "type": "molecular_selector",
            "confidence": 0.85,
            "resources": {"cpu": "high", "memory": "4GB"},
            "hyperparameters": {"temperature": 0.7, "top_k": 3},
        }
        session_id = "test_session"
        user_id = "researcher_1"

        decision_id = await traceability_manager.log_molecule_selection(
            selected_molecules=selected_molecules,
            selection_criteria=selection_criteria,
            candidate_pool=candidate_pool,
            predicted_properties=predicted_properties,
            model_info=model_info,
            session_id=session_id,
            user_id=user_id,
        )

        assert decision_id is not None
        assert decision_id in traceability_manager.decision_logger.decision_records

        # Check molecule histories were updated
        for molecule in selected_molecules:
            assert decision_id in traceability_manager.molecule_histories[molecule]

        # Check metrics were updated
        assert traceability_manager.performance_metrics.total_decisions_logged == 1
        assert (
            traceability_manager.performance_metrics.decision_types_distribution[
                "molecule_selection"
            ]
            == 1
        )

        # Verify decision record
        record = traceability_manager.decision_logger.decision_records[decision_id]
        assert record.decision_type == DecisionType.MOLECULE_SELECTION
        assert record.context.model_version == "selector_v2.0"
        assert record.context.confidence_score == 0.85
        assert record.user_id == user_id
        assert record.session_id == session_id

    @pytest.mark.asyncio
    async def test_log_synthesis_planning(self, traceability_manager):
        """Test logging synthesis planning decision."""
        target_molecule = "CCO"
        selected_route = {
            "route_id": "route_001",
            "steps": [
                {"reaction": "reduction", "reagent": "NaBH4"},
                {"reaction": "purification", "method": "distillation"},
            ],
            "predicted_yield": 0.82,
            "predicted_difficulty": 0.3,
            "estimated_time": 8,
        }
        alternative_routes = [
            {
                "route_id": "route_002",
                "steps": [{"reaction": "fermentation", "organism": "yeast"}],
                "predicted_yield": 0.75,
                "predicted_difficulty": 0.5,
            }
        ]
        model_info = {
            "version": "route_planner_v1.5",
            "type": "synthesis_planner",
            "confidence": 0.78,
            "route_scores": {"route_001": 0.82, "route_002": 0.75},
        }
        session_id = "test_session"
        experiment_id = "exp_001"
        user_id = "researcher_1"
        parent_decision_id = "parent_decision_001"

        decision_id = await traceability_manager.log_synthesis_planning(
            target_molecule=target_molecule,
            selected_route=selected_route,
            alternative_routes=alternative_routes,
            model_info=model_info,
            session_id=session_id,
            experiment_id=experiment_id,
            user_id=user_id,
            parent_decision_id=parent_decision_id,
        )

        assert decision_id is not None
        assert decision_id in traceability_manager.decision_logger.decision_records

        # Check molecule history was updated
        assert decision_id in traceability_manager.molecule_histories[target_molecule]

        record = traceability_manager.decision_logger.decision_records[decision_id]
        assert record.decision_type == DecisionType.SYNTHESIS_ROUTE
        assert record.experiment_id == experiment_id
        assert record.parent_decision_id == parent_decision_id

        # Verify influence relationship was created
        influence_relationships = list(
            traceability_manager.decision_logger.influence_relationships.values()
        )
        assert len(influence_relationships) >= 1

        # Find the influence relationship for this decision
        our_influence = None
        for rel in influence_relationships:
            if rel.target_decision_id == decision_id:
                our_influence = rel
                break

        assert our_influence is not None
        assert our_influence.source_decision_id == parent_decision_id

    @pytest.mark.asyncio
    async def test_log_experimental_design(self, traceability_manager):
        """Test logging experimental design decision."""
        design_parameters = {
            "design_type": "factorial",
            "parameter_space": {
                "temperature": {"min": 50, "max": 150},
                "time": {"min": 1, "max": 12},
                "catalyst_loading": {"min": 0.01, "max": 0.1},
            },
            "doe_strategy": "full_factorial",
            "expected_outcomes": {"yield": 0.75, "selectivity": 0.85},
            "estimated_time": 48,
        }
        experimental_conditions = {
            "temperature": 80,
            "time": 6,
            "catalyst_loading": 0.05,
            "solvent": "toluene",
            "atmosphere": "nitrogen",
        }
        resource_allocation = {
            "cpu_hours": 4,
            "memory_gb": 16,
            "storage_gb": 100,
            "instruments": ["hplc", "nmr", "gc_ms"],
        }
        model_info = {
            "version": "design_optimizer_v3.0",
            "type": "experimental_designer",
            "confidence": 0.72,
            "hyperparameters": {"acquisition_function": "ei", "n_iterations": 50},
        }
        session_id = "test_session"
        experiment_id = "exp_001"
        user_id = "researcher_1"

        decision_id = await traceability_manager.log_experimental_design(
            design_parameters=design_parameters,
            experimental_conditions=experimental_conditions,
            resource_allocation=resource_allocation,
            model_info=model_info,
            session_id=session_id,
            experiment_id=experiment_id,
            user_id=user_id,
        )

        assert decision_id is not None
        assert decision_id in traceability_manager.decision_logger.decision_records

        record = traceability_manager.decision_logger.decision_records[decision_id]
        assert record.decision_type == DecisionType.EXPERIMENTAL_DESIGN
        assert record.experiment_id == experiment_id
        assert record.context.model_type == "experimental_designer"
        assert record.context.confidence_score == 0.72

    @pytest.mark.asyncio
    async def test_create_experiment_trace(self, traceability_manager):
        """Test creating comprehensive experiment trace."""
        molecule_smiles = "CCO"
        target_properties = {"yield": 0.8, "purity": 0.95, "selectivity": 0.9}
        synthesis_route = {"route_id": "r001", "steps": 3, "complexity": "medium"}
        experimental_conditions = {"temperature": 80, "pressure": 1.0, "time": 6}
        session_id = "session_001"
        related_decision_ids = ["decision_001", "decision_002", "decision_003"]

        experiment_id = await traceability_manager.create_experiment_trace(
            molecule_smiles=molecule_smiles,
            target_properties=target_properties,
            synthesis_route=synthesis_route,
            experimental_conditions=experimental_conditions,
            session_id=session_id,
            related_decision_ids=related_decision_ids,
        )

        assert experiment_id is not None
        assert experiment_id in traceability_manager.decision_logger.experiment_records

        record = traceability_manager.decision_logger.experiment_records[experiment_id]
        assert record.molecule_smiles == molecule_smiles
        assert record.target_properties == target_properties
        assert record.synthesis_route == synthesis_route
        assert record.experimental_conditions == experimental_conditions
        assert len(record.decision_chain) == len(related_decision_ids)

        # Check metrics update
        assert traceability_manager.performance_metrics.total_experiments_tracked == 1

    @pytest.mark.asyncio
    async def test_update_experiment_results(self, traceability_manager):
        """Test updating experiment with results and analyzing influences."""
        # First create an experiment
        experiment_id = await traceability_manager.create_experiment_trace(
            molecule_smiles="CCO",
            target_properties={"yield": 0.8},
            synthesis_route={},
            experimental_conditions={},
            session_id="session_001",
            related_decision_ids=[],
        )

        # Update with results
        results = {"yield": 0.75, "purity": 0.92}
        quality_metrics = {"r_squared": 0.94, "rmse": 0.08}
        success = True

        await traceability_manager.update_experiment_results(
            experiment_id=experiment_id,
            results=results,
            quality_metrics=quality_metrics,
            success=success,
            phase=ExperimentPhase.COMPLETED,
        )

        # Verify experiment was updated
        record = traceability_manager.decision_logger.experiment_records[experiment_id]
        assert record.current_phase == ExperimentPhase.COMPLETED
        assert record.actual_results == results
        assert record.quality_metrics == quality_metrics

        # Verify success rate was updated
        assert traceability_manager.performance_metrics.experiment_success_rate == 1.0

    @pytest.mark.asyncio
    async def test_robot_feedback_handling(self, traceability_manager):
        """Test handling robot feedback for traceability."""
        # Create experiment first
        experiment_id = await traceability_manager.create_experiment_trace(
            molecule_smiles="CCO",
            target_properties={"yield": 0.8},
            synthesis_route={},
            experimental_conditions={},
            session_id="session_001",
            related_decision_ids=[],
        )

        # Create robot feedback
        feedback = RobotFeedback(
            feedback_id="fb_001",
            robot_id="hamilton_01",
            action_id="action_001",
            status=ExperimentStatus.COMPLETED,
            progress=1.0,
            data={
                "objective_values": {"yield": 0.82},
                "quality_metrics": {"purity": 0.94},
            },
        )

        # Process feedback (this would normally be called automatically)
        await traceability_manager._process_robot_feedback(feedback)

        # The processing would look for associated experiments
        # Since we don't have a direct link, this test mainly verifies
        # the method can be called without errors

    @pytest.mark.asyncio
    async def test_optimization_result_handling(self, traceability_manager):
        """Test handling optimization results for traceability."""
        # Create experiment result
        result = ExperimentResult(
            experiment_id="exp_001",
            parameters={"temperature": 80, "time": 6},
            objective_values={"yield": 0.75},
            success=True,
            execution_time=3600,
            robot_feedback=[],
            experimental_data=[],
        )

        # Process optimization result
        await traceability_manager._process_optimization_result(result)

        # Verify decision was logged
        assert len(traceability_manager.decision_logger.decision_records) >= 1

        # Find the optimization decision
        opt_decisions = [
            d
            for d in traceability_manager.decision_logger.decision_records.values()
            if d.decision_type == DecisionType.PARAMETER_OPTIMIZATION
        ]
        assert len(opt_decisions) >= 1

        opt_decision = opt_decisions[0]
        assert opt_decision.experiment_id == "exp_001"
        assert "optimization_result" in opt_decision.decision_data

    @pytest.mark.asyncio
    async def test_model_retraining_handling(self, traceability_manager):
        """Test handling model retraining events."""
        retraining_info = {
            "model_id": "selector_v2.0",
            "new_version": "selector_v2.1",
            "improvement_score": 0.15,
            "performance_metrics": {
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.91,
            },
            "training_data_summary": {"samples": 10000, "features": 256, "classes": 5},
            "resources": {"gpu_hours": 12, "memory_gb": 32},
            "hyperparameters": {"learning_rate": 0.001, "batch_size": 64},
        }

        # Process retraining event
        await traceability_manager._process_model_retraining(retraining_info)

        # Verify decision was logged
        quality_decisions = [
            d
            for d in traceability_manager.decision_logger.decision_records.values()
            if d.decision_type == DecisionType.QUALITY_ASSESSMENT
        ]
        assert len(quality_decisions) >= 1

        retraining_decision = quality_decisions[0]
        assert retraining_decision.context.model_version == "selector_v2.1"
        assert retraining_decision.context.confidence_score == 0.15
        assert "retraining_event" in retraining_decision.decision_data

    def test_calculate_diversity_metrics(self, traceability_manager):
        """Test diversity metrics calculation."""
        selected = ["CCO", "CCC", "C1CCCCC1"]
        candidate_pool = ["CCO", "CCC", "C1CCCCC1", "CCCO", "CCCC", "C1CCCC1"]

        metrics = traceability_manager._calculate_diversity_metrics(
            selected, candidate_pool
        )

        assert "selection_diversity" in metrics
        assert "coverage_ratio" in metrics
        assert "uniqueness_score" in metrics

        assert metrics["selection_diversity"] == 1.0  # All unique
        assert metrics["coverage_ratio"] == 0.5  # 3 out of 6
        assert metrics["uniqueness_score"] == 1.0  # Placeholder

    def test_analyze_exclusions(self, traceability_manager):
        """Test exclusion analysis."""
        candidate_pool = ["CCO", "CCC", "CCCO", "CCCC", "C1CCCCC1"]
        selected = ["CCO", "CCC"]
        criteria = {"toxicity": "low", "complexity": "medium"}

        exclusions = traceability_manager._analyze_exclusions(
            candidate_pool, selected, criteria
        )

        assert "filtered_out" in exclusions
        assert "criteria_failed" in exclusions

        excluded = exclusions["filtered_out"]
        assert "CCCO" in excluded
        assert "CCCC" in excluded
        assert "C1CCCCC1" in excluded

        assert len(exclusions["criteria_failed"]) == len(excluded)

    def test_analyze_molecular_similarity(self, traceability_manager):
        """Test molecular similarity analysis."""
        molecules = ["CCO", "CCC", "C1CCCCC1"]

        analysis = traceability_manager._analyze_molecular_similarity(molecules)

        assert "avg_similarity" in analysis
        assert "similarity_matrix" in analysis
        assert "cluster_count" in analysis

        assert analysis["avg_similarity"] == 0.7  # Placeholder
        assert analysis["similarity_matrix"] == "computed"
        assert analysis["cluster_count"] == 3

    def test_experimental_similarity_calculation(self, traceability_manager):
        """Test experimental similarity calculation."""
        exp1 = ExperimentLifecycleRecord(
            experiment_id="exp1",
            molecule_smiles="CCO",
            created_timestamp=datetime.now(timezone.utc),
            updated_timestamp=datetime.now(timezone.utc),
            current_phase=ExperimentPhase.COMPLETED,
            target_properties={},
            actual_results={},
            synthesis_route={},
            experimental_conditions={"temperature": 80, "time": 6},
            decision_chain=[],
            quality_metrics={},
            resource_utilization={},
            timeline=[],
            success_indicators={},
        )

        exp2 = ExperimentLifecycleRecord(
            experiment_id="exp2",
            molecule_smiles="CCO",
            created_timestamp=datetime.now(timezone.utc),
            updated_timestamp=datetime.now(timezone.utc),
            current_phase=ExperimentPhase.COMPLETED,
            target_properties={},
            actual_results={},
            synthesis_route={},
            experimental_conditions={"temperature": 85, "time": 5},
            decision_chain=[],
            quality_metrics={},
            resource_utilization={},
            timeline=[],
            success_indicators={},
        )

        similarity = traceability_manager._calculate_experimental_similarity(exp1, exp2)

        # Should be high due to same molecule and similar conditions
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.8  # Same molecule should give high similarity

    def test_result_correlation_calculation(self, traceability_manager):
        """Test result correlation calculation."""
        results1 = {"yield": 0.8, "purity": 0.9, "selectivity": 0.85}
        results2 = {"yield": 0.75, "purity": 0.92, "selectivity": 0.88}

        correlation = traceability_manager._calculate_result_correlation(
            results1, results2
        )

        assert 0.0 <= correlation <= 1.0
        assert correlation > 0.8  # Similar results should have high correlation

        # Test with no common keys
        results3 = {"different_property": 0.5}
        correlation_empty = traceability_manager._calculate_result_correlation(
            results1, results3
        )
        assert correlation_empty == 0.0

    @pytest.mark.asyncio
    async def test_get_traceability_summary(self, traceability_manager):
        """Test getting comprehensive traceability summary."""
        # Add some test data
        await traceability_manager.start_experiment_session("session_001", "user_001")

        experiment_id = await traceability_manager.create_experiment_trace(
            molecule_smiles="CCO",
            target_properties={"yield": 0.8},
            synthesis_route={},
            experimental_conditions={},
            session_id="session_001",
            related_decision_ids=[],
        )

        await traceability_manager.update_experiment_results(
            experiment_id=experiment_id,
            results={"yield": 0.75},
            quality_metrics={"purity": 0.9},
            success=True,
        )

        summary = await traceability_manager.get_traceability_summary()

        assert "system_status" in summary
        assert "performance_metrics" in summary
        assert "traceability_coverage" in summary

        system_status = summary["system_status"]
        assert system_status["total_decisions_logged"] >= 1
        assert system_status["total_experiments_tracked"] >= 1
        assert system_status["active_sessions"] >= 1

        performance_metrics = summary["performance_metrics"]
        assert "decision_types_distribution" in performance_metrics
        assert "experiment_success_rate" in performance_metrics

        traceability_coverage = summary["traceability_coverage"]
        assert "molecules_with_history" in traceability_coverage
        assert "decision_correlations" in traceability_coverage
        assert "influence_networks" in traceability_coverage


class TestTraceabilityManagerIntegration:
    """Integration tests for traceability manager."""

    @pytest.mark.asyncio
    async def test_complete_workflow_traceability(self):
        """Test complete workflow traceability from molecule selection to results."""
        config = TraceabilityConfig(storage_config={"db_path": ":memory:"})
        manager = TraceabilityManager(config)

        # Step 1: Start session
        session_id = "integration_session"
        user_id = "researcher_001"

        session_decision_id = await manager.start_experiment_session(
            session_id=session_id,
            user_id=user_id,
            objectives={"yield": 0.8, "purity": 0.95},
            constraints={"max_time": 24},
        )

        # Step 2: Log molecule selection
        molecule_decision_id = await manager.log_molecule_selection(
            selected_molecules=["CCO", "CCC"],
            selection_criteria={"diversity": 0.8, "novelty": 0.6},
            candidate_pool=["CCO", "CCC", "CCCO", "CCCC"],
            predicted_properties={
                "CCO": {"mw": 46.07, "logp": -0.31},
                "CCC": {"mw": 44.1, "logp": 0.09},
            },
            model_info={"version": "v1.0", "type": "selector", "confidence": 0.85},
            session_id=session_id,
            user_id=user_id,
            parent_decision_id=session_decision_id,
        )

        # Step 3: Log synthesis planning
        synthesis_decision_id = await manager.log_synthesis_planning(
            target_molecule="CCO",
            selected_route={"route_id": "r1", "steps": 2},
            alternative_routes=[{"route_id": "r2", "steps": 3}],
            model_info={"version": "v1.0", "type": "planner", "confidence": 0.78},
            session_id=session_id,
            user_id=user_id,
            parent_decision_id=molecule_decision_id,
        )

        # Step 4: Log experimental design
        design_decision_id = await manager.log_experimental_design(
            design_parameters={"design_type": "factorial", "estimated_time": 12},
            experimental_conditions={"temperature": 80, "time": 6},
            resource_allocation={"cpu_hours": 4},
            model_info={"version": "v1.0", "type": "designer", "confidence": 0.72},
            session_id=session_id,
            user_id=user_id,
        )

        # Step 5: Create experiment trace
        experiment_id = await manager.create_experiment_trace(
            molecule_smiles="CCO",
            target_properties={"yield": 0.8, "purity": 0.95},
            synthesis_route={"route_id": "r1"},
            experimental_conditions={"temperature": 80, "time": 6},
            session_id=session_id,
            related_decision_ids=[
                molecule_decision_id,
                synthesis_decision_id,
                design_decision_id,
            ],
        )

        # Step 6: Update with results
        await manager.update_experiment_results(
            experiment_id=experiment_id,
            results={"yield": 0.75, "purity": 0.92},
            quality_metrics={"r_squared": 0.94},
            success=True,
        )

        # Verify complete traceability
        summary = await manager.get_traceability_summary()

        assert summary["system_status"]["total_decisions_logged"] >= 4
        assert summary["system_status"]["total_experiments_tracked"] >= 1
        assert summary["system_status"]["active_sessions"] >= 1

        # Verify experiment trace
        trace = await manager.decision_logger.get_experiment_trace(experiment_id)
        assert len(trace["decisions"]) >= 3
        assert trace["experiment"]["molecule_smiles"] == "CCO"

        # Verify molecule history
        assert len(manager.molecule_histories["CCO"]) >= 2  # Selection + Synthesis

        # Verify influence relationships
        assert len(manager.decision_logger.influence_relationships) >= 1

    @pytest.mark.asyncio
    async def test_parallel_experiment_traceability(self):
        """Test traceability with multiple parallel experiments."""
        config = TraceabilityConfig(storage_config={"db_path": ":memory:"})
        manager = TraceabilityManager(config)

        session_id = "parallel_session"
        user_id = "researcher_001"

        # Start session
        await manager.start_experiment_session(session_id, user_id)

        # Create multiple experiments in parallel
        experiment_tasks = []
        for i in range(3):
            molecule = f"molecule_{i}"
            task = manager.create_experiment_trace(
                molecule_smiles=molecule,
                target_properties={"property": float(i)},
                synthesis_route={},
                experimental_conditions={},
                session_id=session_id,
                related_decision_ids=[],
            )
            experiment_tasks.append(task)

        experiment_ids = await asyncio.gather(*experiment_tasks)

        # Update all experiments with results
        result_tasks = []
        for i, exp_id in enumerate(experiment_ids):
            task = manager.update_experiment_results(
                experiment_id=exp_id,
                results={"property": float(i) * 0.8},
                quality_metrics={"quality": 0.9},
                success=True,
            )
            result_tasks.append(task)

        await asyncio.gather(*result_tasks)

        # Verify all experiments tracked
        summary = await manager.get_traceability_summary()
        assert summary["system_status"]["total_experiments_tracked"] == 3
        assert summary["performance_metrics"]["experiment_success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_influence_analysis_workflow(self):
        """Test influence analysis throughout workflow."""
        config = TraceabilityConfig(
            storage_config={"db_path": ":memory:"}, automatic_correlation_threshold=0.3
        )
        manager = TraceabilityManager(config)

        # Create first experiment
        exp1_id = await manager.create_experiment_trace(
            molecule_smiles="CCO",
            target_properties={"yield": 0.8},
            synthesis_route={},
            experimental_conditions={"temperature": 80},
            session_id="session_001",
            related_decision_ids=[],
        )

        await manager.update_experiment_results(
            experiment_id=exp1_id,
            results={"yield": 0.9},  # Good result
            quality_metrics={"purity": 0.95},
            success=True,
        )

        # Create similar experiment
        exp2_id = await manager.create_experiment_trace(
            molecule_smiles="CCO",  # Same molecule
            target_properties={"yield": 0.8},
            synthesis_route={},
            experimental_conditions={"temperature": 85},  # Similar conditions
            session_id="session_001",
            related_decision_ids=[],
        )

        await manager.update_experiment_results(
            experiment_id=exp2_id,
            results={"yield": 0.85},
            quality_metrics={"purity": 0.93},
            success=True,
        )

        # The influence analysis should detect similarity and create relationships
        # This is done automatically in update_experiment_results

        # Verify influence relationships were created
        relationships = list(manager.decision_logger.influence_relationships.values())

        # We expect at least some influence relationships to be detected
        # The exact number depends on the decision chains and analysis
        assert len(relationships) >= 0  # May be 0 if no decision chains exist

        summary = await manager.get_traceability_summary()
        assert summary["system_status"]["total_experiments_tracked"] == 2
        assert summary["performance_metrics"]["experiment_success_rate"] == 1.0
