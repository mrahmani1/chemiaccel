"""
Tests for AI-Robot Integration System

This module contains comprehensive tests for the AI-robot integration system including
AI decision engines, experimental design, and closed-loop AI-driven automation.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import List, Dict, Any

from chemi_accel.autonomous.ai_robot_integration import (
    AIDecisionType, SynthesisComplexity, AIDecision, SynthesisTask,
    ExperimentalDesign, AIDecisionEngine, ActiveLearningDecisionEngine,
    InverseDesignDecisionEngine, AIRobotIntegrationManager,
    create_comprehensive_ai_robot_system, create_active_learning_robot_system,
    create_inverse_design_robot_system
)
from chemi_accel.autonomous.robotic_control import RobotManager, RobotCapability
from chemi_accel.autonomous.adaptive_learning import AdaptiveLearningManager
from chemi_accel.autonomous.closed_loop_optimization import ClosedLoopOptimizer


class TestAIDecision:
    """Test cases for AIDecision dataclass."""
    
    def test_ai_decision_creation(self):
        """Test basic AI decision creation."""
        decision = AIDecision(
            decision_id="ai_decision_001",
            decision_type=AIDecisionType.ACTIVE_LEARNING_QUERY,
            ai_system="active_learning_engine",
            confidence=0.85,
            reasoning={
                "uncertainty_level": 0.7,
                "information_gain": 0.6,
                "exploration_score": 0.8
            },
            parameters={
                "temperature": 120,
                "reaction_time": 4,
                "concentration": 0.5
            },
            priority=3,
            constraints={"max_temperature": 150},
            metadata={"model_version": "v1.2"}
        )
        
        assert decision.decision_id == "ai_decision_001"
        assert decision.decision_type == AIDecisionType.ACTIVE_LEARNING_QUERY
        assert decision.ai_system == "active_learning_engine"
        assert decision.confidence == 0.85
        assert decision.reasoning["uncertainty_level"] == 0.7
        assert decision.parameters["temperature"] == 120
        assert decision.priority == 3
        assert decision.constraints["max_temperature"] == 150
        assert decision.metadata["model_version"] == "v1.2"
        assert decision.created_at is not None


class TestSynthesisTask:
    """Test cases for SynthesisTask dataclass."""
    
    def test_synthesis_task_creation(self):
        """Test synthesis task creation."""
        target_molecule = {
            "smiles": "c1ccc(cc1)C",
            "molecular_weight": 92.14,
            "predicted_properties": {"logp": 2.73}
        }
        
        synthesis_route = [
            {
                "step_number": 1,
                "reaction_type": "methylation",
                "reactants": ["benzene", "methyl_iodide"],
                "conditions": {"temperature": 80, "time": 4},
                "estimated_yield": 0.85
            }
        ]
        
        task = SynthesisTask(
            task_id="synthesis_001",
            target_molecule=target_molecule,
            synthesis_route=synthesis_route,
            complexity=SynthesisComplexity.SIMPLE,
            success_probability=0.9,
            estimated_duration=14400.0,
            resource_requirements={"reactor": 1, "catalyst": 1},
            safety_considerations=["standard_precautions"],
            quality_metrics={"target_yield": 0.8, "target_purity": 0.95},
            ai_decision_id="ai_decision_001"
        )
        
        assert task.task_id == "synthesis_001"
        assert task.target_molecule["smiles"] == "c1ccc(cc1)C"
        assert len(task.synthesis_route) == 1
        assert task.complexity == SynthesisComplexity.SIMPLE
        assert task.success_probability == 0.9
        assert task.estimated_duration == 14400.0
        assert task.resource_requirements["reactor"] == 1
        assert "standard_precautions" in task.safety_considerations
        assert task.quality_metrics["target_yield"] == 0.8
        assert task.ai_decision_id == "ai_decision_001"
        assert task.created_at is not None


class TestExperimentalDesign:
    """Test cases for ExperimentalDesign dataclass."""
    
    def test_experimental_design_creation(self):
        """Test experimental design creation."""
        synthesis_task = SynthesisTask(
            task_id="task_001",
            target_molecule={"smiles": "CCO"},
            synthesis_route=[{"step": 1}],
            complexity=SynthesisComplexity.SIMPLE,
            success_probability=0.8,
            estimated_duration=3600.0,
            resource_requirements={},
            safety_considerations=[],
            quality_metrics={},
            ai_decision_id="decision_001"
        )
        
        ai_decision = AIDecision(
            decision_id="decision_001",
            decision_type=AIDecisionType.INVERSE_DESIGN_CANDIDATE,
            ai_system="inverse_design",
            confidence=0.8,
            reasoning={},
            parameters={}
        )
        
        design = ExperimentalDesign(
            design_id="design_001",
            synthesis_tasks=[synthesis_task],
            analysis_protocols=[{"protocol": "NMR", "techniques": ["1H-NMR", "13C-NMR"]}],
            quality_controls=[{"type": "blank", "frequency": 0.1}],
            success_criteria={"min_yield": 0.7, "min_purity": 0.9},
            risk_assessment={"safety_level": "low", "complexity": "simple"},
            ai_decisions=[ai_decision],
            total_estimated_time=3600.0,
            resource_utilization={"reactor_time": 1.0}
        )
        
        assert design.design_id == "design_001"
        assert len(design.synthesis_tasks) == 1
        assert design.synthesis_tasks[0].task_id == "task_001"
        assert len(design.analysis_protocols) == 1
        assert design.analysis_protocols[0]["protocol"] == "NMR"
        assert len(design.quality_controls) == 1
        assert design.success_criteria["min_yield"] == 0.7
        assert design.risk_assessment["safety_level"] == "low"
        assert len(design.ai_decisions) == 1
        assert design.total_estimated_time == 3600.0
        assert design.resource_utilization["reactor_time"] == 1.0
        assert design.created_at is not None


class TestActiveLearningDecisionEngine:
    """Test cases for ActiveLearningDecisionEngine."""
    
    @pytest.fixture
    def mock_adaptive_manager(self):
        """Create mock adaptive learning manager."""
        manager = Mock(spec=AdaptiveLearningManager)
        manager.get_model_version.return_value = "v2.1"
        return manager
    
    @pytest.fixture
    def al_engine(self, mock_adaptive_manager):
        """Create ActiveLearningDecisionEngine instance."""
        return ActiveLearningDecisionEngine(mock_adaptive_manager)
    
    @pytest.mark.asyncio
    async def test_make_decision(self, al_engine):
        """Test AI decision making."""
        context = {
            "parameter_space": {
                "temperature": (50, 200),
                "time": (1, 12),
                "concentration": (0.1, 2.0)
            },
            "best_parameters": {"temperature": 120, "time": 6, "concentration": 0.8},
            "previous_experiments": []
        }
        
        constraints = {
            "safety_limits": {"temperature": 180},
            "resource_limits": {"time": 10}
        }
        
        decision = await al_engine.make_decision(context, constraints)
        
        assert isinstance(decision, AIDecision)
        assert decision.decision_type == AIDecisionType.ACTIVE_LEARNING_QUERY
        assert decision.ai_system == "active_learning_engine"
        assert 0.0 <= decision.confidence <= 1.0
        assert 1 <= decision.priority <= 5
        assert "temperature" in decision.parameters
        assert "time" in decision.parameters
        assert "concentration" in decision.parameters
        assert decision.constraints == constraints
        assert "query_strategy" in decision.metadata
    
    @pytest.mark.asyncio
    async def test_assess_model_uncertainty(self, al_engine, mock_adaptive_manager):
        """Test model uncertainty assessment."""
        # Mock performance monitor
        mock_performance_monitor = Mock()
        mock_performance_monitor.get_performance_summary.return_value = {
            "recent_performance": [
                {"confidence": 0.8},
                {"confidence": 0.7},
                {"confidence": 0.9},
                {"confidence": 0.75}
            ]
        }
        mock_adaptive_manager.performance_monitor = mock_performance_monitor
        
        context = {}
        uncertainty = await al_engine._assess_model_uncertainty(context)
        
        assert 0.0 <= uncertainty <= 1.0
    
    @pytest.mark.asyncio
    async def test_generate_query_candidates(self, al_engine):
        """Test query candidate generation."""
        context = {
            "parameter_space": {
                "temperature": (60, 180),
                "catalyst_loading": (0.01, 0.1)
            },
            "best_parameters": {"temperature": 120, "catalyst_loading": 0.05}
        }
        
        constraints = {"safety_limits": {"temperature": 160}}
        
        candidates = await al_engine._generate_query_candidates(context, constraints)
        
        assert len(candidates) == 20  # Should generate 20 candidates
        for candidate in candidates:
            assert "parameters" in candidate
            assert "information_gain" in candidate
            assert "diversity_score" in candidate
            assert "exploration_score" in candidate
            assert "feasibility_score" in candidate
            assert "combined_score" in candidate
            
            # Check parameter bounds
            params = candidate["parameters"]
            assert 60 <= params["temperature"] <= 180
            assert 0.01 <= params["catalyst_loading"] <= 0.1
    
    def test_calculate_diversity_score(self, al_engine):
        """Test diversity score calculation."""
        context = {
            "previous_experiments": [
                {"parameters": {"temperature": 100, "time": 4}},
                {"parameters": {"temperature": 150, "time": 8}}
            ],
            "parameter_space": {
                "temperature": (50, 200),
                "time": (1, 12)
            }
        }
        
        # Test point far from existing experiments
        far_params = {"temperature": 60, "time": 2}
        far_score = al_engine._calculate_diversity_score(far_params, context)
        
        # Test point close to existing experiments
        close_params = {"temperature": 105, "time": 4.5}
        close_score = al_engine._calculate_diversity_score(close_params, context)
        
        assert far_score > close_score  # Far point should have higher diversity
    
    def test_assess_feasibility(self, al_engine):
        """Test feasibility assessment."""
        parameters = {"temperature": 140, "catalyst_loading": 0.08}
        
        # Test with no constraints
        feasibility1 = al_engine._assess_feasibility(parameters, {})
        assert feasibility1 == 1.0
        
        # Test with resource constraints
        constraints_ok = {"resource_limits": {"temperature": 150}}
        feasibility2 = al_engine._assess_feasibility(parameters, constraints_ok)
        assert feasibility2 == 1.0
        
        # Test with violated constraints
        constraints_violated = {"resource_limits": {"temperature": 120}}
        feasibility3 = al_engine._assess_feasibility(parameters, constraints_violated)
        assert feasibility3 < 1.0
        
        # Test with safety violations
        safety_violated = {"safety_limits": {"temperature": 130}}
        feasibility4 = al_engine._assess_feasibility(parameters, safety_violated)
        assert feasibility4 < 0.5  # Should be heavily penalized
    
    @pytest.mark.asyncio
    async def test_create_fallback_decision(self, al_engine):
        """Test fallback decision creation."""
        context = {
            "parameter_space": {
                "temperature": (70, 170),
                "pressure": (1, 5)
            }
        }
        constraints = {"max_pressure": 3}
        
        decision = al_engine._create_fallback_decision(context, constraints)
        
        assert isinstance(decision, AIDecision)
        assert decision.decision_type == AIDecisionType.EXPLORATORY_EXPERIMENT
        assert decision.confidence == 0.3  # Low confidence for fallback
        assert decision.reasoning["fallback"] is True
        assert "temperature" in decision.parameters
        assert "pressure" in decision.parameters
        assert 70 <= decision.parameters["temperature"] <= 170
        assert 1 <= decision.parameters["pressure"] <= 5
    
    def test_get_engine_info(self, al_engine):
        """Test engine information retrieval."""
        info = al_engine.get_engine_info()
        
        assert info["engine_type"] == "active_learning"
        assert "description" in info
        assert "performance_metrics" in info
        assert "query_history_size" in info
        assert "supported_strategies" in info
        assert "uncertainty_sampling" in info["supported_strategies"]


class TestInverseDesignDecisionEngine:
    """Test cases for InverseDesignDecisionEngine."""
    
    @pytest.fixture
    def id_engine(self):
        """Create InverseDesignDecisionEngine instance."""
        return InverseDesignDecisionEngine()
    
    @pytest.mark.asyncio
    async def test_make_decision(self, id_engine):
        """Test inverse design decision making."""
        context = {
            "target_properties": {
                "molecular_weight": (200, 400),
                "logp": (2, 4),
                "activity": "maximize"
            }
        }
        
        constraints = {
            "max_molecular_weight": 350,
            "min_synthetic_accessibility": 0.6
        }
        
        decision = await id_engine.make_decision(context, constraints)
        
        assert isinstance(decision, AIDecision)
        assert decision.decision_type == AIDecisionType.INVERSE_DESIGN_CANDIDATE
        assert decision.ai_system == "inverse_design_engine"
        assert 0.0 <= decision.confidence <= 1.0
        assert "target_molecule" in decision.parameters
        assert "synthesis_route" in decision.parameters
        assert "synthesis_parameters" in decision.parameters
        assert decision.constraints == constraints
        assert "design_method" in decision.metadata
    
    @pytest.mark.asyncio
    async def test_generate_molecular_candidates(self, id_engine):
        """Test molecular candidate generation."""
        target_properties = {
            "molecular_weight": (150, 300),
            "logp": (1, 3),
            "activity": "maximize"
        }
        
        constraints = {"max_molecular_weight": 250}
        
        candidates = await id_engine._generate_molecular_candidates(target_properties, constraints)
        
        assert len(candidates) <= 10  # May be filtered by constraints
        for candidate in candidates:
            assert "candidate_id" in candidate
            assert "smiles" in candidate
            assert "predicted_properties" in candidate
            assert "sa_score" in candidate
            assert "novelty" in candidate
            assert 0.0 <= candidate["sa_score"] <= 1.0
            assert 0.0 <= candidate["novelty"] <= 1.0
            
            # Check predicted properties
            for prop_name in target_properties:
                if prop_name in candidate["predicted_properties"]:
                    if isinstance(target_properties[prop_name], tuple):
                        min_val, max_val = target_properties[prop_name]
                        pred_val = candidate["predicted_properties"][prop_name]
                        # Values should be in reasonable range (allowing some flexibility)
                        assert min_val * 0.8 <= pred_val <= max_val * 1.2
    
    def test_generate_random_smiles(self, id_engine):
        """Test SMILES generation."""
        smiles = id_engine._generate_random_smiles()
        
        assert isinstance(smiles, str)
        assert len(smiles) > 0
        # Basic SMILES validation - should contain some typical chemistry symbols
        assert any(char in smiles for char in ['c', 'C', '(', ')', '='])
    
    def test_satisfies_constraints(self, id_engine):
        """Test constraint satisfaction checking."""
        candidate = {
            "predicted_properties": {"molecular_weight": 250},
            "sa_score": 0.7
        }
        
        # Test passing constraints
        constraints_ok = {
            "max_molecular_weight": 300,
            "min_synthetic_accessibility": 0.6
        }
        assert id_engine._satisfies_constraints(candidate, constraints_ok) is True
        
        # Test failing molecular weight constraint
        constraints_mw_fail = {
            "max_molecular_weight": 200,
            "min_synthetic_accessibility": 0.6
        }
        assert id_engine._satisfies_constraints(candidate, constraints_mw_fail) is False
        
        # Test failing SA constraint
        constraints_sa_fail = {
            "max_molecular_weight": 300,
            "min_synthetic_accessibility": 0.8
        }
        assert id_engine._satisfies_constraints(candidate, constraints_sa_fail) is False
    
    def test_select_best_design(self, id_engine):
        """Test best design selection."""
        candidates = [
            {
                "candidate_id": "mol_1",
                "predicted_properties": {"activity": 0.8, "logp": 2.5},
                "sa_score": 0.7,
                "novelty": 0.6
            },
            {
                "candidate_id": "mol_2",
                "predicted_properties": {"activity": 0.9, "logp": 2.2},
                "sa_score": 0.8,
                "novelty": 0.5
            },
            {
                "candidate_id": "mol_3",
                "predicted_properties": {"activity": 0.7, "logp": 2.8},
                "sa_score": 0.6,
                "novelty": 0.8
            }
        ]
        
        target_properties = {
            "activity": "maximize",
            "logp": (2.0, 3.0)
        }
        
        best_candidate = id_engine._select_best_design(candidates, target_properties)
        
        assert best_candidate["candidate_id"] in ["mol_1", "mol_2", "mol_3"]
        # The selection should favor candidates with good property matching and SA scores
    
    @pytest.mark.asyncio
    async def test_generate_synthesis_route(self, id_engine):
        """Test synthesis route generation."""
        molecular_candidate = {
            "smiles": "c1ccc(cc1)CCO",
            "sa_score": 0.8
        }
        
        synthesis_route = await id_engine._generate_synthesis_route(molecular_candidate)
        
        assert isinstance(synthesis_route, list)
        assert len(synthesis_route) >= 1
        
        for step in synthesis_route:
            assert "step_number" in step
            assert "reaction_type" in step
            assert "reactants" in step
            assert "conditions" in step
            assert "estimated_yield" in step
            assert "estimated_time" in step
            assert "safety_rating" in step
            
            assert 0.0 <= step["estimated_yield"] <= 1.0
            assert step["estimated_time"] > 0
            assert step["safety_rating"] in ["low", "medium", "high"]
    
    def test_extract_synthesis_parameters(self, id_engine):
        """Test synthesis parameter extraction."""
        synthesis_route = [
            {
                "conditions": {"temperature": 80, "catalyst_loading": 0.05},
                "estimated_time": 4,
                "reactants": ["benzene", "ethanol"]
            },
            {
                "conditions": {"temperature": 120, "catalyst_loading": 0.02},
                "estimated_time": 6,
                "reactants": ["intermediate", "acid"]
            }
        ]
        
        parameters = id_engine._extract_synthesis_parameters(synthesis_route)
        
        assert "step_1_temperature" in parameters
        assert "step_1_time" in parameters
        assert "step_1_catalyst_loading" in parameters
        assert "step_2_temperature" in parameters
        assert "step_2_time" in parameters
        assert "step_2_catalyst_loading" in parameters
        
        assert parameters["step_1_temperature"] == 80
        assert parameters["step_1_time"] == 4
        assert parameters["step_2_temperature"] == 120
        assert parameters["step_2_time"] == 6
    
    @pytest.mark.asyncio
    async def test_create_fallback_design_decision(self, id_engine):
        """Test fallback design decision creation."""
        context = {"target_properties": {"activity": "maximize"}}
        constraints = {"max_complexity": 3}
        
        decision = id_engine._create_fallback_design_decision(context, constraints)
        
        assert isinstance(decision, AIDecision)
        assert decision.decision_type == AIDecisionType.INVERSE_DESIGN_CANDIDATE
        assert decision.confidence == 0.4  # Low confidence for fallback
        assert decision.reasoning["fallback"] is True
        assert "target_molecule" in decision.parameters
        assert "synthesis_route" in decision.parameters
        assert decision.parameters["target_molecule"]["smiles"] == "c1ccc(cc1)C"  # Toluene fallback
    
    def test_get_engine_info(self, id_engine):
        """Test engine information retrieval."""
        info = id_engine.get_engine_info()
        
        assert info["engine_type"] == "inverse_design"
        assert "description" in info
        assert "performance_metrics" in info
        assert "design_history_size" in info
        assert "supported_representations" in info
        assert "supported_properties" in info
        assert "smiles" in info["supported_representations"]
        assert "molecular_weight" in info["supported_properties"]


class TestAIRobotIntegrationManager:
    """Test cases for AIRobotIntegrationManager."""
    
    @pytest.fixture
    def mock_robot_manager(self):
        """Create mock robot manager."""
        manager = Mock(spec=RobotManager)
        manager.execute_experiment_plan = AsyncMock(return_value="plan_001")
        manager.add_feedback_callback = Mock()
        manager.find_robots_with_capability.return_value = ["synthesis_robot", "analysis_robot"]
        manager.get_robot_capabilities.return_value = [
            RobotCapability("chemical_synthesis", "Synthesis", "Chemical reactions"),
            RobotCapability("analysis", "Analysis", "Analytical techniques")
        ]
        return manager
    
    @pytest.fixture
    def mock_adaptive_manager(self):
        """Create mock adaptive learning manager."""
        manager = Mock(spec=AdaptiveLearningManager)
        manager.add_new_data = AsyncMock()
        return manager
    
    @pytest.fixture
    def mock_optimizer(self):
        """Create mock closed-loop optimizer."""
        optimizer = Mock(spec=ClosedLoopOptimizer)
        optimizer.optimization_state = Mock()
        optimizer.experiment_results = []
        optimizer.config = Mock()
        optimizer.config.parameter_space = {"temperature": (50, 150)}
        return optimizer
    
    @pytest.fixture
    def integration_manager(self, mock_robot_manager, mock_adaptive_manager, mock_optimizer):
        """Create AIRobotIntegrationManager instance."""
        return AIRobotIntegrationManager(
            mock_robot_manager, 
            mock_adaptive_manager, 
            mock_optimizer
        )
    
    def test_initialization(self, integration_manager):
        """Test integration manager initialization."""
        assert len(integration_manager.ai_engines) >= 2  # At least AL and ID engines
        assert "active_learning" in integration_manager.ai_engines
        assert "inverse_design" in integration_manager.ai_engines
        assert integration_manager.integration_metrics["total_ai_decisions"] == 0
        assert not integration_manager._is_running
    
    @pytest.mark.asyncio
    async def test_start_stop_integration(self, integration_manager):
        """Test starting and stopping integration."""
        await integration_manager.start_integration()
        assert integration_manager._is_running is True
        assert integration_manager._integration_task is not None
        
        await integration_manager.stop_integration()
        assert integration_manager._is_running is False
    
    @pytest.mark.asyncio
    async def test_identify_decision_needs(self, integration_manager, mock_optimizer):
        """Test identification of decision needs."""
        # Mock optimization state with low success rate
        mock_optimizer.experiment_results = [
            Mock(success=True),
            Mock(success=False),
            Mock(success=False),
            Mock(success=True),
            Mock(success=False)
        ]
        
        decision_requests = await integration_manager._identify_decision_needs()
        
        assert len(decision_requests) >= 1
        # Should request active learning due to low success rate
        al_requests = [req for req in decision_requests if req["type"] == "active_learning"]
        assert len(al_requests) >= 1
    
    @pytest.mark.asyncio
    async def test_process_decision_request(self, integration_manager):
        """Test processing decision requests."""
        request = {
            "type": "active_learning",
            "context": {
                "parameter_space": {"temperature": (50, 150)},
                "recent_results": []
            },
            "priority": 3
        }
        
        # Mock the AI engine to return a decision
        mock_engine = Mock()
        mock_decision = AIDecision(
            decision_id="test_decision",
            decision_type=AIDecisionType.ACTIVE_LEARNING_QUERY,
            ai_system="active_learning_engine",
            confidence=0.8,
            reasoning={},
            parameters={"temperature": 120}
        )
        mock_engine.make_decision = AsyncMock(return_value=mock_decision)
        integration_manager.ai_engines["active_learning"] = mock_engine
        
        await integration_manager._process_decision_request(request)
        
        # Check that decision was processed
        assert len(integration_manager.decision_history) == 1
        assert integration_manager.integration_metrics["total_ai_decisions"] == 1
        mock_engine.make_decision.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_active_learning_query(self, integration_manager, mock_robot_manager):
        """Test execution of active learning query."""
        decision = AIDecision(
            decision_id="al_decision_001",
            decision_type=AIDecisionType.ACTIVE_LEARNING_QUERY,
            ai_system="active_learning_engine",
            confidence=0.85,
            reasoning={},
            parameters={"temperature": 120, "time": 4, "concentration": 0.5}
        )
        
        await integration_manager._execute_active_learning_query(decision)
        
        # Verify experiment plan was executed
        mock_robot_manager.execute_experiment_plan.assert_called_once()
        
        # Check that experimental design was created
        assert len(integration_manager.experimental_designs) == 1
        design_id = f"al_design_{decision.decision_id}"
        assert design_id in integration_manager.experimental_designs
        
        design = integration_manager.experimental_designs[design_id]
        assert len(design.synthesis_tasks) == 1
        assert design.ai_decisions[0] == decision
    
    @pytest.mark.asyncio
    async def test_execute_inverse_design_synthesis(self, integration_manager, mock_robot_manager):
        """Test execution of inverse design synthesis."""
        target_molecule = {
            "smiles": "c1ccc(cc1)CCO",
            "molecular_weight": 122.16
        }
        
        synthesis_route = [
            {
                "step_number": 1,
                "reaction_type": "friedel_crafts",
                "conditions": {"temperature": 80, "time": 4},
                "estimated_yield": 0.75,
                "estimated_time": 4
            }
        ]
        
        decision = AIDecision(
            decision_id="id_decision_001",
            decision_type=AIDecisionType.INVERSE_DESIGN_CANDIDATE,
            ai_system="inverse_design_engine",
            confidence=0.9,
            reasoning={},
            parameters={
                "target_molecule": target_molecule,
                "synthesis_route": synthesis_route,
                "synthesis_parameters": {"step_1_temperature": 80, "step_1_time": 4}
            }
        )
        
        await integration_manager._execute_inverse_design_synthesis(decision)
        
        # Verify experiment plan was executed
        mock_robot_manager.execute_experiment_plan.assert_called_once()
        
        # Check that experimental design was created
        assert len(integration_manager.experimental_designs) == 1
        design_id = f"id_design_{decision.decision_id}"
        assert design_id in integration_manager.experimental_designs
        
        design = integration_manager.experimental_designs[design_id]
        assert len(design.synthesis_tasks) == 1
        assert design.synthesis_tasks[0].target_molecule == target_molecule
        assert design.synthesis_tasks[0].synthesis_route == synthesis_route
    
    def test_assess_synthesis_complexity(self, integration_manager):
        """Test synthesis complexity assessment."""
        # Simple synthesis (1 step)
        simple_route = [{"step": 1}]
        complexity = integration_manager._assess_synthesis_complexity(simple_route)
        assert complexity == SynthesisComplexity.SIMPLE
        
        # Moderate synthesis (2 steps)
        moderate_route = [{"step": 1}, {"step": 2}]
        complexity = integration_manager._assess_synthesis_complexity(moderate_route)
        assert complexity == SynthesisComplexity.MODERATE
        
        # Complex synthesis (3-4 steps)
        complex_route = [{"step": 1}, {"step": 2}, {"step": 3}]
        complexity = integration_manager._assess_synthesis_complexity(complex_route)
        assert complexity == SynthesisComplexity.COMPLEX
        
        # Novel synthesis (5+ steps)
        novel_route = [{"step": i} for i in range(1, 6)]
        complexity = integration_manager._assess_synthesis_complexity(novel_route)
        assert complexity == SynthesisComplexity.NOVEL
    
    def test_calculate_resource_requirements(self, integration_manager):
        """Test resource requirements calculation."""
        synthesis_route = [
            {"reaction_type": "suzuki_coupling", "conditions": {}},
            {"reaction_type": "reduction", "conditions": {}},
            {"reaction_type": "oxidation", "conditions": {}}
        ]
        
        requirements = integration_manager._calculate_resource_requirements(synthesis_route)
        
        assert "reactor" in requirements
        assert "analytical_time" in requirements
        assert requirements["reactor"] == 3  # One per step
        assert requirements["analytical_time"] == 1.5  # 0.5 per step
        
        # Check special requirements
        assert "inert_atmosphere" in requirements  # Due to coupling reaction
        assert "hydrogen_source" in requirements  # Due to reduction
    
    def test_assess_safety_considerations(self, integration_manager):
        """Test safety considerations assessment."""
        synthesis_route = [
            {
                "conditions": {"temperature": 150, "pressure": 3},
                "reactants": ["benzene", "sodium"]
            },
            {
                "conditions": {"temperature": 80, "pressure": 1},
                "reactants": ["toluene", "catalyst"]
            }
        ]
        
        safety_considerations = integration_manager._assess_safety_considerations(synthesis_route)
        
        assert "high_temperature_handling" in safety_considerations  # Due to 150°C
        assert "pressure_vessel_required" in safety_considerations  # Due to 3 bar
        assert "handle_alkali_metals" in safety_considerations  # Due to sodium
    
    def test_assess_design_risk(self, integration_manager):
        """Test design risk assessment."""
        # Low risk (simple synthesis)
        simple_route = [{"step": 1}]
        risk = integration_manager._assess_design_risk(simple_route)
        assert risk["safety_level"] == "low"
        assert risk["complexity_risk"] == "low"
        
        # High risk (complex synthesis)
        complex_route = [{"step": i} for i in range(1, 6)]
        risk = integration_manager._assess_design_risk(complex_route)
        assert risk["safety_level"] == "high"
        assert risk["complexity_risk"] == "high"
    
    @pytest.mark.asyncio
    async def test_convert_design_to_robot_plans(self, integration_manager, mock_robot_manager):
        """Test conversion of experimental design to robot plans."""
        synthesis_task = SynthesisTask(
            task_id="task_001",
            target_molecule={"smiles": "CCO"},
            synthesis_route=[
                {
                    "reaction_type": "reduction",
                    "conditions": {"temperature": 60, "pressure": 1},
                    "reactants": ["aldehyde", "hydrogen"],
                    "estimated_yield": 0.8,
                    "estimated_time": 3
                }
            ],
            complexity=SynthesisComplexity.SIMPLE,
            success_probability=0.9,
            estimated_duration=10800,
            resource_requirements={},
            safety_considerations=[],
            quality_metrics={},
            ai_decision_id="decision_001"
        )
        
        design = ExperimentalDesign(
            design_id="design_001",
            synthesis_tasks=[synthesis_task],
            analysis_protocols=[{
                "protocol": "standard_analysis",
                "techniques": ["NMR", "HPLC"]
            }],
            quality_controls=[],
            success_criteria={},
            risk_assessment={},
            ai_decisions=[],
            total_estimated_time=10800,
            resource_utilization={}
        )
        
        plans = await integration_manager._convert_design_to_robot_plans(design)
        
        assert len(plans) == 1
        plan = plans[0]
        
        assert plan.plan_id == f"{design.design_id}_plan_0"
        assert plan.experiment_id == f"{design.design_id}_exp_0"
        assert len(plan.actions) >= 1  # At least synthesis action
        
        # Check synthesis action
        synthesis_actions = [action for action in plan.actions if action.capability_id == "chemical_synthesis"]
        assert len(synthesis_actions) == 1
        synthesis_action = synthesis_actions[0]
        assert synthesis_action.robot_id == "synthesis_robot"
        assert synthesis_action.parameters["reaction_type"] == "reduction"
        
        # Check analysis action (if analysis robots available)
        analysis_actions = [action for action in plan.actions if action.capability_id == "analysis"]
        if mock_robot_manager.find_robots_with_capability("analysis"):
            assert len(analysis_actions) >= 1
    
    def test_get_integration_status(self, integration_manager):
        """Test integration status retrieval."""
        status = integration_manager.get_integration_status()
        
        assert "is_running" in status
        assert "ai_engines" in status
        assert "performance_metrics" in status
        assert "active_designs" in status
        assert "active_synthesis_tasks" in status
        assert "decision_history_size" in status
        assert "recent_decisions" in status
        
        assert status["is_running"] is False  # Initially not running
        assert len(status["ai_engines"]) >= 2  # AL and ID engines
        assert "active_learning" in status["ai_engines"]
        assert "inverse_design" in status["ai_engines"]
    
    def test_add_callbacks(self, integration_manager):
        """Test callback registration."""
        decision_callback = Mock()
        synthesis_callback = Mock()
        design_callback = Mock()
        
        integration_manager.add_decision_callback(decision_callback)
        integration_manager.add_synthesis_callback(synthesis_callback)
        integration_manager.add_design_callback(design_callback)
        
        assert decision_callback in integration_manager.decision_callbacks
        assert synthesis_callback in integration_manager.synthesis_callbacks
        assert design_callback in integration_manager.design_callbacks


class TestFactoryFunctions:
    """Test cases for factory functions."""
    
    @pytest.fixture
    def mock_robot_manager(self):
        """Create mock robot manager."""
        return Mock(spec=RobotManager)
    
    @pytest.fixture
    def mock_adaptive_manager(self):
        """Create mock adaptive learning manager."""
        return Mock(spec=AdaptiveLearningManager)
    
    @pytest.fixture
    def mock_optimizer(self):
        """Create mock optimizer."""
        return Mock(spec=ClosedLoopOptimizer)
    
    def test_create_comprehensive_ai_robot_system(self, mock_robot_manager, mock_adaptive_manager, mock_optimizer):
        """Test comprehensive AI-robot system creation."""
        integration_manager = create_comprehensive_ai_robot_system(
            mock_robot_manager,
            mock_adaptive_manager,
            mock_optimizer
        )
        
        assert isinstance(integration_manager, AIRobotIntegrationManager)
        assert integration_manager.robot_manager == mock_robot_manager
        assert integration_manager.adaptive_manager == mock_adaptive_manager
        assert integration_manager.optimizer == mock_optimizer
        assert len(integration_manager.ai_engines) >= 2
    
    def test_create_active_learning_robot_system(self, mock_robot_manager, mock_adaptive_manager):
        """Test active learning focused system creation."""
        integration_manager = create_active_learning_robot_system(
            mock_robot_manager,
            mock_adaptive_manager
        )
        
        assert isinstance(integration_manager, AIRobotIntegrationManager)
        assert integration_manager.optimizer is None  # No optimizer provided
        assert "active_learning" in integration_manager.ai_engines
    
    def test_create_inverse_design_robot_system(self, mock_robot_manager, mock_adaptive_manager):
        """Test inverse design focused system creation."""
        integration_manager = create_inverse_design_robot_system(
            mock_robot_manager,
            mock_adaptive_manager
        )
        
        assert isinstance(integration_manager, AIRobotIntegrationManager)
        assert integration_manager.optimizer is None  # No optimizer provided
        assert "inverse_design" in integration_manager.ai_engines


class TestIntegrationScenarios:
    """Integration test scenarios for AI-robot system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_ai_driven_experiment(self):
        """Test complete AI-driven experimental workflow."""
        # Create mock components
        robot_manager = Mock(spec=RobotManager)
        robot_manager.execute_experiment_plan = AsyncMock(return_value="plan_001")
        robot_manager.add_feedback_callback = Mock()
        robot_manager.find_robots_with_capability.return_value = ["synthesis_robot"]
        
        adaptive_manager = Mock(spec=AdaptiveLearningManager)
        adaptive_manager.add_new_data = AsyncMock()
        
        # Create integration manager
        integration_manager = create_comprehensive_ai_robot_system(
            robot_manager, adaptive_manager, None
        )
        
        # Start integration
        await integration_manager.start_integration()
        
        # Let it run briefly to process decisions
        await asyncio.sleep(0.1)
        
        # Stop integration
        await integration_manager.stop_integration()
        
        # Verify system operated correctly
        assert len(integration_manager.ai_engines) >= 2
        robot_manager.add_feedback_callback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ai_decision_to_robot_execution_workflow(self):
        """Test workflow from AI decision to robot execution."""
        # Create integration manager
        robot_manager = Mock(spec=RobotManager)
        robot_manager.execute_experiment_plan = AsyncMock(return_value="plan_001")
        robot_manager.find_robots_with_capability.return_value = ["synthesis_robot"]
        
        adaptive_manager = Mock(spec=AdaptiveLearningManager)
        integration_manager = AIRobotIntegrationManager(robot_manager, adaptive_manager, None)
        
        # Create AI decision
        decision = AIDecision(
            decision_id="workflow_test",
            decision_type=AIDecisionType.INVERSE_DESIGN_CANDIDATE,
            ai_system="inverse_design_engine",
            confidence=0.85,
            reasoning={},
            parameters={
                "target_molecule": {"smiles": "CCO"},
                "synthesis_route": [{
                    "step_number": 1,
                    "reaction_type": "reduction",
                    "conditions": {"temperature": 80}
                }],
                "synthesis_parameters": {"step_1_temperature": 80}
            }
        )
        
        # Execute the decision
        await integration_manager._execute_ai_decision(decision)
        
        # Verify robot execution was triggered
        robot_manager.execute_experiment_plan.assert_called_once()
        
        # Verify experimental design was created
        assert len(integration_manager.experimental_designs) == 1
    
    @pytest.mark.asyncio
    async def test_multiple_ai_engines_coordination(self):
        """Test coordination between multiple AI engines."""
        robot_manager = Mock(spec=RobotManager)
        robot_manager.execute_experiment_plan = AsyncMock(return_value="plan_001")
        robot_manager.find_robots_with_capability.return_value = ["robot_01"]
        
        adaptive_manager = Mock(spec=AdaptiveLearningManager)
        integration_manager = AIRobotIntegrationManager(robot_manager, adaptive_manager, None)
        
        # Test decision requests for both engines
        decision_requests = [
            {
                "type": "active_learning",
                "context": {"parameter_space": {"temp": (50, 150)}},
                "priority": 2
            },
            {
                "type": "inverse_design",
                "context": {"target_properties": {"activity": "maximize"}},
                "priority": 3
            }
        ]
        
        # Process both requests
        for request in decision_requests:
            await integration_manager._process_decision_request(request)
        
        # Verify both decisions were processed
        assert len(integration_manager.decision_history) == 2
        assert integration_manager.integration_metrics["total_ai_decisions"] == 2
        
        # Check that different AI engines were used
        decision_types = [d.decision_type for d in integration_manager.decision_history]
        assert AIDecisionType.ACTIVE_LEARNING_QUERY in decision_types
        assert AIDecisionType.INVERSE_DESIGN_CANDIDATE in decision_types
    
    @pytest.mark.asyncio
    async def test_concurrent_experimental_designs(self):
        """Test handling of concurrent experimental designs."""
        robot_manager = Mock(spec=RobotManager)
        robot_manager.execute_experiment_plan = AsyncMock(return_value="plan_001")
        robot_manager.find_robots_with_capability.return_value = ["robot_01", "robot_02"]
        
        adaptive_manager = Mock(spec=AdaptiveLearningManager)
        integration_manager = AIRobotIntegrationManager(robot_manager, adaptive_manager, None)
        
        # Create multiple AI decisions
        decisions = []
        for i in range(3):
            decision = AIDecision(
                decision_id=f"concurrent_decision_{i}",
                decision_type=AIDecisionType.ACTIVE_LEARNING_QUERY,
                ai_system="active_learning_engine",
                confidence=0.8,
                reasoning={},
                parameters={"temperature": 100 + i * 10}
            )
            decisions.append(decision)
        
        # Execute decisions concurrently
        execution_tasks = [
            integration_manager._execute_ai_decision(decision)
            for decision in decisions
        ]
        
        await asyncio.gather(*execution_tasks)
        
        # Verify all designs were created
        assert len(integration_manager.experimental_designs) == 3
        
        # Verify robot execution was called for each
        assert robot_manager.execute_experiment_plan.call_count == 3
    
    def test_ai_engine_performance_tracking(self):
        """Test AI engine performance tracking."""
        robot_manager = Mock(spec=RobotManager)
        adaptive_manager = Mock(spec=AdaptiveLearningManager)
        integration_manager = AIRobotIntegrationManager(robot_manager, adaptive_manager, None)
        
        # Get initial performance metrics
        initial_metrics = integration_manager.integration_metrics.copy()
        
        # Simulate some AI decisions
        for i in range(5):
            decision = AIDecision(
                decision_id=f"perf_test_{i}",
                decision_type=AIDecisionType.ACTIVE_LEARNING_QUERY,
                ai_system="active_learning_engine",
                confidence=0.7 + i * 0.05,
                reasoning={},
                parameters={}
            )
            integration_manager.decision_history.append(decision)
            integration_manager.integration_metrics["total_ai_decisions"] += 1
        
        # Check performance tracking
        assert integration_manager.integration_metrics["total_ai_decisions"] == 5
        assert len(integration_manager.decision_history) == 5
        
        # Test status reporting
        status = integration_manager.get_integration_status()
        assert status["decision_history_size"] == 5
        assert len(status["recent_decisions"]) == 5  # All decisions are recent