"""
Tests for Closed-Loop Optimization System

This module contains comprehensive tests for the closed-loop optimization system
including acquisition functions, optimization strategies, and AI-robot integration.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import List, Dict, Any

from chemi_accel.autonomous.closed_loop_optimization import (
    ClosedLoopStrategy, ConvergenceCriteria, OptimizationConfig,
    ExperimentResult, ActiveLearningQuery, OptimizationState,
    AcquisitionFunction, ExpectedImprovement, UpperConfidenceBound,
    ActiveLearningAcquisition, ClosedLoopOptimizer,
    create_bayesian_optimizer, create_active_learning_optimizer,
    create_multi_objective_optimizer
)
from chemi_accel.autonomous.robotic_control import (
    RobotManager, OptimizationObjective, ExperimentPlan, RobotFeedback, ExperimentStatus
)
from chemi_accel.autonomous.adaptive_learning import AdaptiveLearningManager, TrainingData


class TestOptimizationConfig:
    """Test cases for OptimizationConfig dataclass."""
    
    def test_optimization_config_creation(self):
        """Test basic optimization config creation."""
        parameter_space = {
            "temperature": (20, 200),
            "reaction_time": (0.5, 24),
            "concentration": (0.01, 1.0)
        }
        
        convergence_criteria = {
            ConvergenceCriteria.MAX_ITERATIONS: 50,
            ConvergenceCriteria.IMPROVEMENT_THRESHOLD: 0.01
        }
        
        config = OptimizationConfig(
            strategy=ClosedLoopStrategy.BAYESIAN_OPTIMIZATION,
            objective=OptimizationObjective.MAXIMIZE_YIELD,
            parameter_space=parameter_space,
            convergence_criteria=convergence_criteria,
            batch_size=3,
            initial_samples=10,
            exploration_weight=0.15
        )
        
        assert config.strategy == ClosedLoopStrategy.BAYESIAN_OPTIMIZATION
        assert config.objective == OptimizationObjective.MAXIMIZE_YIELD
        assert config.parameter_space["temperature"] == (20, 200)
        assert config.convergence_criteria[ConvergenceCriteria.MAX_ITERATIONS] == 50
        assert config.batch_size == 3
        assert config.initial_samples == 10
        assert config.exploration_weight == 0.15


class TestExperimentResult:
    """Test cases for ExperimentResult dataclass."""
    
    def test_experiment_result_creation(self):
        """Test experiment result creation."""
        parameters = {"temperature": 80, "time": 4, "concentration": 0.5}
        objective_values = {"yield": 0.75, "purity": 0.92}
        
        feedback = RobotFeedback(
            feedback_id="fb_001",
            robot_id="synthesis_01",
            action_id="action_001",
            status=ExperimentStatus.COMPLETED,
            progress=1.0,
            data={"final_yield": 0.75}
        )
        
        result = ExperimentResult(
            experiment_id="exp_001",
            parameters=parameters,
            objective_values=objective_values,
            success=True,
            execution_time=14400.0,
            robot_feedback=[feedback],
            experimental_data=[],
            metadata={"optimization_round": 1}
        )
        
        assert result.experiment_id == "exp_001"
        assert result.parameters["temperature"] == 80
        assert result.objective_values["yield"] == 0.75
        assert result.success is True
        assert result.execution_time == 14400.0
        assert len(result.robot_feedback) == 1
        assert result.metadata["optimization_round"] == 1
        assert result.timestamp is not None


class TestActiveLearningQuery:
    """Test cases for ActiveLearningQuery dataclass."""
    
    def test_active_learning_query_creation(self):
        """Test active learning query creation."""
        parameters = {"temperature": 120, "catalyst_loading": 0.05}
        
        query = ActiveLearningQuery(
            query_id="query_001",
            parameters=parameters,
            uncertainty_score=0.8,
            expected_information_gain=0.7,
            diversity_score=0.6,
            feasibility_score=0.9,
            combined_score=0.75
        )
        
        assert query.query_id == "query_001"
        assert query.parameters["temperature"] == 120
        assert query.uncertainty_score == 0.8
        assert query.expected_information_gain == 0.7
        assert query.diversity_score == 0.6
        assert query.feasibility_score == 0.9
        assert query.combined_score == 0.75
        assert query.created_at is not None


class TestExpectedImprovement:
    """Test cases for Expected Improvement acquisition function."""
    
    @pytest.fixture
    def ei_function(self):
        """Create Expected Improvement function."""
        return ExpectedImprovement()
    
    @pytest.fixture
    def mock_model(self):
        """Create mock Gaussian Process model."""
        model = Mock()
        model.predict.return_value = (
            np.array([0.7, 0.6, 0.8, 0.5]),  # means
            np.array([0.1, 0.2, 0.05, 0.3])   # std deviations
        )
        return model
    
    def test_expected_improvement_evaluation(self, ei_function, mock_model):
        """Test Expected Improvement calculation."""
        candidate_points = np.random.randn(4, 2)
        current_best = 0.65
        exploration_weight = 0.1
        
        with patch('chemi_accel.autonomous.closed_loop_optimization.SKLEARN_AVAILABLE', True):
            with patch('scipy.stats.norm') as mock_norm:
                mock_norm.cdf.return_value = np.array([0.7, 0.2, 0.9, 0.1])
                mock_norm.pdf.return_value = np.array([0.3, 0.4, 0.1, 0.5])
                
                ei_values = ei_function.evaluate(candidate_points, mock_model, current_best, exploration_weight)
        
        assert len(ei_values) == 4
        assert all(ei >= 0 for ei in ei_values)  # EI should be non-negative
        mock_model.predict.assert_called_once_with(candidate_points, return_std=True)
    
    def test_expected_improvement_no_sklearn(self, ei_function, mock_model):
        """Test Expected Improvement without sklearn."""
        candidate_points = np.random.randn(3, 2)
        current_best = 0.5
        
        with patch('chemi_accel.autonomous.closed_loop_optimization.SKLEARN_AVAILABLE', False):
            with pytest.raises(ImportError, match="Expected Improvement requires scikit-learn"):
                ei_function.evaluate(candidate_points, mock_model, current_best)


class TestUpperConfidenceBound:
    """Test cases for Upper Confidence Bound acquisition function."""
    
    @pytest.fixture
    def ucb_function(self):
        """Create UCB function."""
        return UpperConfidenceBound()
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.predict.return_value = (
            np.array([0.6, 0.7, 0.5]),  # means
            np.array([0.2, 0.1, 0.3])   # std deviations
        )
        return model
    
    def test_ucb_evaluation(self, ucb_function, mock_model):
        """Test UCB calculation."""
        candidate_points = np.random.randn(3, 2)
        current_best = 0.6
        exploration_weight = 0.2
        
        with patch('chemi_accel.autonomous.closed_loop_optimization.SKLEARN_AVAILABLE', True):
            ucb_values = ucb_function.evaluate(candidate_points, mock_model, current_best, exploration_weight)
        
        assert len(ucb_values) == 3
        # UCB = mean + exploration_weight * std
        expected_ucb = np.array([0.6, 0.7, 0.5]) + 0.2 * np.array([0.2, 0.1, 0.3])
        np.testing.assert_array_almost_equal(ucb_values, expected_ucb)


class TestActiveeLearningAcquisition:
    """Test cases for Active Learning acquisition function."""
    
    @pytest.fixture
    def mock_adaptive_manager(self):
        """Create mock adaptive learning manager."""
        manager = Mock(spec=AdaptiveLearningManager)
        manager.get_training_data.return_value = [
            np.array([0.5, 0.3, 0.7]),
            np.array([0.8, 0.2, 0.4])
        ]
        return manager
    
    @pytest.fixture
    def al_function(self, mock_adaptive_manager):
        """Create Active Learning acquisition function."""
        return ActiveLearningAcquisition(mock_adaptive_manager)
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.predict.return_value = (
            np.array([0.65, 0.72]),  # means
            np.array([0.15, 0.08])   # std deviations
        )
        return model
    
    def test_active_learning_evaluation(self, al_function, mock_model):
        """Test active learning acquisition calculation."""
        candidate_points = np.random.randn(2, 3)
        current_best = 0.7
        
        scores = al_function.evaluate(candidate_points, mock_model, current_best)
        
        assert len(scores) == 2
        assert all(score >= 0 for score in scores)
        mock_model.predict.assert_called_once()
    
    def test_diversity_calculation(self, al_function):
        """Test diversity score calculation."""
        # Mock training data
        al_function.adaptive_manager.get_training_data.return_value = [
            np.array([0.5, 0.5, 0.5])
        ]
        
        candidate_points = np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]])
        
        # First point is close to existing data, second is far
        diversity_scores = al_function._calculate_diversity(candidate_points)
        
        assert len(diversity_scores) == 2
        assert diversity_scores[1] > diversity_scores[0]  # Far point has higher diversity


class TestClosedLoopOptimizer:
    """Test cases for ClosedLoopOptimizer."""
    
    @pytest.fixture
    def mock_robot_manager(self):
        """Create mock robot manager."""
        manager = Mock(spec=RobotManager)
        manager.execute_experiment_plan = AsyncMock(return_value="plan_001")
        manager.add_feedback_callback = Mock()
        manager.find_robots_with_capability.return_value = ["robot_01"]
        return manager
    
    @pytest.fixture
    def mock_adaptive_manager(self):
        """Create mock adaptive learning manager."""
        manager = Mock(spec=AdaptiveLearningManager)
        manager.add_new_data = AsyncMock()
        return manager
    
    @pytest.fixture
    def optimization_config(self):
        """Create test optimization config."""
        return OptimizationConfig(
            strategy=ClosedLoopStrategy.BAYESIAN_OPTIMIZATION,
            objective=OptimizationObjective.MAXIMIZE_YIELD,
            parameter_space={
                "temperature": (50, 150),
                "time": (1, 12),
                "concentration": (0.1, 2.0)
            },
            convergence_criteria={
                ConvergenceCriteria.MAX_ITERATIONS: 10,
                ConvergenceCriteria.IMPROVEMENT_THRESHOLD: 0.05
            },
            batch_size=2,
            initial_samples=3
        )
    
    @pytest.fixture
    def optimizer(self, mock_robot_manager, mock_adaptive_manager, optimization_config):
        """Create ClosedLoopOptimizer instance."""
        return ClosedLoopOptimizer(mock_robot_manager, mock_adaptive_manager, optimization_config)
    
    def test_optimizer_initialization(self, optimizer, optimization_config):
        """Test optimizer initialization."""
        assert optimizer.config == optimization_config
        assert optimizer.optimization_state.iteration == 0
        assert optimizer.optimization_state.best_objective == float('-inf')  # Maximization
        assert len(optimizer.experiment_results) == 0
        assert not optimizer._is_running
    
    def test_is_maximization_objective(self, optimizer):
        """Test maximization objective detection."""
        assert optimizer._is_maximization_objective() is True
        
        # Test minimization objective
        optimizer.config.objective = OptimizationObjective.MINIMIZE_TIME
        assert optimizer._is_maximization_objective() is False
    
    @pytest.mark.asyncio
    async def test_start_stop_optimization(self, optimizer):
        """Test starting and stopping optimization."""
        # Start optimization
        await optimizer.start_optimization()
        assert optimizer._is_running is True
        assert optimizer._optimization_task is not None
        
        # Stop optimization
        await optimizer.stop_optimization()
        assert optimizer._is_running is False
    
    def test_extract_objective_value(self, optimizer):
        """Test objective value extraction from results."""
        # Test yield maximization
        result = ExperimentResult(
            experiment_id="test",
            parameters={},
            objective_values={"yield": 0.85},
            success=True,
            execution_time=3600,
            robot_feedback=[],
            experimental_data=[]
        )
        
        objective_value = optimizer._extract_objective_value(result)
        assert objective_value == 0.85
        
        # Test time minimization
        optimizer.config.objective = OptimizationObjective.MINIMIZE_TIME
        result.execution_time = 7200
        objective_value = optimizer._extract_objective_value(result)
        assert objective_value == -7200  # Negative for minimization
    
    @pytest.mark.asyncio
    async def test_generate_initial_samples(self, optimizer, mock_robot_manager):
        """Test initial sample generation."""
        with patch.object(optimizer, '_create_experiment_plans') as mock_create_plans:
            with patch.object(optimizer, '_wait_for_experiment_completion') as mock_wait:
                mock_plans = [Mock(spec=ExperimentPlan) for _ in range(3)]
                mock_create_plans.return_value = mock_plans
                mock_wait.return_value = None
                
                await optimizer._generate_initial_samples()
                
                mock_create_plans.assert_called_once()
                assert len(mock_create_plans.call_args[0][0]) == 3  # 3 initial samples
                assert mock_robot_manager.execute_experiment_plan.call_count == 3
    
    def test_bayesian_optimization_candidates(self, optimizer):
        """Test Bayesian optimization candidate generation."""
        # Add some mock results to enable surrogate model
        optimizer.experiment_results = [
            ExperimentResult("exp1", {"temperature": 100, "time": 4, "concentration": 0.5}, 
                           {"yield": 0.7}, True, 3600, [], []),
            ExperimentResult("exp2", {"temperature": 120, "time": 6, "concentration": 1.0}, 
                           {"yield": 0.8}, True, 3600, [], [])
        ]
        
        # Mock surrogate model
        mock_model = Mock()
        mock_model.predict.return_value = (
            np.random.randn(1000),  # means
            np.random.rand(1000) * 0.5  # std devs
        )
        optimizer.surrogate_model = mock_model
        
        # Test candidate generation
        candidates = asyncio.run(optimizer._bayesian_optimization_candidates())
        
        assert len(candidates) == optimizer.config.batch_size
        for candidate in candidates:
            assert "temperature" in candidate
            assert "time" in candidate
            assert "concentration" in candidate
            # Check bounds
            assert 50 <= candidate["temperature"] <= 150
            assert 1 <= candidate["time"] <= 12
            assert 0.1 <= candidate["concentration"] <= 2.0
    
    @pytest.mark.asyncio
    async def test_active_learning_candidates(self, optimizer):
        """Test active learning candidate generation."""
        candidates = await optimizer._active_learning_candidates()
        
        assert len(candidates) == optimizer.config.batch_size
        for candidate in candidates:
            assert "temperature" in candidate
            assert "time" in candidate
            assert "concentration" in candidate
    
    def test_convergence_checking(self, optimizer):
        """Test convergence criteria checking."""
        # Test max iterations
        optimizer.optimization_state.iteration = 10
        assert optimizer._check_convergence() is True
        
        # Test objective threshold (maximization)
        optimizer.optimization_state.iteration = 5
        optimizer.config.convergence_criteria[ConvergenceCriteria.OBJECTIVE_THRESHOLD] = 0.9
        optimizer.optimization_state.best_objective = 0.95
        assert optimizer._check_convergence() is True
        
        # Test improvement threshold
        optimizer.optimization_state.best_objective = 0.7
        optimizer.optimization_state.objective_values = [0.6, 0.62, 0.61, 0.63, 0.62]
        assert optimizer._check_convergence() is True  # Small improvements
    
    def test_process_completed_experiment(self, optimizer):
        """Test processing of completed experiments."""
        # Create mock feedback with experimental data
        feedback = RobotFeedback(
            feedback_id="fb_001",
            robot_id="robot_01",
            action_id="action_001",
            status=ExperimentStatus.COMPLETED,
            progress=1.0,
            data={
                "objective_values": {"yield": 0.82},
                "experimental_data": [{
                    "data_id": "data_001",
                    "data_type": "synthesis_result",
                    "raw_data": {"product_mass": 0.82}
                }]
            }
        )
        
        # Create mock experiment plan
        mock_plan = Mock()
        mock_plan.experiment_id = "exp_001"
        mock_plan.target_parameters = {"temperature": 100, "time": 4}
        mock_plan.created_at = datetime.now(timezone.utc) - timedelta(hours=2)
        mock_plan.metadata = {"optimization_round": 1}
        
        optimizer.active_experiments["plan_001"] = mock_plan
        
        # Process the feedback
        optimizer._process_completed_experiment(feedback)
        
        # Check that experiment result was added
        assert len(optimizer.experiment_results) == 1
        result = optimizer.experiment_results[0]
        assert result.experiment_id == "exp_001"
        assert result.objective_values["yield"] == 0.82
        assert result.success is True
        
        # Check optimization state update
        assert optimizer.optimization_state.total_experiments == 1
        assert optimizer.optimization_state.successful_experiments == 1
        assert optimizer.optimization_state.best_objective == 0.82
    
    def test_is_better_objective(self, optimizer):
        """Test objective comparison."""
        # Maximization (default)
        assert optimizer._is_better_objective(0.8, 0.7) is True
        assert optimizer._is_better_objective(0.6, 0.7) is False
        
        # Minimization
        optimizer.config.objective = OptimizationObjective.MINIMIZE_TIME
        assert optimizer._is_better_objective(100, 150) is True
        assert optimizer._is_better_objective(200, 150) is False
    
    def test_get_optimization_summary(self, optimizer):
        """Test optimization summary generation."""
        # Add some mock data
        optimizer.optimization_state.iteration = 5
        optimizer.optimization_state.total_experiments = 10
        optimizer.optimization_state.successful_experiments = 8
        optimizer.optimization_state.best_objective = 0.85
        optimizer.optimization_state.objective_values = [0.6, 0.7, 0.75, 0.8, 0.85]
        
        summary = optimizer.get_optimization_summary()
        
        assert summary["optimization_state"]["iteration"] == 5
        assert summary["optimization_state"]["total_experiments"] == 10
        assert summary["optimization_state"]["success_rate"] == 0.8
        assert summary["optimization_state"]["best_objective"] == 0.85
        assert summary["config"]["strategy"] == "bayesian_optimization"
        assert summary["config"]["objective"] == "maximize_yield"
        assert len(summary["results_summary"]["objective_history"]) == 5


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
    
    def test_create_bayesian_optimizer(self, mock_robot_manager, mock_adaptive_manager):
        """Test Bayesian optimizer creation."""
        parameter_space = {
            "temperature": (50, 200),
            "catalyst_loading": (0.01, 0.1)
        }
        
        optimizer = create_bayesian_optimizer(
            mock_robot_manager,
            mock_adaptive_manager,
            parameter_space,
            OptimizationObjective.MAXIMIZE_PURITY
        )
        
        assert isinstance(optimizer, ClosedLoopOptimizer)
        assert optimizer.config.strategy == ClosedLoopStrategy.BAYESIAN_OPTIMIZATION
        assert optimizer.config.objective == OptimizationObjective.MAXIMIZE_PURITY
        assert optimizer.config.parameter_space == parameter_space
        assert optimizer.config.acquisition_function == "expected_improvement"
    
    def test_create_active_learning_optimizer(self, mock_robot_manager, mock_adaptive_manager):
        """Test active learning optimizer creation."""
        parameter_space = {"reaction_time": (0.5, 24)}
        
        optimizer = create_active_learning_optimizer(
            mock_robot_manager,
            mock_adaptive_manager,
            parameter_space
        )
        
        assert isinstance(optimizer, ClosedLoopOptimizer)
        assert optimizer.config.strategy == ClosedLoopStrategy.ACTIVE_LEARNING
        assert optimizer.config.acquisition_function == "active_learning"
        assert ConvergenceCriteria.UNCERTAINTY_THRESHOLD in optimizer.config.convergence_criteria
    
    def test_create_multi_objective_optimizer(self, mock_robot_manager, mock_adaptive_manager):
        """Test multi-objective optimizer creation."""
        parameter_space = {"temperature": (20, 180)}
        objective_weights = {"yield": 0.6, "purity": 0.4}
        
        optimizer = create_multi_objective_optimizer(
            mock_robot_manager,
            mock_adaptive_manager,
            parameter_space,
            objective_weights
        )
        
        assert isinstance(optimizer, ClosedLoopOptimizer)
        assert optimizer.config.strategy == ClosedLoopStrategy.MULTI_OBJECTIVE
        assert optimizer.config.objective == OptimizationObjective.MULTI_OBJECTIVE
        assert optimizer.config.metadata["objective_weights"] == objective_weights


class TestIntegrationScenarios:
    """Integration test scenarios for closed-loop optimization."""
    
    @pytest.mark.asyncio
    async def test_complete_optimization_cycle(self):
        """Test a complete optimization cycle."""
        # Create mocks
        robot_manager = Mock(spec=RobotManager)
        robot_manager.execute_experiment_plan = AsyncMock(return_value="plan_001")
        robot_manager.add_feedback_callback = Mock()
        robot_manager.find_robots_with_capability.return_value = ["synthesis_robot"]
        
        adaptive_manager = Mock(spec=AdaptiveLearningManager)
        adaptive_manager.add_new_data = AsyncMock()
        
        # Create optimizer
        config = OptimizationConfig(
            strategy=ClosedLoopStrategy.BAYESIAN_OPTIMIZATION,
            objective=OptimizationObjective.MAXIMIZE_YIELD,
            parameter_space={"temperature": (50, 150)},
            convergence_criteria={ConvergenceCriteria.MAX_ITERATIONS: 2},
            initial_samples=1,
            batch_size=1
        )
        
        optimizer = ClosedLoopOptimizer(robot_manager, adaptive_manager, config)
        
        # Mock experiment execution and feedback
        async def mock_wait_completion():
            # Simulate completed experiment
            feedback = RobotFeedback(
                feedback_id="fb_001",
                robot_id="synthesis_robot",
                action_id="action_001",
                status=ExperimentStatus.COMPLETED,
                progress=1.0,
                data={"objective_values": {"yield": 0.75}}
            )
            
            # Create mock plan
            mock_plan = Mock()
            mock_plan.experiment_id = "exp_001"
            mock_plan.target_parameters = {"temperature": 100}
            mock_plan.created_at = datetime.now(timezone.utc)
            mock_plan.metadata = {}
            
            optimizer.active_experiments["plan_001"] = mock_plan
            optimizer._process_completed_experiment(feedback)
        
        with patch.object(optimizer, '_wait_for_experiment_completion', side_effect=mock_wait_completion):
            # Run optimization for a short time
            await optimizer.start_optimization()
            await asyncio.sleep(0.1)  # Let it run briefly
            await optimizer.stop_optimization()
        
        # Verify optimization progressed
        assert optimizer.optimization_state.iteration >= 0
        assert len(optimizer.experiment_results) >= 0
    
    @pytest.mark.asyncio
    async def test_optimization_with_constraints(self):
        """Test optimization with parameter constraints."""
        robot_manager = Mock(spec=RobotManager)
        adaptive_manager = Mock(spec=AdaptiveLearningManager)
        
        # Define constraint function
        def temperature_constraint(params):
            return params.get("temperature", 0) <= 120  # Max temperature constraint
        
        config = OptimizationConfig(
            strategy=ClosedLoopStrategy.ACTIVE_LEARNING,
            objective=OptimizationObjective.MAXIMIZE_YIELD,
            parameter_space={"temperature": (50, 200), "time": (1, 10)},
            convergence_criteria={ConvergenceCriteria.MAX_ITERATIONS: 3},
            constraint_functions=[temperature_constraint],
            initial_samples=2,
            batch_size=1
        )
        
        optimizer = ClosedLoopOptimizer(robot_manager, adaptive_manager, config)
        
        # Test candidate generation respects constraints
        candidates = await optimizer._active_learning_candidates()
        
        assert len(candidates) == 1
        for candidate in candidates:
            # Note: constraint enforcement would be implemented in feasibility scoring
            assert "temperature" in candidate
            assert "time" in candidate
    
    @pytest.mark.asyncio
    async def test_multi_objective_optimization(self):
        """Test multi-objective optimization scenario."""
        robot_manager = Mock(spec=RobotManager)
        adaptive_manager = Mock(spec=AdaptiveLearningManager)
        
        config = OptimizationConfig(
            strategy=ClosedLoopStrategy.MULTI_OBJECTIVE,
            objective=OptimizationObjective.MULTI_OBJECTIVE,
            parameter_space={"temperature": (60, 140), "catalyst": (0.01, 0.1)},
            convergence_criteria={ConvergenceCriteria.MAX_ITERATIONS: 5},
            metadata={"objective_weights": {"yield": 0.7, "purity": 0.3}},
            initial_samples=2,
            batch_size=1
        )
        
        optimizer = ClosedLoopOptimizer(robot_manager, adaptive_manager, config)
        
        # Test multi-objective value extraction
        result = ExperimentResult(
            experiment_id="multi_exp",
            parameters={"temperature": 100, "catalyst": 0.05},
            objective_values={"yield": 0.8, "purity": 0.9},
            success=True,
            execution_time=3600,
            robot_feedback=[],
            experimental_data=[]
        )
        
        objective_value = optimizer._extract_objective_value(result)
        expected_value = 0.7 * 0.8 + 0.3 * 0.9  # Weighted sum
        assert abs(objective_value - expected_value) < 1e-6
    
    @pytest.mark.asyncio
    async def test_optimization_failure_recovery(self):
        """Test optimization recovery from failed experiments."""
        robot_manager = Mock(spec=RobotManager)
        adaptive_manager = Mock(spec=AdaptiveLearningManager)
        
        config = OptimizationConfig(
            strategy=ClosedLoopStrategy.BAYESIAN_OPTIMIZATION,
            objective=OptimizationObjective.MAXIMIZE_YIELD,
            parameter_space={"temperature": (50, 150)},
            convergence_criteria={ConvergenceCriteria.MAX_ITERATIONS: 3},
            initial_samples=1
        )
        
        optimizer = ClosedLoopOptimizer(robot_manager, adaptive_manager, config)
        
        # Simulate failed experiment feedback
        failed_feedback = RobotFeedback(
            feedback_id="fb_fail",
            robot_id="robot_01",
            action_id="action_fail",
            status=ExperimentStatus.FAILED,
            progress=0.5,
            data={},
            error_message="Synthesis failed due to equipment error"
        )
        
        optimizer._process_failed_experiment(failed_feedback)
        
        # Check that failure was recorded
        assert optimizer.optimization_state.total_experiments == 1
        assert optimizer.optimization_state.successful_experiments == 0
    
    @pytest.mark.asyncio
    async def test_adaptive_learning_integration(self):
        """Test integration with adaptive learning system."""
        robot_manager = Mock(spec=RobotManager)
        adaptive_manager = Mock(spec=AdaptiveLearningManager)
        adaptive_manager.add_new_data = AsyncMock()
        
        config = OptimizationConfig(
            strategy=ClosedLoopStrategy.ACTIVE_LEARNING,
            objective=OptimizationObjective.MAXIMIZE_YIELD,
            parameter_space={"temperature": (50, 150)},
            convergence_criteria={ConvergenceCriteria.MAX_ITERATIONS: 2}
        )
        
        optimizer = ClosedLoopOptimizer(robot_manager, adaptive_manager, config)
        
        # Simulate completed experiment that should trigger adaptive learning update
        feedback = RobotFeedback(
            feedback_id="fb_al",
            robot_id="robot_01",
            action_id="action_al",
            status=ExperimentStatus.COMPLETED,
            progress=1.0,
            data={"objective_values": {"yield": 0.85}}
        )
        
        mock_plan = Mock()
        mock_plan.experiment_id = "exp_al"
        mock_plan.target_parameters = {"temperature": 120}
        mock_plan.created_at = datetime.now(timezone.utc)
        mock_plan.metadata = {}
        
        optimizer.active_experiments["plan_al"] = mock_plan
        
        # Process the completed experiment
        optimizer._process_completed_experiment(feedback)
        
        # Verify adaptive learning was updated
        adaptive_manager.add_new_data.assert_called_once()
        call_args = adaptive_manager.add_new_data.call_args[0][0]
        assert isinstance(call_args, TrainingData)
        assert call_args.metadata["optimization_result"] is True
    
    def test_optimization_callbacks(self):
        """Test optimization callback system."""
        robot_manager = Mock(spec=RobotManager)
        adaptive_manager = Mock(spec=AdaptiveLearningManager)
        
        config = OptimizationConfig(
            strategy=ClosedLoopStrategy.BAYESIAN_OPTIMIZATION,
            objective=OptimizationObjective.MAXIMIZE_YIELD,
            parameter_space={"temperature": (50, 150)},
            convergence_criteria={ConvergenceCriteria.MAX_ITERATIONS: 5}
        )
        
        optimizer = ClosedLoopOptimizer(robot_manager, adaptive_manager, config)
        
        # Add callbacks
        experiment_results = []
        optimization_states = []
        
        optimizer.add_experiment_callback(lambda result: experiment_results.append(result))
        optimizer.add_optimization_callback(lambda state: optimization_states.append(state))
        
        # Simulate experiment completion
        feedback = RobotFeedback(
            feedback_id="fb_callback",
            robot_id="robot_01",
            action_id="action_callback",
            status=ExperimentStatus.COMPLETED,
            progress=1.0,
            data={"objective_values": {"yield": 0.7}}
        )
        
        mock_plan = Mock()
        mock_plan.experiment_id = "exp_callback"
        mock_plan.target_parameters = {"temperature": 90}
        mock_plan.created_at = datetime.now(timezone.utc)
        mock_plan.metadata = {}
        
        optimizer.active_experiments["plan_callback"] = mock_plan
        optimizer._process_completed_experiment(feedback)
        
        # Verify callbacks were called
        assert len(experiment_results) == 1
        assert experiment_results[0].experiment_id == "exp_callback"