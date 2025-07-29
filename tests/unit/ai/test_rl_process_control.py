"""
Comprehensive tests for RL Process Control framework.

This test suite covers all components of the RL system including:
- RL agents (PPO, SAC, etc.)
- Chemical process environments
- Cost-aware acquisition functions
- Real-time data management
- Constraint-based experimental planning
"""

import asyncio
import tempfile
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

# Import the RL components
from src.chemi_accel.ai.rl_process_control import (
    CUPY_AVAILABLE,
    JAX_AVAILABLE,
    TORCH_AVAILABLE,
    AgentType,
    ChemicalEnvironment,
    ConstraintType,
    ContinuousManufacturingEnvironment,
    CostAwareAcquisitionFunction,
    EnvironmentType,
    ExperimentalCost,
    MultiStepSynthesisEnvironment,
    ProcessConstraint,
    ProcessControlTrainer,
    RealTimeData,
    RealTimeDataManager,
    RLConfig,
    RLError,
    create_cost_aware_acquisition_function,
    create_cost_constraints,
    create_quality_constraints,
    create_real_time_data_manager,
    create_safety_constraints,
    energy_cost_model,
    material_cost_model,
    train_continuous_manufacturing_agent,
    train_synthesis_agent,
)

if TORCH_AVAILABLE:
    from src.chemi_accel.ai.rl_process_control import PolicyNetwork, PPOAgent, QNetwork


class TestRLConfig:
    """Test RL configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RLConfig()

        assert config.agent_type == AgentType.PPO
        assert config.environment_type == EnvironmentType.SYNTHESIS
        assert config.learning_rate == 3e-4
        assert config.batch_size == 64
        assert config.gamma == 0.99
        assert config.use_gpu is True
        assert config.max_episodes == 1000
        assert config.max_steps_per_episode == 200

    def test_custom_config(self):
        """Test custom configuration."""
        config = RLConfig(
            agent_type=AgentType.SAC,
            environment_type=EnvironmentType.CONTINUOUS_MANUFACTURING,
            learning_rate=1e-3,
            batch_size=128,
            gamma=0.95,
            use_gpu=False,
            max_episodes=500,
            cost_weight=0.2,
            constraint_weight=2.0,
        )

        assert config.agent_type == AgentType.SAC
        assert config.environment_type == EnvironmentType.CONTINUOUS_MANUFACTURING
        assert config.learning_rate == 1e-3
        assert config.batch_size == 128
        assert config.gamma == 0.95
        assert config.use_gpu is False
        assert config.max_episodes == 500
        assert config.cost_weight == 0.2
        assert config.constraint_weight == 2.0


class TestExperimentalCost:
    """Test experimental cost modeling."""

    def test_cost_creation(self):
        """Test cost object creation."""
        cost = ExperimentalCost(
            material_cost=50.0,
            energy_cost=20.0,
            time_cost=10.0,
            equipment_cost=5.0,
            labor_cost=100.0,
            environmental_cost=15.0,
        )

        assert cost.material_cost == 50.0
        assert cost.energy_cost == 20.0
        assert cost.time_cost == 10.0
        assert cost.equipment_cost == 5.0
        assert cost.labor_cost == 100.0
        assert cost.environmental_cost == 15.0
        assert cost.total_cost() == 200.0

    def test_cost_to_dict(self):
        """Test cost conversion to dictionary."""
        cost = ExperimentalCost(material_cost=30.0, energy_cost=10.0)
        cost_dict = cost.to_dict()

        assert cost_dict["material_cost"] == 30.0
        assert cost_dict["energy_cost"] == 10.0
        assert cost_dict["total_cost"] == 40.0
        assert "time_cost" in cost_dict
        assert "equipment_cost" in cost_dict


class TestProcessConstraint:
    """Test process constraint functionality."""

    def test_constraint_creation(self):
        """Test constraint object creation."""
        constraint = ProcessConstraint(
            name="temperature",
            constraint_type=ConstraintType.SAFETY,
            min_value=273.15,
            max_value=473.15,
            target_value=350.0,
            tolerance=10.0,
            weight=2.0,
            violation_penalty=100.0,
        )

        assert constraint.name == "temperature"
        assert constraint.constraint_type == ConstraintType.SAFETY
        assert constraint.min_value == 273.15
        assert constraint.max_value == 473.15
        assert constraint.target_value == 350.0
        assert constraint.tolerance == 10.0
        assert constraint.weight == 2.0
        assert constraint.violation_penalty == 100.0

    def test_constraint_evaluation_satisfied(self):
        """Test constraint evaluation when satisfied."""
        constraint = ProcessConstraint(
            name="temperature",
            constraint_type=ConstraintType.SAFETY,
            min_value=300.0,
            max_value=400.0,
            target_value=350.0,
            tolerance=5.0,
            violation_penalty=100.0,
        )

        # Value within target tolerance
        satisfied, penalty = constraint.evaluate(352.0)
        assert satisfied is True
        assert penalty == 0.0

        # Value within bounds but outside target tolerance
        satisfied, penalty = constraint.evaluate(370.0)
        assert satisfied is False
        assert penalty > 0.0

    def test_constraint_evaluation_violated(self):
        """Test constraint evaluation when violated."""
        constraint = ProcessConstraint(
            name="pressure",
            constraint_type=ConstraintType.SAFETY,
            min_value=1e5,
            max_value=5e5,
            violation_penalty=200.0,
        )

        # Below minimum
        satisfied, penalty = constraint.evaluate(0.5e5)
        assert satisfied is False
        assert penalty > 0.0

        # Above maximum
        satisfied, penalty = constraint.evaluate(6e5)
        assert satisfied is False
        assert penalty > 0.0


class TestRealTimeData:
    """Test real-time data structures."""

    def test_data_creation(self):
        """Test real-time data creation."""
        data = RealTimeData(
            timestamp=time.time(),
            experiment_id="exp_001",
            measurements={"temperature": 350.0, "pressure": 2e5, "yield": 0.85},
            metadata={"operator": "test", "batch": "B001"},
        )

        assert data.experiment_id == "exp_001"
        assert data.measurements["temperature"] == 350.0
        assert data.measurements["pressure"] == 2e5
        assert data.measurements["yield"] == 0.85
        assert data.metadata["operator"] == "test"

    def test_data_to_dict(self):
        """Test data conversion to dictionary."""
        data = RealTimeData(
            timestamp=1234567890.0,
            experiment_id="exp_002",
            measurements={"pH": 7.2, "concentration": 1.5},
        )

        data_dict = data.to_dict()
        assert data_dict["timestamp"] == 1234567890.0
        assert data_dict["experiment_id"] == "exp_002"
        assert data_dict["measurements"]["pH"] == 7.2
        assert data_dict["measurements"]["concentration"] == 1.5
        assert data_dict["metadata"] == {}


class TestMultiStepSynthesisEnvironment:
    """Test multi-step synthesis environment."""

    def create_test_environment(self):
        """Create test synthesis environment."""
        config = RLConfig(max_steps_per_episode=10)
        return MultiStepSynthesisEnvironment(config)

    def test_environment_initialization(self):
        """Test environment initialization."""
        env = self.create_test_environment()

        assert env.state_dim == 10
        assert env.action_dim == 4
        assert env.max_steps == 20
        assert len(env.reaction_network["reactions"]) > 0
        assert "A" in env.reaction_network["initial_concentrations"]

    def test_environment_reset(self):
        """Test environment reset functionality."""
        env = self.create_test_environment()

        initial_state = env.reset()

        assert len(initial_state) == env.state_dim
        assert env.episode_step == 0
        assert env.total_reward == 0.0
        assert len(env.history) == 0

        # Check initial concentrations
        assert initial_state[0] == 1.0  # Initial A concentration
        assert initial_state[3] == 0.0  # Initial product concentration

    def test_environment_step(self):
        """Test environment step functionality."""
        env = self.create_test_environment()

        state = env.reset()
        action = np.array([0.1, 5.0, 0.05, 0.5])  # Small actions

        next_state, reward, done, info = env.step(action)

        assert len(next_state) == env.state_dim
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

        # Check state evolution
        assert next_state[8] > state[8]  # Time should increase
        assert env.episode_step == 1
        assert len(env.history) == 1

        # Check info content
        assert "product_concentration" in info
        assert "selectivity" in info
        assert "step_reward" in info

    def test_environment_constraints(self):
        """Test environment with constraints."""
        env = self.create_test_environment()

        # Add safety constraint
        constraint = ProcessConstraint(
            name="temperature",
            constraint_type=ConstraintType.SAFETY,
            min_value=300.0,
            max_value=400.0,
            violation_penalty=500.0,
        )
        env.add_constraint(constraint)

        assert len(env.constraints) == 1

        # Test constraint evaluation
        measurements = {"temperature": 350.0}
        satisfied, penalty = env.evaluate_constraints(measurements)
        assert satisfied is True
        assert penalty == 0.0

        # Test constraint violation
        measurements = {"temperature": 450.0}
        satisfied, penalty = env.evaluate_constraints(measurements)
        assert satisfied is False
        assert penalty > 0.0

    def test_environment_cost_model(self):
        """Test environment with cost model."""
        env = self.create_test_environment()

        # Set cost model
        env.set_cost_model(material_cost_model)

        action = np.array([0.1, 5.0, 0.05, 0.5])
        cost = env.calculate_cost(action)

        assert isinstance(cost, ExperimentalCost)
        assert cost.total_cost() > 0.0


class TestContinuousManufacturingEnvironment:
    """Test continuous manufacturing environment."""

    def create_test_environment(self):
        """Create test continuous environment."""
        config = RLConfig(max_steps_per_episode=50)
        return ContinuousManufacturingEnvironment(config)

    def test_environment_initialization(self):
        """Test environment initialization."""
        env = self.create_test_environment()

        assert env.state_dim == 15
        assert env.action_dim == 6
        assert env.target_throughput == 100.0
        assert env.target_quality == 0.95
        assert "reactor_volume" in env.process_config

    def test_environment_reset(self):
        """Test environment reset."""
        env = self.create_test_environment()

        initial_state = env.reset()

        assert len(initial_state) == env.state_dim
        assert initial_state[0] == 100.0  # Initial feed flow
        assert initial_state[2] == 350.0  # Initial temperature
        assert initial_state[8] == 0.95  # Initial conversion

    def test_environment_step(self):
        """Test environment step."""
        env = self.create_test_environment()

        state = env.reset()
        action = np.array([0.1, 0.2, 0.1, 0.0, 0.0, 0.0])  # Small control actions

        next_state, reward, done, info = env.step(action)

        assert len(next_state) == env.state_dim
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)

        # Check state evolution
        assert next_state[14] > state[14]  # Time should increase
        assert env.episode_step == 1

        # Check info content
        assert "throughput" in info
        assert "quality" in info
        assert "conversion" in info
        assert "selectivity" in info

    def test_physical_constraints(self):
        """Test physical constraint enforcement."""
        env = self.create_test_environment()

        # Create extreme state
        extreme_state = np.array(
            [
                1000.0,  # Extreme flow rate
                -50.0,  # Negative flow (should be clipped)
                500.0,  # Extreme temperature
                250.0,  # Low cooling temp
                5.0,  # Extreme concentration
                -1.0,  # Negative concentration (should be clipped)
                10e5,  # Extreme pressure
                0.5e5,  # Low pressure
                1.5,  # Impossible conversion (should be clipped)
                -0.2,  # Negative selectivity (should be clipped)
                2.0,  # Impossible quality (should be clipped)
                2.0,  # Extreme fouling (should be clipped)
                0.0,  # Vibration
                0.0,  # Throughput
                0.0,  # Time
            ]
        )

        constrained_state = env._apply_physical_constraints(extreme_state)

        # Check flow rate constraints
        assert 10.0 <= constrained_state[0] <= 500.0
        assert 5.0 <= constrained_state[1] <= 450.0

        # Check temperature constraints
        assert 300.0 <= constrained_state[2] <= 400.0
        assert 280.0 <= constrained_state[3] <= 380.0

        # Check concentration constraints
        assert 0.1 <= constrained_state[4] <= 3.0
        assert 0.0 <= constrained_state[5] <= 2.5

        # Check quality metrics
        assert 0.0 <= constrained_state[8] <= 1.0
        assert 0.0 <= constrained_state[9] <= 1.0
        assert 0.0 <= constrained_state[10] <= 1.0
        assert 0.0 <= constrained_state[11] <= 1.0


class TestCostAwareAcquisitionFunction:
    """Test cost-aware acquisition function."""

    def create_test_acquisition_function(self):
        """Create test acquisition function."""
        config = RLConfig(cost_weight=0.1, constraint_weight=0.5)
        return CostAwareAcquisitionFunction(config)

    def test_acquisition_function_creation(self):
        """Test acquisition function creation."""
        acq_func = self.create_test_acquisition_function()

        assert acq_func.config.cost_weight == 0.1
        assert acq_func.config.constraint_weight == 0.5
        assert len(acq_func.cost_models) == 0
        assert len(acq_func.constraint_models) == 0

    def test_add_cost_model(self):
        """Test adding cost models."""
        acq_func = self.create_test_acquisition_function()

        acq_func.add_cost_model("material", material_cost_model)
        acq_func.add_cost_model("energy", energy_cost_model)

        assert len(acq_func.cost_models) == 2
        assert "material" in acq_func.cost_models
        assert "energy" in acq_func.cost_models

    def test_add_constraint(self):
        """Test adding constraints."""
        acq_func = self.create_test_acquisition_function()

        constraint = ProcessConstraint(
            name="temperature",
            constraint_type=ConstraintType.SAFETY,
            min_value=300.0,
            max_value=400.0,
            violation_penalty=100.0,
        )
        acq_func.add_constraint(constraint)

        assert len(acq_func.constraint_models) == 1
        assert "temperature" in acq_func.constraint_models

    def test_acquisition_evaluation(self):
        """Test acquisition function evaluation."""
        acq_func = self.create_test_acquisition_function()

        # Add cost model and constraint
        acq_func.add_cost_model("material", material_cost_model)

        constraint = ProcessConstraint(
            name="temperature",
            constraint_type=ConstraintType.SAFETY,
            min_value=300.0,
            max_value=400.0,
            violation_penalty=100.0,
        )
        acq_func.add_constraint(constraint)

        # Test data
        candidates = np.array(
            [
                [350.0, 0.1, 0.05],  # Good candidate
                [450.0, 0.5, 0.2],  # Bad candidate (high temp)
                [320.0, 0.2, 0.1],  # Medium candidate
            ]
        )

        surrogate_mean = np.array([0.8, 0.6, 0.7])
        surrogate_variance = np.array([0.1, 0.2, 0.15])
        best_value = 0.5

        acquisition_values = acq_func.evaluate(
            candidates, surrogate_mean, surrogate_variance, best_value
        )

        assert len(acquisition_values) == 3
        assert np.all(np.isfinite(acquisition_values))

        # Good candidate should have higher acquisition value than bad candidate
        assert acquisition_values[0] > acquisition_values[1]


class TestRealTimeDataManager:
    """Test real-time data management."""

    def create_test_data_manager(self):
        """Create test data manager."""
        config = RLConfig(data_buffer_size=100, model_update_frequency=10)
        return RealTimeDataManager(config)

    def test_data_manager_creation(self):
        """Test data manager creation."""
        manager = self.create_test_data_manager()

        assert manager.config.data_buffer_size == 100
        assert manager.config.model_update_frequency == 10
        assert len(manager.data_buffer) == 0
        assert len(manager.subscribers) == 0
        assert manager.is_running is False

    def test_add_subscriber(self):
        """Test adding subscribers."""
        manager = self.create_test_data_manager()

        async def test_callback(data: RealTimeData):
            pass

        manager.add_subscriber(test_callback)
        assert len(manager.subscribers) == 1

    def test_add_preprocessing(self):
        """Test adding preprocessing functions."""
        manager = self.create_test_data_manager()

        def test_preprocessing(data: RealTimeData) -> RealTimeData:
            return data

        manager.add_preprocessing(test_preprocessing)
        assert len(manager.preprocessing_functions) == 1

    def test_add_validation_rule(self):
        """Test adding validation rules."""
        manager = self.create_test_data_manager()

        def test_validation(data: RealTimeData) -> bool:
            return True

        manager.add_validation_rule(test_validation)
        assert len(manager.validation_rules) == 1

    @pytest.mark.asyncio
    async def test_data_ingestion(self):
        """Test data ingestion."""
        manager = self.create_test_data_manager()

        # Add validation rule
        def validate_data(data: RealTimeData) -> bool:
            return len(data.measurements) > 0

        manager.add_validation_rule(validate_data)

        # Create test data
        test_data = RealTimeData(
            timestamp=time.time(),
            experiment_id="test_001",
            measurements={"temperature": 350.0, "yield": 0.85},
        )

        # Ingest data
        await manager.ingest_data(test_data)

        assert len(manager.data_buffer) == 1
        assert manager.data_buffer[0].experiment_id == "test_001"

    @pytest.mark.asyncio
    async def test_data_validation_failure(self):
        """Test data validation failure."""
        manager = self.create_test_data_manager()

        # Add strict validation rule
        def validate_data(data: RealTimeData) -> bool:
            return "required_field" in data.measurements

        manager.add_validation_rule(validate_data)

        # Create invalid test data
        test_data = RealTimeData(
            timestamp=time.time(),
            experiment_id="test_002",
            measurements={"temperature": 350.0},  # Missing required_field
        )

        # Ingest data (should be rejected)
        await manager.ingest_data(test_data)

        assert len(manager.data_buffer) == 0

    def test_get_recent_data(self):
        """Test getting recent data."""
        manager = self.create_test_data_manager()

        # Add some test data directly to buffer
        for i in range(5):
            data = RealTimeData(
                timestamp=time.time() + i,
                experiment_id=f"test_{i}",
                measurements={"value": i},
            )
            manager.data_buffer.append(data)

        # Get recent data
        recent_data = manager.get_recent_data(3)
        assert len(recent_data) == 3
        assert recent_data[-1].experiment_id == "test_4"

    def test_get_data_statistics(self):
        """Test data statistics calculation."""
        manager = self.create_test_data_manager()

        # Empty buffer
        stats = manager.get_data_statistics()
        assert stats["status"] == "empty"

        # Add test data
        base_time = time.time()
        for i in range(10):
            data = RealTimeData(
                timestamp=base_time + i,
                experiment_id=f"test_{i}",
                measurements={"temperature": 300 + i, "pressure": 1e5 + i * 1000},
            )
            manager.data_buffer.append(data)

        # Calculate statistics
        stats = manager.get_data_statistics()

        assert stats["buffer_size"] == 10
        assert "measurement_stats" in stats
        assert "temperature" in stats["measurement_stats"]
        assert "pressure" in stats["measurement_stats"]

        temp_stats = stats["measurement_stats"]["temperature"]
        assert temp_stats["count"] == 10
        assert temp_stats["mean"] == 304.5  # 300 + (0+1+...+9)/10
        assert temp_stats["min"] == 300.0
        assert temp_stats["max"] == 309.0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPPOAgent:
    """Test PPO agent implementation."""

    def create_test_agent(self):
        """Create test PPO agent."""
        config = RLConfig(
            hidden_sizes=[64, 64],
            learning_rate=1e-3,
            batch_size=32,
            use_gpu=False,
            device="cpu",
        )
        return PPOAgent(config, state_dim=5, action_dim=2)

    def test_agent_creation(self):
        """Test agent creation."""
        agent = self.create_test_agent()

        assert agent.state_dim == 5
        assert agent.action_dim == 2
        assert agent.device == "cpu"
        assert hasattr(agent, "policy_network")
        assert hasattr(agent, "optimizer")

    @pytest.mark.asyncio
    async def test_action_selection(self):
        """Test action selection."""
        agent = self.create_test_agent()

        state = np.random.randn(5)

        # Training mode
        action = await agent.select_action(state, training=True)
        assert len(action) == 2
        assert np.all(np.abs(action) <= 1.0)  # Actions should be clipped

        # Evaluation mode
        action = await agent.select_action(state, training=False)
        assert len(action) == 2
        assert np.all(np.abs(action) <= 1.0)

    def test_buffer_management(self):
        """Test experience buffer management."""
        agent = self.create_test_agent()

        # Initially empty
        assert len(agent.states) == 0
        assert len(agent.actions) == 0
        assert len(agent.rewards) == 0

        # Store experience
        agent.store_reward_and_done(1.0, False)
        assert len(agent.rewards) == 1
        assert len(agent.dones) == 1

        # Clear buffer
        agent.clear_buffer()
        assert len(agent.states) == 0
        assert len(agent.rewards) == 0

    @pytest.mark.asyncio
    async def test_agent_update(self):
        """Test agent update with minimal data."""
        agent = self.create_test_agent()

        # Generate minimal experience
        for _ in range(agent.config.batch_size):
            state = np.random.randn(5)
            await agent.select_action(state, training=True)
            agent.store_reward_and_done(np.random.randn(), False)

        # Update agent
        metrics = await agent.update()

        # Should return metrics
        assert isinstance(metrics, dict)
        if metrics:  # May be empty if not enough data
            assert "policy_loss" in metrics or len(metrics) == 0

    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        agent = self.create_test_agent()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            # Save checkpoint
            agent.save_checkpoint(tmp.name)

            # Modify agent state
            agent.episode_count = 100

            # Load checkpoint
            agent.load_checkpoint(tmp.name)

            # Should restore original state
            assert agent.episode_count == 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestProcessControlTrainer:
    """Test process control trainer."""

    def create_test_trainer(self):
        """Create test trainer."""
        config = RLConfig(
            max_episodes=3, max_steps_per_episode=5, use_gpu=False, device="cpu"
        )

        env = MultiStepSynthesisEnvironment(config)
        agent = PPOAgent(config, env.state_dim, env.action_dim)
        trainer = ProcessControlTrainer(agent, env, config)

        return trainer

    @pytest.mark.asyncio
    async def test_trainer_creation(self):
        """Test trainer creation."""
        trainer = self.create_test_trainer()

        assert hasattr(trainer, "agent")
        assert hasattr(trainer, "environment")
        assert hasattr(trainer, "config")
        assert len(trainer.episode_rewards) == 0
        assert trainer.best_reward == -np.inf

    @pytest.mark.asyncio
    async def test_training_loop(self):
        """Test basic training loop."""
        trainer = self.create_test_trainer()

        # Run short training
        summary = await trainer.train()

        assert isinstance(summary, dict)
        assert "total_episodes" in summary
        assert "total_steps" in summary
        assert "best_reward" in summary
        assert "training_time" in summary

        assert summary["total_episodes"] == 3
        assert summary["total_steps"] > 0
        assert summary["training_time"] > 0

        # Check that rewards were recorded
        assert len(trainer.episode_rewards) == 3

    @pytest.mark.asyncio
    async def test_evaluation(self):
        """Test agent evaluation."""
        trainer = self.create_test_trainer()

        # Run evaluation
        eval_summary = await trainer.evaluate(num_episodes=2)

        assert isinstance(eval_summary, dict)
        assert "num_episodes" in eval_summary
        assert "mean_reward" in eval_summary
        assert "std_reward" in eval_summary
        assert "rewards" in eval_summary
        assert "lengths" in eval_summary

        assert eval_summary["num_episodes"] == 2
        assert len(eval_summary["rewards"]) == 2
        assert len(eval_summary["lengths"]) == 2


class TestUtilityFunctions:
    """Test utility functions."""

    @pytest.mark.asyncio
    async def test_train_synthesis_agent(self):
        """Test synthesis agent training utility."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        config = RLConfig(
            max_episodes=2, max_steps_per_episode=3, use_gpu=False, device="cpu"
        )

        # Custom reaction network
        reaction_network = {
            "reactions": [
                {"reactants": ["A"], "products": ["B"], "rate_constant": 0.1}
            ],
            "initial_concentrations": {"A": 1.0, "B": 0.0},
        }

        # Safety constraints
        constraints = create_safety_constraints()

        agent, summary = await train_synthesis_agent(
            reaction_network=reaction_network, config=config, constraints=constraints
        )

        assert agent is not None
        assert isinstance(summary, dict)
        assert "total_episodes" in summary
        assert summary["total_episodes"] == 2

    @pytest.mark.asyncio
    async def test_train_continuous_manufacturing_agent(self):
        """Test continuous manufacturing agent training utility."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        config = RLConfig(
            max_episodes=2, max_steps_per_episode=5, use_gpu=False, device="cpu"
        )

        # Custom process config
        process_config = {
            "reactor_volume": 500.0,
            "residence_time": 1800.0,
            "nominal_flow_rate": 50.0,
            "nominal_temperature": 340.0,
        }

        # Quality constraints
        constraints = create_quality_constraints()

        agent, summary = await train_continuous_manufacturing_agent(
            process_config=process_config, config=config, constraints=constraints
        )

        assert agent is not None
        assert isinstance(summary, dict)
        assert "total_episodes" in summary
        assert summary["total_episodes"] == 2

    def test_create_cost_aware_acquisition_function(self):
        """Test cost-aware acquisition function creation utility."""
        cost_models = {"material": material_cost_model, "energy": energy_cost_model}

        constraints = create_safety_constraints() + create_cost_constraints()

        acq_func = create_cost_aware_acquisition_function(
            cost_models=cost_models, constraints=constraints
        )

        assert isinstance(acq_func, CostAwareAcquisitionFunction)
        assert len(acq_func.cost_models) == 2
        assert len(acq_func.constraint_models) == len(constraints)

    def test_create_real_time_data_manager(self):
        """Test real-time data manager creation utility."""
        manager = create_real_time_data_manager()

        assert isinstance(manager, RealTimeDataManager)
        assert len(manager.preprocessing_functions) == 1  # Default standardization
        assert len(manager.validation_rules) == 1  # Default validation

    def test_cost_models(self):
        """Test example cost models."""
        action = np.array([0.1, 5.0, 0.05, 0.2])

        # Material cost model
        material_cost = material_cost_model(action)
        assert isinstance(material_cost, ExperimentalCost)
        assert material_cost.material_cost > 0.0
        assert material_cost.total_cost() > 0.0

        # Energy cost model
        energy_cost = energy_cost_model(action)
        assert isinstance(energy_cost, ExperimentalCost)
        assert energy_cost.energy_cost > 0.0
        assert energy_cost.total_cost() > 0.0

    def test_constraint_creation_utilities(self):
        """Test constraint creation utilities."""
        # Safety constraints
        safety_constraints = create_safety_constraints()
        assert len(safety_constraints) > 0
        assert all(
            c.constraint_type == ConstraintType.SAFETY for c in safety_constraints
        )

        # Quality constraints
        quality_constraints = create_quality_constraints()
        assert len(quality_constraints) > 0
        assert all(
            c.constraint_type == ConstraintType.QUALITY for c in quality_constraints
        )

        # Cost constraints
        cost_constraints = create_cost_constraints(max_cost_per_step=150.0)
        assert len(cost_constraints) > 0
        assert all(c.constraint_type == ConstraintType.COST for c in cost_constraints)
        assert cost_constraints[0].max_value == 150.0


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_missing_torch_error(self):
        """Test error when PyTorch is not available."""
        if TORCH_AVAILABLE:
            pytest.skip("PyTorch is available")

        config = RLConfig(use_gpu=False)

        with pytest.raises(RLError):
            await train_synthesis_agent(config=config)

    def test_invalid_environment_type(self):
        """Test handling of invalid environment types."""
        config = RLConfig(environment_type=EnvironmentType.SYNTHESIS)

        # This should work
        env = MultiStepSynthesisEnvironment(config)
        assert env is not None

        # Test with continuous manufacturing config
        config.environment_type = EnvironmentType.CONTINUOUS_MANUFACTURING
        env2 = ContinuousManufacturingEnvironment(config)
        assert env2 is not None

    def test_constraint_edge_cases(self):
        """Test constraint evaluation edge cases."""
        # Constraint with only minimum
        constraint_min = ProcessConstraint(
            name="test",
            constraint_type=ConstraintType.SAFETY,
            min_value=10.0,
            violation_penalty=100.0,
        )

        satisfied, penalty = constraint_min.evaluate(15.0)
        assert satisfied is True
        assert penalty == 0.0

        satisfied, penalty = constraint_min.evaluate(5.0)
        assert satisfied is False
        assert penalty > 0.0

        # Constraint with only maximum
        constraint_max = ProcessConstraint(
            name="test",
            constraint_type=ConstraintType.SAFETY,
            max_value=100.0,
            violation_penalty=50.0,
        )

        satisfied, penalty = constraint_max.evaluate(80.0)
        assert satisfied is True
        assert penalty == 0.0

        satisfied, penalty = constraint_max.evaluate(120.0)
        assert satisfied is False
        assert penalty > 0.0

    def test_data_manager_edge_cases(self):
        """Test data manager edge cases."""
        manager = create_real_time_data_manager()

        # Empty data statistics
        stats = manager.get_data_statistics()
        assert stats["status"] == "empty"

        # Get recent data from empty buffer
        recent = manager.get_recent_data(10)
        assert len(recent) == 0

        # Test with None n_points
        recent = manager.get_recent_data(None)
        assert len(recent) == 0


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_full_rl_pipeline(self):
        """Test full RL pipeline integration."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        # Configuration
        config = RLConfig(
            max_episodes=2,
            max_steps_per_episode=3,
            use_gpu=False,
            device="cpu",
            cost_weight=0.1,
            constraint_weight=0.5,
        )

        # Create environment with constraints and cost model
        env = MultiStepSynthesisEnvironment(config)

        # Add constraints
        safety_constraints = create_safety_constraints()
        for constraint in safety_constraints:
            env.add_constraint(constraint)

        # Set cost model
        env.set_cost_model(material_cost_model)

        # Create agent
        agent = PPOAgent(config, env.state_dim, env.action_dim)

        # Create trainer
        trainer = ProcessControlTrainer(agent, env, config)

        # Train
        summary = await trainer.train()

        # Evaluate
        eval_summary = await trainer.evaluate(num_episodes=1)

        assert summary["total_episodes"] == 2
        assert eval_summary["num_episodes"] == 1
        assert len(env.constraints) == len(safety_constraints)

    @pytest.mark.asyncio
    async def test_real_time_integration(self):
        """Test real-time data integration."""
        # Create data manager
        data_manager = create_real_time_data_manager()

        # Start streaming
        await data_manager.start_streaming()
        assert data_manager.is_running is True

        # Create test data
        test_data = RealTimeData(
            timestamp=time.time(),
            experiment_id="integration_test",
            measurements={"temperature": 350.0, "yield": 0.85, "cost": 45.0},
        )

        # Ingest data
        await data_manager.ingest_data(test_data)

        # Check data was processed
        assert len(data_manager.data_buffer) == 1

        # Get statistics
        stats = data_manager.get_data_statistics()
        assert stats["buffer_size"] == 1
        assert "temperature" in stats["measurement_stats"]

        # Stop streaming
        await data_manager.stop_streaming()
        assert data_manager.is_running is False

    def test_cost_aware_optimization_integration(self):
        """Test cost-aware optimization integration."""
        # Create acquisition function
        cost_models = {"material": material_cost_model, "energy": energy_cost_model}
        constraints = create_safety_constraints() + create_quality_constraints()

        acq_func = create_cost_aware_acquisition_function(
            cost_models=cost_models, constraints=constraints
        )

        # Test evaluation
        candidates = np.array(
            [
                [350.0, 0.1, 0.05],  # Good candidate
                [450.0, 0.5, 0.2],  # Bad candidate (high temp)
                [320.0, 0.2, 0.1],  # Medium candidate
            ]
        )

        surrogate_mean = np.array([0.8, 0.6, 0.7])
        surrogate_variance = np.array([0.1, 0.2, 0.15])
        best_value = 0.5

        acquisition_values = acq_func.evaluate(
            candidates, surrogate_mean, surrogate_variance, best_value
        )

        assert len(acquisition_values) == 3
        assert np.all(np.isfinite(acquisition_values))


if __name__ == "__main__":
    # Run specific test categories

    print("Testing RL Configuration...")
    test_config = TestRLConfig()
    test_config.test_default_config()
    test_config.test_custom_config()
    print("✓ RL Configuration tests passed")

    print("\nTesting Cost and Constraint Models...")
    test_cost = TestExperimentalCost()
    test_cost.test_cost_creation()
    test_cost.test_cost_to_dict()

    test_constraint = TestProcessConstraint()
    test_constraint.test_constraint_creation()
    test_constraint.test_constraint_evaluation_satisfied()
    test_constraint.test_constraint_evaluation_violated()
    print("✓ Cost and Constraint tests passed")

    print("\nTesting Chemical Environments...")
    test_synthesis = TestMultiStepSynthesisEnvironment()
    test_synthesis.test_environment_initialization()
    test_synthesis.test_environment_reset()
    test_synthesis.test_environment_step()

    test_continuous = TestContinuousManufacturingEnvironment()
    test_continuous.test_environment_initialization()
    test_continuous.test_environment_reset()
    test_continuous.test_environment_step()
    print("✓ Chemical Environment tests passed")

    print("\nTesting Cost-Aware Acquisition...")
    test_acq = TestCostAwareAcquisitionFunction()
    test_acq.test_acquisition_function_creation()
    test_acq.test_add_cost_model()
    test_acq.test_add_constraint()
    print("✓ Cost-Aware Acquisition tests passed")

    print("\nTesting Real-Time Data Management...")
    test_rtd = TestRealTimeDataManager()
    test_rtd.test_data_manager_creation()
    test_rtd.test_add_subscriber()
    test_rtd.test_get_recent_data()
    test_rtd.test_get_data_statistics()
    print("✓ Real-Time Data Management tests passed")

    print("\nTesting Utility Functions...")
    test_util = TestUtilityFunctions()
    test_util.test_create_cost_aware_acquisition_function()
    test_util.test_create_real_time_data_manager()
    test_util.test_cost_models()
    test_util.test_constraint_creation_utilities()
    print("✓ Utility Function tests passed")

    print("\nAll manual tests completed successfully!")
    print(
        "Run 'pytest test_rl_process_control.py -v' for full test suite including async tests"
    )
