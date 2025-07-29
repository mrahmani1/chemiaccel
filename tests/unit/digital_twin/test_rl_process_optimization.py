"""
Unit tests for reinforcement learning process optimization module.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from src.chemi_accel.core.exceptions import ChemiAccelError
from src.chemi_accel.digital_twin.manufacturing_integration import (
    ManufacturingDigitalTwin,
    ProcessConditions,
    ProcessOutput,
    ProcessType,
)
from src.chemi_accel.digital_twin.multiscale_framework import MultiScaleFramework
from src.chemi_accel.digital_twin.rl_process_optimization import (
    Actor,
    Critic,
    EnergyEfficiencyRewardFunction,
    MultiObjectiveRewardFunction,
    OptimizationObjective,
    ProcessAction,
    ProcessEnvironment,
    ProcessState,
    QualityRewardFunction,
    ReplayBuffer,
    RLAlgorithm,
    RLConfig,
)


class TestRLConfig:
    """Test RLConfig dataclass."""

    def test_default_initialization(self):
        """Test default initialization."""
        config = RLConfig()

        assert config.algorithm == RLAlgorithm.PPO
        assert config.device in ["cuda", "cpu"]
        assert config.hidden_sizes == [256, 256, 128]
        assert config.learning_rate == 3e-4
        assert config.gamma == 0.99
        assert config.max_episodes == 1000

    def test_custom_initialization(self):
        """Test custom initialization."""
        config = RLConfig(
            algorithm=RLAlgorithm.SAC,
            learning_rate=1e-3,
            hidden_sizes=[128, 128],
            max_episodes=500,
        )

        assert config.algorithm == RLAlgorithm.SAC
        assert config.learning_rate == 1e-3
        assert config.hidden_sizes == [128, 128]
        assert config.max_episodes == 500


class TestProcessState:
    """Test ProcessState dataclass."""

    @pytest.fixture
    def sample_state(self):
        """Create sample process state."""
        return ProcessState(
            temperature=473.15,
            pressure=2e5,
            flow_rate=0.1,
            shear_rate=100.0,
            residence_time=60.0,
            viscosity=1000.0,
            density=1000.0,
            thermal_conductivity=0.2,
            part_quality=0.8,
            dimensional_accuracy=0.85,
            surface_finish=0.75,
            quality_trend=[0.7, 0.75, 0.8],
            energy_consumption_trend=[1000.0, 950.0, 900.0],
        )

    def test_to_tensor(self, sample_state):
        """Test conversion to tensor."""
        tensor = sample_state.to_tensor()

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == sample_state.state_size
        assert torch.all(torch.isfinite(tensor))

    def test_to_tensor_gpu(self, sample_state):
        """Test conversion to GPU tensor."""
        if torch.cuda.is_available():
            tensor = sample_state.to_tensor("cuda")
            assert tensor.device.type == "cuda"

    def test_state_size(self, sample_state):
        """Test state size property."""
        assert sample_state.state_size == 19

    def test_empty_trends(self):
        """Test state with empty trends."""
        state = ProcessState(
            temperature=473.15,
            pressure=2e5,
            flow_rate=0.1,
            shear_rate=100.0,
            residence_time=60.0,
            viscosity=1000.0,
            density=1000.0,
            thermal_conductivity=0.2,
            part_quality=0.8,
            dimensional_accuracy=0.85,
            surface_finish=0.75,
        )

        tensor = state.to_tensor()
        assert tensor.shape[0] == state.state_size
        assert torch.all(torch.isfinite(tensor))


class TestProcessAction:
    """Test ProcessAction dataclass."""

    @pytest.fixture
    def sample_action(self):
        """Create sample process action."""
        return ProcessAction(
            temperature_delta=0.1,
            pressure_delta=-0.05,
            flow_rate_delta=0.02,
            shear_rate_delta=0.15,
            residence_time_delta=-0.1,
            heating_power=0.5,
            cooling_rate=0.3,
            pump_speed=0.8,
        )

    def test_to_tensor(self, sample_action):
        """Test conversion to tensor."""
        tensor = sample_action.to_tensor()

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == sample_action.action_size
        assert torch.all(torch.isfinite(tensor))

    def test_from_tensor(self, sample_action):
        """Test creation from tensor."""
        tensor = sample_action.to_tensor()
        reconstructed = ProcessAction.from_tensor(tensor)

        assert (
            abs(reconstructed.temperature_delta - sample_action.temperature_delta)
            < 1e-6
        )
        assert abs(reconstructed.pressure_delta - sample_action.pressure_delta) < 1e-6
        assert abs(reconstructed.flow_rate_delta - sample_action.flow_rate_delta) < 1e-6

    def test_action_size(self, sample_action):
        """Test action size property."""
        assert sample_action.action_size == 8


class TestQualityRewardFunction:
    """Test QualityRewardFunction class."""

    @pytest.fixture
    def reward_function(self):
        """Create QualityRewardFunction instance."""
        return QualityRewardFunction(
            quality_weight=1.0,
            dimensional_weight=0.8,
            surface_weight=0.6,
            target_quality=0.95,
        )

    @pytest.fixture
    def sample_state(self):
        """Create sample process state."""
        return ProcessState(
            temperature=473.15,
            pressure=2e5,
            flow_rate=0.1,
            shear_rate=100.0,
            residence_time=60.0,
            viscosity=1000.0,
            density=1000.0,
            thermal_conductivity=0.2,
            part_quality=0.7,
            dimensional_accuracy=0.75,
            surface_finish=0.7,
        )

    @pytest.fixture
    def sample_action(self):
        """Create sample process action."""
        return ProcessAction(0.1, -0.05, 0.02, 0.15, -0.1)

    @pytest.fixture
    def sample_output(self):
        """Create sample process output."""
        return ProcessOutput(
            part_quality=0.85,
            dimensional_accuracy=0.9,
            surface_finish=0.8,
            processing_time=80.0,
        )

    def test_calculate_reward_improvement(
        self, reward_function, sample_state, sample_action, sample_output
    ):
        """Test reward calculation with improvement."""
        next_state = ProcessState(
            temperature=473.15,
            pressure=2e5,
            flow_rate=0.1,
            shear_rate=100.0,
            residence_time=60.0,
            viscosity=1000.0,
            density=1000.0,
            thermal_conductivity=0.2,
            part_quality=0.85,
            dimensional_accuracy=0.9,
            surface_finish=0.8,
            quality_trend=[0.7, 0.75, 0.8, 0.85],
        )

        reward = reward_function.calculate_reward(
            sample_state, sample_action, next_state, sample_output
        )

        assert isinstance(reward, float)
        assert reward > 0  # Should be positive for improvement
        assert -5.0 <= reward <= 5.0  # Should be clipped to range

    def test_calculate_reward_target_reached(
        self, reward_function, sample_state, sample_action
    ):
        """Test reward when target quality is reached."""
        high_quality_output = ProcessOutput(
            part_quality=0.98,
            dimensional_accuracy=0.97,
            surface_finish=0.96,
            processing_time=80.0,
        )

        next_state = ProcessState(
            temperature=473.15,
            pressure=2e5,
            flow_rate=0.1,
            shear_rate=100.0,
            residence_time=60.0,
            viscosity=1000.0,
            density=1000.0,
            thermal_conductivity=0.2,
            part_quality=0.98,
            dimensional_accuracy=0.97,
            surface_finish=0.96,
        )

        reward = reward_function.calculate_reward(
            sample_state, sample_action, next_state, high_quality_output
        )

        assert reward > 1.0  # Should get target bonus

    def test_calculate_reward_poor_quality(
        self, reward_function, sample_state, sample_action
    ):
        """Test reward with poor quality."""
        poor_output = ProcessOutput(
            part_quality=0.3,
            dimensional_accuracy=0.25,
            surface_finish=0.2,
            processing_time=80.0,
        )

        next_state = ProcessState(
            temperature=473.15,
            pressure=2e5,
            flow_rate=0.1,
            shear_rate=100.0,
            residence_time=60.0,
            viscosity=1000.0,
            density=1000.0,
            thermal_conductivity=0.2,
            part_quality=0.3,
            dimensional_accuracy=0.25,
            surface_finish=0.2,
        )

        reward = reward_function.calculate_reward(
            sample_state, sample_action, next_state, poor_output
        )

        assert reward < 0  # Should be negative for poor quality

    def test_get_objectives(self, reward_function):
        """Test getting optimization objectives."""
        objectives = reward_function.get_objectives()
        assert OptimizationObjective.QUALITY in objectives


class TestEnergyEfficiencyRewardFunction:
    """Test EnergyEfficiencyRewardFunction class."""

    @pytest.fixture
    def reward_function(self):
        """Create EnergyEfficiencyRewardFunction instance."""
        return EnergyEfficiencyRewardFunction(
            energy_weight=1.0, quality_threshold=0.7, baseline_energy=1000.0
        )

    @pytest.fixture
    def sample_state(self):
        """Create sample process state."""
        return ProcessState(
            temperature=473.15,
            pressure=2e5,
            flow_rate=0.1,
            shear_rate=100.0,
            residence_time=60.0,
            viscosity=1000.0,
            density=1000.0,
            thermal_conductivity=0.2,
            part_quality=0.8,
            dimensional_accuracy=0.85,
            surface_finish=0.8,
        )

    def test_calculate_reward_efficient(self, reward_function, sample_state):
        """Test reward for energy efficient process."""
        action = ProcessAction(0.1, -0.05, 0.02, 0.15, -0.1)

        efficient_output = ProcessOutput(
            part_quality=0.85,
            dimensional_accuracy=0.9,
            surface_finish=0.8,
            energy_consumption=500.0,  # Half of baseline
        )

        next_state = ProcessState(
            temperature=473.15,
            pressure=2e5,
            flow_rate=0.1,
            shear_rate=100.0,
            residence_time=60.0,
            viscosity=1000.0,
            density=1000.0,
            thermal_conductivity=0.2,
            part_quality=0.85,
            dimensional_accuracy=0.9,
            surface_finish=0.8,
        )

        reward = reward_function.calculate_reward(
            sample_state, action, next_state, efficient_output
        )

        assert reward > 0.5  # Should get high reward for efficiency

    def test_calculate_reward_inefficient(self, reward_function, sample_state):
        """Test reward for energy inefficient process."""
        action = ProcessAction(0.1, -0.05, 0.02, 0.15, -0.1)

        inefficient_output = ProcessOutput(
            part_quality=0.85, energy_consumption=3000.0  # Three times baseline
        )

        next_state = ProcessState(
            temperature=473.15,
            pressure=2e5,
            flow_rate=0.1,
            shear_rate=100.0,
            residence_time=60.0,
            viscosity=1000.0,
            density=1000.0,
            thermal_conductivity=0.2,
            part_quality=0.85,
            dimensional_accuracy=0.85,
            surface_finish=0.8,
        )

        reward = reward_function.calculate_reward(
            sample_state, action, next_state, inefficient_output
        )

        assert reward < 0.5  # Should get low reward for inefficiency

    def test_get_objectives(self, reward_function):
        """Test getting optimization objectives."""
        objectives = reward_function.get_objectives()
        assert OptimizationObjective.ENERGY_EFFICIENCY in objectives


class TestMultiObjectiveRewardFunction:
    """Test MultiObjectiveRewardFunction class."""

    @pytest.fixture
    def reward_function(self):
        """Create MultiObjectiveRewardFunction instance."""
        return MultiObjectiveRewardFunction(
            objective_weights={
                OptimizationObjective.QUALITY: 0.5,
                OptimizationObjective.ENERGY_EFFICIENCY: 0.3,
                OptimizationObjective.CYCLE_TIME: 0.2,
            }
        )

    def test_calculate_reward(self, reward_function):
        """Test multi-objective reward calculation."""
        state = ProcessState(
            temperature=473.15,
            pressure=2e5,
            flow_rate=0.1,
            shear_rate=100.0,
            residence_time=60.0,
            viscosity=1000.0,
            density=1000.0,
            thermal_conductivity=0.2,
            part_quality=0.7,
            dimensional_accuracy=0.75,
            surface_finish=0.7,
        )

        action = ProcessAction(0.1, -0.05, 0.02, 0.15, -0.1)

        next_state = ProcessState(
            temperature=473.15,
            pressure=2e5,
            flow_rate=0.1,
            shear_rate=100.0,
            residence_time=60.0,
            viscosity=1000.0,
            density=1000.0,
            thermal_conductivity=0.2,
            part_quality=0.85,
            dimensional_accuracy=0.9,
            surface_finish=0.8,
        )

        output = ProcessOutput(
            part_quality=0.85,
            dimensional_accuracy=0.9,
            surface_finish=0.8,
            processing_time=80.0,
            energy_consumption=800.0,
        )

        reward = reward_function.calculate_reward(state, action, next_state, output)

        assert isinstance(reward, float)
        assert -5.0 <= reward <= 5.0

    def test_pareto_archive_update(self, reward_function):
        """Test Pareto archive updating."""
        initial_size = len(reward_function.pareto_archive)

        # Add a good solution
        state = ProcessState(
            temperature=473.15,
            pressure=2e5,
            flow_rate=0.1,
            shear_rate=100.0,
            residence_time=60.0,
            viscosity=1000.0,
            density=1000.0,
            thermal_conductivity=0.2,
            part_quality=0.85,
            dimensional_accuracy=0.9,
            surface_finish=0.8,
        )

        action = ProcessAction(0.1, -0.05, 0.02, 0.15, -0.1)
        output = ProcessOutput(
            part_quality=0.85, dimensional_accuracy=0.9, surface_finish=0.8
        )

        reward_function.calculate_reward(state, action, state, output)

        assert len(reward_function.pareto_archive) > initial_size

    def test_get_objectives(self, reward_function):
        """Test getting optimization objectives."""
        objectives = reward_function.get_objectives()
        assert OptimizationObjective.QUALITY in objectives
        assert OptimizationObjective.ENERGY_EFFICIENCY in objectives
        assert OptimizationObjective.CYCLE_TIME in objectives


class TestActor:
    """Test Actor network."""

    @pytest.fixture
    def actor(self):
        """Create Actor instance."""
        return Actor(
            state_size=19, action_size=8, hidden_sizes=[64, 64], activation="relu"
        )

    def test_forward(self, actor):
        """Test forward pass."""
        batch_size = 5
        state = torch.randn(batch_size, 19)

        mean, log_std = actor.forward(state)

        assert mean.shape == (batch_size, 8)
        assert log_std.shape == (batch_size, 8)
        assert torch.all(torch.isfinite(mean))
        assert torch.all(torch.isfinite(log_std))
        assert torch.all(torch.abs(mean) <= 1.0)  # Should be bounded by tanh

    def test_sample_action(self, actor):
        """Test action sampling."""
        state = torch.randn(1, 19)

        action, log_prob = actor.sample_action(state)

        assert action.shape == (1, 8)
        assert log_prob.shape == (1, 1)
        assert torch.all(torch.isfinite(action))
        assert torch.all(torch.isfinite(log_prob))


class TestCritic:
    """Test Critic network."""

    def test_value_critic(self):
        """Test value function critic."""
        critic = Critic(state_size=19, action_size=None, hidden_sizes=[64, 64])

        batch_size = 5
        state = torch.randn(batch_size, 19)

        value = critic.forward(state)

        assert value.shape == (batch_size, 1)
        assert torch.all(torch.isfinite(value))

    def test_q_function_critic(self):
        """Test Q-function critic."""
        critic = Critic(state_size=19, action_size=8, hidden_sizes=[64, 64])

        batch_size = 5
        state = torch.randn(batch_size, 19)
        action = torch.randn(batch_size, 8)

        q_value = critic.forward(state, action)

        assert q_value.shape == (batch_size, 1)
        assert torch.all(torch.isfinite(q_value))


class TestReplayBuffer:
    """Test ReplayBuffer class."""

    @pytest.fixture
    def buffer(self):
        """Create ReplayBuffer instance."""
        return ReplayBuffer(capacity=1000, state_size=19, action_size=8, device="cpu")

    def test_add_experience(self, buffer):
        """Test adding experience to buffer."""
        state = torch.randn(19)
        action = torch.randn(8)
        reward = 1.5
        next_state = torch.randn(19)
        done = False

        initial_size = len(buffer)
        buffer.add(state, action, reward, next_state, done)

        assert len(buffer) == initial_size + 1
        assert buffer.size == 1

    def test_sample_batch(self, buffer):
        """Test sampling batch from buffer."""
        # Add some experiences
        for _ in range(10):
            state = torch.randn(19)
            action = torch.randn(8)
            reward = np.random.random()
            next_state = torch.randn(19)
            done = np.random.random() < 0.1

            buffer.add(state, action, reward, next_state, done)

        # Sample batch
        batch_size = 5
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        assert states.shape == (batch_size, 19)
        assert actions.shape == (batch_size, 8)
        assert rewards.shape == (batch_size, 1)
        assert next_states.shape == (batch_size, 19)
        assert dones.shape == (batch_size, 1)

    def test_buffer_capacity(self, buffer):
        """Test buffer capacity limits."""
        capacity = 10
        small_buffer = ReplayBuffer(capacity, 19, 8, "cpu")

        # Add more experiences than capacity
        for i in range(15):
            state = torch.randn(19)
            action = torch.randn(8)
            reward = float(i)
            next_state = torch.randn(19)
            done = False

            small_buffer.add(state, action, reward, next_state, done)

        assert len(small_buffer) == capacity
        assert small_buffer.size == capacity


class TestProcessEnvironment:
    """Test ProcessEnvironment class."""

    @pytest.fixture
    def mock_digital_twin(self):
        """Create mock digital twin."""
        mock_twin = Mock(spec=ManufacturingDigitalTwin)
        mock_twin.run_manufacturing_simulation.return_value = ProcessOutput(
            part_quality=0.8,
            dimensional_accuracy=0.85,
            surface_finish=0.75,
            processing_time=100.0,
            energy_consumption=1000.0,
        )
        mock_twin.process_conditions = None
        mock_twin.multiscale_framework = Mock()
        return mock_twin

    @pytest.fixture
    def reward_function(self):
        """Create reward function."""
        return QualityRewardFunction()

    @pytest.fixture
    def parameter_ranges(self):
        """Create parameter ranges."""
        return {
            "temperature": (400.0, 550.0),
            "pressure": (1e5, 5e5),
            "flow_rate": (0.01, 0.5),
            "shear_rate": (10.0, 500.0),
            "residence_time": (10.0, 300.0),
        }

    @pytest.fixture
    def environment(self, mock_digital_twin, reward_function, parameter_ranges):
        """Create ProcessEnvironment instance."""
        return ProcessEnvironment(
            mock_digital_twin,
            ProcessType.EXTRUSION,
            reward_function,
            parameter_ranges,
            max_steps=10,
        )

    def test_reset(self, environment):
        """Test environment reset."""
        state = environment.reset()

        assert isinstance(state, ProcessState)
        assert environment.current_step == 0
        assert environment.current_state == state
        assert len(environment.episode_rewards) == 0

    def test_step(self, environment):
        """Test environment step."""
        initial_state = environment.reset()
        action = ProcessAction(0.1, -0.05, 0.02, 0.15, -0.1)

        next_state, reward, done, info = environment.step(action)

        assert isinstance(next_state, ProcessState)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert environment.current_step == 1

    def test_episode_completion(self, environment):
        """Test complete episode."""
        environment.reset()

        for step in range(environment.max_steps + 1):
            action = ProcessAction(0.1, -0.05, 0.02, 0.15, -0.1)
            next_state, reward, done, info = environment.step(action)

            if done:
                break

        assert done  # Should be done after max steps
        assert len(environment.episode_rewards) > 0

    def test_get_episode_summary(self, environment):
        """Test episode summary."""
        environment.reset()

        # Run a few steps
        for _ in range(3):
            action = ProcessAction(0.1, -0.05, 0.02, 0.15, -0.1)
            environment.step(action)

        summary = environment.get_episode_summary()

        assert "total_steps" in summary
        assert "total_reward" in summary
        assert "average_reward" in summary
        assert summary["total_steps"] == 3


if __name__ == "__main__":
    pytest.main([__file__])
