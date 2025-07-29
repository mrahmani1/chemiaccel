"""
Unit tests for RL agents module.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from src.chemi_accel.core.exceptions import ChemiAccelError
from src.chemi_accel.digital_twin.manufacturing_integration import ProcessType
from src.chemi_accel.digital_twin.rl_agents import (
    DDPGAgent,
    PPOAgent,
    ProcessOptimizer,
    RLAgent,
    RLAgentFactory,
    SACAgent,
)
from src.chemi_accel.digital_twin.rl_process_optimization import (
    ProcessAction,
    ProcessEnvironment,
    ProcessState,
    QualityRewardFunction,
    RLAlgorithm,
    RLConfig,
)


class TestPPOAgent:
    """Test PPOAgent class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RLConfig(
            algorithm=RLAlgorithm.PPO,
            device="cpu",
            hidden_sizes=[32, 32],
            learning_rate=1e-3,
            batch_size=16,
            ppo_epochs=2,
            max_episodes=10,
        )

    @pytest.fixture
    def agent(self, config):
        """Create PPOAgent instance."""
        return PPOAgent(config, state_size=19, action_size=8)

    def test_initialization(self, agent, config):
        """Test agent initialization."""
        assert agent.config == config
        assert agent.state_size == 19
        assert agent.action_size == 8
        assert agent.actor is not None
        assert agent.critic is not None
        assert agent.actor_optimizer is not None
        assert agent.critic_optimizer is not None

    def test_select_action_training(self, agent):
        """Test action selection in training mode."""
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

        action = agent.select_action(state, training=True)

        assert isinstance(action, ProcessAction)
        # Check that experience was stored
        assert len(agent.states) == 1
        assert len(agent.actions) == 1
        assert len(agent.log_probs) == 1
        assert len(agent.values) == 1

    def test_select_action_evaluation(self, agent):
        """Test action selection in evaluation mode."""
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

        action = agent.select_action(state, training=False)

        assert isinstance(action, ProcessAction)
        # Check that no experience was stored
        assert len(agent.states) == 0
        assert len(agent.actions) == 0

    def test_store_experience(self, agent):
        """Test experience storage."""
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

        action = ProcessAction(0.1, -0.05, 0.02, 0.15, -0.1)
        next_state = state

        agent._store_experience(state, action, 1.0, next_state, False)

        assert len(agent.rewards) == 1
        assert len(agent.dones) == 1
        assert agent.rewards[0] == 1.0
        assert agent.dones[0] == False

    def test_calculate_advantages(self, agent):
        """Test advantage calculation."""
        # Add some dummy experience
        for i in range(5):
            agent.rewards.append(1.0)
            agent.dones.append(False)
            agent.values.append(torch.tensor([0.5]))

        returns, advantages = agent._calculate_advantages()

        assert len(returns) == 5
        assert len(advantages) == 5
        assert all(isinstance(r, float) for r in returns)
        assert all(isinstance(a, float) for a in advantages)

    def test_clear_experience(self, agent):
        """Test experience clearing."""
        # Add some dummy data
        agent.states.append(torch.randn(19))
        agent.actions.append(torch.randn(8))
        agent.rewards.append(1.0)
        agent.log_probs.append(torch.tensor([0.1]))
        agent.values.append(torch.tensor([0.5]))
        agent.dones.append(False)

        agent._clear_experience()

        assert len(agent.states) == 0
        assert len(agent.actions) == 0
        assert len(agent.rewards) == 0
        assert len(agent.log_probs) == 0
        assert len(agent.values) == 0
        assert len(agent.dones) == 0

    def test_save_load(self, agent, tmp_path):
        """Test saving and loading agent."""
        filepath = tmp_path / "test_agent.pth"

        # Save agent
        agent.save(str(filepath))
        assert filepath.exists()

        # Create new agent and load
        new_agent = PPOAgent(agent.config, agent.state_size, agent.action_size)
        new_agent.load(str(filepath))

        # Test that parameters are the same
        for param1, param2 in zip(
            agent.actor.parameters(), new_agent.actor.parameters()
        ):
            assert torch.allclose(param1, param2)


class TestDDPGAgent:
    """Test DDPGAgent class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RLConfig(
            algorithm=RLAlgorithm.DDPG,
            device="cpu",
            hidden_sizes=[32, 32],
            learning_rate=1e-3,
            buffer_size=1000,
            batch_size=16,
        )

    @pytest.fixture
    def agent(self, config):
        """Create DDPGAgent instance."""
        return DDPGAgent(config, state_size=19, action_size=8)

    def test_initialization(self, agent, config):
        """Test agent initialization."""
        assert agent.config == config
        assert agent.actor is not None
        assert agent.critic is not None
        assert agent.target_actor is not None
        assert agent.target_critic is not None
        assert agent.replay_buffer is not None

    def test_select_action_training(self, agent):
        """Test action selection with noise."""
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

        action = agent.select_action(state, training=True)

        assert isinstance(action, ProcessAction)

    def test_select_action_evaluation(self, agent):
        """Test action selection without noise."""
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

        action1 = agent.select_action(state, training=False)
        action2 = agent.select_action(state, training=False)

        # Actions should be deterministic in evaluation mode
        assert action1.temperature_delta == action2.temperature_delta

    def test_soft_update(self, agent):
        """Test soft update of target networks."""
        # Get initial target parameters
        initial_target_params = [
            param.clone() for param in agent.target_actor.parameters()
        ]

        # Modify source network
        for param in agent.actor.parameters():
            param.data += 0.1

        # Perform soft update
        agent._soft_update(agent.target_actor, agent.actor)

        # Check that target parameters changed but not completely
        for initial_param, updated_param in zip(
            initial_target_params, agent.target_actor.parameters()
        ):
            assert not torch.allclose(initial_param, updated_param)
            assert not torch.allclose(
                updated_param, agent.actor.state_dict()[updated_param]
            )


class TestSACAgent:
    """Test SACAgent class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RLConfig(
            algorithm=RLAlgorithm.SAC,
            device="cpu",
            hidden_sizes=[32, 32],
            learning_rate=1e-3,
            buffer_size=1000,
            batch_size=16,
        )

    @pytest.fixture
    def agent(self, config):
        """Create SACAgent instance."""
        return SACAgent(config, state_size=19, action_size=8)

    def test_initialization(self, agent, config):
        """Test agent initialization."""
        assert agent.config == config
        assert agent.actor is not None
        assert agent.critic1 is not None
        assert agent.critic2 is not None
        assert agent.target_critic1 is not None
        assert agent.target_critic2 is not None
        assert agent.log_alpha is not None
        assert agent.replay_buffer is not None

    def test_alpha_property(self, agent):
        """Test alpha property."""
        alpha = agent.alpha
        assert isinstance(alpha, torch.Tensor)
        assert alpha.item() > 0

    def test_select_action_stochastic(self, agent):
        """Test stochastic action selection."""
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

        action1 = agent.select_action(state, training=True)
        action2 = agent.select_action(state, training=True)

        # Actions should be different due to stochasticity
        assert isinstance(action1, ProcessAction)
        assert isinstance(action2, ProcessAction)
        # Note: Actions might be the same by chance, so we don't assert inequality

    def test_select_action_deterministic(self, agent):
        """Test deterministic action selection in evaluation."""
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

        action1 = agent.select_action(state, training=False)
        action2 = agent.select_action(state, training=False)

        # Actions should be deterministic in evaluation mode
        assert action1.temperature_delta == action2.temperature_delta


class TestRLAgentFactory:
    """Test RLAgentFactory class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RLConfig(device="cpu", hidden_sizes=[32, 32], learning_rate=1e-3)

    def test_create_ppo_agent(self, config):
        """Test creating PPO agent."""
        config.algorithm = RLAlgorithm.PPO
        agent = RLAgentFactory.create_agent(
            RLAlgorithm.PPO, config, state_size=19, action_size=8
        )

        assert isinstance(agent, PPOAgent)
        assert agent.state_size == 19
        assert agent.action_size == 8

    def test_create_ddpg_agent(self, config):
        """Test creating DDPG agent."""
        config.algorithm = RLAlgorithm.DDPG
        agent = RLAgentFactory.create_agent(
            RLAlgorithm.DDPG, config, state_size=19, action_size=8
        )

        assert isinstance(agent, DDPGAgent)

    def test_create_sac_agent(self, config):
        """Test creating SAC agent."""
        config.algorithm = RLAlgorithm.SAC
        agent = RLAgentFactory.create_agent(
            RLAlgorithm.SAC, config, state_size=19, action_size=8
        )

        assert isinstance(agent, SACAgent)

    def test_create_unsupported_agent(self, config):
        """Test creating unsupported agent."""
        with pytest.raises(ChemiAccelError):
            RLAgentFactory.create_agent(
                RLAlgorithm.TD3, config, state_size=19, action_size=8
            )


class TestProcessOptimizer:
    """Test ProcessOptimizer class."""

    @pytest.fixture
    def mock_digital_twin(self):
        """Create mock digital twin."""
        mock_twin = Mock()
        mock_twin.run_manufacturing_simulation.return_value = Mock(
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
        }

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RLConfig(
            algorithm=RLAlgorithm.PPO,
            device="cpu",
            hidden_sizes=[32, 32],
            learning_rate=1e-3,
            max_episodes=5,
            max_steps_per_episode=10,
        )

    @pytest.fixture
    def optimizer(self, mock_digital_twin, reward_function, parameter_ranges, config):
        """Create ProcessOptimizer instance."""
        return ProcessOptimizer(
            mock_digital_twin,
            ProcessType.EXTRUSION,
            reward_function,
            parameter_ranges,
            config,
        )

    def test_initialization(self, optimizer, config):
        """Test optimizer initialization."""
        assert optimizer.digital_twin is not None
        assert optimizer.process_type == ProcessType.EXTRUSION
        assert optimizer.reward_function is not None
        assert optimizer.parameter_ranges is not None
        assert optimizer.config == config
        assert optimizer.environment is not None
        assert optimizer.agent is not None

    @patch("src.chemi_accel.digital_twin.rl_agents.RLAgent.train")
    def test_optimize(self, mock_train, optimizer):
        """Test optimization process."""
        # Mock training results
        mock_train.return_value = {
            "episode_rewards": [1.0, 1.5, 2.0, 2.5, 3.0],
            "episode_lengths": [10, 10, 10, 10, 10],
            "average_rewards": [1.0, 1.25, 1.5, 1.75, 2.0],
            "loss_values": [0.1, 0.08, 0.06, 0.04, 0.02],
        }

        results = optimizer.optimize(num_episodes=5)

        assert "algorithm" in results
        assert "total_episodes" in results
        assert "training_time" in results
        assert "best_reward" in results
        assert "final_performance" in results
        assert results["algorithm"] == "ppo"
        assert results["total_episodes"] == 5
        mock_train.assert_called_once_with(optimizer.environment, 5)

    def test_find_convergence_episode(self, optimizer):
        """Test convergence detection."""
        # Test with no convergence
        optimizer.training_history = {
            "episode_rewards": list(range(30))  # Increasing rewards
        }
        convergence_episode = optimizer._find_convergence_episode()
        assert convergence_episode is None

        # Test with convergence
        optimizer.training_history = {
            "episode_rewards": [1.0] * 100  # Constant rewards (converged)
        }
        convergence_episode = optimizer._find_convergence_episode()
        assert convergence_episode is not None
        assert convergence_episode >= 50

    @patch("src.chemi_accel.digital_twin.rl_agents.RLAgent.select_action")
    @patch("src.chemi_accel.digital_twin.rl_agents.ProcessEnvironment.step")
    @patch("src.chemi_accel.digital_twin.rl_agents.ProcessEnvironment.reset")
    def test_get_optimal_conditions(
        self, mock_reset, mock_step, mock_select_action, optimizer
    ):
        """Test getting optimal conditions."""
        # Mock environment responses
        mock_state = Mock(spec=ProcessState)
        mock_reset.return_value = mock_state
        mock_step.return_value = (mock_state, 2.0, True, {})
        mock_select_action.return_value = Mock(spec=ProcessAction)

        best_state, best_reward = optimizer.get_optimal_conditions()

        assert best_state is not None
        assert isinstance(best_reward, float)
        assert best_reward > 0

    def test_save_load_agent(self, optimizer, tmp_path):
        """Test saving and loading agent."""
        filepath = tmp_path / "test_optimizer_agent.pth"

        # Save agent
        optimizer.save_agent(str(filepath))
        assert filepath.exists()

        # Load agent
        optimizer.load_agent(str(filepath))
        # No assertion needed, just check it doesn't crash

    def test_get_optimization_summary_empty(self, optimizer):
        """Test optimization summary with no results."""
        summary = optimizer.get_optimization_summary()

        assert "message" in summary
        assert "No optimization runs completed yet" in summary["message"]

    def test_get_optimization_summary_with_results(self, optimizer):
        """Test optimization summary with results."""
        # Add mock results
        optimizer.optimization_results = [
            {
                "algorithm": "ppo",
                "best_reward": 2.5,
                "final_performance": 2.3,
                "training_time": 100.0,
                "convergence_episode": 50,
            }
        ]

        summary = optimizer.get_optimization_summary()

        assert "total_runs" in summary
        assert "latest_algorithm" in summary
        assert "latest_best_reward" in summary
        assert summary["total_runs"] == 1
        assert summary["latest_algorithm"] == "ppo"
        assert summary["latest_best_reward"] == 2.5


if __name__ == "__main__":
    pytest.main([__file__])
