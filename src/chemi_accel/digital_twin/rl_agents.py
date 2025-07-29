"""
GPU-Accelerated Reinforcement Learning Agents

Implementation of state-of-the-art RL algorithms optimized for GPU acceleration
and manufacturing process optimization. Includes PPO, DDPG, SAC, and TD3 agents
with custom training loops and process-specific optimizations.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from ..core.exceptions import ChemiAccelError
from .rl_process_optimization import (
    Actor,
    Critic,
    ProcessAction,
    ProcessEnvironment,
    ProcessState,
    ReplayBuffer,
    RewardFunction,
    RLAlgorithm,
    RLConfig,
)

logger = logging.getLogger(__name__)


class RLAgent(ABC):
    """Abstract base class for RL agents."""

    def __init__(self, config: RLConfig, state_size: int, action_size: int):
        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(config.device)

        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        self.evaluation_scores = []

        # Initialize logger
        self.writer = None
        if hasattr(config, "log_dir") and config.log_dir:
            self.writer = SummaryWriter(config.log_dir)

    @abstractmethod
    def select_action(
        self, state: ProcessState, training: bool = True
    ) -> ProcessAction:
        """Select action given current state."""
        pass

    @abstractmethod
    def update(self, batch_data: Any) -> Dict[str, float]:
        """Update agent parameters."""
        pass

    @abstractmethod
    def save(self, filepath: str):
        """Save agent parameters."""
        pass

    @abstractmethod
    def load(self, filepath: str):
        """Load agent parameters."""
        pass

    def train(
        self, environment: ProcessEnvironment, num_episodes: int
    ) -> Dict[str, List[float]]:
        """Train agent in given environment."""
        training_history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "average_rewards": [],
            "loss_values": [],
        }

        for episode in range(num_episodes):
            episode_reward, episode_length, episode_loss = self._train_episode(
                environment
            )

            # Record metrics
            training_history["episode_rewards"].append(episode_reward)
            training_history["episode_lengths"].append(episode_length)
            training_history["loss_values"].append(episode_loss)

            # Calculate moving average
            window_size = min(100, len(training_history["episode_rewards"]))
            avg_reward = np.mean(training_history["episode_rewards"][-window_size:])
            training_history["average_rewards"].append(avg_reward)

            # Log progress
            if episode % 10 == 0:
                logger.info(
                    f"Episode {episode}: Reward={episode_reward:.2f}, "
                    f"Length={episode_length}, Avg={avg_reward:.2f}"
                )

            # Tensorboard logging
            if self.writer:
                self.writer.add_scalar(
                    "Training/EpisodeReward", episode_reward, episode
                )
                self.writer.add_scalar(
                    "Training/EpisodeLength", episode_length, episode
                )
                self.writer.add_scalar("Training/AverageReward", avg_reward, episode)
                if episode_loss is not None:
                    self.writer.add_scalar("Training/Loss", episode_loss, episode)

            # Evaluation
            if episode % self.config.eval_frequency == 0 and episode > 0:
                eval_score = self.evaluate(environment, num_episodes=5)
                training_history.setdefault("evaluation_scores", []).append(eval_score)
                logger.info(f"Evaluation at episode {episode}: Score={eval_score:.2f}")

                if self.writer:
                    self.writer.add_scalar("Evaluation/Score", eval_score, episode)

            # Save checkpoint
            if episode % self.config.save_frequency == 0 and episode > 0:
                self.save(f"checkpoint_episode_{episode}.pth")

        return training_history

    def _train_episode(
        self, environment: ProcessEnvironment
    ) -> Tuple[float, int, Optional[float]]:
        """Train single episode."""
        state = environment.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_loss = None

        while True:
            # Select action
            action = self.select_action(state, training=True)

            # Take environment step
            next_state, reward, done, info = environment.step(action)

            # Store experience (implementation specific)
            self._store_experience(state, action, reward, next_state, done)

            # Update policy (implementation specific)
            loss = self._update_policy()
            if loss is not None:
                episode_loss = loss

            episode_reward += reward
            episode_length += 1
            state = next_state

            if done:
                break

        return episode_reward, episode_length, episode_loss

    def evaluate(
        self, environment: ProcessEnvironment, num_episodes: int = 10
    ) -> float:
        """Evaluate agent performance."""
        total_rewards = []

        for _ in range(num_episodes):
            state = environment.reset()
            episode_reward = 0.0

            while True:
                action = self.select_action(state, training=False)
                next_state, reward, done, _ = environment.step(action)

                episode_reward += reward
                state = next_state

                if done:
                    break

            total_rewards.append(episode_reward)

        return np.mean(total_rewards)

    def _store_experience(
        self,
        state: ProcessState,
        action: ProcessAction,
        reward: float,
        next_state: ProcessState,
        done: bool,
    ):
        """Store experience (to be overridden by specific algorithms)."""
        pass

    def _update_policy(self) -> Optional[float]:
        """Update policy (to be overridden by specific algorithms)."""
        return None

    def __del__(self):
        """Cleanup resources."""
        if self.writer:
            self.writer.close()


class PPOAgent(RLAgent):
    """Proximal Policy Optimization agent."""

    def __init__(self, config: RLConfig, state_size: int, action_size: int):
        super().__init__(config, state_size, action_size)

        # Networks
        self.actor = Actor(
            state_size,
            action_size,
            config.hidden_sizes,
            config.activation,
            config.dropout_rate,
        ).to(self.device)

        self.critic = Critic(
            state_size,
            None,
            config.hidden_sizes,
            config.activation,
            config.dropout_rate,
        ).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=config.learning_rate
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config.learning_rate
        )

        # Experience storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def select_action(
        self, state: ProcessState, training: bool = True
    ) -> ProcessAction:
        """Select action using current policy."""
        state_tensor = state.to_tensor(str(self.device)).unsqueeze(0)

        with torch.no_grad():
            action_tensor, log_prob = self.actor.sample_action(state_tensor)
            value = self.critic(state_tensor)

        if training:
            # Store for training
            self.states.append(state_tensor.squeeze(0))
            self.actions.append(action_tensor.squeeze(0))
            self.log_probs.append(log_prob.squeeze(0))
            self.values.append(value.squeeze(0))

        return ProcessAction.from_tensor(action_tensor.squeeze(0))

    def _store_experience(
        self,
        state: ProcessState,
        action: ProcessAction,
        reward: float,
        next_state: ProcessState,
        done: bool,
    ):
        """Store experience for PPO."""
        self.rewards.append(reward)
        self.dones.append(done)

    def _update_policy(self) -> Optional[float]:
        """Update PPO policy."""
        if len(self.rewards) < self.config.batch_size:
            return None

        # Calculate advantages and returns
        returns, advantages = self._calculate_advantages()

        # Convert to tensors
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        returns = torch.tensor(returns, device=self.device)
        advantages = torch.tensor(advantages, device=self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0

        # PPO update epochs
        for _ in range(self.config.ppo_epochs):
            # Get current policy values
            _, new_log_probs = self.actor.sample_action(states)
            new_values = self.critic(states).squeeze()

            # Calculate ratios
            ratios = torch.exp(new_log_probs.squeeze() - old_log_probs.squeeze())

            # Calculate policy loss
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(
                    ratios, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio
                )
                * advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            # Calculate value loss
            value_loss = F.mse_loss(new_values, returns)

            # Calculate entropy loss
            entropy_loss = -new_log_probs.mean()

            # Total loss
            loss = (
                policy_loss
                + self.config.value_coef * value_loss
                + self.config.entropy_coef * entropy_loss
            )

            # Update networks
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            total_loss += loss.item()

        # Clear experience
        self._clear_experience()

        return total_loss / self.config.ppo_epochs

    def _calculate_advantages(self) -> Tuple[List[float], List[float]]:
        """Calculate advantages using GAE."""
        returns = []
        advantages = []

        # Calculate returns
        discounted_return = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_return = 0
            discounted_return = reward + self.config.gamma * discounted_return
            returns.insert(0, discounted_return)

        # Calculate advantages
        values_list = [v.item() for v in self.values]
        for i in range(len(returns)):
            if i == len(returns) - 1:
                next_value = 0 if self.dones[i] else values_list[i]
            else:
                next_value = values_list[i + 1]

            td_error = self.rewards[i] + self.config.gamma * next_value - values_list[i]
            advantages.append(td_error)

        return returns, advantages

    def _clear_experience(self):
        """Clear stored experience."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def save(self, filepath: str):
        """Save agent parameters."""
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "config": self.config,
            },
            filepath,
        )
        logger.info(f"PPO agent saved to {filepath}")

    def load(self, filepath: str):
        """Load agent parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        logger.info(f"PPO agent loaded from {filepath}")


class DDPGAgent(RLAgent):
    """Deep Deterministic Policy Gradient agent."""

    def __init__(self, config: RLConfig, state_size: int, action_size: int):
        super().__init__(config, state_size, action_size)

        # Networks
        self.actor = Actor(
            state_size,
            action_size,
            config.hidden_sizes,
            config.activation,
            config.dropout_rate,
        ).to(self.device)

        self.critic = Critic(
            state_size,
            action_size,
            config.hidden_sizes,
            config.activation,
            config.dropout_rate,
        ).to(self.device)

        # Target networks
        self.target_actor = Actor(
            state_size,
            action_size,
            config.hidden_sizes,
            config.activation,
            config.dropout_rate,
        ).to(self.device)

        self.target_critic = Critic(
            state_size,
            action_size,
            config.hidden_sizes,
            config.activation,
            config.dropout_rate,
        ).to(self.device)

        # Copy parameters to target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=config.learning_rate
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config.learning_rate
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            config.buffer_size, state_size, action_size, str(self.device)
        )

        # Noise for exploration
        self.noise_std = config.noise_std
        self.exploration_noise = Normal(0, self.noise_std)

    def select_action(
        self, state: ProcessState, training: bool = True
    ) -> ProcessAction:
        """Select action using deterministic policy with exploration noise."""
        state_tensor = state.to_tensor(str(self.device)).unsqueeze(0)

        with torch.no_grad():
            action_mean, _ = self.actor(state_tensor)

            if training:
                # Add exploration noise
                noise = self.exploration_noise.sample(action_mean.shape).to(self.device)
                action_tensor = torch.clamp(action_mean + noise, -1, 1)
            else:
                action_tensor = action_mean

        return ProcessAction.from_tensor(action_tensor.squeeze(0))

    def _store_experience(
        self,
        state: ProcessState,
        action: ProcessAction,
        reward: float,
        next_state: ProcessState,
        done: bool,
    ):
        """Store experience in replay buffer."""
        state_tensor = state.to_tensor(str(self.device))
        action_tensor = action.to_tensor(str(self.device))
        next_state_tensor = next_state.to_tensor(str(self.device))

        self.replay_buffer.add(
            state_tensor, action_tensor, reward, next_state_tensor, done
        )

    def _update_policy(self) -> Optional[float]:
        """Update DDPG policy."""
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )

        # Update critic
        with torch.no_grad():
            next_actions, _ = self.target_actor(next_states)
            next_q_values = self.target_critic(next_states, next_actions)
            target_q_values = (
                rewards + self.config.gamma * next_q_values * (~dones).float()
            )

        current_q_values = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q_values, target_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Update actor
        predicted_actions, _ = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Update target networks
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)

        return critic_loss.item() + actor_loss.item()

    def _soft_update(self, target_net: nn.Module, source_net: nn.Module):
        """Soft update target network."""
        for target_param, source_param in zip(
            target_net.parameters(), source_net.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * source_param.data
                + (1 - self.config.tau) * target_param.data
            )

    def save(self, filepath: str):
        """Save agent parameters."""
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "target_actor_state_dict": self.target_actor.state_dict(),
                "target_critic_state_dict": self.target_critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "config": self.config,
            },
            filepath,
        )
        logger.info(f"DDPG agent saved to {filepath}")

    def load(self, filepath: str):
        """Load agent parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.target_actor.load_state_dict(checkpoint["target_actor_state_dict"])
        self.target_critic.load_state_dict(checkpoint["target_critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        logger.info(f"DDPG agent loaded from {filepath}")


class SACAgent(RLAgent):
    """Soft Actor-Critic agent."""

    def __init__(self, config: RLConfig, state_size: int, action_size: int):
        super().__init__(config, state_size, action_size)

        # Networks
        self.actor = Actor(
            state_size,
            action_size,
            config.hidden_sizes,
            config.activation,
            config.dropout_rate,
        ).to(self.device)

        self.critic1 = Critic(
            state_size,
            action_size,
            config.hidden_sizes,
            config.activation,
            config.dropout_rate,
        ).to(self.device)

        self.critic2 = Critic(
            state_size,
            action_size,
            config.hidden_sizes,
            config.activation,
            config.dropout_rate,
        ).to(self.device)

        # Target critics
        self.target_critic1 = Critic(
            state_size,
            action_size,
            config.hidden_sizes,
            config.activation,
            config.dropout_rate,
        ).to(self.device)

        self.target_critic2 = Critic(
            state_size,
            action_size,
            config.hidden_sizes,
            config.activation,
            config.dropout_rate,
        ).to(self.device)

        # Copy parameters
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=config.learning_rate
        )
        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(), lr=config.learning_rate
        )
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(), lr=config.learning_rate
        )

        # Automatic entropy tuning
        self.target_entropy = -action_size
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            config.buffer_size, state_size, action_size, str(self.device)
        )

    @property
    def alpha(self) -> torch.Tensor:
        """Get current alpha value."""
        return self.log_alpha.exp()

    def select_action(
        self, state: ProcessState, training: bool = True
    ) -> ProcessAction:
        """Select action using stochastic policy."""
        state_tensor = state.to_tensor(str(self.device)).unsqueeze(0)

        with torch.no_grad():
            if training:
                action_tensor, _ = self.actor.sample_action(state_tensor)
            else:
                # Use mean for evaluation
                action_mean, _ = self.actor(state_tensor)
                action_tensor = torch.tanh(action_mean)

        return ProcessAction.from_tensor(action_tensor.squeeze(0))

    def _store_experience(
        self,
        state: ProcessState,
        action: ProcessAction,
        reward: float,
        next_state: ProcessState,
        done: bool,
    ):
        """Store experience in replay buffer."""
        state_tensor = state.to_tensor(str(self.device))
        action_tensor = action.to_tensor(str(self.device))
        next_state_tensor = next_state.to_tensor(str(self.device))

        self.replay_buffer.add(
            state_tensor, action_tensor, reward, next_state_tensor, done
        )

    def _update_policy(self) -> Optional[float]:
        """Update SAC policy."""
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )

        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample_action(next_states)
            next_q1 = self.target_critic1(next_states, next_actions)
            next_q2 = self.target_critic2(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + self.config.gamma * next_q * (~dones).float()

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update actor
        new_actions, log_probs = self.actor.sample_action(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha
        alpha_loss = -(
            self.log_alpha * (log_probs + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Update target networks
        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)

        return critic1_loss.item() + critic2_loss.item() + actor_loss.item()

    def _soft_update(self, target_net: nn.Module, source_net: nn.Module):
        """Soft update target network."""
        for target_param, source_param in zip(
            target_net.parameters(), source_net.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * source_param.data
                + (1 - self.config.tau) * target_param.data
            )

    def save(self, filepath: str):
        """Save agent parameters."""
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic1_state_dict": self.critic1.state_dict(),
                "critic2_state_dict": self.critic2.state_dict(),
                "target_critic1_state_dict": self.target_critic1.state_dict(),
                "target_critic2_state_dict": self.target_critic2.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic1_optimizer_state_dict": self.critic1_optimizer.state_dict(),
                "critic2_optimizer_state_dict": self.critic2_optimizer.state_dict(),
                "log_alpha": self.log_alpha,
                "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
                "config": self.config,
            },
            filepath,
        )
        logger.info(f"SAC agent saved to {filepath}")

    def load(self, filepath: str):
        """Load agent parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic1.load_state_dict(checkpoint["critic1_state_dict"])
        self.critic2.load_state_dict(checkpoint["critic2_state_dict"])
        self.target_critic1.load_state_dict(checkpoint["target_critic1_state_dict"])
        self.target_critic2.load_state_dict(checkpoint["target_critic2_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic1_optimizer.load_state_dict(
            checkpoint["critic1_optimizer_state_dict"]
        )
        self.critic2_optimizer.load_state_dict(
            checkpoint["critic2_optimizer_state_dict"]
        )
        self.log_alpha = checkpoint["log_alpha"]
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
        logger.info(f"SAC agent loaded from {filepath}")


class RLAgentFactory:
    """Factory for creating RL agents."""

    @staticmethod
    def create_agent(
        algorithm: RLAlgorithm, config: RLConfig, state_size: int, action_size: int
    ) -> RLAgent:
        """Create RL agent based on algorithm type."""
        if algorithm == RLAlgorithm.PPO:
            return PPOAgent(config, state_size, action_size)
        elif algorithm == RLAlgorithm.DDPG:
            return DDPGAgent(config, state_size, action_size)
        elif algorithm == RLAlgorithm.SAC:
            return SACAgent(config, state_size, action_size)
        elif algorithm == RLAlgorithm.TD3:
            # TD3 would be similar to DDPG with additional features
            raise NotImplementedError("TD3 algorithm not yet implemented")
        elif algorithm == RLAlgorithm.A3C:
            # A3C would require multi-process implementation
            raise NotImplementedError("A3C algorithm not yet implemented")
        else:
            raise ChemiAccelError(f"Unsupported RL algorithm: {algorithm}")


class ProcessOptimizer:
    """High-level interface for process optimization using RL."""

    def __init__(
        self,
        digital_twin,
        process_type,
        reward_function: RewardFunction,
        parameter_ranges: Dict[str, Tuple[float, float]],
        config: RLConfig = None,
    ):
        self.digital_twin = digital_twin
        self.process_type = process_type
        self.reward_function = reward_function
        self.parameter_ranges = parameter_ranges
        self.config = config or RLConfig()

        # Create environment
        self.environment = ProcessEnvironment(
            digital_twin,
            process_type,
            reward_function,
            parameter_ranges,
            self.config.max_steps_per_episode,
        )

        # Initialize agent
        state_size = ProcessState(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).state_size
        action_size = ProcessAction(0, 0, 0, 0, 0).action_size

        self.agent = RLAgentFactory.create_agent(
            self.config.algorithm, self.config, state_size, action_size
        )

        # Training history
        self.training_history = {}
        self.optimization_results = []

    def optimize(self, num_episodes: int = None) -> Dict[str, Any]:
        """Run optimization using RL."""
        if num_episodes is None:
            num_episodes = self.config.max_episodes

        logger.info(
            f"Starting RL optimization with {self.config.algorithm.value.upper()} "
            f"for {num_episodes} episodes"
        )

        start_time = time.time()

        # Train agent
        self.training_history = self.agent.train(self.environment, num_episodes)

        training_time = time.time() - start_time

        # Evaluate final performance
        final_performance = self.agent.evaluate(self.environment, num_episodes=10)

        # Get best episode
        best_episode_idx = np.argmax(self.training_history["episode_rewards"])
        best_reward = self.training_history["episode_rewards"][best_episode_idx]

        results = {
            "algorithm": self.config.algorithm.value,
            "total_episodes": num_episodes,
            "training_time": training_time,
            "best_episode": best_episode_idx,
            "best_reward": best_reward,
            "final_performance": final_performance,
            "average_final_reward": np.mean(
                self.training_history["episode_rewards"][-10:]
            ),
            "convergence_episode": self._find_convergence_episode(),
            "training_history": self.training_history,
        }

        self.optimization_results.append(results)

        logger.info(
            f"Optimization completed. Best reward: {best_reward:.2f}, "
            f"Final performance: {final_performance:.2f}"
        )

        return results

    def _find_convergence_episode(self) -> Optional[int]:
        """Find episode where agent converged."""
        rewards = self.training_history["episode_rewards"]
        if len(rewards) < 100:
            return None

        # Look for convergence (stable performance for last 50 episodes)
        for i in range(50, len(rewards)):
            recent_std = np.std(rewards[i - 50 : i])
            if recent_std < 0.1:  # Converged if std < 0.1
                return i

        return None

    def get_optimal_conditions(self) -> Tuple[ProcessState, float]:
        """Get optimal process conditions from trained agent."""
        # Run evaluation to find best conditions
        best_state = None
        best_reward = -float("inf")

        for _ in range(10):
            state = self.environment.reset()
            episode_reward = 0.0
            episode_states = [state]

            while True:
                action = self.agent.select_action(state, training=False)
                next_state, reward, done, _ = self.environment.step(action)

                episode_reward += reward
                episode_states.append(next_state)
                state = next_state

                if done:
                    break

            if episode_reward > best_reward:
                best_reward = episode_reward
                best_state = episode_states[-1]  # Final state

        return best_state, best_reward

    def save_agent(self, filepath: str):
        """Save trained agent."""
        self.agent.save(filepath)

    def load_agent(self, filepath: str):
        """Load trained agent."""
        self.agent.load(filepath)

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        if not self.optimization_results:
            return {"message": "No optimization runs completed yet"}

        latest_result = self.optimization_results[-1]

        return {
            "total_runs": len(self.optimization_results),
            "latest_algorithm": latest_result["algorithm"],
            "latest_best_reward": latest_result["best_reward"],
            "latest_final_performance": latest_result["final_performance"],
            "latest_training_time": latest_result["training_time"],
            "average_best_reward": np.mean(
                [r["best_reward"] for r in self.optimization_results]
            ),
            "best_overall_reward": max(
                [r["best_reward"] for r in self.optimization_results]
            ),
            "convergence_rates": [
                r.get("convergence_episode") for r in self.optimization_results
            ],
        }
