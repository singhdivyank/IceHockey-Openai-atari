"""DQN agent implementation with experience replay"""

import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from neuralnet import DQN


@dataclass
class AgentConfig:
    """Configuration parameters for DQN Agent"""

    buffer_size : int = 10000
    batch_size : int = 32
    gamma: float = 0.99
    epsilon : float = 1.0
    epsilon_decay : float = 0.995
    epsilon_min : float = 0.1
    learning_rate : float = 0.001
    momentum : float = 0.9
    update_target_every : int = 5000
    grad_clip_value : float = 100.0
    input_shape : Tuple[int] = (1, 84, 84)

@dataclass
class TrainingMetrics:
    """Conatiner for tracking training metrics"""

    reward_history : List[float] = field(default_factory=list)
    policy_loss_history : List[float] = field(default_factory=list)
    target_loss_history : List[float] = field(default_factory=list)
    running_reward : float = 0.0
    best_reward : float = float("-inf")


class DQNAgent:
    """Deep Q-Network Agent with experience replay"""

    def __init__(self, env, config: AgentConfig = None):
        """
        Initialise the DQN agent.

        Args:
            env: Gymnasium environment
            config: Agent configuration parameters
        """
        self.env = env
        self.config = config or AgentConfig
        self.metrics = TrainingMetrics()
        self.buffer = deque(maxlen=self.config.buffer_size)
        self.epsilon = self.config.epsilon
        self.step_counter = 0
        self.current_policy_loss = 0.0
        self.current_target_loss = 0.0
        self._setup_device()
        self._setup_network()
        self._setup_optimizer()
    
    def _setup_device(self) -> None:
        """configure the compute device"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def _setup_newtork(self) -> None:
        """Initialise and configure neural networks"""
        num_actions = self.env.action_space.n
        self.policy_net = DQN(self.config.input_shape, num_actions).float()
        self.target_net = DQN(self.config.input_shape, num_actions).float()
        
        DQN.initialise_weights(self.policy_net)
        
        self.sync_target_net()
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.eval()
    
    def _setup_optimizer(self) -> None:
        """Configure the optimizer"""
        self.optimizer = optim.SGD(
            params = self.policy_net.parameters(),
            lr = self.config.learning_rate,
            momentum = self.config.momentum
        )
    
    def sync_target_net(self) -> None:
        """Copy weights from policy network to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print("Target network updated")

    def preprocess(self, obs: np.ndarray) -> torch.Tensor:
        """
        Preprocess observation for the network.

        Args:
            obs: RGB observation from environment
        
        Returns:
            Preprocessed tensor ready for network input
        """
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        normalised = np.expand_dims(resized, axis=0) / 255.0
        return torch.FloatTensor(normalised).unsqueeze(0).to(self.device)

    def select_action(self, state: torch.Tensor) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state tensor
        
        Returns:
            Selected action index
        """
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def store_experience(
            self, 
            state: torch.Tensor, 
            action: int, 
            reward: float, 
            next_state: torch.Tensor, 
            done: bool
        ) -> None:
        """Store experience in replay buffer"""
        clipped_reward = np.clip(reward, -1.0, 1.0)
        self.buffer.append((state, action, clipped_reward, next_state, done))

    def _sample_batch(self) -> Tuple[torch.Tensor, ...]:
        """Sample a batch from the replay buffer"""
        batch = random.sample(self.buffer, self.config.batch_size)
        states = torch.cat([b[0] for b in batch])
        actions = torch.tensor(
            [b[1] for b in batch], dtype = torch.long, device = self.device
        )
        rewards = torch.tensor(
            [b[2] for b in batch], dtype = torch.float32, device = self.device
        )
        next_states = torch.cat([b[3] for b in batch])
        dones = torch.tensor(
            [b[4] for b in batch], dtype = torch.float32, device = self.device
        )
        return states, actions, rewards, next_states, dones

    def compute_q_values(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            next_states: torch.Tensor,
            dones: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute current and target Q-values"""
        current_val = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_val = self.target_net(next_states).max(1)[0]
            target_val = rewards + (1 - dones) * self.config.gamma * next_val
        
        return current_val, target_val

    def _compute_losses(
            self, 
            current_q: torch.Tensor, 
            target_q: torch.Tensor, 
            states: torch.Tensor, 
            actions: torch.Tensor
        ) -> torch.Tensor:
        """Compute policy and target network losses"""
        policy_loss = nn.SmoothL1Loss()(current_q, target_q)
        with torch.no_grad():
            target_pred = (
                self.target_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            )
            target_loss = nn.SmoothL1Loss()(target_pred, target_q)
        
        self.current_policy_loss = policy_loss.item()
        self.current_target_loss = target_loss.item()
        return policy_loss

    def update_policy_network(self, loss: torch.Tensor) -> None:
        """Perform backpropogation and update policy network."""
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(
            self.policy_net.parameters(), self.config.grad_clip_value
        )
        self.optimizer.step()
    
    def replay(self):
        """sample from experience and update network"""
        if len(self.buffer) < self.config.batch_size:
            return
        
        states, actions, rewards, next_steps, dones = self._sample_batch()
        current_q, target_q = self._compute_q_values(
            states, actions, rewards, next_steps, dones
        )
        policy_loss = self._compute_losses(
            current_q, target_q, states, actions
        )
        self.update_policy_network(policy_loss)

    def decay_epsilon(self) -> None:
        """Apply epsilon decay"""
        if self.epsilon > self.config.epsilon_min:
            self.epsilon *= self.config.epsilon_decay
    
    def update_target(self) -> None:
        """Update target network if step threshold reached"""
        self.step_counter += 1
        if not self.step_counter % self.config.update_target_every:
            self.sync_target_net()


def compute_avg_loss(
        policy_loss: List[float], 
        target_loss: List[float]
    ) -> Tuple[float, float]:
    """
    Calculate avg losses for the episode.

    Args:
        policy_loss: list of policy net losses
        target_loss: list of target net losses
    
    Returns:
        Tuple of average policy and target net losses
    """
    avg_policy_loss = np.mean(policy_loss) if policy_loss else 0
    avg_target_loss = np.mean(target_loss) if target_loss else 0
    return avg_policy_loss, avg_target_loss

def update_running_reward(
        running_reward: float, 
        episode_reward: float, 
        episode: int
    ) -> float:
    """
    Update exponential moving average of rewards.
    
    Args:
        running_reward: current running reward
        episode_reward: reward from latest episode
        episode: current episode number
    
    Returns:
        updated running reward
    """

    if not episode:
        return episode_reward
    
    return 0.05 * episode_reward + 0.95 * running_reward

def save_model(agent: DQNAgent, episode_reward: float) -> None:
    """
    Save model if it achieves a new best reward.
    
    Args:
        agent: DQNAgent instance
        episode_reward: total reward from current episode
    """
    
    if episode_reward > agent.metrics.best_reward:
        agent.best_reward = episode_reward
        torch.save(agent.policy_net.state_dict(), "best_model.pth")
        print(f"New best model saved with reward: {episode_reward}")
