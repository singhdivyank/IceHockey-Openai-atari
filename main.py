"""Main training loop for DQN agent on IceHockey-v5 environment"""

import gymnasium as gym

from utils.agent import (
    AgentConfig,
    DQNAgent,
    compute_avg_loss,
    save_model,
    update_running_reward
)

from utils.visualise import plot_fig, plot_loss

def create_environment(render: bool = True) -> gym.Env:
    """
    Create and configure the IceHockey-v5 environment.

    Args:
        render (bool): Whether to enable rendering.
    
    Returns:
        gym.Env: Configured IceHockey-v5 environment.
    """

    render_mode = 'human' if render else None
    env = gym.make(
        id = 'ALE/IceHockey-v5', 
        obs_type = 'rgb',
        render_mode = render_mode
    )
    return gym.wrappers.TimeLimit(env=env, max_episode_steps=3600)

def run_episode(agent: DQNAgent, max_steps: int) -> tuple:
    """
    Execute a single training episode.

    Args:
        agent (DQNAgent): The DQN agent.
        max_steps (int): Maximum steps for the episode.

    Returns:
        tuple: Total reward, policy loss list, target loss list.
    """

    state, _ = agent.env.reset()
    state = agent.preprocess(state)
    total_reward = 0.0
    policy_loss = []
    target_loss = []

    for _ in range(max_steps):
        action = agent.select_action_action(state)
        next_obs, reward, done, truncated, _ = agent.env.step(action)
        
        next_state = agent.preprocess(next_obs)
        agent.store_experience(state, action, reward, next_state, done or truncated)
        
        state = next_state
        total_reward += reward

        agent.replay()
        collect_losses(agent, policy_loss, target_loss)
        agent.update_target()
        if done or truncated:
            break
        
    return total_reward, policy_loss, target_loss

def collect_losses(
    agent: DQNAgent, policy_loss: list, target_loss: list
) -> None:
    """
    Collect loss values from the agent if available.

    Args:
        agent (DQNAgent): The DQN agent.
        policy_loss (list): List to store policy losses.
        target_loss (list): List to store target losses.
    """

    if agent.current_policy_loss:
        policy_loss.append(agent.current_policy_loss)
    if agent.current_target_loss:
        target_loss.append(agent.current_target_loss)

def log_progress(episode: int, reward: float, epsilon: float) -> None:
    """Print training progress."""
    print(f"Episode: {episode}, Reward: {reward}, Epsilon: {epsilon:.4f}")

def save_plots(agent: DQNAgent, episode: int, final: bool = False) -> None:
    """Save reward and loss plots."""
    if final:
        plot_fig(rewards=agent.metrics.reward_history, fig_name="rewards_plot.png")
        plot_loss(
            fig_name = "loss_plot.png",
            policy_loss = agent.metrics.policy_loss_history,
            target_loss = agent.metrics.target_loss_history
        )
    elif episode > 0 and episode % 100 == 0:
        plot_fig(
            rewards=agent.metrics.reward_history,
            fig_name=f"rewards_plot_episode_{episode}.png"
        )

def train(episodes: int = 500, max_steps: int = 1000):
    """
    Main training loop for DQN agent on IceHockey-v5 environment.
    
    Args:
        episodes (int): Number of training episodes
        max_steps (int): Maximum steps per episode
    """
    print(f"training {episodes} episodes")
    
    env = create_environment(render=False)
    agent = DQNAgent(env=env, config=AgentConfig())

    for episode in range(episodes):
        total_reward, policy_loss, target_loss = run_episode(agent, max_steps)
        agent.decay_epsilon()
        agent.metrics.reward_history.append(total_reward)
        agent.metrics.running_reward = update_running_reward(
            agent.metrics.running_reward, total_reward, episode
        )
        save_model(agent, total_reward)
        avg_policy, avg_target = compute_avg_loss(policy_loss, target_loss)
        agent.metrics.policy_loss_history.append(avg_policy)
        agent.metrics.target_loss_history.append(avg_target)

        if not episode % 10:
            log_progress(episode, total_reward, agent.epsilon)
            save_plots(agent, episode)
    
    print("Training finished!")
    save_plots(agent, episodes, final=True)


if __name__ == '__main__':
    train(episodes=500, max_steps=1000)
