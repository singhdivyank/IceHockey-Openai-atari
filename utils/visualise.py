"""
Module for visualizing training rewards and losses using matplotlib
"""

import matplotlib.pyplot as plt

def plot_fig(fig_name: str, rewards: list) -> None:
    """
    Plot the training rewards
    
    Args:
        fig_name (str): name of the figure file to save
        rewards (list): list of rewards per episode
    """

    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig(fig_name)
    plt.close()

def plot_loss(fig_name: str, policy_loss: list, target_loss: list) -> None:
    """
    Plot the loss for each epoch of policy_net and target_net
    
    Args:
        fig_name (str): name of the figure file to save
        policy_loss (list): list of policy net losses per episode
        target_loss (list): list of target net losses per episode
    """
    
    plt.figure(figsize=(10, 5))
    plt.plot(policy_loss, label='Policy Net Loss')
    plt.plot(target_loss, label='Target Net Loss')
    plt.title('Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_name)
    plt.close()
