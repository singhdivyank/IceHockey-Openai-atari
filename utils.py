import matplotlib.pyplot as plt
import numpy as np
import torch

def compute_avg_loss(policy_loss, target_loss):
    """calculate avg loss for policy_net and target_net"""
    avg_policy_loss = np.mean(policy_loss) if policy_loss else 0
    avg_target_loss = np.mean(target_loss) if target_loss else 0
    return avg_policy_loss, avg_target_loss

# SUGGESTED BY GPT
def update_running_reward(running_reward, total_reward, episode):
    """update agent's running reward for each episode"""
    if not episode>0:
        running_reward = total_reward
    else:
        running_reward = 0.05 * total_reward + 0.95 * running_reward
    return running_reward

def save_model(agent, total_reward):
    """save the best PyTorch model"""
    if total_reward-agent.best_reward>0:
        agent.best_reward = total_reward
        torch.save(agent.policy_net.state_dict(), "best_model.pth")
        print(f"New best model saved with reward: {total_reward}")

def plot_fig(fig_name, rewards):
    """Plot the training rewards"""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig(fig_name)
    plt.close()

def plot_loss(fig_name, policy_loss, target_loss):
    """Plot the loss for each epoch of policy_net and target_net"""
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
