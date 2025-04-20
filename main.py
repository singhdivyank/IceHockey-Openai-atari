import numpy as np
import gymnasium as gym

from neuralnet import *
from utils import *

def train(episodes=500, max_steps=1000):
    print(f"training {episodes} episodes")
    agent = DQNAgent(env=env)

    for episode in range(episodes):
        total_reward, policy_loss, target_loss = 0, [], []
        # reset environment
        state, _ = agent.env.reset()
        state = agent.preprocess(state)
        
        for _ in range(max_steps):
            # obtain action
            action = agent.obtain_action(state)
            # perform action
            next_state, reward, done, truncated, _ = agent.env.step(action)
            
            # preprocess observation
            next_state = agent.preprocess(next_state)
            # clip rewards between -1 and 1 for stability
            clipped_reward = np.clip(reward, -1.0, 1.0)
            # store experience
            agent.buffer.append((state, action, clipped_reward, next_state, done or truncated))
            # update state
            state = next_state
            # accumulate reward
            total_reward += reward
            # experience replay
            agent.replay()
            # store calculated policy_net loss
            if hasattr(agent, 'current_policy_loss'):
                policy_loss.append(agent.current_policy_loss)
            # store calculated target_net loss
            if hasattr(agent, 'current_target_loss'):
                target_loss.append(agent.current_target_loss)
            # update target network
            agent.step_counter += 1

            if not agent.step_counter % agent.update_target_every:
                agent.load_model()
            
            if done or truncated:
                break
        
        # epsilon decay
        if agent.epsilon-0.1>0:
            agent.epsilon *= 0.995
        # track rewards
        agent.rewards_history.append(total_reward)
        # update running reward
        agent.running_reward = update_running_reward(running_reward=agent.running_reward, total_reward=total_reward, episode=episode)
        
        # save model
        save_model(agent=agent, total_reward=total_reward)
        # compute average policy_loss and targte_loss
        avg_policy_loss, avg_target_loss = compute_avg_loss()
        # store average loss
        agent.policy_loss_history.append(avg_policy_loss)
        agent.target_loss_history.append(avg_target_loss)

        # print progress
        if not episode%10:
            print(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")
            # plot progress every 100 episodes
            if episode > 0 and episode % 100 == 0:
                plot_fig(rewards=agent.rewards_history, fig_name = f"rewards_plot_episode_{episode}.png")
    
    print("Training finished!")
    # Plot final results
    plot_fig(rewards=agent.rewards_history, fig_name = 'rewards_plot.png')
    # plot loss
    plot_loss(fig_name='loss_plot.png', policy_loss=agent.policy_loss_history, target_loss=agent.target_loss_history)


if __name__ == '__main__':
    # initialise environment -- https://ale.farama.org/environments/ice_hockey/
    env = gym.make(
        id='ALE/IceHockey-v5', 
        obs_type='rgb',
        # Uncomment to visualize
        render_mode='human'
    )
    env = gym.wrappers.TimeLimit(env=env, max_episode_steps=3600)
    # call train method
    train(episodes=500, max_steps=1000)
