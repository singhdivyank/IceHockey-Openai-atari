import cv2
import numpy as np
import gymnasium as gym
import random
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.out_size = self.get_size(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.out_size, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_actions)
        )
    
    def get_size(self, input_shape):
        with torch.no_grad():
            return self.conv(torch.zeros(1, *input_shape)).view(1, -1).shape[1]

    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))


class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.buffer = deque(maxlen=100)
        self.epsilon = 1.0
        self.gamma = 0.99
        # specify decay from assignment2
        self.decay = 0.999995
        # initialise neural networks
        if self.init_nn():
            self.load_model()
        # after model training
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=0.0001)
    
    def init_nn(self):

        self.num_actions = self.env.action_space.n

        try:
            self.policy_net = DQN(input_shape=(3, 84, 84), num_actions=self.num_actions).float()
            self.target_net = DQN(input_shape=(3, 84, 84), num_actions=self.num_actions).float()
            return True
        except Exception as e:
            print(f"unable to initialise NN ... {str(e)}")
            exit(1)
    
    def load_model(self):
        state_dict = self.policy_net.state_dict()
        self.target_net.load_state_dict(state_dict)

    def preprocess(self, obs):
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        obs = obs.transpose((2, 0, 1))
        # normalise
        return torch.tensor(data=obs, dtype=torch.float32).unsqueeze(0)/255.0
    
    def obtain_action(self, state):
        """
        determine action selection (epsilon-greedy)
        """

        # implemented as per assignment2
        selection_strategy = self.epsilon-np.random.random()
        if selection_strategy>0:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            return self.policy_net(state).argmax().item()
    
    def optimize_model(self, batch_size=32):
        """
        function to obtain state, reward, actions 
        and perform backpropogation 
        """

        if len(self.buffer)<batch_size:
            return 

        # obs, actions, rewards, next_obs, done = zip(*random.sample(population=self.buffer, k=batch_size))

        indices = np.random.choice(a=len(self.buffer), size=batch_size, replace=False)
        samples = np.array(self.buffer, dtype=object)[indices]
        # extract components
        obs, next_obs = [sample[0] for sample in samples], [sample[-2] for sample in samples]
        actions = np.array([sample[1] for sample in samples], dtype=np.int64)
        rewards = np.array([sample[2] for sample in samples], dtype=np.float32)
        done = np.array([sample[-1] for sample in samples], dtype=bool)
        
        # convert to torch tensors
        states, next_states = torch.cat(obs), torch.cat(next_obs)
        actions = torch.tensor(data=actions, dtype=torch.long)
        rewards = torch.tensor(data=rewards, dtype=torch.float32)
        done = torch.tensor(data=done, dtype=torch.bool)

        # updte current, next, and target states
        current = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next = self.target_net(next_states).max(1)[0].detach()
        increment = (1-done.float())*self.gamma*next
        target = rewards + increment

        # compute loss
        loss = nn.MSELoss()(current.squeeze(), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def train(self, episodes):
        """
        function to train agent over number of episodes

        Args:
        episodes (int): number of episodes
        """
        
        for episode in range(episodes):
            # to compute reward for each episode
            episode_reward = 0
            # reset environment
            obs, _ = self.env.reset()
            state = self.preprocess(obs=obs)
            
            while True:
                # determine action
                action = self.obtain_action(state)
                new_obs, reward, done, truncated, _ = self.env.step(action)
                # convert to greyscale and update observation
                new_state = self.preprocess(obs=new_obs)
                self.buffer.append((state, action, reward, new_state, done))
                
                # update state
                state = new_state
                # update reward
                episode_reward += reward

                if len(self.buffer)>=32:
                    self.optimize_model(batch_size=32)

                if done or truncated:
                    break
            
            # epsilon decay --- same logic as assignment2
            self.epsilon = max(0.1, self.epsilon*self.decay)
            # evaluate episode reward
            if not episode%10:
                self.load_model()
                print(f"Episode: {episode}, Reward: {episode_reward}, Epsilon: {self.epsilon}")

if __name__ == '__main__':
    # initialise environment -- https://ale.farama.org/environments/ice_hockey/
    env = gym.make(
        id='ALE/IceHockey-v5', 
        render_mode='human', 
        obs_type='rgb'
    )
    env = gym.wrappers.TimeLimit(env=env, max_episode_steps=3600)
    agent = DQNAgent(env=env)
    agent.train(episodes=100)
