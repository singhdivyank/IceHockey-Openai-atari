import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape[0], 
                out_channels=32, 
                kernel_size=8, 
                stride=4
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, 
                out_channels=64, 
                kernel_size=4, 
                stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, 
                out_channels=64, 
                kernel_size=3, 
                stride=1
            ),
            nn.ReLU()
        )
        self.out_size = self.get_size(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(
                in_features=self.out_size, 
                out_features=512
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=512, 
                out_features=num_actions
            )
        )
    
    def get_size(self, input_shape):
        with torch.no_grad():
            return self.conv(torch.zeros(1, *input_shape)).view(1, -1).shape[1]

    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))


class DQNAgent:
    def __init__(self, env):
        self.env = env
        # For tracking progress
        self.rewards_history = []
        self.policy_loss_history = []
        self.target_loss_history = []
        self.running_reward = 0
        self.best_reward = -float('inf')
        self.update_target_every = 5000
        self.step_counter = 0
        # buffer to save progress
        self.buffer = deque(maxlen=10000)
        # for epsilon decay
        self.epsilon = 1.0
        # Initialize neural networks
        if self.init_nn():
            self.load_model()
        # set device and set ploicy_net, targte_net to use on the device
        self.get_device()
                # optimizer --- earlier Adam was used, changed it later
        self.optimizer = optim.SGD(
            params=self.policy_net.parameters(), 
            lr=0.001,
            momentum=0.9
        )
    
    def get_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)

    def init_nn(self):
        try:
            self.num_actions = self.env.action_space.n
            self.policy_net = DQN(input_shape=(1, 84, 84), num_actions=self.num_actions).float()
            self.target_net = DQN(input_shape=(1, 84, 84), num_actions=self.num_actions).float()
            
            # NOTE: model training was very slow --- below steps suggested by GPT
            for param in self.policy_net.parameters():
                if param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param)
                else:
                    torch.nn.init.constant_(param, 0)
            
            self.target_net.eval()
            return True
        except Exception as e:
            print(f"Unable to initialize NN: {str(e)}")
            exit(1)
    
    def load_model(self):
        state_dict = self.policy_net.state_dict()
        self.target_net.load_state_dict(state_dict=state_dict)
        print("Target network updated")
        
    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        processed = np.expand_dims(obs, axis=0)/ 255.0
        return torch.FloatTensor(processed).unsqueeze(0).to(self.device)
    
    def obtain_action(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        
        # Greedy action
        with torch.no_grad():
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def get_curr_target(self, states, actions, next_states, done, rewards):
        # Current Q values
        current = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # Next Q values
        with torch.no_grad():
            next = self.target_net(next_states).max(1)[0]
        # Target Q values
        increment = (1 - done) * 0.99 * next
        target = rewards + increment
        return current, target
    
    def obtain_loss(self, current, target, states, actions):
        # policy net loss
        policy_loss = nn.SmoothL1Loss()(current, target)
        # target net loss
        with torch.no_grad():
            target_pred = self.target_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            target_loss = nn.SmoothL1Loss()(target_pred, target)
        # update policy and target loss
        self.current_policy_loss, self.current_target_loss = policy_loss.item(), target_loss.item()
        return policy_loss
    
    def perform_backprop(self, policy_loss):
        self.optimizer.zero_grad()
        policy_loss.backward()
        # clip gradients to help with training stability
        torch.nn.utils.clip_grad_value_(parameters=self.policy_net.parameters(), clip_value=100)
        self.optimizer.step()

    def replay(self):
        """sample from experience and update network"""
        # buffer size cannot be less than batch size
        if len(self.buffer) < 32:
            return
        
        # Sample batch
        batches = random.sample(self.buffer, 32)
        # update state, next_sate, action, reward, done
        states, next_states = torch.cat([batch[0] for batch in batches]), torch.cat([batch[-2] for batch in batches])
        actions = torch.tensor([batch[1] for batch in batches], dtype=torch.long).to(self.device)
        rewards = torch.tensor([batch[2] for batch in batches], dtype=torch.float32).to(self.device)
        done = torch.tensor([batch[-1] for batch in batches], dtype=torch.float32).to(self.device)
        # obtain current_q and target_q values  
        current, target = self.get_curr_target(states=states, actions=actions, next_states=next_states, done=done, rewards=rewards)
        # obtain policy_net_loss and compute policy_net_loss and target_net_loss
        policy_loss = self.obtain_loss(current=current, target=target, states=states, actions=actions)
        # backpropagation
        self.perform_backprop(policy_loss=policy_loss)
