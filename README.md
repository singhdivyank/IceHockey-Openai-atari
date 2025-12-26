# Deep Q-Network for Atari Ice Hockey

![](utils/imgs/Python-logo.png)
![](utils/imgs/pytorch_logo.png)

A Deep Q-Network (DQN) implementation for playing the Atari Ice Hockey game using reinforcement learning

## Overview

This project implements a DQN agent that learns to play Ice Hockey directly from raw pixel observations. The agent uses experience replay and a target network to stabilize training, following the architecture introduced by Mnih et al. (2015).

**Key Features**:
- **Deep Q-Network Architecture**: Three convolutional layers followed by fully connected layers
- **Experience Replay**: Buffer of 10,000 transitions to break temporal correlations
- **Target Network**: Synchronized every 5,000 steps to stabilize learning
- **Epsilon-Greedy Exploration**: Annealed from 1.0 to 0.1 during training
- **Configurable Hyperparameters**: Easy-to-modify training parameters via dataclass

## Project Structure

```
dqn-ice-hockey/
├── utils/
│   └── visualise.py      # Plotting utilities
|   ├── agent.py          # DQN agent and utilities
├── requirements.txt
├── main.py
├── neuralnet.py
└── README.md
```

## Installation

**Prerequisites**:
- Python 3.8 or higher
- CUDA-compatible GPU

## Setup

1. Clone the repository

```
git clone https://github.com/yourusername/dqn-ice-hockey.git
cd dqn-ice-hockey
```

2. Create a virtual environment

```
python -m venv venv
source venv/bin/activate
```

3. Install dependencies

```
pip install -r requirements.txt
```

## Usage

**Training**

Run the training script with default parameters- `python3 main.py`

**Configuration Options**

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `buffer_size` | 10,000 | Experience replay buffer capacity |
| `batch_size` | 32 | Mini-batch size for training |
| `gamma` | 0.99 | Discount factor |
| `epsilon` | 1.0 | Initial exploration rate |
| `epsilon_decay` | 0.995 | Exploration decay rate |
| `epsilon_min` | 0.1 | Minimum exploration rate |
| `learning_rate` | 0.001 | SGD learning rate |
| `update_target_every` | 5,000 | Target network sync frequency |

## Results

Training over 500 episodes yielded the following results:

| Metric | Value |
| ------ | ----- |
| Best Reward | +3.0 |
| Initial Reward | -4.0 |
| First Positive Reward | Episode 157 |
| Average Reward | -3.2 |

The agent demonstrated learning capability in this challenging sparse-reward environment, occasionally achieving positive rewards (scoring more goals than the opponent)

## Architecture

The DQN follows the canonical Atari architecture:

```
Input (84x84 grayscale) 
    → Conv2D(32, 8x8, stride=4) → ReLU
    → Conv2D(64, 4x4, stride=2) → ReLU  
    → Conv2D(64, 3x3, stride=1) → ReLU
    → Flatten
    → Linear(512) → ReLU
    → Linear(18)
```

## Challenges

Ice Hockey presents several RL challenges:

1. **Sparse Rewards**: Goals are rare, making credit assignment difficult
2. **Large Action Space**: 18 possible actions (joystick + fire combinations)
3. **Adversarial Dynamics**: Opponent behavior introduces variance
4. **High-Dimensional Input**: Raw pixel observations require efficient feature extraction
