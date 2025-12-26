"""Deep Q-Network Agent implementation using PyTorch"""

import torch
import torch.nn as nn


class DQN(nn.Module):
    """Deep Q-Network with convolution layers for image procesing"""

    def __init__(self, input_shape: tuple, num_actions: int):
        """
        Initialise the DQN architecture.

        Args:
            input_shape (tuple): shape of the input
            num_actions (int): number of possible actions
        """
        super(DQN, self).__init__()
        self.conv = self.build_conv_layer(input_shape[0])
        self.out_size = self.compute_output_size(input_shape)
        self.fc = self.build_fc_layer(num_actions)
    
    def build_conv_layer(self, input_channels: int) -> nn.Module:
        """
        Build convolutional layers.
        
        Args:
            input_channels (int): number of input channels
        
        Returns:
            nn.Module: convolutional layers
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )
    
    def get_size(self, input_shape: tuple) -> int:
        """
        Compute size of conv output.
        
        Args:
            input_shape (tuple): shape of the input

        Returns:
            int: size of conv output
        """
        with torch.no_grad():
            return self.conv(torch.zeros(1, *input_shape)).view(1, -1).shape[1]
    
    def compute_output_size(self, input_shape: tuple) -> int:
        """
        Compute output size of conv layers.
        
        Args:
            input_shape (tuple): shape of the input

        Returns:
            int: output size
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            return self.conv(dummy_input).view(1, -1).shape[1]
    
    def build_fc_layer(self, num_actions: int) -> nn.Module:
        """
        Build fully connected layers.
        
        Args:
            num_actions (int): number of actions

        Returns:
            nn.Module: fully connected layers
        """
        return nn.Sequential(
            nn.Linear(in_features=self.out_size, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        conv_out = self.conv(x).view(x.size(0), -1)
        return self.fc(conv_out)
    
    @staticmethod
    def initialise_weights(model: nn.Module) -> None:
        """
        Apply Xavier initialization to model weights
        
        Args:
            model (nn.Module): neural network model
        """

        for param in model.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.constant_(param, 0)
