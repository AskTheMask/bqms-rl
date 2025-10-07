"""
models.py

Neural network architectures for DQN variants.

This module defines the following models:
- `VariationalBayesianLinear`: Linear layer with variational Bayesian weights.
- `OPS_VBQN`: Two-hidden-layer network with a variational Bayesian output layer.
- `DQN`: Standard fully connected Deep Q-Network.
- `BootstrapDQN`: DQN with multiple heads for bootstrapped Q-learning.

Each model is implemented as a PyTorch `nn.Module`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

class VariationalBayesianLinear(nn.Module):
    """Linear layer with variational Bayesian weight sampling.

    The layer samples its weights from a learned Gaussian posterior during
    each forward pass, allowing Bayesian uncertainty estimation in outputs.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): Whether to include a bias term. Defaults to True.
        log_sig2_init (float, optional): Initial log variance for the weights. Defaults to -4.6.

    Attributes:
        weight_mu (nn.Parameter): Mean of the weight posterior.
        weight_log_sig2 (nn.Parameter): Log variance of the weight posterior.
        bias_mu (nn.Parameter or None): Mean of the bias posterior.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        log_sig2_init=-4.6,   
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.weight_mu = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_log_sig2 = nn.Parameter(torch.empty((out_features, in_features)))
        if self.has_bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias_mu", None)
        self.reset_parameters(log_sig2_init)


    def reset_parameters(self, log_sig2_init) -> None:
        """Initializes parameters using Kaiming uniform and constant log variance."""
        init.kaiming_uniform_(self.weight_mu, a=math.sqrt(self.weight_mu.shape[1]))
        init.constant_(self.weight_log_sig2, log_sig2_init)
        if self.has_bias:
            init.zeros_(self.bias_mu)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass by sampling weights from the posterior.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        output_mu = F.linear(input, self.weight_mu, self.bias_mu)
        output_sig2 = F.linear(
            input.pow(2), self.weight_log_sig2.exp(), bias=None
        ) 

        return output_mu + output_sig2.sqrt() * torch.randn_like(output_sig2)
    
    def name() -> str:
        """Returns the model name."""
        return "variational_bayesian_linear"

    
class OPS_VBQN(nn.Module):
    """Two-hidden-layer network with a Variational Bayesian output layer.

    Args:
        in_features (int): Number of input features.
        hidden_layer_size (int): Number of units in the hidden layers.
        out_features (int): Number of output features.
        bias (bool, optional): Whether the output layer has a bias. Defaults to True.
        log_sig2_init (float, optional): Initial log variance for the output layer. Defaults to -4.6.
    """
    def __init__(self, in_features, hidden_layer_size, out_features, bias=True, log_sig2_init=-4.6):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_layer_size)   # Fully connected layer 1
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)    # Fully connected layer 2
        self.out = VariationalBayesianLinear(hidden_layer_size, out_features, bias=bias, log_sig2_init=log_sig2_init)  # Output layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = F.relu(self.fc1(x)) # Apply ReLU activation function
        x = F.relu(self.fc2(x))
        x = self.out(x)         # Calculate output 
        return x
    
    def name() -> str:
        """Returns the model name."""
        return "OPS-VBQN"


class DQN(nn.Module):
    """Standard fully connected Deep Q-Network.

    Args:
        in_features (int): Number of input features.
        hidden_layer_size (int): Number of units in the hidden layers.
        out_features (int): Number of output features.
    """
    def __init__(self, in_features, hidden_layer_size, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_layer_size)   # Fully connected layer 1
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)    # Fully connected layer 2
        self.out = nn.Linear(hidden_layer_size, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = F.relu(self.fc1(x)) # Apply ReLU activation function
        x = F.relu(self.fc2(x))
        x = self.out(x)         # Calculate output 
        return x
    
    def name() -> str:
        """Returns the model name."""
        return "DQN"
    

class BootstrapDQN(nn.Module):
    """Deep Q-Network with multiple heads for bootstrapped Q-learning.

    Args:
        in_features (int): Number of input features.
        hidden_layer_size (int): Number of units in the shared hidden layers.
        out_features (int): Number of output features per head.
        bootstrap_heads (int, optional): Number of heads. Defaults to 10.

    Attributes:
        shared_fc1 (nn.Linear): Shared first hidden layer.
        shared_fc2 (nn.Linear): Shared second hidden layer.
        heads (nn.ModuleList): List of output layers, one per head.
    """
    def __init__(self, in_features, hidden_layer_size, out_features, bootstrap_heads: int = 10):
        super().__init__()
        self.bootstrap_heads = bootstrap_heads

        self.shared_fc1 = nn.Linear(in_features, hidden_layer_size)
        self.shared_fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_layer_size, out_features) for _ in range(bootstrap_heads)
        ])

    def forward(self, x: torch.Tensor, head_idx: int) -> torch.Tensor:
        """Forward pass through shared layers and selected head.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            head_idx (int): Index of the head to use.

        Returns:
            torch.Tensor: Q-values predicted by the selected head.

        Raises:
            ValueError: If head_idx is None.
        """
        if head_idx is None:
            raise ValueError("Bootstrapped DQN requires a specific head index for forward pass.")

        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))
        return self.heads[head_idx](x)

    def name() -> str:
        """Returns the model name."""
        return "BootstrapDQN"
