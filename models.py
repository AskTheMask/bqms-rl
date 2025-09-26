import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

class VariationalBayesianLinear(nn.Module):

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
        init.kaiming_uniform_(self.weight_mu, a=math.sqrt(self.weight_mu.shape[1]))
        init.constant_(self.weight_log_sig2, log_sig2_init)
        if self.has_bias:
            init.zeros_(self.bias_mu)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output_mu = F.linear(input, self.weight_mu, self.bias_mu)
        output_sig2 = F.linear(
            input.pow(2), self.weight_log_sig2.exp(), bias=None
        ) 

        return output_mu + output_sig2.sqrt() * torch.randn_like(output_sig2)
    
    def name():
        return "variational_bayesian_linear"

    
class BQMS(nn.Module):
    def __init__(self, in_features, hidden_layer_size, out_features, bias=True, log_sig2_init=-4.6):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_layer_size)   # Fully connected layer 1
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)    # Fully connected layer 2
        self.out = VariationalBayesianLinear(hidden_layer_size, out_features, bias=bias, log_sig2_init=log_sig2_init)  # Output layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass by sampling weights from the learned posterior
        and applying them to the input.
        """
        x = F.relu(self.fc1(x)) # Apply ReLU activation function
        x = F.relu(self.fc2(x))
        x = self.out(x)         # Calculate output 
        return x
    
    def name() -> str:
        return "BQMS"


class DQN(nn.Module):
    def __init__(self, in_features, hidden_layer_size, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_layer_size)   # Fully connected layer 1
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)    # Fully connected layer 2
        self.out = nn.Linear(hidden_layer_size, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass by sampling weights from the learned posterior
        and applying them to the input.
        """
        x = F.relu(self.fc1(x)) # Apply ReLU activation function
        x = F.relu(self.fc2(x))
        x = self.out(x)         # Calculate output 
        return x
    
    def name() -> str:
        return "DQN"
    

class BootstrapDQN(nn.Module):
    def __init__(self, in_features, hidden_layer_size, out_features, bootstrap_heads: int = 10):
        super().__init__()
        self.bootstrap_heads = bootstrap_heads

        self.shared_fc1 = nn.Linear(in_features, hidden_layer_size)
        self.shared_fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_layer_size, out_features) for _ in range(bootstrap_heads)
        ])

    def forward(self, x: torch.Tensor, head_idx: int) -> torch.Tensor:
        """
        Forward pass through the shared layers and selected head.

        Args:
            x (torch.Tensor): Input tensor.
            head_idx (int): Index of the head to use.

        Returns:
            torch.Tensor: Q-values predicted by the selected head.
        """
        if head_idx is None:
            raise ValueError("Bootstrapped DQN requires a specific head index for forward pass.")

        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))
        return self.heads[head_idx](x)

    def name() -> str:
        return "BootstrapDQN"
