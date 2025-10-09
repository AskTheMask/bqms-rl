"""
agent.py

Defines the Agent class for training and evaluating reinforcement learning agents
across different algorithms such as DQN, OPS-VBQN, and BootstrapDQN.
"""

from memory import ReplayMemory, Transition
from utils import encode_state
from data_handler import save_seed_data, save_model, load_model
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
from typing import List, Tuple, Optional, Any

class Agent:

    def __init__(self, 
                 alpha: float, 
                 gamma: float, 
                 start_e: float, 
                 end_e: float, 
                 exploration_fraction: float, 
                 train_frequency: int, 
                 learning_start: int, 
                 network_sync_rate: int, 
                 replay_memory_size: int, 
                 mini_batch_size: int, 
                 model, 
                 hidden_layer_size: int, 
                 env=None,
                 env_name: str = None
    ):
        """
        Initializes a reinforcement learning agent.

        Sets up the agent's hyperparameters, neural network model, optimizer placeholders,
        exploration schedule, and replay memory for training in the given environment.
        """
         
        self.alpha = alpha                                  # Learning rate
        self.gamma = gamma                                  # Discount factor
        self.start_e = start_e                              # Initial exploration rate
        self.end_e  = end_e                                 # Final exploration rate
        self.exploration_fraction = exploration_fraction    # Fraction of total steps over which the exploration rate is annealed
        self.train_frequency = train_frequency              # Frequency of training the model
        self.learning_start = learning_start                # Number of steps before starting training
        self.network_sync_rate = network_sync_rate          # Synchronize the target network with the policy network every 10 steps
        self.replay_memory_size = replay_memory_size        # Size of replay memory
        self.mini_batch_size = mini_batch_size              # Size of training data set sampled from the replay memory
        self.model = model                                  # Model to use for the agent
        self.hidden_layer_size = hidden_layer_size          # Size of each hidden layer in the model
        self.loss_fn = nn.MSELoss()                         # Loss function for the model
        self.optimizer = None                               # Optimizer for the model
        self.env = env                                      # Environment for the agent
        self.env_name = env_name                            # Name of the environment


    def linear_schedule(self, start_e: float, end_e: float, duration: int, t: int) -> float:
        """
        Computes the linearly annealed epsilon (exploration rate) for epsilon-greedy action selection.

        The epsilon value starts at `start_e` and decreases linearly to `end_e` over `duration` steps.
        After `duration` steps, epsilon remains constant at `end_e`. This schedule is typically used 
        to gradually reduce exploration in reinforcement learning agents.

        Args:
            start_e (float): Initial exploration rate at the start of training (usually close to 1.0).
            end_e (float): Final exploration rate after annealing (usually a small value like 0.05).
            duration (int): Number of time steps over which epsilon is linearly annealed.
            t (int): Current time step for which to compute epsilon.

        Returns:
            float: The exploration rate (epsilon) at time step `t`, clipped to not go below `end_e`.
        """

        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)
 

    def optimize(self, 
                 mini_batch: List[Tuple[torch.Tensor, int, torch.Tensor, float, bool, Any]], 
                 policy_dqn: nn.Module, 
                 target_dqn: nn.Module, 
                 head_idx: Optional[int] = None
        ) -> None:
        """
        Performs a single optimization step on the policy network using a mini-batch of transitions.

        If the network is a bootstrapped DQN network with multiple heads, `head_idx` selects which head to update.
        Otherwise, the standard DQN head is used. The function computes the TD target and performs a 
        gradient descent step to minimize the loss between current Q-values and target Q-values.

        Args:
            mini_batch (List[Tuple[torch.Tensor, int, torch.Tensor, float, bool, Any]]): 
                A batch of transitions, each containing:
                    - state (torch.Tensor): The current state.
                    - action (int): The action taken.
                    - next_state (torch.Tensor): The resulting next state.
                    - reward (float): The reward received after taking the action.
                    - terminated (bool): Whether the episode ended after this transition.
                    - info (Any): Additional info (ignored in optimization).
            policy_dqn (nn.Module): The DQN policy network to be optimized.
            target_dqn (nn.Module): The target DQN network used for computing target Q-values.
            head_idx (Optional[int], default=None): Index of the bootstrap head to use. 
                If None, the standard DQN is used.

        Returns:
            None
        """

        # Unpack transitions
        states, actions, next_states, rewards, terminated, _ = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long)
        next_states = torch.stack(next_states)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        terminated = torch.tensor(terminated, dtype=torch.bool)

        # Forward pass through the correct head or normal model
        if head_idx is not None:
            current_q_values = policy_dqn(states, head_idx).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q_values = target_dqn(next_states, head_idx).max(1)[0]
        else:
            current_q_values = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q_values = target_dqn(next_states).max(1)[0]

        target_q_values = rewards.clone()
        target_q_values[~terminated] += self.gamma * next_q_values[~terminated]

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def create_networks(self, model_class: type, num_states: int, num_actions: int, bootstrap_heads: Optional[int] = None):
        """
        Initializes the policy and target networks for training.

        If `model_class` is `BootstrapDQN`, `bootstrap_heads` must be provided and both networks
        will have the specified number of bootstrap heads. Otherwise, standard DQN networks are created.

        The target network weights are synchronized with the policy network upon creation.

        Args:
            model_class (type): The neural network class to instantiate (e.g., DQN or BootstrapDQN).
            num_states (int): Number of input features (state dimensions) for the network.
            num_actions (int): Number of possible actions (output dimensions) for the network.
            bootstrap_heads (int, optional): Number of bootstrap heads if using `BootstrapDQN`. 
                                            Required for bootstrap networks. Defaults to None.

        Returns:
            tuple: A tuple `(policy_dqn, target_dqn)` of the initialized networks.
        """

        if model_class.__name__ == "BootstrapDQN":
            if bootstrap_heads is None:
                raise ValueError("bootstrap_heads must be provided for BootstrapDQN")
            # Pass bootstrap_heads to constructor if needed
            policy_dqn = model_class(in_features=num_states, hidden_layer_size=self.hidden_layer_size,
                                    out_features=num_actions, bootstrap_heads=bootstrap_heads)
            target_dqn = model_class(in_features=num_states, hidden_layer_size=self.hidden_layer_size,
                                    out_features=num_actions, bootstrap_heads=bootstrap_heads)
        else:
            # Standard model initialization
            policy_dqn = model_class(in_features=num_states, hidden_layer_size=self.hidden_layer_size,
                                    out_features=num_actions)
            target_dqn = model_class(in_features=num_states, hidden_layer_size=self.hidden_layer_size,
                                    out_features=num_actions)

        target_dqn.load_state_dict(policy_dqn.state_dict())
        return policy_dqn, target_dqn
    
    def get_env_dimensions(self):
        """
        Returns key environment dimensions required for the agent.

        Determines the number of states, number of actions, and returns the 
        observation space object from the environment.

        Supports:
            - Discrete observation spaces: returns the number of discrete states.
            - Box observation spaces: 
                - For flat vectors: returns the vector length.
                - For images (3D arrays): returns the full shape (C, H, W).
            - Discrete action spaces: returns the number of possible actions.

        Raises:
            NotImplementedError: If the environment uses an unsupported observation or action space type.

        Returns:
            num_states (int or tuple of ints): Number of states or observation shape.
            num_actions (int): Number of discrete actions.
            obs_space (gym.Space): Original observation space object.
        """
        env = self.env
        obs_space = env.observation_space
        action_space = env.action_space

        # Determine state shape/size
        if isinstance(obs_space, gym.spaces.Discrete):
            num_states = obs_space.n
        elif isinstance(obs_space, gym.spaces.Box):
            # Handle image observation (e.g. Atari)
            if len(obs_space.shape) == 3:
                # For images, return the full shape (channels, height, width)
                num_states = obs_space.shape
            else:
                # For flat vector observations
                num_states = obs_space.shape[0]
        else:
            raise NotImplementedError(f"Unsupported observation space type: {type(obs_space)}")

        # Determine number of actions
        if isinstance(action_space, gym.spaces.Discrete):
            num_actions = action_space.n
        else:
            raise NotImplementedError(f"Unsupported action space type: {type(action_space)}")

        return num_states, num_actions, obs_space
    
    def select_action(self, 
                      state: torch.Tensor, 
                      policy_dqn: nn.Module, 
                      step: int, 
                      max_steps: int, 
                      num_actions: int, 
                      num_samples: Optional[int] = None, 
                      bootstrap_head: Optional[int] = None, 
                      eval_mode: bool = False
        ) -> int:
        """
        Selects an action using an epsilon-greedy strategy based on the policy network.

        During training:
            - With probability epsilon, a random action is selected (exploration).
            - Otherwise, the greedy action is chosen from the policy network (exploitation).
            - Epsilon is linearly annealed from start_e to end_e over the training steps.

        During evaluation (`eval_mode=True`), actions are always chosen greedily.

        Special handling for specific algorithms:
            - OPS-VBQN: requires `num_samples` to generate multiple posterior samples; the action
            is selected by taking the argmax over the sampled Q-values.
            - BootstrapDQN: requires `bootstrap_head` to select the head of the ensemble
            from which to compute Q-values.

        Args:
            state (torch.Tensor): Current state, expected shape depends on environment.
            policy_dqn (torch.nn.Module): The policy network used to estimate Q-values.
            step (int): Current training step, used for epsilon annealing.
            max_steps (int): Total number of training steps for annealing.
            num_actions (int): Number of discrete actions available in the environment.
            num_samples (int, optional): Number of posterior samples (required for OPS-VBQN).
            bootstrap_head (int, optional): Head index to use for BootstrapDQN action selection.
            eval_mode (bool, optional): If True, actions are chosen greedily without exploration.

        Returns:
            int: Selected action index.
        
        Raises:
            ValueError: If required arguments are missing for OPS-VBQN or BootstrapDQN.
        """
        model_name = self.model.name()
        
        epsilon = self.linear_schedule(self.start_e, self.end_e, max_steps * self.exploration_fraction, step)

        greedy = eval_mode or (np.random.uniform(0, 1) > epsilon)

        if greedy:
            with torch.no_grad():
                if model_name == "OPS-VBQN":
                    if num_samples is None:
                        raise ValueError("num_samples must be specified for OPS-VBQN action selection.")
                    batched_states = state.expand(num_samples, *state.shape)
                    action_samples = policy_dqn(batched_states)
                    best_index = action_samples.argmax().item()
                    action = best_index % num_actions
                elif model_name == "BootstrapDQN":
                    if bootstrap_head is None:
                        raise ValueError("bootstrap_head must be specified for BootstrapDQN action selection.")
                    action_values = policy_dqn(state.unsqueeze(0), head_idx=bootstrap_head)
                    action = action_values.argmax(dim=1).item()
                else:
                    action_values = policy_dqn(state.unsqueeze(0))
                    action = action_values.argmax(dim=1).item()
            return action

        # Random action (exploration)
        return self.env.action_space.sample()

    def seed_everything(self, seed: int) -> None:
        """
        Set the random seed for reproducibility across Python, NumPy, PyTorch, and the environment.

        This function ensures that experiments are deterministic to the extent possible by controlling
        all sources of randomness, including:
        - Python's `random` module
        - NumPy random generator
        - PyTorch CPU and CUDA random generators
        - The environment's random number generators for actions and observations

        Args:
            seed (int): The integer seed to use for all random number generators.

        Returns:
            None
        """

        # Seed Python, NumPy, and PyTorch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Seed the environment
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)


    def train(
            self, 
            max_steps: int, 
            max_episodes: int, 
            reward_limit: int, 
            seed: int, 
            num_samples: Optional[int] = None, 
            bootstrap_heads: Optional[int] = None
        ) -> Tuple[np.ndarray, int, float]:
        """
        Train the agent on the environment for a given number of steps.

        Implements epsilon-greedy exploration, experience replay, and target network updates.
        Tracks episodic rewards, cumulative regret, and the number of episodes required for the
        agent to consistently reach a specified reward threshold (achieved in 5 consecutive episodes).
        Saves the trained model upon completion.

        Args:
            max_steps (int): Total number of environment steps to train for.
            max_episodes (int): Maximum number of episodes considered when computing cumulative regret.
            reward_limit (int): Reward threshold that defines when the task is considered solved.
            seed (int): Random seed for reproducibility.
            num_samples (Optional[int], default=None): Number of samples for OPS-VBQN action selection (if applicable).
            bootstrap_heads (Optional[int], default=None): Number of bootstrap heads for BootstrapDQN (if applicable).

        Returns:
            tuple[np.ndarray, int, float]: A tuple containing:
                - episodic_rewards (np.ndarray): Array of total rewards collected per episode.
                - episodes_to_solve (int): Number of episodes required to reach the reward threshold
                for 5 consecutive episodes. If the threshold is not met within `max_episodes`,
                `max_episodes` is returned.
                - final_cumulative_regret (float): Total cumulative regret up to the episode when the
                task was considered solved, or up to `max_episodes` if the task was not solved earlier.
        """

        # Get the number of states and actions
        num_states, num_actions, obs_space = self.get_env_dimensions()  

        # Initialize the replay memory
        memory = ReplayMemory(self.replay_memory_size)

        # Initialize the policy network and the target network
        policy_dqn, target_dqn = self.create_networks(self.model, num_states, num_actions, bootstrap_heads)
        


        # Initialize the optimizer using the Adam optimizer
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.alpha)

        # Track the episode numberand episodic rewards
        terminated = False           # True when the agent reaches a terminal state
        truncated = False            # True when the agent exceeds the step limit 
        episode_reward = 0
        episode_number = 1
        cumulative_regret = 0
        episode_rewards = np.zeros(20000)
        episodes_to_solve = 0
        final_cumulative_regret = 0
        task_solved = False
        current_head = np.random.choice(bootstrap_heads) if bootstrap_heads else None

        episodic_rewards = np.zeros(max_steps)
        current_step = 0

        # Initialize the progress bar
        progress_bar = tqdm(total=max_steps, desc="Training Progress", smoothing=0.1)
        update_interval = 5000
        self.seed_everything(seed)  # Seed the environment and random number generators

        for step in range(max_steps):
            
            if step == 0 or terminated or truncated:
                state = self.env.reset()[0]  # Reset the environment and get the initial state
                state = encode_state(state, obs_space)  # Encode the state
            
            if step % update_interval == 0:
                progress_bar.update(update_interval)

            action = self.select_action(state, policy_dqn, step, max_steps, num_actions, 
                                      num_samples=num_samples, bootstrap_head=current_head)

            # Execute action and observe the next state and reward
            new_state, reward, terminated, truncated, infos = self.env.step(action)

            # Episodic reward is stored when the episode ends
            if terminated or truncated: 

                regret = max(0, reward_limit - episode_reward)
                cumulative_regret += regret
                episode_rewards[episode_number] = episode_reward
                last_5_rewards = episode_rewards[episode_number - 4 : episode_number + 1]
                if not task_solved and ((episode_number >= 4 and np.all(last_5_rewards >= reward_limit)) or (max_episodes == episode_number)):
                    final_cumulative_regret = cumulative_regret
                    episodes_to_solve = episode_number
                    task_solved = True

                episodic_rewards[current_step:step + 1] = episode_reward
                current_step = step + 1
                new_state = self.env.reset()[0]
                episode_reward = 0
                episode_number += 1

                # During training:

                current_head = np.random.choice(bootstrap_heads) if bootstrap_heads else None

            new_state = encode_state(new_state, obs_space)  # Encode the state

            if self.model.name() == "BootstrapDQN":
                mask = np.random.binomial(1, 0.8, size=bootstrap_heads).tolist()
                if sum(mask) == 0:
                    mask[current_head] = 1
                transition = Transition(state, action, new_state, reward, terminated, mask)
            else:
                transition = Transition(state, action, new_state, reward, terminated, None)



            memory.push(transition)

            episode_reward += reward

            # Move to the next state
            state = new_state
            
            # Check if if enough experiences have been collected 
            if step > self.learning_start:   
                if step % self.train_frequency == 0:
                    mini_batch = memory.sample(self.mini_batch_size, head=current_head if self.model.name() == "BootstrapDQN" else None)
                    self.optimize(mini_batch, policy_dqn, target_dqn, head_idx=current_head)

                # Copy policy network weights to target network after every network_sync_rate steps
                if step % self.network_sync_rate == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())


        # Close environment
        self.env.close()


        # Save policy
        save_model(model=policy_dqn, env_name=self.env_name, alg_name = self.model.name(), seed = seed, posterior_samples = num_samples, bootstrap_heads= bootstrap_heads)

        return episodic_rewards, episodes_to_solve, final_cumulative_regret

    

    def evaluate(
            self, 
            episodes: int, 
            seed: int, 
            num_samples: Optional[int] = None, 
            bootstrap_heads: Optional[int] = None
        ) -> float:
        """
        Evaluate a trained agent on the environment for a given number of episodes.

        This method loads a previously trained policy and runs it in evaluation mode
        without further training. It supports evaluation with posterior sampling
        (for OPS-VBQN) or bootstrap heads (for bootstrapped DQN).


        Args:
            episodes (int): Number of evaluation episodes to run.
            seed (int): Random seed for reproducibility.
            num_samples (Optional[int], default=None): Number of posterior samples 
                for OPS-VBQN action selection (if applicable).
            bootstrap_heads (Optional[int], default=None): Number of bootstrap heads 
                for BootstrapDQN (if applicable).

        Returns:
            float: The average total reward obtained per episode during evaluation.
        """

        # Get the number of states and actions
        num_states, num_actions, obs_space = self.get_env_dimensions()  

        # Load learned policy
        policy_dqn = policy_dqn = load_model(model_class = self.model, env_name = self.env_name, alg_name = self.model.name(), seed = seed, 
                                in_features=num_states, hidden_layer_size=self.hidden_layer_size, out_features=num_actions, posterior_samples=num_samples, 
                                bootstrap_heads=bootstrap_heads)
        
        policy_dqn.eval() # Set the model to evaluation mode

        episode_rewards = np.zeros(episodes)
        self.seed_everything(seed)

        for episode in tqdm(range(episodes), desc = "Test Progress"):
            state = self.env.reset()[0]  # Reset the environment and get the initial state
            state = encode_state(state, obs_space)  # Encode the state
            terminated = False      # True when the agent reaches a terminal state
            truncated = False       # True when the agent exceeds the step limit
            episode_reward = 0
            current_head = np.random.choice(bootstrap_heads) if bootstrap_heads else None

            while(not terminated and not truncated):
                action = self.select_action(state, policy_dqn, step=0, max_steps=1, num_actions=num_actions,
                                            num_samples=num_samples, bootstrap_head=current_head, eval_mode=True)

                # Execute action
                state, reward, terminated, truncated, _ = self.env.step(action)
                state = encode_state(state, obs_space)  # Encode the state

                episode_reward += reward

            # Store the episode reward
            episode_rewards[episode] = episode_reward

        self.env.close()

        return np.average(episode_rewards)



    def benchmarks(
            self, 
            seeds: List[int],
            max_steps: int, 
            episodes: int, 
            max_episodes: int, 
            reward_limit: int, 
            num_samples: Optional[int] = None, 
            bootstrap_heads: Optional[int] = None 
            ) -> None:
        """
        Runs benchmark training and evaluation across multiple random seeds.

        For each seed, the agent is trained using the specified environment and hyperparameters,
        evaluated using the learned policy, and saves the episodic returns and benchmark results.

        Args:
            seeds (List[int]): List of random seeds to run the benchmark across.
            max_steps (int): Maximum number of training steps per seed.
            episodes (int): Number of episodes to use for evaluation.
            max_episodes (int): Maximum number of episodes considered for cumulative regret calculation.
            reward_limit (int): Reward threshold for determining task success.
            num_samples (Optional[int], default=None): Number of posterior samples for OPS-VBQN (if applicable).
            bootstrap_heads (Optional[int], default=None): Number of bootstrap heads for BootstrapDQN (if applicable).

        Returns:
            None
        """
        for run, seed in enumerate(seeds, start=1):
            train_kwargs = {
                "max_steps": max_steps,
                "seed": seed,
                "max_episodes": max_episodes,
                "reward_limit": reward_limit,
                "num_samples": num_samples,
                "bootstrap_heads": bootstrap_heads,
            }

            episodic_return, episodes_to_solve, cumulative_regret = self.train(**train_kwargs)
            average_episodic_return = self.evaluate(episodes, seed, num_samples, bootstrap_heads)

            print(f"Iteration {run}: Average episodic return: {average_episodic_return:.2f}")

            save_seed_data(
                env_name=self.env_name,
                alg_name=self.model.name(),
                seed=seed,
                returns=episodic_return,
                benchmark_score=average_episodic_return,
                cumulative_regret=cumulative_regret,
                episodes_to_solve=episodes_to_solve,
                posterior_samples=num_samples,
                bootstrap_heads=bootstrap_heads,
            )