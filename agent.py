from memory import ReplayMemory, Transition
from utils import encode_state
from data_handler import save_seed_data, save_model, load_model
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from collections import deque, namedtuple, Counter
from typing import Any, List

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
        Initializes the agent with hyperparameters, model, and environment.

        Sets learning rate, exploration schedule, replay memory size, model architecture,
        and prepares placeholders for optimizer and environment.
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
        Computes the epsilon value linearly annealed from start to end over duration steps.

        Returns the exploration rate at time step t.
        """

        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)
 

    def optimize(self, mini_batch, policy_dqn, target_dqn, head_idx=None) -> None:
        """
        Performs a single optimization step on the policy network using a mini-batch.
        Assumes mini_batch is already filtered by head if head_idx is provided.
        """

        # Unpack transitions — mask isn't needed here because filtering already done
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

    def create_networks(self, model_class, num_states, num_actions, bootstrap_heads=None):
        """
        Initializes policy and target networks according to the model and special params.
        Ensures target network weights are synced with policy.
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
    
    def select_action(self, state, policy_dqn, step, max_steps, num_actions, 
                  num_samples=None, bootstrap_head=None, eval_mode=False):
        model_name = self.model.name()
        
        epsilon = self.linear_schedule(self.start_e, self.end_e, max_steps * self.exploration_fraction, step)

        greedy = eval_mode or (np.random.uniform(0, 1) > epsilon)

        if greedy:
            with torch.no_grad():
                if model_name == "BQMS":
                    if num_samples is None:
                        raise ValueError("num_samples must be specified for BQMS action selection.")
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

    def seed_everything(self, seed: int):
        # Seed Python, NumPy, and PyTorch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Seed the environment
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)


    def train(self, max_steps: int, max_episodes: int, reward_limit: int, seed: int, num_samples: int = None, bootstrap_heads: int = None) -> np.ndarray:
        """
        Trains the agent on the environment for a given number of steps.

        Uses epsilon-greedy action selection, experience replay, and target network updates.

        Returns an array of episodic rewards recorded during training.

        Saves the trained model and episodic rewards to disk.
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

    

    def evaluate(self, episodes: int, seed: int, num_samples: int = None, bootstrap_heads: int = None) -> float:
        """
        Evaluates the trained policy over a specified number of episodes with a given number of posterior samples.

        Executes the learned policy without exploration and returns the average reward over all episodes.

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



    def benchmarks(self, max_steps: int, episodes: int, max_episodes: int, reward_limit: int, num_samples: int, bootstrap_heads: int, seeds: List[int]) -> tuple[float, float]:
        """
        Runs a benchmark test on the environment specified by self, training the agent and the evaluating the agent 
        across multiple seeds. Returns the average episodic return and standard deviation for each seed as the benchmark result.
        """
        run = 1
        for seed in seeds:
            episodic_return, episode_number, cumulative_regret = self.train(max_steps=max_steps, 
                                                                            seed=seed, 
                                                                            max_episodes=max_episodes, 
                                                                            reward_limit=reward_limit, 
                                                                            num_samples=num_samples, 
                                                                            bootstrap_heads=bootstrap_heads)
            average_episodic_return = self.evaluate(episodes, seed, num_samples, bootstrap_heads)
            print(f"Iteration {run}: Average episodic return: {average_episodic_return:.2f}")
            save_seed_data(
                env_name = self.env_name, 
                alg_name = self.model.name(), 
                seed = seed, 
                returns = episodic_return, 
                benchmark_score = average_episodic_return,
                cumulative_regret=cumulative_regret,
                episodes_to_solve=episode_number, 
                posterior_samples = num_samples, 
                bootstrap_heads = bootstrap_heads
            )
            run += 1
