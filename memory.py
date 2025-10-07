"""
replay_memory.py

Implements a replay buffer for storing and sampling transitions for DQN agents.

The ReplayMemory class stores transitions in a fixed-size deque and allows:
- Random sampling of transitions (standard DQN)
- Head-specific sampling for bootstrapped DQN agents, using a mask.

Transitions are represented as namedtuples with the following fields:
    - state
    - action
    - new_state
    - reward
    - terminated
    - mask (for head-specific sampling, optional)
"""


from collections import deque, namedtuple
import random

Transition = namedtuple('Transition', ['state', 'action', 'new_state', 'reward', 'terminated', 'mask'])


class ReplayMemory:
    """
    Replay buffer for storing and sampling transitions in DQN training.

    Attributes:
        memory (deque): Fixed-length buffer storing Transition namedtuples.

    Methods:
        push(transition): Adds a transition to the memory.
        sample(sample_size, head=None): Returns a list of sampled transitions.
    """
    def __init__(self, maxlen: int):
        """Initializes the replay memory.

        Args:
            maxlen (int): Maximum number of transitions to store.
        """
        self.memory = deque([], maxlen=maxlen)
    
    def push(self, transition: Transition) -> None:
        """Adds a transition to the replay buffer.

        Args:
            transition (Transition): Transition namedtuple to store.
        """
        self.memory.append(transition)
        
    def sample(self, sample_size: int, head: int = None) -> list:
        """Samples a batch of transitions from memory.

        If `head` is None, samples transitions randomly (standard DQN).  
        If `head` is specified, samples only transitions where mask[head] == 1 
        (used for Bootstrapped DQN).

        Args:
            sample_size (int): Number of transitions to sample.
            head (int, optional): Head index for bootstrapped DQN sampling. Defaults to None.

        Returns:
            list: A list of sampled Transition namedtuples.

        Raises:
            ValueError: If there are not enough transitions for the specified head.
        """
        if head is None:
            return random.sample(self.memory, sample_size)
        else:
            filtered = [t for t in self.memory if t.mask is not None and t.mask[head] == 1]
            if len(filtered) < sample_size:
                raise ValueError(f"Not enough transitions for head {head} to sample {sample_size} transitions.")
            return random.sample(filtered, sample_size)
    
    def __len__(self) -> int:
        """Returns the current number of transitions stored in memory.

        Returns:
            int: Number of transitions in the buffer.
        """
        return len(self.memory)