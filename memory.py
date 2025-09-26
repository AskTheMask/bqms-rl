from collections import deque, namedtuple
import random

Transition = namedtuple('Transition', ['state', 'action', 'new_state', 'reward', 'terminated', 'mask'])


class ReplayMemory:
    def __init__(self, maxlen: int):
        self.memory = deque([], maxlen=maxlen)
    
    def push(self, transition: Transition) -> None:
        self.memory.append(transition)
        
    def sample(self, sample_size: int, head: int = None) -> list:
        """
        Samples transitions:
        - If head is None, sample randomly as usual (standard DQN).
        - If head is specified, sample only transitions where mask[head] == 1.
        """
        if head is None:
            return random.sample(self.memory, sample_size)
        else:
            filtered = [t for t in self.memory if t.mask is not None and t.mask[head] == 1]
            if len(filtered) < sample_size:
                raise ValueError(f"Not enough transitions for head {head} to sample {sample_size} transitions.")
            return random.sample(filtered, sample_size)
    
    def __len__(self):
        return len(self.memory)