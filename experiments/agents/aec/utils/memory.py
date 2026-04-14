from collections import namedtuple
import numpy as np
import json
import os

State = namedtuple('State', ('obs',))
Transition = namedtuple('Transition', ('state', 'act', 'reward', 'next_state', 'next_acts', 'done'))


def sample(rng: np.random.RandomState, data: list, k: int):
    """Choose k unique random elements from a list."""
    return [data[i] for i in rng.choice(len(data), k, replace=False)]


class EpisodicMemory:
    def __init__(self, capacity, seed=20210824, filepath=None):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.rng = np.random.RandomState(seed)
        self.filepath = filepath

        if filepath and os.path.exists(self.filepath):
            os.remove(self.filepath)

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return sample(self.rng, self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, index):
        return self.memory[index]

    def update_ec(self, sequence, gamma=0.99):
        Rtd = 0.0
        for seq in reversed(sequence):
            s = seq['state']
            a = seq['action']
            r = seq['reward']
            Rtd = r + gamma * Rtd
            seq['reward'] = float(Rtd)

            found = False
            for idx, trans in enumerate(self.memory):
                if trans is not None and trans.state == s and trans.act == a:
                    if Rtd > trans.reward:
                        self.memory[idx] = Transition(s, a, Rtd, None, None, False)
                    found = True
                    break

            if not found:
                if len(self.memory) < self.capacity:
                    self.memory.append(None)
                self.memory[self.position] = Transition(s, a, Rtd, None, None, False)
                self.position = (self.position + 1) % self.capacity

            if self.filepath:
                if os.path.exists(self.filepath):
                    with open(self.filepath, 'r') as f:
                        data = json.load(f)
                else:
                    data = seq
                with open(self.filepath, 'w') as f:
                    json.dump(data, f, indent=4)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)
