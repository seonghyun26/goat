import torch
import proxy
from tqdm import tqdm
import openmm.unit as unit
from utils.utils import *
from torch.distributions import Normal


class ReplayBuffer:
    def __init__(self, config, md):
        self.positions = torch.zeros(
            (config['agent']['buffer']['size'], config['agent']['num_steps'] + 1, md.num_particles, 3),
            device=config['system']['device'],
        )
        self.actions = torch.zeros(
            (config['agent']['buffer']['size'], config['agent']['num_steps'], md.num_particles, 3), device=config['system']['device']
        )
        self.log_reward = torch.zeros(config['agent']['buffer']['size'], device=config['system']['device'])
        self.priorities = torch.ones(config['agent']['buffer']['size'], device=config['system']['device'])

        self.idx = 0
        self.prioritized = config['agent']['buffer']['prioritized']
        self.prioritized_exp = config['agent']['buffer']['prioritized_exp']
        self.buffer_size = config['agent']['buffer']['size']
        self.num_samples = config['agent']['num_samples']
        self.batch_size = config['training']['batch_size']

    def add(self, data):
        indices = torch.arange(self.idx, self.idx + self.num_samples) % self.buffer_size
        self.idx += self.num_samples

        self.positions[indices], self.actions[indices], self.log_reward[indices] = data

    def sample(self):
        if self.prioritized:
            probs = self.priorities[: min(self.idx, self.buffer_size)].pow(
                self.prioritized_exp
            )
        else:
            probs = self.priorities[: min(self.idx, self.buffer_size)]
        
        probs = probs / probs.sum()
        indices = torch.multinomial(
            probs, min(self.idx, self.batch_size), replacement=False
        )
        
        return (
            indices,
            self.positions[indices],
            self.actions[indices],
            self.log_reward[indices],
        )

    def update_priorities(self, indices, weight):
        self.priorities[indices] = weight
