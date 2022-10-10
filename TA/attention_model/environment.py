import torch
import random
from attention_model.parameters import params
from attention_model.function import characteristic_function as func

GRID = params['environment']['map']
MIN_COLLECTIVE_SIZE = params['environment']['min_collective_size']
MAX_COLLECTIVE_SIZE = params['environment']['max_collective_size']
CAPACITY = params['environment']['capacity']
PENALTY = params['environment']['penalty']

class Collective:
    def __init__(self, agents, tasks, device):
        self._batch_size = len(agents)
        self._device = device

        self.agents = agents
        self.tasks = tasks

        self.indices = torch.empty(
            self._batch_size, MAX_COLLECTIVE_SIZE,
            dtype=torch.long, device=self._device
        ).fill_(-1)

        self.is_terminal = torch.zeros(self._batch_size, dtype=torch.bool, device=self._device)

    def add_participant(self, action):
        action = action - 1

        insert_idx = (self.indices != -1).sum(dim=1)
        insert_idx = torch.where((insert_idx < MAX_COLLECTIVE_SIZE - 1) & (~self.is_terminal),
            insert_idx, torch.empty_like(insert_idx).fill_(-1))

        collective = Collective(self.agents, self.tasks, self._device)
        collective.indices = self.indices.clone()
        collective.indices[range(self._batch_size), insert_idx] = torch.where(self.is_terminal, collective.indices[:, -1], action)
        collective.is_terminal = torch.logical_or(action == -1, insert_idx == -1)

        return collective

    def get_reward(self):
        reward = torch.zeros(self._batch_size, dtype=torch.float, device=self._device)

        n = (self.indices != -1).sum(dim=1)
        is_feasible = (MIN_COLLECTIVE_SIZE <= n)
        
        penalty = torch.empty_like(reward).fill_(PENALTY)
        reward = torch.where(is_feasible, reward, penalty)

        compute_reward = is_feasible & self.is_terminal  
        for b, compute in enumerate(compute_reward):
            if compute:
                idx, a, t, size = self.indices[b], self.agents[b], self.tasks[b], n[b]
                reward[b] = func(idx.cpu(), a.cpu(), t.cpu()) / size

        return reward

def get_random_partial_collective(agents, tasks, device):
    batch_size = len(agents)
    pool_size = torch.sum((tasks != -1).all(dim=-1), dim=-1)

    collective = Collective(agents, tasks, device)
    indices = torch.cat([
        torch.randint(s, size=(1, MAX_COLLECTIVE_SIZE), device=device)
        for s in pool_size], dim=0)

    collective_size = torch.randint(CAPACITY, size=(batch_size, 1), device=device)
    collective.indices = torch.where(
        torch.arange(MAX_COLLECTIVE_SIZE, device=device).repeat(batch_size, 1) < collective_size,
        indices, collective.indices)

    return collective

with open(GRID, 'r') as f:
    grid = [l.strip().split(',') for l in f.readlines()]

def sample_agents_tasks(n_agents, n_tasks):
    typecell = {'0': [], '1': [], '2': []}
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            typecell[cell].append((i, j))
    random.shuffle(typecell['1'])
    return ([typecell['1'].pop() for _ in range(n_agents)],
        [[typecell['1'].pop(), typecell['1'].pop()] for _ in range(n_tasks)])
