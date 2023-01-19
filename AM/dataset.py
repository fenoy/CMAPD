import torch
from random import randint
from torch.utils.data import Dataset
from environment import sample_agents_tasks

class TADataset(Dataset):
    def __init__(self, size, n_agents, n_tasks):
        super(TADataset, self).__init__()
        self.agents, self.tasks = list(zip(*[
            sample_agents_tasks(n_agents, randint(n_tasks[0], n_tasks[1])) for _ in range(size)]))

        self.agents = [torch.tensor(a) for a in self.agents]
        self.tasks = [torch.nn.functional.pad(
            torch.tensor(t).view(-1, 4), (0, 0, 0, n_tasks[1] - len(t)), value=-1)
            for t in self.tasks]

        self.assignments = []
        self.paths = []
        self.waypoints = []

    def __len__(self):
        return len(self.agents)

    def __getitem__(self, idx):
        if len(self.assignments): return {
            'agents': self.agents[idx], 'tasks': self.tasks[idx],
            'assignments': self.assignments[idx], 'paths': self.paths[idx], 'waypoints': self.waypoints[idx]}
        return {'agents': self.agents[idx], 'tasks': self.tasks[idx]}
