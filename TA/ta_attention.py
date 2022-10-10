import torch
import pulp
from torch.distributions import Categorical
from attention_model.model import Transformer
from attention_model.parameters import params
from attention_model.environment import Collective
from attention_model.function import characteristic_function as f
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(
    input_size=params['model']['input_size'],
    d_model=params['model']['d_model'],
    nhead=params['model']['nhead'],
    dim_feedforward=params['model']['dim_feedforward'],
    num_layers=params['model']['num_layers']).to(device)
model.load_state_dict(torch.load('attention_model/transformer.pth'))

def _get_mask(tasks, collective):
    batch_size, seq_len, _ = tasks.size()
    mask = torch.zeros(batch_size, 1 + seq_len, dtype=torch.bool, device=collective.device)
    mask.scatter_(1, 1 + collective, True)
    return mask[:, 1:]

@torch.no_grad()
def attention_assignment(agents, tasks, tbudget=10):
    model.eval()

    agents_tensor = torch.tensor(agents, device=device).view(-1, 1, 2)
    tasks_tensor = torch.tensor(tasks, device=device).view(1, -1, 4).repeat(len(agents), 1, 1)

    candidates = []
    costs = []

    start = time.time()
    while time.time() - start < tbudget:
        collective = Collective(agents_tensor, tasks_tensor, device)
        while True:
            mask = _get_mask(tasks_tensor, collective.indices)
            if collective.is_terminal.all():
                for i, (a, t) in enumerate(zip(agents_tensor.squeeze(1).tolist(), collective.indices.tolist())):
                    c = (tuple(a), tuple((tuple(tasks_tensor[0][i][:2].tolist()), tuple(tasks_tensor[0][i][2:4].tolist())) for i in t if i != -1))
                    if c not in candidates:
                        candidates.append(c)
                        costs.append(f(t, agents_tensor[i], tasks_tensor[i]))
                break
            probs, _ = model(agents_tensor, tasks_tensor, collective.indices, mask)
            distribution = Categorical(probs)
            action = distribution.sample()
            collective = collective.add_participant(action)

    x = pulp.LpVariable.dicts(
        "collective", candidates, lowBound=0, upBound=1, cat=pulp.LpInteger)

    ilp = pulp.LpProblem("ilp", pulp.LpMinimize)
    ilp += pulp.lpSum([f * x[c] for c, f in zip(candidates, costs)])

    for a in agents:
        ilp += (pulp.lpSum([x[c] for c in candidates if a == c[0]]) == 1, "Must_be_%s" % (a,))

    for t in tasks:
        ilp += (pulp.lpSum([x[c] for c in candidates if tuple(t) in c[1]]) == 1, "Must_be_%s" % (t,))

    ilp.solve(pulp.PULP_CBC_CMD(msg=0))

    return [[c[0], [list(t) for t in c[1]]] for c in candidates if x[c].value() == 1.0]
