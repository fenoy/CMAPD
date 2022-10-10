import numpy as np

CAPACITY = 3

def greedy_assignment(agents, tasks):
    distance_matrix = np.load('./env/distance_matrix.npy')

    assignment = [[a] for a in agents]
    agents_loc = [a for a in agents]
    count = [0] * len(agents)
    tracker = [False] * len(tasks)
    dist = np.zeros((len(tasks), len(agents)))
    while True:
        cost_matrix = np.array([
            [distance_matrix[a[0], a[1], t[0], t[1]] for a in agents_loc]
            for t in tasks]) + dist

        cost_matrix[tracker, :] = float('inf')
        cost_matrix[:, np.array(count) >= CAPACITY] = float('inf')

        if (cost_matrix == float('inf')).all():
            break

        task_idx, agent_idx = np.unravel_index(cost_matrix.argmin(), cost_matrix.shape)

        assignment[agent_idx].append(tasks[task_idx][:2])
        assignment[agent_idx].append(tasks[task_idx][2:])
        agents_loc[agent_idx] = tasks[task_idx][2:]
        tracker[task_idx] = True
        count[agent_idx] += 1
        dist[task_idx, agent_idx] += np.min(cost_matrix)

    return assignment
