from cbs import cbs
from ta_ortools import ta_ortools
import time
import numpy as np

# PARAMETERS
CAPACITY = 3
N_ITERATIONS = 50

distance_matrix = np.load('./env/distance_matrix.npy')

xp, yp, xd, yd = distance_matrix.shape
distance_matrix = distance_matrix.reshape(xp * yp, xd * yd)

# MAIN LOOP
print('t1,t2,cost,nconflicts')
for k in range(1, N_ITERATIONS):
    with open('./instances/maps/' + str(k) + '.map', 'r') as f:
        for _ in range(4):
            f.readline()
        grid = [list(l.strip()) for l in f.readlines()]

    agents = [(i, j) for i, row in enumerate(grid) for j, item in enumerate(row) if item == 'r']

    with open('./instances/tasks/' + str(k) + '.task', 'r') as f:
        f.readline()
        tasks = [list(map(int, row.split('\t'))) for row in f.readlines()]
        tasks = [[
            (t[1] // len(grid[0]), t[1] % len(grid[0])),
            (t[2] // len(grid[0]), t[2] % len(grid[0]))
            ] for t in tasks]

    # TASK ASSIGNMENT

    t = time.time()

    agents = [a[0] * yp + a[1] for a in agents]
    tasks = [[p[0] * yp + p[1] , d[0] * yp + d[1]] for p, d in tasks]

    trips = ta_ortools(distance_matrix, agents, tasks, CAPACITY)

    t1_greedy = time.time() - t
    t = time.time()

    # PATH FINDING WITH CBS
    solution, c_greedy, nc_greedy = cbs(grid, trips)

    t2_greedy = time.time() - t
    t = time.time()

    print(
        str(t1_greedy) + ',' +
        str(t2_greedy) + ',' +
        str(c_greedy) + ',' +
        str(nc_greedy))
