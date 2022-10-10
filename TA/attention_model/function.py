import numpy as np
from itertools import accumulate, groupby
from attention_model.parameters import params
from attention_model.oracle import oracle

CAPACITY = params['environment']['capacity']

distance_matrix = np.load('./env/distance_matrix.npy')
xp, yp, xd, yd = distance_matrix.shape
distance_matrix = distance_matrix.reshape(xp * yp, xd * yd)

def characteristic_function(indices, agent, tasks):
    collective = [tasks[i].view(-1, 2).tolist() for i in indices if i != -1]
    collective = [[p[0] * yp + p[1] , d[0] * yp + d[1]] for p, d in collective]

    agent = agent.flatten().tolist()
    agent = agent[0] * yp + agent[1]

    best = oracle(distance_matrix, agent, collective, CAPACITY)

    '''
    best = float("inf")
    for trip in permutations(collective, capacity=CAPACITY):
        trip = [agent] + trip

        trip_dist = 0
        for x, y in zip(trip[:-1], trip[1:]):
            trip_dist += distance_matrix[x[0]][x[1]][y[0]][y[1]]

        if trip_dist < best:
            best = trip_dist
    '''

    return best

def permutations_(elements):
    size = 2 * len(elements)
    if size == 0:
        yield elements
        return
    if size <= 2:
        yield elements[0]
        return
    for perm in permutations(elements[1:]):
        for i in range(size-1):
            for j in range(i, size-1):
                yield perm[:i] + [elements[0][0]] + perm[i:j] + [elements[0][1]] + perm[j:]

def permutations(elements, capacity=float('inf'), flat=False):
    if flat:
        items = elements
    else:
        items = [e for tup in elements for e in tup]

    size = len(items)
    locs = [i for i in range(size)]

    if size <= 2:
        yield items
        return

    for perm in permutations(locs[2:], capacity=capacity, flat=True):
        pair = [-2*(item % 2) + 1 for item in perm]
        load = [i == capacity for i in accumulate(pair)]
        load = [load[0]] + [l or load[i] for i, l in enumerate(load[1:])]
        groups = [(k, [x[0] for x in g]) for k, g in groupby(zip(perm, load), lambda x: x[1])]
        for k, (skip, g) in enumerate(groups):
            if skip: continue
            head = [items[idx] for group in groups[:k] for idx in group[1]]
            tail = [items[idx] for group in groups[k+1:] for idx in group[1]]
            for i in range(len(g)+1):
                for j in range(i, len(g)+1):
                    gg = g[:i] + [locs[0]] + g[i:j] + [locs[1]] + g[j:]
                    yield head + [items[idx] for idx in gg] + tail
