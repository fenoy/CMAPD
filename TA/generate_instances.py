import argparse
import os
import shutil
import random

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='./instances/', help='Where to place instances')
parser.add_argument('--a', type=int, help='Number of agents')
parser.add_argument('--t', type=int, help='Number of tasks')
parser.add_argument('--n', type=int, help='Number of instances')
args = parser.parse_args()

# CREATE DIRECTORY

dirpath = args.path + 'a' + str(args.a) + '-t' + str(args.t)
if os.path.isdir(dirpath):
    shutil.rmtree(dirpath)
os.makedirs(dirpath)

# LOAD GRID
with open('./env/grid.map', 'r') as f:
    f.readline()
    grid = [l.strip() for l in f.readlines()]

# GENERATE FILES
for k in range(args.n):
    endpoints = [(i, j) for i, row in enumerate(grid) for j, cell in enumerate(row) if cell == 'e']
    random.shuffle(endpoints)

    agents = [endpoints.pop() for _ in range(args.a)]
    tasks = [[endpoints.pop(), endpoints.pop()] for _ in range(args.t)]

    with open(dirpath + '/' + str(k) + '.agents', 'w') as f:
        f.write(str(args.a) + '\n')
        for a in agents:
            f.write(','.join(map(str, a)) + '\n')

    with open(dirpath + '/' + str(k) + '.tasks', 'w') as f:
        f.write(str(args.t) + '\n')
        for t in tasks:
            f.write(','.join(','.join(map(str, tt)) for tt in t) + '\n')
