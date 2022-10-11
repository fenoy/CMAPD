import argparse
import os
import shutil
import time
from ta_greedy import greedy_assignment
from ta_attention import attention_assignment
from ta_ortools import ortools_assignment

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='Where to read instances from')
parser.add_argument('--ta', choices=['greedy', 'attention', 'ortools'], help='Type of task assignment')
args = parser.parse_args()

dirpath = os.path.join(args.path, 'ta-' + args.ta)
if os.path.isdir(dirpath):
    shutil.rmtree(dirpath)
os.makedirs(dirpath)

# MAIN LOOP

files = {'agents': [], 'tasks': []}
for file in os.listdir(os.fsencode(args.path)):
    filename = os.fsdecode(file)
    if filename.endswith(".agents"):
        files['agents'].append(os.path.join(args.path, filename))
    if filename.endswith(".tasks"):
        files['tasks'].append(os.path.join(args.path, filename))

for k, (agents_file, task_file) in enumerate(zip(sorted(files['agents']), sorted(files['tasks']))):
    with open('./env/grid.map', 'r') as f:
        f.readline()
        grid = [l.strip() for l in f.readlines()]

    with open(agents_file, 'r') as f:
        f.readline()
        agents = [list(map(int, row.split(','))) for row in f.readlines()]

    with open(task_file, 'r') as f:
        f.readline()
        tasks = [list(map(int, row.split(','))) for row in f.readlines()]

    # TASK ASSIGNMENT

    if args.ta == 'greedy':
        task_assignent = greedy_assignment
    if args.ta == 'attention':
        task_assignent = attention_assignment
    if args.ta == 'ortools':
        task_assignent = ortools_assignment

    t1 = time.time()
    assignment = task_assignent(agents, tasks)
    t2 = time.time()

    with open(os.path.join(args.path, 'ta-' + args.ta, str(k) + '.assignment'), 'w') as f:
        print(t2 - t1)
        f.write(str(len(assignment)) + '\n')
        for a in assignment:
            f.write(str(len(a)) + ',')
            f.write(','.join(','.join(map(str, aa)) for aa in a) + '\n')
