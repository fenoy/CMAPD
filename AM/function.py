import numpy as np
from oracle.oracle import oracle

with open('env/grid.map') as f:
    size = tuple(map(int, f.readline().strip().split(',')))
grid, grid_size = np.genfromtxt('env/grid.map', delimiter=1, skip_header=1, dtype='S').ravel() == b'@', size

def characteristic_function(waypoints):
    '''
    Input:
        - waypoints: Pytorch tensor of (x, y) coordinates of points [number_of_agents, 2 * max_collective_size + 1, 2]
    
    Output:
        - value: Characteristic function value provided by the oracle (float)
    '''

    num_agents = len(waypoints)
    waypoints = [[p for p in a if p[0] != -1] for a in waypoints]   # Pytorch tensor to list of variable size lists of waypoints
    sep = np.array(list(map(len, waypoints)), dtype=np.int32)       # Numpy array with the number of waypoints for each agent (required to parse waypoints inside oracle)
    waypoints = np.array([p[0] * grid_size[1] + p[1] for a in waypoints for p in a], dtype=np.int32) # Format waypoints to 1d numpy array (x * d_x + y)
    return oracle(grid, waypoints, sep, num_agents, len(waypoints), grid_size[0], grid_size[1]) # Call the oracle
