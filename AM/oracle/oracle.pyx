#cython: language_level=3
import cython
from cython.parallel import prange

import numpy as np
cimport numpy as np
from libcpp cimport bool, int

cdef extern from "cpp_oracle.hpp":
    float cpp_oracle(const bool *grid, const int *agents, const int *sep,
        const int num_agents, const int num_locs, const int num_rows, const int num_cols)

@cython.boundscheck(False)
@cython.wraparound(False)

def oracle(
    np.ndarray[bool, ndim = 1, mode = "c"] grid,
    np.ndarray[int, ndim = 1, mode = "c"] agents,
    np.ndarray[int, ndim = 1, mode = "c"] sep,
    num_agents, num_locs, num_rows, num_cols):
    
    return cpp_oracle(&grid[0], &agents[0], &sep[0], num_agents, num_locs, num_rows, num_cols)
