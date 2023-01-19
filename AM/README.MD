# Attention-based approach for capacitated MAPD

This repo implements a training setup to train the attention model for the capacitated MAPD task.

## Description of the repo

### Files

- "parameters.py" : File to tune the parameters of the attention model, training and environment.
- "train.py" : Initializes the attention model and calls the training setup.
- "trainer.py" : Implements a training setup based on the REINFORCE algorithm with a rollout baseline and a soft-update rule for the baseline.
- "model.py" : Implements the attention-based encoder-decoder model.
- "environment.py" : Implements the rules to update collectives, paths and waypoints. Also calls the reward function.
- "function.py" : Calls the oracle to compute the characteristic function value.
- "dataset.py" : Pytorch dataset implementation to store and pass the data to each training iteration.

### Directories

- "env" : Stores the map and the distance matrix of the map.
- "oracle" : Stores the files to compute the characteristic function values in C++ and Cython.

## Running experiments

To train the attention model:

```
python3 train.py
```

The console will print if cpu or cuda is being used, and then proceed with the training iterations.

## Work to be done

1. Incorporate a faster oracle . Compare the performance with the actual oracle in terms of speed and values.
2. The code presents a memory leak, which is likely to be in the oracle implementation. Check if step 1 fixed the memory leak by performing several training iterations and checking if, after each one, the free memory keeps decreasing. If not fixed, try to spot where is the memory leak and fix it.
3. The insertion method of task pick-up and delivery points into the sequence of waypoints is naive and slow. Try to improve it, maybe with an approximated approach.
4. Currently, the path encoding only contins the first and last elements of the path. A better encoding needs to be designed taking into account that the attention model does not admit variable size representation of a path.
