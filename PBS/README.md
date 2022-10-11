# Prioritized Based Search for Capacitated Multi-Agent Path Finding

A suboptimal solver for Capacitated Multi-Agent Path Finding. Adapted from the [implementation](https://github.com/Jiaoyang-Li/PBS) by [Jiaoyang-Li](https://github.com/Jiaoyang-Li).

## Usage

The code requires the external library [boost](https://www.boost.org/). If you are using Ubantu, you can install it simply by

```shell script
sudo apt install libboost-all-dev
``` 

After you installed boost and downloaded the source code, go into the directory of the source code and compile it with CMake:
```shell script
cmake -DCMAKE_BUILD_TYPE=RELEASE .
make
```

The solver receives a file where each row is a path of waypoints:

```
./pbs -m env/grid.map -a instances/a20-t40/ta-greedy/0.assignment -o test.csv --outputPaths=paths.txt -k 20 -t 60
```

- m: the map file from the MAPF benchmark
- a: the scenario file from the MAPF benchmark
- o: the output file that contains the search statistics
- outputPaths: the output file that contains the paths 
- k: the number of agents
- t: the runtime limit

To run PBS for 50 assignments generated with the greedy task assignment code with 20 agents and 40 tasks do:

```
sh ./evaluate.sh greedy 20 40 50
```

## License

PBS is released under USC – Research License. See license.md for further details.
