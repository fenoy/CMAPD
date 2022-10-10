# Task assignment for capacitated MAPD

## Generate instances

Display instance generation options:

```
python3 generate_instances -h
```

Example to generate 50 instances with 20 agents and 40 tasks:

```
python3 generate_instances.py --a 20 --t 40 --n 50 
```

## Perform task assignment

Display task assignemnt options:

```
python3 generate_instances -h
```

Compute greedy task assignment for instances in path "./instances/a20-t40":

```
python3 evaluation.py --path ./instances/a20-t40 --ta greedy
```

Compute attention task assignment for instances in path "./instances/a20-t40":

```
python3 evaluation.py --path ./instances/a20-t40 --ta attention
```

Compute ortools task assignment for instances in path "./instances/a20-t40":

```
python3 evaluation.py --path ./instances/a20-t40 --ta ortools
```

## Use assignment as instance for PBS

Copy assignment folder to PBS:

```
cp -r instances ../PBS
```
