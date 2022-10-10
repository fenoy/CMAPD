import numpy as np
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

CAPACITY = 3

distance_matrix = np.load('./env/distance_matrix.npy')

xp, yp, xd, yd = distance_matrix.shape
distance_matrix = distance_matrix.reshape(xp * yp, xd * yd)

def ta_ortools_(distance_matrix, agents, tasks, capacity, makespan):
    # SETUP VAHICLES AND CAPACITY
    num_vehicles = len(agents)
    capacities = [capacity] * num_vehicles

    # DEMANDS AND INDEX TRACKER
    demands = [0] * num_vehicles
    original_idx = {i: a for i, a in enumerate(agents)}
    for i, req in enumerate(tasks):
        demands.extend([1, -1])
        original_idx[num_vehicles + 2 * i + 0] = req[0]
        original_idx[num_vehicles + 2 * i + 1] = req[1]
    reduced_idx = {v: k for k, v in original_idx.items()}

    # REDUCED DISTANCE MATRIX AND NO RETURN TO DEPOT
    indices = [original_idx[i] for i in range(len(original_idx))]
    distance_matrix = distance_matrix[indices, :][:, indices]

    distance_matrix = np.append(distance_matrix, np.zeros((num_vehicles, len(distance_matrix))), 0)
    distance_matrix = np.append(distance_matrix, np.zeros((len(distance_matrix), num_vehicles)), 1)
    demands.extend([0] * num_vehicles)

    # MANAGER AND ROUTING
    manager = pywrapcp.RoutingIndexManager(
        len(distance_matrix),
        num_vehicles,
        [reduced_idx[a] for a in agents],
        [len(distance_matrix) - i for i in range(num_vehicles, 0, -1)])
    routing = pywrapcp.RoutingModel(manager)

    # DISTANCE CALLBACK
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,     # no slack
        makespan,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # DEMAND CALLBACK
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,           # null capacity slack
        capacities,  # vehicle maximum capacities
        True,        # start cumul to zero
        'Capacity')

    # PICKUP AND DELIVERIES
    for req in tasks:
        pickup_index = manager.NodeToIndex(reduced_idx[req[0]])
        delivery_index = manager.NodeToIndex(reduced_idx[req[1]])
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        routing.solver().Add(
            routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index))
        routing.solver().Add(
            distance_dimension.CumulVar(pickup_index) <= distance_dimension.CumulVar(delivery_index))
    
    # SEARCH PARAMETERS
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)

    # SOLUTION
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        print(solution)

        trips = []
        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            plan_output = []
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                plan_output.append((original_idx[node_index] // 35, original_idx[node_index] % 35))
                index = solution.Value(routing.NextVar(index))
            trips.append(plan_output)
        return trips

    return solution

def ortools_assignment(agents, tasks):
    agents = [a[0] * yp + a[1] for a in agents]
    tasks = [[t[0] * yp + t[1] , t[2] * yp + t[3]] for t in tasks]

    for makespan in range(20, 100, 1):
        solution = ta_ortools_(distance_matrix, agents, tasks, CAPACITY, makespan)
        if solution:
            return solution

    return None
