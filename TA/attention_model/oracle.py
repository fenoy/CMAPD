from ortools.constraint_solver import routing_enums_pb2, pywrapcp

def oracle(distance_matrix, agent, collective, capacity):
    # SETUP VAHICLES AND CAPACITY
    num_vehicles = 1
    capacities = [capacity]

    # DEMANDS AND INDEX TRACKER
    demands = [0]
    original_idx = {0: agent}
    for i, req in enumerate(collective):
        demands.extend([1, -1])
        original_idx[2*i+1] = req[0]
        original_idx[2*i+2] = req[1]
    reduced_idx = {v: k for k, v in original_idx.items()}

    # REDUCED DISTANCE MATRIX AND NO RETURN TO DEPOT
    indices = [original_idx[i] for i in range(len(original_idx))]
    distance_matrix = distance_matrix[indices, :][:, indices]
    distance_matrix[:, 0] = 0

    # MANAGER AND ROUTING
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_vehicles, reduced_idx[agent])
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
        6000,  # vehicle maximum travel distance
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
    for req in collective:
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
        #plan = []
        distance = []

        to_idx = routing.Start(reduced_idx[agent])
        while not routing.IsEnd(to_idx):
            #to_node = manager.IndexToNode(to_idx)
            #plan.append((original_idx[to_node] // 35, original_idx[to_node] % 35))
            from_idx = to_idx
            to_idx = solution.Value(routing.NextVar(to_idx))
            distance.append(routing.GetArcCostForVehicle(from_idx, to_idx, 0))
    
        return sum(distance)

    return solution

if __name__ == '__main__':
    from parameters import params
    import numpy as np

    CAPACITY = params['environment']['capacity']
    distance_matrix = np.load('./env/distance_matrix.npy')

    xp, yp, xd, yd = distance_matrix.shape
    distance_matrix = distance_matrix.reshape(xp * yp, xd * yd)

    agent = [7, 9]
    collective = [
        [[19, 10], [9, 2]],
        [[4, 29], [15, 11]],
        [[18, 32], [1, 18]],
        [[17, 1], [3, 2]],
        [[13, 22], [11, 19]],
        [[20, 12], [15, 21]]]

    agent = agent[0] * yp + agent[1]
    collective = [[p[0] * yp + p[1] , d[0] * yp + d[1]] for p, d in collective]

    print(oracle(distance_matrix, agent, collective, CAPACITY))
