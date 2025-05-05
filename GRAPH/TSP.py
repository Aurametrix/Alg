import math
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def haversine(lat1, lon1, lat2, lon2):
    """Calculates the great-circle distance between two points on Earth."""
    R = 3959  # Earth radius in miles

    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    # Using integer distances for OR-Tools (multiply by a large factor)
    # Scale factor can impact precision but needed for the solver
    scale_factor = 10000
    return int(distance * scale_factor)


def create_data_model():
    """Stores the data for the problem."""
    data = {}
    # Coordinates list: [ (lat, lon), (lat, lon), ... ]
    data['locations'] = [
        (35.5651268427082, -84.23655488714552), # Start
        (35.56772, -84.24081), # Box 1
        (35.56464, -84.24595), # Box 2
        (35.56451, -84.25142), # Box 3
        (35.56494, -84.253),   # Box 4
        (35.56771, -84.24851), # Box 5
        (35.56793, -84.24535), # Box 6
        (35.57049, -84.24142), # Box 7
        (35.57122, -84.24028), # Box 8
        (35.57315, -84.23745), # Box 9
        (35.57408, -84.23451), # Box 10
        (35.57386, -84.23365), # Box 11
        (35.56866, -84.23944), # Box 12
        (35.57317, -84.23027), # Box 13
        (35.56445, -84.23224), # Box 14
        (35.56288, -84.23188), # Box 15
        (35.56098, -84.23589), # Box 16
        (35.56369, -84.23899), # Box 17
        (35.56508, -84.23867)  # Box 18
    ]
    data['num_vehicles'] = 1
    data['depot'] = 0 # Starting point index
    data['box_names'] = [
        "Start", "Kahite Box #1", "Kahite Box #2", "Kahite Box #3", "Kahite Box #4",
        "Kahite Box #5", "Kahite Box #6", "Kahite Box #7", "Kahite Box #8", "Kahite Box #9",
        "Kahite Box #10", "Kahite Box #11", "Kahite Box #12", "Kahite Box #13",
        "Kahite Box #14", "Kahite Box #15", "Kahite Box #16", "Kahite Box #17", "Kahite Box #18"
    ]
    return data

def compute_distance_matrix(locations):
    """Creates callback to return distance between points."""
    num_locations = len(locations)
    distances = {}
    for from_counter in range(num_locations):
        distances[from_counter] = {}
        for to_counter in range(num_locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                lat1, lon1 = locations[from_counter]
                lat2, lon2 = locations[to_counter]
                distances[from_counter][to_counter] = haversine(lat1, lon1, lat2, lon2)
    return distances

def get_solution_sequence(manager, routing, solution, data):
    """Returns the solution sequence."""
    index = routing.Start(0)
    route = []
    route_distance_scaled = 0
    while not routing.IsEnd(index):
        node_index = manager.IndexToNode(index)
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route.append(data['box_names'][node_index])
        route_distance_scaled += routing.GetArcCostForVehicle(previous_index, index, 0)

    # Add the end node (depot) to complete the loop display if needed
    node_index = manager.IndexToNode(index)
    route.append(data['box_names'][node_index])

    # Convert scaled distance back to miles
    scale_factor = 10000
    route_distance_miles = route_distance_scaled / scale_factor
    return route, route_distance_miles


# --- Main execution ---
data = create_data_model()
distance_matrix = compute_distance_matrix(data['locations'])

manager = pywrapcp.RoutingIndexManager(len(data['locations']),
                                       data['num_vehicles'], data['depot'])
routing = pywrapcp.RoutingModel(manager)

def distance_callback(from_index, to_index):
    """Returns the distance between the two nodes."""
    # Convert from routing variable Index to distance matrix NodeIndex.
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return distance_matrix[from_node][to_node]

transit_callback_index = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
search_parameters.time_limit.seconds = 30 # Set a time limit

solution = routing.SolveWithParameters(search_parameters)

# --- Print solution ---
if solution:
    route_sequence, total_distance = get_solution_sequence(manager, routing, solution, data)
    # The route_sequence includes Start at the beginning and end.
    # The sequence requested starts *after* the starting point.
    optimal_box_sequence = route_sequence[1:-1] # Exclude Start and the return to Start

    print("Optimal Sequence of Boxes to Visit:")
    for i, box_name in enumerate(optimal_box_sequence):
        print(f"{i+1}. {box_name}")

    print(f"\nTotal Estimated Distance (including return to start): {total_distance:.4f} miles")

else:
    print('No solution found!')
