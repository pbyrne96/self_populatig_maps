# solved WITH Self-populating maps
from read_coor import read_files
import pandas as pd
from utils.numpy_utils import generate_network,get_neighborhood, select_closest, get_route, route_distance, normalize_points
import numpy as np
from EDA import plot_network, plot_route

TARGET_COLUMNS = ['latitude','longitude']

def __main__() -> np.floating:

    all_locations = read_files()
    country_names = list(all_locations.keys())
    problem = all_locations[country_names[0]]
    route = calculate_route(problem, iterations=100)
    problem = problem.reindex(route)
    distance = route_distance(problem, TARGET_COLUMNS)
    return distance

def calculate_route(df: pd.DataFrame, iterations, learning_rate = 0.8) -> pd.DataFrame:
    # Obtain the normalized set of cities (w/ coord in [0,1])
    cities = df.copy()

    cities[TARGET_COLUMNS] = normalize_points(cities[TARGET_COLUMNS])

    steps = [i for i in range(0,iterations,5)]

    # The population size is 8 times the number of cities
    n = cities.shape[0] * 8

    # Generate an adequate network of neurons:
    network = generate_network(n)

    for i in range(iterations):
        city = df.sample(1)[TARGET_COLUMNS].values
        winner_idx = select_closest(network, city)
        # Generate a filter that applies changes to the winner's gaussian
        gaussian = get_neighborhood(winner_idx, n//10, network.shape[0])
        # Update the network's weights (closer to the city)
        network += gaussian[:,np.newaxis] * learning_rate * (city - network)
        # Decay the variables
        learning_rate = learning_rate * 0.99997
        n = n * 0.9997

        if i in steps:
            plot_route(cities, get_route(cities, network), name=f'diagrams/iter-no{i}.jpg')
        # Check if any parameter has completely decayed.
        if n < 1:
            print('Radius has completely decayed, finishing execution at {} iterations'.format(i))
            break
        if learning_rate < 0.001:
            print('Learning rate has completely decayed, finishing execution at {} iterations'.format(i))
            break

    route = get_route(cities, network)
    plot_network(cities, network, name='diagrams/final.jpg')
    plot_route(cities, route, 'diagrams/route.jpg')
    return get_route(cities, network)


def create_gif()-> None:
    ...