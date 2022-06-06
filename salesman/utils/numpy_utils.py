import numpy as np
from typing import Any, List
from pandas import DataFrame


def select_closest(candidates, origin):
    """Return the index of the closest candidate to a given point."""
    return euclidean_distance(candidates, origin).argmin()

def euclidean_distance(a: np.array, b: np.array):
    # Return an array of the instances of two numpy arrays of points.
    return np.linalg.norm(a - b, axis=1)

def route_distance(locations: DataFrame, columns: List[str] ) -> np.floating:
    # returns the cost of traversing between routes of a location of a graph/node
    points: DataFrame = locations[columns]
    distance: Any = euclidean_distance(points, np.roll(points, 1 , axis=1))
    return np.sum(distance)

def normalize_points(points: DataFrame) -> DataFrame:
    ratio = (points.latitude.max() - points.latitude.min()) / (points.longitude.max() - points.longitude.min()), 1
    ratio = np.array(ratio) / max(ratio)
    norm = points.apply(lambda c: (c - c.min()) / (c.max() - c.min()))
    return norm.apply(lambda p: ratio * p, axis=1)

def generate_network(size):
    """
    Generate a neuron network of a given size.
    Return a vector of two dimensional points in the interval [0,1].
    """
    return np.random.rand(size, 2)

def get_neighborhood(center, radix, domain):
    """Get the range gaussian of given radix around a center index."""

    # Impose an upper bound on the radix to prevent NaN and blocks
    if radix < 1:
        radix = 1

    # Compute the circular network distance to the center
    deltas = np.absolute(center - np.arange(domain))
    distances = np.minimum(deltas, domain - deltas)

    # Compute Gaussian distribution around the given center
    return np.exp(-(distances*distances) / (2*(radix*radix)))

def get_route(cities: DataFrame, network: np.ndarray) -> DataFrame:
    """Return the route computed by a network."""
    cities['winner'] = cities[['latitude', 'longitude']].apply(
        lambda c: select_closest(network, c),
        axis=1, raw=True)

    return cities.sort_values('winner').index