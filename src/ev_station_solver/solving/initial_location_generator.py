import igraph as ig
import numpy as np
from sklearn.cluster import KMeans

from ev_station_solver.helper_functions import get_distance_matrix
from ev_station_solver.logging import get_logger
from ev_station_solver.solving.sample import Sample

logger = get_logger(__name__)


class InitialLocationGenerator:
    def __init__(self, vehicle_locations: np.ndarray) -> None:
        self.vehicle_locations = vehicle_locations

    def get_random_locations(self, n_stations: int, seed: int | None = None) -> np.ndarray:
        # random generator
        rng = np.random.default_rng(seed=seed)
        # scale random locations to grid
        x_min, y_min = self.vehicle_locations.min(axis=0)
        x_max, y_max = self.vehicle_locations.max(axis=0)
        new_locations = rng.random((n_stations, 2)) * np.array([x_max - x_min, y_max - y_min]) + np.array([x_min, y_min])

        return new_locations

    def get_k_means_locations(self, n_stations: int, seed: int | None = None, verbose: int = 0) -> np.ndarray:
        kmeans = KMeans(n_clusters=n_stations, n_init=1, random_state=seed, verbose=verbose)
        new_locations = kmeans.fit(self.vehicle_locations).cluster_centers_

        return new_locations

    def get_clique_locations(self, samples: list[Sample], n: int, q: int, seed: int | None = None) -> np.ndarray:
        # concatenate locations and ranges of all vehicles in all samples
        all_vehicle_locations = np.concatenate([sample.vehicle_locations for sample in samples], axis=0)
        all_ranges = np.concatenate([sample.ranges for sample in samples], axis=0)

        # obtain unique vehicle locations and take the lowest range
        unique_locations, unique_index = np.unique(all_vehicle_locations, axis=0, return_index=True)
        # update ranges
        unique_ranges = all_ranges[unique_index]

        # compute the distance matrix and construct adjacency matrix (vehicles are connected if the other location is within range)
        distance_matrix = get_distance_matrix(unique_locations, unique_locations)
        ranges_row = unique_ranges[:, np.newaxis]  # Shape (n, 1)
        ranges_col = unique_ranges[np.newaxis, :]  # Shape (1, n)

        # Compare distances with both ranges
        within_range1 = distance_matrix < ranges_row  # Compare with first point's range
        within_range2 = distance_matrix < ranges_col  # Compare with second point's range

        # Points are adjacent if within either range
        adjacency = np.logical_or(within_range1, within_range2)

        # create a graph object
        G = ig.Graph.Adjacency(adjacency.tolist(), mode=ig.ADJ_UNDIRECTED)

        # store the new locations
        new_locations = np.empty((0, 2))

        # while the graph is not empty keep going

        while G.vcount() > 0:
            # find the largest cliques
            cliques = G.largest_cliques()

            removed_nodes = []

            for clique in cliques:
                if any([node in removed_nodes for node in clique]):
                    # if any of the nodes in the clique have been removed, skip this clique
                    continue
                else:
                    # apply k means

                    kmeans = KMeans(n_clusters=-(len(clique) // -(n * q)), random_state=seed)
                    kmeans.fit(unique_locations[clique])

                    # append centers to new locations
                    new_locations = np.vstack((new_locations, kmeans.cluster_centers_))

                    # remove the clique from the graph
                    G.delete_vertices(clique)
                    removed_nodes.extend(clique)
                    # TODO: are the indices updated as well? or do we need to update ranges and

        return new_locations
