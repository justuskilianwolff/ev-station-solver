import igraph as ig
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ev_station_solver.helper_functions import get_distance_matrix
from ev_station_solver.logging import get_logger
from ev_station_solver.solving.sample import Sample

logger = get_logger(__name__, "DEBUG")


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
        logger.info("Computing clique locations")

        # across all samples take all locations for a specific vehicle
        all_samples = []

        for sample in samples:
            sample_id = sample.index
            sample_vehicle_ids = sample.indices
            sample_locations = sample.vehicle_locations
            sample_ranges = sample.ranges

            sample_df = pd.DataFrame(
                {
                    "sample_id": sample_id,
                    "vehicle_id": sample_vehicle_ids,
                    "x": sample_locations[:, 0],
                    "y": sample_locations[:, 1],
                    "range": sample_ranges,
                }
            )

            all_samples.append(sample_df)

        location_information = pd.concat(all_samples, ignore_index=True)

        # compute df where each vehicle df is unique with the lowest range
        min_range_df = location_information.sort_values("range").drop_duplicates(subset=["vehicle_id"], keep="first")
        min_range_locations = min_range_df[["x", "y"]].values
        min_ranges = min_range_df["range"].values

        # compute the distance matrix and construct adjacency matrix (vehicles are connected if the other location is within range)
        distance_matrix = get_distance_matrix(min_range_locations, min_range_locations)
        ranges_row = min_ranges[:, np.newaxis]  # Shape (n, 1)
        ranges_col = min_ranges[np.newaxis, :]  # Shape (1, n)

        # Points are adjacent if within either range
        adjacency = np.logical_and(distance_matrix <= ranges_row, distance_matrix <= ranges_col)

        # create a graph object
        G = ig.Graph.Adjacency(adjacency.tolist(), mode=ig.ADJ_UNDIRECTED)
        # give each vertex an id since deleting vertices does not keep the indexing
        G.vs["id"] = range(G.vcount())

        # store the new locations
        new_locations = np.empty((0, 2))

        # while the graph is not empty keep going
        while G.vcount() > 0:
            logger.debug(f"Number of vertices: {G.vcount()}")

            # find the largest cliques
            clique = G.largest_cliques()[0]
            # for each clique get the ids
            clique_ids = np.array([G.vs[node]["id"] for node in clique])

            # compute the vehicle ids present in the clique
            vehicle_ids = min_range_df.iloc[clique_ids]["vehicle_id"]
            # filter original df for only those vehicle ids
            clique_df = location_information[location_information["vehicle_id"].isin(vehicle_ids)]
            # compute cl
            max_locations_by_cluster = clique_df["sample_id"].value_counts()
            max_locations = max_locations_by_cluster.max()

            # apply k means
            kmeans = KMeans(n_clusters=-(max_locations // -(n * q)), random_state=seed)
            kmeans.fit(min_range_locations[clique_ids])

            # append centers to new locations
            new_locations = np.vstack((new_locations, kmeans.cluster_centers_))

            # remove the clique from the graph
            G.delete_vertices(clique)

        logger.info("Finished computing clique locations")

        return new_locations
