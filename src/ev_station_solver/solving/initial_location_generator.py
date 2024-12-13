import numpy as np
from sklearn.cluster import KMeans

from ev_station_solver.logging import get_logger

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
