import numpy as np

from ev_station_solver.helper_functions import (
    get_distance_matrix,
)
from ev_station_solver.stochastic_functions import (
    ev_charging,
    ev_charging_probabilities,
    generate_ranges,
)


class Sample:
    def __init__(
        self,
        vehicle_locations: np.ndarray,
        ranges: np.ndarray,
        distance_matrix: np.ndarray,
    ):
        self.vehicle_locations: np.ndarray = vehicle_locations
        self.n_vehicles: int = vehicle_locations.shape[0]  # list of number of vehicles in sample
        self.I: range = range(self.n_vehicles)  # indices of vehicles in sample
        self.ranges: np.ndarray = ranges  # numpy 1D array of ranges of vehicles in sample
        self.distance_matrix: np.ndarray = distance_matrix  # distance matrix of vehicles in sample to cl
        self.reachability_matrix: np.ndarray = (
            self.distance_matrix.T <= self.ranges
        ).T  # reachabilitye matrix of vehicles in sample

    @classmethod
    def create_sample(cls, total_vehicle_locations: np.ndarray, coordinates_potential_cl: np.ndarray):
        ranges = generate_ranges(num=total_vehicle_locations.shape[0])  # ranges for each car
        charging_prob = ev_charging_probabilities(ranges=ranges)  # probability of charging for each car
        charging = ev_charging(
            ranges=ranges, charging_probabilites=charging_prob
        )  # binary mask of whether car is charging or not

        # mask with binary values of whether car is charging or not
        vehicle_locations = total_vehicle_locations[charging]
        ranges = ranges[charging]

        # distance matrix of vehicles to cl
        distance_matrix = get_distance_matrix(vehicle_locations, coordinates_potential_cl)

        return cls(vehicle_locations, ranges, distance_matrix)
