import numpy as np

from ev_station_solver.constants import CONSTANTS
from ev_station_solver.helper_functions import get_distance_matrix
from ev_station_solver.stochastic_functions import ev_charging, ev_charging_probabilities, generate_ranges


class Sample:
    def __init__(self, index: int, total_vehicle_locations: np.ndarray):
        ranges = generate_ranges(num=total_vehicle_locations.shape[0])  # ranges for each car
        charging_prob = ev_charging_probabilities(ranges=ranges)  # probability of charging for each car
        # binary mask of whether car is charging or not
        charging = ev_charging(ranges=ranges, charging_probabilites=charging_prob)

        # distance matrix of vehicles to cl
        self.index: int = index
        self.indices: np.ndarray = np.where(charging)[0]  # idices of charging vehicles within original locations
        self.vehicle_locations: np.ndarray = total_vehicle_locations[self.indices]
        self.ranges: np.ndarray = ranges[self.indices]  # numpy 1D array of ranges of vehicles in sample
        self.n_vehicles: int = self.vehicle_locations.shape[0]  # list of number of vehicles in sample
        self.I: range = range(self.n_vehicles)  # indices of vehicles in sample

        # distance matrix of vehicles in sample to cl
        self.distance_matrix: np.ndarray = np.empty(())  # distance matrix of vehicles in sample to cl
        # reachabilitye matrix of vehicles in sample
        self.reachable: np.ndarray = np.empty(())  # reachability matrix of vehicles in sample

    def __str__(self) -> str:
        return str(self.index)

    def get_fixed_charge_cost(self, charge_cost_param: float, ub_range: int = CONSTANTS["ub_range"]) -> float:
        """Obtain the fixed charge cost for the sample.

        Args:
            charge_cost_param (float): cost of charging
            ub_range (int, optional): the upper range. Defaults to CONSTANTS["ub_range"].

        Returns:
            float: total fixed charge cost for the sample
        """
        return charge_cost_param * (ub_range - self.ranges).sum()

    def set_distance_and_reachable(self, coordinates_cl: np.ndarray) -> None:
        """Set the distance matrix and reachable matrix for the sample.

        Args:
            coordinates_cl (np.ndarray): coordinates of charging locations
        """
        self.distance_matrix = get_distance_matrix(self.vehicle_locations, coordinates_cl)
        self.reachable = (self.distance_matrix.T <= self.ranges).T
