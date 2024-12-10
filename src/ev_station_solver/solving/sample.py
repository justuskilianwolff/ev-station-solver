import numpy as np

from ev_station_solver.constants import CONSTANTS
from ev_station_solver.helper_functions import get_distance_matrix
from ev_station_solver.stochastic_functions import ev_charging, ev_charging_probabilities, generate_ranges


class Sample:
    def __init__(self, index: int, total_vehicle_locations: np.ndarray, coordinates_cl: np.ndarray):
        ranges = generate_ranges(num=total_vehicle_locations.shape[0])  # ranges for each car
        charging_prob = ev_charging_probabilities(ranges=ranges)  # probability of charging for each car
        # binary mask of whether car is charging or not
        charging = ev_charging(ranges=ranges, charging_probabilites=charging_prob)

        # distance matrix of vehicles to cl
        self.index = index
        self.vehicle_locations: np.ndarray = total_vehicle_locations[charging]
        self.ranges: np.ndarray = ranges[charging]  # numpy 1D array of ranges of vehicles in sample
        self.n_vehicles: int = self.vehicle_locations.shape[0]  # list of number of vehicles in sample
        self.I: range = range(self.n_vehicles)  # indices of vehicles in sample
        # distance matrix of vehicles in sample to cl
        self.distance_matrix: np.ndarray = get_distance_matrix(self.vehicle_locations, coordinates_cl)
        # reachabilitye matrix of vehicles in sample
        self.reachable: np.ndarray = (self.distance_matrix.T <= self.ranges).T

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
