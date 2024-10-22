import logging

from ev_station_solver.constants import MOPTA_CONSTANTS
from ev_station_solver.loading import load_locations
from ev_station_solver.mopta_solver import MOPTASolver

# use given starting solutions
locations = load_locations("small").sample(100).values
n_clusters = int(len(locations) * MOPTA_CONSTANTS["mu_charging"] / (2 * MOPTA_CONSTANTS["max_size"]))


mopta_solver = MOPTASolver(
    vehicle_locations=locations,
    loglevel=logging.INFO,
    service_level=0.95,
)

# compute number of initial locations
mopta_solver.add_initial_locations(n_clusters + 50, mode="k-means", seed=0)
mopta_solver.add_samples(num=1)

n, L, mip_gap, mip_gap_relative, iterations = mopta_solver.solve(verbose=False, timelimit=10, epsilon_stable=100)

objective_values, build_cost, distance_cost, service_levels, mip_gaps = mopta_solver.allocation_problem(
    L_sol=L, n_sol=n, n_iter=10
)
