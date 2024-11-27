import logging

from ev_station_solver.constants import MOPTA_CONSTANTS
from ev_station_solver.loading import load_locations
from ev_station_solver.solving.solver import Solver
from ev_station_solver.solving.validator import Validator

# use given starting solutions
locations = load_locations("small").sample(100).values
n_clusters = int(len(locations) * MOPTA_CONSTANTS["mu_charging"] / (2 * MOPTA_CONSTANTS["station_ub"]))
service_level = 0.95

s = Solver(
    vehicle_locations=locations,
    loglevel=logging.INFO,
    service_level=service_level,
)

# compute number of initial locations
s.add_initial_locations(n_clusters, mode="k-means", seed=0)
s.add_initial_locations(n_clusters, mode="random")
s.add_samples(num=2)
s.solve()

best_sol = s.solutions[-1]  # take last solution (the one with optimal locations without filtering)

v = Validator(
    coordinates_cl=s.coordinates_potential_cl,
    vehicle_locations=locations,
    service_level=service_level,
    sol=best_sol,
)
print("finished")


# objective_values, build_cost, distance_cost, service_levels, mip_gaps = mopta_solver.allocation_problem(
#     locations_built=locations_built, v_sol_built=v_sol_built, n_iter=10
# )

# print("Test")
