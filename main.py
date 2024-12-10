import logging

from ev_station_solver.constants import CONSTANTS
from ev_station_solver.loading import load_locations
from ev_station_solver.logging import get_logger
from ev_station_solver.solving.solver import Solver
from ev_station_solver.solving.validator import Validator

logger = get_logger(__name__)


# use given starting solutions
locations = load_locations("small").sample(100).values
n_clusters = int(len(locations) * CONSTANTS["mu_charging"] / (2 * CONSTANTS["station_ub"]))
service_level = 0.95

s = Solver(vehicle_locations=locations, loglevel=logging.INFO, service_level=service_level)

# compute number of initial locations
s.add_initial_locations(n_clusters, mode="k-means", seed=0)
s.add_initial_locations(n_clusters, mode="random")
s.add_samples(num=2)
location_solutions = s.solve()

best_sol = location_solutions[-1]  # take last solution (the one with optimal locations without filtering)

v = Validator(coordinates_cl=s.coordinates_potential_cl, vehicle_locations=locations, sol=best_sol)
validation_solutions = v.validate(desired_service_level=service_level)
logger.info("Finished")
