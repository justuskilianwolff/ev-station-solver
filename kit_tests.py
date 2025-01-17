import logging

from src.ev_station_solver.solving.linear_solver import LinearSolver
from src.ev_station_solver.loading import load_locations
from src.ev_station_solver.solving.solver import Solver
from src.ev_station_solver.logging import get_logger
import xpress as xp
logger = get_logger(__name__)

locations = load_locations("medium").sample(25).values
print(locations)
service_level = 0.95

ds = Solver(vehicle_locations=locations, loglevel=logging.INFO, service_level=service_level)

# compute number of initial locations
ds.add_initial_locations(15, mode="k-means", seed=0)
# ds.add_initial_locations(3, mode="random")
ds.add_samples(num=5)

location_solutions = ds.solve(verbose=False,
                timelimit=60,
                epsilon_stable=100)
# for s in ds.S:
#     print(s)
#     print(s.ranges)
# print(location_solutions[-1].u_sol)
# print(location_solutions[-1].v_sol)
# print(location_solutions[-1].w_sol)

ls = LinearSolver(vehicle_locations=locations, loglevel=logging.DEBUG, service_level=service_level)
ls.J = range(ls.n_vehicles)
ls.S = ds.S

# ls.add_samples(num=2)
ls.build_xpress_model(300)
for s in ls.S:
    print(s)
    print(s.ranges)

# ls.add_dvars()
# ls.add_constraints()
# ls.update_objective(K=ls.J)


# location_solutions = ls.solve()
# print(ls.sol)
# print(location_solutions.u_sol)
# print(location_solutions.v_sol)
# print(location_solutions.w_sol)
# ls.m.export_as_lp(basename="linear_model", path='C:\\Users\\ksearle\\OneDrive - University of Edinburgh\\Research\\location_chargers\\ev-station-solver\\src\\ev_station_solver')

