import numpy as np
from docplex.mp.sdetails import SolveDetails
from docplex.mp.solution import SolveSolution

from ev_station_solver.logging import get_logger
from ev_station_solver.solving.sample import Sample

# create logger
logger = get_logger(__name__)


class Solution:
    def __init__(self, v, w, u, sol: SolveSolution, sol_det: SolveDetails, S: list[Sample]) -> None:
        dtype = float  # desired dtype for numpy
        # extract solution
        self.v_sol = np.array(sol.get_value_list(dvars=v)).round().astype(dtype)
        self.w_sol = np.array(sol.get_value_list(dvars=w)).round().astype(dtype)

        self.u_sol = []
        for s in S:
            self.u_sol.append(np.zeros(s.reachable.shape))
            self.u_sol[-1][s.reachable] = np.array(sol.get_value_list(dvars=u[s.index][s.reachable].flatten()))
            self.u_sol[-1] = self.u_sol[-1].round().astype(dtype)

        # get solve information
        self.objective_value = sol.objective_value
        self.mip_gap = sol_det.gap  # obtain mip gap
        self.mip_gap_relative = sol_det.mip_relative_gap  # obtain relative mip gap

        # set indice sets for solution
        self.cl_built_indices, self.cl_not_built_indices = self.set_location_indice_sets()
        self.added_locations = None

        # set cost terms

    def set_location_indice_sets(self):
        cl_built_indices = np.argwhere(self.v_sol == 1).flatten()
        cl_not_built_indices = np.argwhere(self.v_sol == 0).flatten()

        logger.debug(f"There are {len(cl_built_indices )} built and {len(cl_not_built_indices)} not built locations.")
        return cl_built_indices, cl_not_built_indices
