import numpy as np
from docplex.mp.solution import SolveSolution

from ev_station_solver.logging import get_logger

# create logger
logger = get_logger(__name__)


class Solution:
    def __init__(self, v, w, sol: SolveSolution) -> None:
        logger.info("Extracting solution.")
        dtype = float  # desired dtype for numpy
        # extract solution
        self.v_sol = np.array(sol.get_value_list(dvars=v)).round().astype(dtype)
        self.w_sol = np.array(sol.get_value_list(dvars=w)).round().astype(dtype)

        self.u_sol = []
        for s in self.S_range:
            self.u_sol.append(np.zeros(self.u[s].shape))
            self.u_sol[s][self.S[s].reachability_matrix[i, k]] = np.array(
                sol.get_value_list(dvars=self.u[s][self.S[s].reachability_matrix[i, k]].flatten())
            )
            self.u_sol[s] = self.u_sol[s].round().astype(dtype)

        self.objective_value = sol.objective_value
        self.added_locations = []  # TODO fill

    def get_built_stations_indice_sets(self):
        return np.argwhere(b_sol == 1).flatten(), np.argwhere(b_sol == 0).flatten()
