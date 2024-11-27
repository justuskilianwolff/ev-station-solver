import numpy as np
from docplex.mp.model import Model
from docplex.mp.sdetails import SolveDetails
from docplex.mp.solution import SolveSolution

from ev_station_solver.logging import get_logger
from ev_station_solver.solving.sample import Sample

# create logger
logger = get_logger(__name__)


class Solution:
    def __init__(self, v, w, u, sol: SolveSolution, sol_det: SolveDetails, S: list[Sample], m: Model) -> None:
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

        # costs as dict
        self.costs = {kpi.name: round(m.kpi_value_by_name(name=kpi.name, solution=sol), 2) for kpi in m.iter_kpis()}

        # report both solutions
        self.kpis = self.get_kpis(model=m, solution=sol)

    def __repr__(self) -> str:
        return f"Solution(obj:{round(self.objective_value, 2)}, mip_gap:{self.mip_gap}, mip_gap_r:{self.mip_gap_relative}, n_pot_cl:{len(self.v_sol)})"

    def __str__(self) -> str:
        return self.__repr__()

    def set_location_indice_sets(self):
        cl_built_indices = np.argwhere(self.v_sol == 1).flatten()
        cl_not_built_indices = np.argwhere(self.v_sol == 0).flatten()

        logger.debug(f"There are {len(cl_built_indices )} built and {len(cl_not_built_indices)} not built locations.")
        return cl_built_indices, cl_not_built_indices

    def get_kpis(self, model: Model, solution: SolveSolution, log: bool = True):
        # build dict
        kpis = {}

        for kpi in model.iter_kpis():
            kpis[kpi.name] = kpi.solution_value

        # if log
        if log:
            logger.info(f"KPIs {solution.name}:")
            for kpi_name, kpi_value in kpis.items():
                logger.info(f"  - {kpi_name}: {round(kpi_value, 2)}")

        return kpis
