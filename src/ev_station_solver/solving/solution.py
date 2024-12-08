import numpy as np
from docplex.mp.model import Model
from docplex.mp.sdetails import SolveDetails
from docplex.mp.solution import SolveSolution

from ev_station_solver.logging import get_logger
from ev_station_solver.solving.sample import Sample

# create logger
logger = get_logger(__name__)


class Solution:
    def __init__(self, sol_det: SolveDetails, sol: SolveSolution, m: Model) -> None:
        # get solve information
        self.mip_gap = sol_det.gap  # obtain mip gap
        self.mip_gap_relative = sol_det.mip_relative_gap  # obtain relative mip gap

        self.costs = {kpi.name: round(m.kpi_value_by_name(name=kpi.name, solution=sol), 2) for kpi in m.iter_kpis()}

        # report both solutions
        self.kpis = self.get_kpis(model=m, solution=sol)

    def __repr__(self) -> str:
        return f"Solution(obj:{round(self.kpis['total_cost'], 2)}, mip_gap:{self.mip_gap}, mip_gap_r:{self.mip_gap_relative})"

    def __str__(self) -> str:
        return self.__repr__()

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


class LocationSolution(Solution):
    def __init__(self, v, w, u, sol: SolveSolution, sol_det: SolveDetails, S: list[Sample], m: Model) -> None:
        super().__init__(sol_det=sol_det, sol=sol, m=m)

        # extract solution
        self.v_sol = np.array(sol.get_value_list(dvars=v)).round().astype(float)
        self.w_sol = np.array(sol.get_value_list(dvars=w)).round().astype(float)

        self.u_sol = []
        for s in S:
            self.u_sol.append(np.zeros(s.reachable.shape))
            self.u_sol[-1][s.reachable] = np.array(sol.get_value_list(dvars=u[s.index][s.reachable].flatten()))
            self.u_sol[-1] = self.u_sol[-1].round().astype(float)

        # set indice sets for solution
        self.cl_built_indices, self.cl_not_built_indices = self.set_location_indice_sets()

    def set_location_indice_sets(self):
        cl_built_indices = np.argwhere(self.v_sol == 1).flatten()
        cl_not_built_indices = np.argwhere(self.v_sol == 0).flatten()

        logger.debug(f"There are {len(cl_built_indices )} built and {len(cl_not_built_indices)} not built locations.")
        return cl_built_indices, cl_not_built_indices


class ValidationSolution(Solution):
    def __init__(
        self,
        u: np.ndarray,
        sol: SolveSolution,
        sol_det: SolveDetails,
        s: Sample,
        m: Model,
        service_level: float,
        desired_service_level: float,
    ) -> None:
        super().__init__(sol_det=sol_det, sol=sol, m=m)

        # obtain u solutions
        self.u_sol = np.zeros(s.reachable.shape)
        self.u_sol[s.reachable] = np.array(sol.get_value_list(dvars=u[s.reachable].flatten()))
        self.u_sol = self.u_sol[-1].round().astype(float)

        # set sample
        self.s = s

        # set service level
        self.service_level = service_level
        self.feasible = service_level >= desired_service_level
