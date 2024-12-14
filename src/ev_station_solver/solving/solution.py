import numpy as np
from docplex.mp.model import Model
from docplex.mp.sdetails import SolveDetails
from docplex.mp.solution import SolveSolution

from ev_station_solver.logging import get_logger
from ev_station_solver.solving.sample import Sample

# create logger
logger = get_logger(__name__)


class Solution:
    def __init__(self, sol_det: SolveDetails, sol: SolveSolution, m: Model, log: bool = True) -> None:
        # get solve information
        self.mip_gap = sol_det.gap  # obtain mip gap
        self.mip_gap_relative = sol_det.mip_relative_gap  # obtain relative mip gap

        self.costs = {kpi.name: round(m.kpi_value_by_name(name=kpi.name, solution=sol), 2) for kpi in m.iter_kpis()}

        # report both solutions
        self.kpis = self.get_kpis(model=m, solution=sol, log=log)

    def __repr__(self) -> str:
        return f"Solution(obj:{round(self.kpis['total_cost'], 2)}, mip_gap:{self.mip_gap}, mip_gap_r:{self.mip_gap_relative})"

    def __str__(self) -> str:
        return self.__repr__()

    def get_kpis(self, model: Model, solution: SolveSolution, log: bool):
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
    """Solution for the location improvement problem."""

    def __init__(
        self,
        v: np.ndarray,
        w: np.ndarray,
        u: list[np.ndarray],
        sol: SolveSolution,
        sol_det: SolveDetails,
        S: list[Sample],
        m: Model,
    ) -> None:
        """Create a location solution object.

        Args:
            v (np.ndarray): decision variable for building a charging location
            w (np.ndarray): decision variable for how many chargers to build at charging location
            u (list[np.ndarray]): allocatin solution
            sol (SolveSolution): docplex solution object
            sol_det (SolveDetails): docplex solve details object
            S (list[Sample]): list of samples
            m (Model): docplex model
        """
        super().__init__(sol_det=sol_det, sol=sol, m=m, log=True)

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

    def set_location_indice_sets(self) -> tuple[np.ndarray, np.ndarray]:
        """Obtain the indices of built and not built locations.


        Returns:
            tuple[np.ndarray, np.ndarray]: list of indices for built and not built locations
        """
        cl_built_indices = np.argwhere(self.v_sol == 1).flatten()
        cl_not_built_indices = np.argwhere(self.v_sol == 0).flatten()

        logger.debug(f"There are {len(cl_built_indices )} built and {len(cl_not_built_indices)} not built locations.")
        return cl_built_indices, cl_not_built_indices


class LocationSolution_linear(Solution):
    """Solution for the location improvement problem."""

    def __init__(
        self,
        v: np.ndarray,
        w: np.ndarray,
        u: list[np.ndarray],
        sol: SolveSolution,
        sol_det: SolveDetails,
        S: list[Sample],
        m: Model,
    ) -> None:
        """Create a location solution object.

        Args:
            v (np.ndarray): decision variable for building a charging location
            w (np.ndarray): decision variable for how many chargers to build at charging location
            u (list[np.ndarray]): allocatin solution
            sol (SolveSolution): docplex solution object
            sol_det (SolveDetails): docplex solve details object
            S (list[Sample]): list of samples
            m (Model): docplex model
        """
        super().__init__(sol_det=sol_det, sol=sol, m=m, log=True)

        # extract solution
        self.v_sol = np.array(sol.get_value_list(dvars=v)).round().astype(float)
        self.w_sol = np.array(sol.get_value_list(dvars=w)).round().astype(float)

        self.u_sol = {}
        for s in S:
            for i in s.I:
                for j in range(len(self.v_sol)):
                    self.u_sol[(s, s.I_s[i], j)] = sol.get_value(u[s.index][i, j])

        # set indice sets for solution
        self.cl_built_indices, self.cl_not_built_indices = self.set_location_indice_sets()

    def set_location_indice_sets(self) -> tuple[np.ndarray, np.ndarray]:
        """Obtain the indices of built and not built locations.


        Returns:
            tuple[np.ndarray, np.ndarray]: list of indices for built and not built locations
        """
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
        """Create a validation solution object.

        Args:
            u (np.ndarray): the allocation solution
            sol (SolveSolution): docplex solution object
            sol_det (SolveDetails): docplex solve details object
            s (Sample): sample object
            m (Model): docplex model
            service_level (float): attained service level
            desired_service_level (float): desired service level
        """
        super().__init__(sol_det=sol_det, sol=sol, m=m, log=False)

        # obtain u solutions
        self.u_sol = np.zeros(s.reachable.shape)
        self.u_sol[s.reachable] = np.array(sol.get_value_list(dvars=u[s.reachable].flatten()))
        self.u_sol = self.u_sol[-1].round().astype(float)

        # set sample
        self.s = s

        # set service level
        self.service_level = service_level
        self.feasible = service_level >= desired_service_level
