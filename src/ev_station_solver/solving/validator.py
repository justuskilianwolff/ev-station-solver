import numpy as np
from docplex.mp.model import Model
from docplex.mp.sdetails import SolveDetails
from tqdm import tqdm

from ev_station_solver.constants import MOPTA_CONSTANTS
from ev_station_solver.helper_functions import compute_maximum_matching
from ev_station_solver.logging import get_logger
from ev_station_solver.solving.sample import Sample
from ev_station_solver.solving.solution import LocationSolution, ValidationSolution

# create logger
logger = get_logger(__name__)


class Validator:
    def __init__(
        self,
        coordinates_cl: np.ndarray,
        vehicle_locations: np.ndarray,
        sol: LocationSolution,
        drive_cost: float = MOPTA_CONSTANTS["drive_cost"],
        charge_cost: float = MOPTA_CONSTANTS["charge_cost"],
        queue_size: int = MOPTA_CONSTANTS["queue_size"],
        timelimit: int = 60,
    ):
        # vehicles
        self.vehicles_locations = vehicle_locations
        self.n_vehicles = self.vehicles_locations.shape[0]
        self.expected_number_vehicles = int(self.n_vehicles * MOPTA_CONSTANTS["mu_charging"])

        # chargin locations
        self.coordinates_cl = coordinates_cl[sol.cl_built_indices]
        self.w_sol = sol.w_sol[sol.cl_built_indices]
        self.n_cl = len(self.coordinates_cl)
        self.J = range(self.n_cl)
        self.queue_size: int = queue_size

        # solution
        self.sol = sol

        # params
        self.charge_cost_param = charge_cost
        self.drive_charge_cost_param = drive_cost + charge_cost

        # model
        self.m = Model("Validation")
        self.u = np.empty((0, self.n_cl))

        # constraints
        self.allocated_one_charger_constraints: list = []
        self.allocated_up_to_qn_constraints: list = []
        self.service_level_constraints: list = []

        # costs
        self.build_cost = self.sol.kpis["build_cost"]
        self.maintenance_cost = self.sol.kpis["maintenance_cost"]
        self.fixed_charge_cost = None
        self.drive_charge_cost = None

        # kpis
        self.kpi_maintenance = self.m.add_kpi(self.maintenance_cost, "maintenance_cost")  # constant
        self.kpi_build = self.m.add_kpi(self.build_cost, "build_cost")  # constant
        self.kpi_fixed_charge = None
        self.kpi_drive_charge = None
        self.kpi_total = None

        # solutions
        self.solutions: list[ValidationSolution] = []

        # since some decision variables in some samples have no effect -> turn off presolve
        self.m.parameters.preprocessing.presolve = 0  # type: ignore
        self.m.parameters.timelimit.set(timelimit)  # type: ignore

    def validate(self, desired_service_level: float, n_iter: int = 50) -> list[ValidationSolution]:
        """Validate the solution by sampling vehicles and checking if the service level is attainable.

        Args:
            desired_service_level (float): desired service level
            n_iter (int, optional): number of validation iterations. Defaults to 50.

        Raises:
            ValueError: if no solve details are obtained

        Returns:
            list[ValidationSolution]: list of validation solutions
        """
        logger.info(f"Starting allocation problem with {n_iter} iterations.")

        # set decision variables
        self.set_decision_variables()

        logger.info("Starting validation iterations.")
        for i in tqdm(range(n_iter), total=n_iter, desc="Validation Iterations"):
            # clear all constraints from the previous iteration
            self.m.clear_constraints()

            # sample one sample
            s = Sample(index=i, total_vehicle_locations=self.vehicles_locations, coordinates_cl=self.coordinates_cl)

            # compute attainable service level
            attainable_service_level = compute_maximum_matching(w=self.w_sol, queue_size=self.queue_size, reachable=s.reachable)
            if attainable_service_level >= desired_service_level:
                logger.debug("Service level is  attainable.")
                service_level = desired_service_level
            else:
                logger.debug("Service level is not attainable.")
                service_level = attainable_service_level

            # get decision variables for this sample
            u_sample = self.update_decision_variables(s=s)

            # set constraints
            self.add_allocated_to_charger_constrainst(u_sample=u_sample, s=s)
            self.add_allocated_to_qw_constrainst(u_sample=u_sample, s=s, queue_size=self.queue_size)
            self.add_service_level_constraint(u_sample=u_sample, s=s, service_level=service_level)

            self.update_objective(u_sample=u_sample, s=s)
            self.update_kpis()

            # solve the problem
            sol = self.m.solve(clean_before_solve=True)
            # obtain solve details
            sol_det = self.m.solve_details
            if not isinstance(sol_det, SolveDetails):
                raise ValueError("No solve details obtained...")

            # save solution
            validation_solution = ValidationSolution(
                u=u_sample,
                sol=sol,
                sol_det=sol_det,
                s=s,
                m=self.m,
                service_level=service_level,
                desired_service_level=desired_service_level,
            )
            self.solutions.append(validation_solution)

        # Clear model to free resources
        self.m.end()

        # categorize solutions
        feasible_solutions = [sol for sol in self.solutions if sol.feasible]
        infeasible_solutions = [sol for sol in self.solutions if not sol.feasible]

        # Result logging
        logger.info(f"Out of {n_iter} samples, {len(feasible_solutions)} are feasible.")

        # check that lists are actually not empty
        if len(feasible_solutions) != 0:
            mean_objective_value = np.mean([sol.kpis["total_cost"] for sol in feasible_solutions])
            logger.info(f"- Mean objective value (feasible): ${round(mean_objective_value, 2)}.")

        if len(infeasible_solutions) != 0:
            mean_objective_value = np.mean([sol.kpis["total_cost"] for sol in infeasible_solutions])
            mean_service_level = np.mean([sol.service_level for sol in infeasible_solutions]) * 100

            logger.info(
                f"- Mean objective value (infeasible): ${round(mean_objective_value, 2)} with a mean service level of {round(mean_service_level, 2)}%."
            )

        return self.solutions

    def set_decision_variables(self) -> None:
        """Set initial decision variables for the expected number of vehicles"""

        # create a general u for the expexted number of vehicles
        u = np.array(
            [self.m.binary_var(name=f"u_{i}_{j}") for i in range(self.expected_number_vehicles) for j in self.J]
        ).reshape(self.expected_number_vehicles, self.n_cl)

        self.u = u

        logger.info("Initial set of decision variables added.")

    def update_decision_variables(self, s: Sample) -> np.ndarray:
        """Update the decision variables if sample is larger than already existing dvs. Also it returns the reachable vars for this sample

        Args:
            s (Sample): current sample

        Returns:
            np.ndarray: current decision variables
        """
        # check if size of u is sufficient: if not -> extend u
        if s.n_vehicles > self.u.shape[0]:
            # append decision variables onto u
            size = s.n_vehicles - self.u.shape[0]
            new_u = np.array(
                [self.m.binary_var(name=f"u_{i}_{j}") for i in range(s.n_vehicles - size, s.n_vehicles) for j in self.J]
            ).reshape(size, self.n_cl)
            self.u = np.concatenate((self.u, new_u), axis=0)

        return np.where(s.reachable, self.u[: s.n_vehicles, :], 0)  # define u for this sample

    def add_allocated_to_charger_constrainst(self, u_sample: np.ndarray, s: Sample) -> None:
        """Add constraints to the model that vehicles are allocated to at most one charger.

        Args:
            u_sample (np.ndarray): decision variables for the current sample
            s (Sample): current sample
        """
        # allocated up to one charger
        constraints = self.m.add_constraints((self.m.sum(u_sample[i, j] for j in self.J) <= 1 for i in s.I))
        self.allocated_one_charger_constraints.append(constraints)

    def add_allocated_to_qw_constrainst(self, u_sample: np.ndarray, s: Sample, queue_size: int) -> None:
        """Add constraints to the model that vehicles are allocated to at most 2n chargers.

        Args:
            u_sample (np.ndarray): decision variables for the current sample
            s (Sample): current sample
            queue_size (int): queue size
        """
        # allocated up to 2n
        constraints = self.m.add_constraints(
            (self.m.sum(u_sample[i, j] for i in s.I) <= queue_size * self.w_sol[j] for j in self.J)
        )
        self.allocated_up_to_qn_constraints.append(constraints)

    def add_service_level_constraint(self, u_sample: np.ndarray, s: Sample, service_level: float) -> None:
        """Add constraints to the model that the service level is met.

        Args:
            u_sample (np.ndarray): decision variables for the current sample
            s (Sample): current sample
            service_level (float): service level
        """
        constraint = self.m.add_constraint(self.m.sum(u_sample) / s.n_vehicles >= service_level)
        self.service_level_constraints.append(constraint)

    def update_objective(self, u_sample: np.ndarray, s: Sample) -> None:
        """Update the objective function of the model.

        Args:
            u_sample (np.ndarray): current decision variables
            s (Sample): current sample
        """
        # update objective terms
        self.fixed_charge_cost = s.get_fixed_charge_cost(charge_cost_param=self.charge_cost_param)
        self.drive_charge_cost = self.drive_charge_cost_param * self.m.sum(u_sample * s.distance_matrix)

        self.m.minimize(self.build_cost + self.maintenance_cost + self.fixed_charge_cost + 365 * self.drive_charge_cost)

    def update_kpis(self) -> None:
        """Update the KPIs of the model with the current costs."""
        # clear all kpis
        self.m.clear_kpis()

        # add new kpis
        self.kpi_total = self.m.add_kpi(
            self.build_cost + self.maintenance_cost + 365 * self.drive_charge_cost + 365 * self.fixed_charge_cost,
            "total_cost",
        )
        self.kpi_build = self.m.add_kpi(self.build_cost, "build_cost")
        self.kpi_maintenance = self.m.add_kpi(self.maintenance_cost, "maintenance_cost")
        self.kpi_drive_charge = self.m.add_kpi(365 * self.drive_charge_cost, "drive_charge_cost")
        self.kpi_fixed_charge = self.m.add_kpi(365 * self.fixed_charge_cost, "fixed_charge_cost")

        logger.debug("KPIs set.")
