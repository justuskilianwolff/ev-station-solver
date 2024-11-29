import numpy as np
from docplex.mp.model import Model

from ev_station_solver.constants import MOPTA_CONSTANTS
from ev_station_solver.helper_functions import compute_maximum_matching
from ev_station_solver.logging import get_logger
from ev_station_solver.solving.sample import Sample
from ev_station_solver.solving.solution import Solution

# create logger
logger = get_logger(__name__)


class Validator:
    def __init__(
        self,
        coordinates_cl: np.ndarray,
        vehicle_locations: np.ndarray,
        sol: Solution,
        service_level: float,
        drive_cost: float = MOPTA_CONSTANTS["drive_cost"],
        charge_cost: float = MOPTA_CONSTANTS["charge_cost"],
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

        # solution
        self.sol = sol

        # params
        self.service_level = service_level
        self.drive_charge_cost_param = drive_cost + charge_cost
        # model
        self.m = Model("Validation")
        self.u = np.empty((0, self.n_cl))
        # since some decision variables in some samples have no effect -> turn off presolve
        self.m.parameters.preprocessing.presolve = 0  # type: ignore

    def validate(
        self,
        n_iter: int = 50,
        timelimit: int = 60,
        verbose: bool = False,
    ):
        # initialize model
        objective_values = []  # objective values of all solutions
        build_cost = []
        distance_cost = []
        service_levels = []  # service levels of all solutions
        mip_gaps = []  # mip gaps of all solutions

        logger.info(f"Starting allocation problem with {n_iter} iterations.")

        # set time limit
        self.m.parameters.timelimit.set(timelimit)  # type: ignore

        # get fixed terms
        build_cost = self.sol.kpis["build_cost"]
        maintenance_cost = self.sol.kpis["maintenance_cost"]
        fixed_charge_cost = self.sol.kpis["fixed_charge_cost"]
        # build_maintenance_term = self.maintenance_cost_param * np.sum(v_sol_built) + self.build_cost_param * len(
        #     v_sol_built
        # )

        u = self.set_decision_variables()

        for i in range(n_iter):
            logger.info(f"Allocation iteration {i + 1}/{n_iter}.")
            # clear all constraints from the previous iteration
            self.m.clear_constraints()

            # sample one sample
            s = Sample(index=i, total_vehicle_locations=self.vehicles_locations, coordinates_cl=self.coordinates_cl)

            # compute attainable service level
            attainable_sl = compute_maximum_matching(w=self.w_sol, reachable=s.reachable)
            service_level = self.service_level if attainable_sl >= self.service_level else attainable_sl

            logger.debug(
                f"  - Attainable service level: {round(attainable_sl * 100, 2)}% (set to {round(service_level * 100, 2)})"
            )

            # TODO: check if this is correct
            self.update_decision_variables(s=s)
            u_reachable = np.where(s.reachable, self.u[: s.n_vehicles, :], 0)  # define u for this sample
            # TODO: what is u_reachable?

            # allocated up to one charger
            self.m.add_constraints((self.m.sum(u_reachable[i, j] for j in self.J) <= 1 for i in s.I))

            logger.debug("  - Setting the 2 * n constraints.")
            # allocated up to 2n
            self.m.add_constraints((self.m.sum(u_reachable[i, j] for i in s.I) <= 2 * self.w_sol[j] for j in self.J))

            logger.debug(f"  - Setting the service level constraint to {round(service_level * 100, 2)}%.")
            self.m.add_constraint(self.m.sum(u_reachable) / s.n_vehicles >= service_level)

            logger.debug("  - Setting the objective function for the distance minimisation.")
            self.m.minimize(
                build_cost
                + maintenance_cost
                + fixed_charge_cost
                + 365 * self.drive_charge_cost_param * self.m.sum(u_reachable * s.distance_matrix)
            )

            logger.debug("  - Starting the solve process.")
            sol = self.m.solve(log_output=verbose, clean_before_solve=True)

            # report objective values
            objective_value = sol.objective_value
            # logger.debug(f"  - Objective value: ${round(objective_value, 2)}")
            # logger.debug(f"  - Build cost: ${round(build_maintenance_term, 2)}")
            # logger.debug(f"  - Constant term: ${round(constant_term, 2)}")
            # logger.debug(f"  - Distance cost: ${round(objective_value - constant_term - build_maintenance_term, 2)}")

            # TODO: subclass solution to hold also vlaidation resukt
            # add values to lists
            objective_values.append(sol.objective_value)
            build_cost.append(build_maintenance_term)
            distance_cost.append(objective_value - constant_term - build_maintenance_term)
            service_levels.append(service_level)
            mip_gaps.append(m.solve_details.gap)

        # Clear model to free resources
        self.m.end()

        # convert to numpy arrays
        objective_values = np.array(objective_values)
        service_levels = np.array(service_levels)
        mip_gaps = np.array(mip_gaps)

        i_infeasible = np.argwhere(service_levels < self.service_level).flatten()
        feasible = np.argwhere(service_levels >= self.service_level).flatten()

        # Result logging
        logger.info(f"Out of {n_iter} samples, {len(feasible)} are feasible.")
        # check that lists are actually not empty
        if len(feasible) != 0:
            logger.info(f"- Mean objective value (feasible): ${np.round(np.mean(objective_values[feasible]), 2)}.")
        if len(i_infeasible) != 0:
            logger.info(
                f"- Mean objective value (infeasible): ${np.round(np.mean(objective_values[i_infeasible]), 2)} with a mean service level "
                f"of {np.round(np.mean(service_levels[i_infeasible]) * 100, 2)}%."
            )

        return objective_values, build_cost, distance_cost, service_levels, mip_gaps

    def set_decision_variables(self):
        logger.info("Creating decision variables")
        # create a general u for the expexted number of vehicles
        u = np.array(
            [self.m.binary_var(name=f"u_{i}_{j}") for i in range(self.expected_number_vehicles) for j in self.J]
        ).reshape(self.expected_number_vehicles, self.n_cl)

        logger.info("Decision variables added.")

        self.u = u

    def update_decision_variables(self, s: Sample):
        # set up ranges for problem
        # check if size of u is sufficient: if not -> extend u
        if s.n_vehicles > self.u.shape[0]:
            # append decision variables onto u
            size = s.n_vehicles - self.u.shape[0]
            new_u = np.array(
                [self.m.binary_var(name=f"u_{i}_{j}") for i in range(s.n_vehicles - size, s.n_vehicles) for j in self.J]
            ).reshape(size, self.n_cl)
            self.u = np.concatenate((self.u, new_u), axis=0)
