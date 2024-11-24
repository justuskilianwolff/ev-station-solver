import numpy as np
from docplex.mp.model import Model

from ev_station_solver.logging import get_logger

# create logger
logger = get_logger(__name__)


class Validator:
    def allocation_problem(
        self,
        locations_built: np.ndarray,
        v_sol_built: np.ndarray,
        verbose: bool = False,
        n_iter: int = 50,
        timelimit: int = 60,
    ):
        # initialize model
        objective_values = []  # objective values of all solutions
        build_cost = []
        distance_cost = []
        service_levels = []  # service levels of all solutions
        mip_gaps = []  # mip gaps of all solutions

        build_maintenance_term = self.maintenance_cost_param * np.sum(v_sol_built) + self.build_cost_param * len(
            v_sol_built
        )

        w = len(locations_built)
        J = range(w)
        logger.info(f"Starting allocation problem with {n_iter} iterations.")

        # Create model once and then update it
        m_a = Model("Allocation Problem")
        expected_number_vehicles = int(self.n_vehicles * MOPTA_CONSTANTS["mu_charging"])

        logger.info("Creating decision variables")
        # create a general u for the expexted number of vehicles
        u = np.array([m_a.binary_var(name=f"u_{i}_{j}") for i in range(expected_number_vehicles) for j in J]).reshape(
            expected_number_vehicles, w
        )

        logger.info("Decision variables added.")

        # since some decision variables in some samples have no effect -> turn off presolve
        m_a.parameters.preprocessing.presolve = 0
        # set time limit
        m_a.parameters.timelimit.set(timelimit)

        for i in range(n_iter):
            logger.info(f"Allocation iteration {i + 1}/{n_iter}.")
            # clear all constraints from the previous iteration
            m_a.clear_constraints()

            # sample one sample
            ranges, charging_prob, charging = self.get_sample()
            logger.debug("  - Sample generated.")

            # filter for vehicles that are charging
            ranges = ranges[charging]
            locations = self.vehicle_locations[charging]
            distances = get_distance_matrix(locations, locations_built)
            reachable = (distances.T <= ranges).T

            # compute attainable service level
            logger.debug("  - Checking what service level is attainable.")
            attainable_service_level = compute_maximum_matching(n=v_sol_built, reachable=reachable)
            service_level = (
                self.service_level if attainable_service_level >= self.service_level else attainable_service_level
            )

            logger.debug(
                f"  - Attainable service level: {round(attainable_service_level * 100, 2)}% "
                f"(set to {round(service_level * 100, 2)})"
            )

            # set up ranges for problem
            l = charging.sum()
            I = range(l)

            # check if size of u is sufficient: if not -> extend u
            if l > u.shape[0]:
                # append decision variables onto u
                size = l - u.shape[0]
                new_u = np.array([m_a.binary_var(name=f"u_{i}_{j}") for i in range(l - size, l) for j in J]).reshape(
                    size, w
                )
                u = np.concatenate((u, new_u), axis=0)

            u_reachable = np.where(reachable, u[:l, :], 0)  # define u for this sample

            # Add constraints to it
            logger.debug("  - Setting the allocation constraints.")
            m_a.add_constraints((m_a.sum(u_reachable[i, j] for j in J) <= 1 for i in I))  # allocated up to one charger

            logger.debug("  - Setting the 2 * n constraints.")
            m_a.add_constraints(
                (m_a.sum(u_reachable[i, j] for i in I) <= 2 * v_sol_built[j] for j in J)
            )  # allocated up to 2n

            logger.debug(f"  - Setting the service level constraint to {round(service_level * 100, 2)}%.")
            m_a.add_constraint(m_a.sum(u_reachable) / l >= service_level)

            logger.debug("  - Setting the objective function for the distance minimisation.")
            constant_term = self.charge_cost_param * 365 * (250 - ranges).sum()
            m_a.minimize(
                365 * self.drive_charge_cost_param * m_a.sum(u_reachable * distances)
                + build_maintenance_term
                + constant_term
            )

            logger.debug("  - Starting the solve process.")
            sol = m_a.solve(log_output=verbose, clean_before_solve=True)

            # report objective values
            objective_value = sol.objective_value
            logger.debug(f"  - Objective value: ${round(objective_value, 2)}")
            logger.debug(f"  - Build cost: ${round(build_maintenance_term, 2)}")
            logger.debug(f"  - Constant term: ${round(constant_term, 2)}")
            logger.debug(f"  - Distance cost: ${round(objective_value - constant_term - build_maintenance_term, 2)}")

            # add values to lists
            objective_values.append(sol.objective_value)
            build_cost.append(build_maintenance_term)
            distance_cost.append(objective_value - constant_term - build_maintenance_term)
            service_levels.append(service_level)
            mip_gaps.append(m_a.solve_details.gap)

        # Clear model to free resources
        m_a.end()

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
