import logging
import time
from typing import Callable, Literal

import numpy as np
from docplex.mp.model import Model
from docplex.mp.sdetails import SolveDetails
from docplex.mp.solution import SolveSolution
from sklearn.cluster import KMeans
from tqdm import tqdm

from ev_station_solver.constants import MOPTA_CONSTANTS
from ev_station_solver.errors import IntegerInfeasible
from ev_station_solver.helper_functions import compute_maximum_matching, get_distance_matrix
from ev_station_solver.location_improvement import find_optimal_location
from ev_station_solver.logging import get_logger
from ev_station_solver.solving.sample import Sample
from ev_station_solver.solving.solution import LocationSolution

# create logger
logger = get_logger(__name__)


class Solver:
    def __init__(
        self,
        vehicle_locations: np.ndarray,
        loglevel: int = logging.INFO,
        build_cost: float = MOPTA_CONSTANTS["build_cost"],
        maintenance_cost: float = MOPTA_CONSTANTS["maintenance_cost"],
        drive_cost: float = MOPTA_CONSTANTS["drive_cost"],
        charge_cost: float = MOPTA_CONSTANTS["charge_cost"],
        service_level: float = MOPTA_CONSTANTS["service_level"],
        station_ub: int = MOPTA_CONSTANTS["station_ub"],
        queue_size: int = MOPTA_CONSTANTS["queue_size"],
        fixed_station_number: int | None = None,
        streamlit_callback: Callable | None = None,
    ):
        """
        Initialize the MOPTA solver with the given parameters.

        Common abbreviations:
        - dv: decision variable
        - lt: less than
        - cl: charging location
        - n: number

        :param vehicle_locations: the locations of the vehicles
        :param loglevel: logging level, e.g., logging.DEBUG, logging.INFO
        :param build_cost: the cost for building a location
        :param maintenance_cost: the cost for maintaining per charger at a location
        :param drive_cost: the cost per drive mile
        :param charge_cost: the cost per charged mile
        :param service_level: the percentage of vehicles that need to be charged
        :param station_ub: the number of vehicles, that can be charged at a location
        :param fixed_station_number: if wanted to specify the number of locations
        :param streamlit_callback: function to update streamlit user interface
        """
        logger.setLevel(level=loglevel)

        # Sanity checks:
        # vehicle locations are in R^2 and there are at least two vehicle locations
        if vehicle_locations.shape[0] == 1:
            raise ValueError("Please add more than one vehicle location.")
        if vehicle_locations.shape[1] != 2:
            raise ValueError("Please add two dimensional vehicle locations.")

        # Check whether service level is within (0,1]
        if service_level <= 0 or service_level > 1:
            raise ValueError("Service level should be within (0,1].")

        # set total vehicle locations
        self.vehicle_locations = vehicle_locations
        self.n_vehicles = len(self.vehicle_locations)

        # Tightest grid
        self.x_min = np.min(self.vehicle_locations[:, 0])  # most left vehicle
        self.x_max = np.max(self.vehicle_locations[:, 0])  # most right vehicle
        self.y_min = np.min(self.vehicle_locations[:, 1])  # most down vehicle
        self.y_max = np.max(self.vehicle_locations[:, 1])  # most up vehicle

        # charging locations
        self.coordinates_potential_cl: np.ndarray = np.empty((0, 2))  # charging locations
        self.n_potential_cl: int = 0  # number of charging locations
        self.J = range(self.n_potential_cl)  # indices of potential charging locations
        self.queue_size = queue_size

        # Samples
        self.S: list[Sample] = []

        # model and decision variables
        self.m = Model(name="Placement EV Chargers - Location Improvement", cts_by_name=True)  # docplex model
        self.v = np.empty((0,))  # binary decision variables whether to build cl or not
        self.w = np.empty((0,))  # integer decision variables how many to build at cl
        self.u = []  # binary decision variables for allocation per sample

        # cost terms
        self.station_ub = station_ub  # upper bound on number of stations
        self.build_cost_param = build_cost  # build cost
        self.maintenance_cost_param = maintenance_cost  # maintenance cost
        self.charge_cost_param = charge_cost  # charge cost
        self.drive_charge_cost_param = charge_cost + drive_cost  # drive + charge cost

        # objective terms (later add terms)
        self.build_cost = 0
        self.maintenance_cost = 0
        self.drive_charge_cost = 0
        self.fixed_charge_cost = 0
        self.total_cost = 0

        # constraints
        self.fixed_station_number = fixed_station_number  # fixed number of built cl
        self.fixed_station_number_constraint = None  # fixed number of stations constraint
        self.service_level = service_level  # service level
        self.service_constraints: list = []  # service constraints (at least XX% are serviced)
        self.v_lt_w_constraints: list = []  # v only positive if also w positive
        self.w_lt_mv_constraints: list = []  # w only positive if also v positive
        self.max_queue_constraints: list = []  # max queue length constraints
        self.allocation_constraints: list[list] = []  # allocation constraints (allocated to up to one charging station)

        # kpis
        self.kpi_build = None
        self.kpi_maintenance = None
        self.kpi_drive_charge = None
        self.kpi_avg_drive_distance = None
        self.kpi_fixed_charge = None
        self.kpi_total = None

        # solutions and added locations for each iteration
        self.solutions: list[LocationSolution] = []  # solutions for each iteration
        self.added_locations: list[np.ndarray] = []  # added locations in each iteration

        # streamlit
        self.streamlit_callback = streamlit_callback  # callback function for streamlit

    def add_initial_locations(
        self,
        n_stations: int,
        mode: Literal["random", "k-means"] = "random",
        verbose: int = 0,
        seed: int | None = None,
    ) -> None:
        """
        Add initial locations to the model
        :param n_stations: number of locations to add
        :param mode: random, k-means
        :param verbose: verbosity mode
        :param seed: seed for random state
        """

        if mode == "random":
            logger.debug("Adding random locations.")
            # random generator
            rng = np.random.default_rng(seed=seed)
            # scale random locations to grid
            new_locations = rng.random((n_stations, 2)) * np.array([self.x_max - self.x_min, self.y_max - self.y_min]) + np.array(
                [self.x_min, self.y_min]
            )

        elif mode == "k-means":
            logger.debug(f"Adding {n_stations} k-means locations.")
            kmeans = KMeans(n_clusters=n_stations, n_init=1, random_state=seed, verbose=verbose)
            new_locations = kmeans.fit(self.vehicle_locations).cluster_centers_

        else:
            raise Exception('Invalid mode for initial locations. Choose between "random" or "k-means"')

        # add new locations
        self.coordinates_potential_cl = np.concatenate((self.coordinates_potential_cl, new_locations))

        self.n_potential_cl = len(self.coordinates_potential_cl)
        self.J = range(self.n_potential_cl)

    def add_samples(self, num: int):
        def add_sample():
            """
            Adds a sample of charging values to the problem, which is used to optimise over.
            """
            logger.debug("Adding sample.")

            if self.coordinates_potential_cl is None:
                raise Exception("Please add initial locations before adding samples.")

            sample = Sample(
                index=len(self.S),
                total_vehicle_locations=self.vehicle_locations,
                coordinates_cl=self.coordinates_potential_cl,
            )

            # append to samples
            self.S.append(sample)

            # create empty u decision variables
            self.u.append(np.empty((sample.n_vehicles, 0)))
            # add empty list for allocation constraints
            self.allocation_constraints.append([])

        for _ in range(num):
            add_sample()

        logger.info(f"Added {num} samples. Total number of samples: {len(self.S)}.")

    def solve(
        self,
        epsilon_stable: float = 10e-2,
        counting_radius: float = MOPTA_CONSTANTS["counting_radius"],
        min_distance: float = MOPTA_CONSTANTS["min_distance"],
        timelimit: float | None = 10,
        verbose: bool = False,
    ) -> list[LocationSolution]:
        """
        Solves the optimization problem for EV charger placement.

        Parameters:
            epsilon_stable (float): The threshold for determining stability of the solution. Defaults to 10e-2.
            counting_radius (float): The radius within which locations are counted. Defaults to MOPTA_CONSTANTS["counting_radius"].
            min_distance (float): The minimum distance between built and not built locations. Defaults to MOPTA_CONSTANTS["min_distance"].
            timelimit (float): the maximum allowable time in seconds between successive solutions in the branch-and-cut tree. Defaults to 0.5s
            verbose (bool): Whether to log detailed output during the optimization routine. Defaults to False.

        Raises:
            ValueError: If the number of fixed locations is larger than the number of available locations.
            ValueError: If the service level cannot be reached with the given number of locations.
            ValueError: If the model is infeasible.

        Returns:
            list[LocationSolution]: A list of solutions for each iteration of the optimization routine
        """

        # sanity check for at least the number of fixed locations
        if (self.fixed_station_number is not None) and (self.fixed_station_number > self.n_potential_cl):
            raise ValueError(
                "Number of fixed locations is larger than the number of potential charging locations. Please add more locations."
            )
        else:
            # sanity check passed
            self.added_locations.append(self.coordinates_potential_cl)  # add initial locations

        # compute all maximum service levels to check for infeasibility
        # if one is below the minimum sla then raise
        max_service_levels = [
            compute_maximum_matching(w=np.repeat(self.station_ub, self.n_potential_cl), reachable=s.reachable) for s in self.S
        ]
        if min(max_service_levels) < self.service_level:
            raise ValueError("Service level cannot be attained with the given number of locations. Please add more locations.")

        logger.debug(f"Maximum service levels for samples: {max_service_levels}")

        # set solve parameters for cplex
        # turn presolve off to avoid issues after improvement
        self.m.parameters.preprocessing.presolve = 0  # type: ignore
        # stop after every found solution
        self.m.parameters.mip.limits.solutions = 1  # type: ignore

        # monitor number of iterations
        n_outer_iterations = 0
        # start the optimization routine
        logger.info("Starting the optimization routine.")
        # monitor time
        start_time = time.time()

        # intialize the model with inital locations
        self.add_new_decision_variables(K=self.J)
        self.update_constraints(K=self.J)
        self.update_objective(K=self.J)
        # set fixed charge cost()
        self.set_fixed_charge_cost()
        # update kpis
        self.update_kpis()

        # set current objective value as nan (not solved yet)
        current_objective_value = float("nan")

        while True:
            # update iterations number
            n_outer_iterations += 1

            # set timelimit per improvement iteration for future solves
            # This while loop runs until either
            # - the objective value does not increase
            # - no improved location is found

            # count inner iterations for logging and potential future improvement tracking
            n_inner_iterations = 0

            # start improving the current potential cl with our heuristic
            while True:  # run the inner loop while improvement between solutions is good enough
                # get solution and status
                sol = self.m.solve(log_output=verbose, clean_before_solve=False)

                # make sure solve details are set
                solve_details = self.m.solve_details
                if not isinstance(solve_details, SolveDetails):
                    raise Exception("Solve details are not set")

                status = solve_details.status  # get status
                n_inner_iterations += 1

                # check sol object
                if sol is None:
                    if status == "integer infeasible":
                        raise IntegerInfeasible
                    else:
                        raise Exception(f"Failed with {status}.")
                else:
                    sol.name = "Improvement"  # name solution for KPI reporting
                    if timelimit is not None:
                        # set time limit in second iteration if present (first solve might need a bit longer)
                        self.m.parameters.timelimit.set(timelimit)  # type: ignore
                        # set timelimit to none to not enter again
                        timelimit = None

                # solution was returned
                if status == "solution limit exceeded":
                    # new solution was found (not necesarily better) -> keep going
                    improvement = round(current_objective_value - self.m.objective_value, 2)
                    # set current objective value
                    current_objective_value = self.m.objective_value

                    logger.debug(f"Solution found, which is ${improvement} better. Continue with the next iteration.")

                    # logging all 4 iterations
                    if n_inner_iterations % 4 == 0:
                        rounded_objective_value = round(current_objective_value, 2)
                        logger.info(f"Solver is improving the solution, objective value: ${rounded_objective_value}")

                    # next solve iteration
                    continue

                elif status == "time limit exceeded":
                    # Since no improvement was found in the set time we continue with the location improvement
                    logger.info("Time limit exceeded. Continue with location improvement.")
                    break

                elif (status == "integer optimal, tolerance") or (status == "integer optimal solution"):
                    # if an optimal solution is found we can proceed with the location improvement
                    logger.info("Optimal solution found. Continue with location improvement.")
                    break
                else:
                    raise Exception(f"Solve ended with status: {status}")

            # extract current solution
            solution = LocationSolution(v=self.v, w=self.w, u=self.u, sol=sol, sol_det=solve_details, S=self.S, m=self.m)
            self.solutions.append(solution)

            # if a streamlit callback function was added -> call it
            if self.streamlit_callback is not None:
                self.streamlit_callback(self)
            # apply improvement heuristic
            mip_start = self.apply_improvement_heuristic(
                solution=solution, min_distance=min_distance, counting_radius=counting_radius, filter_locations=True
            )
            if not isinstance(mip_start, SolveSolution):
                # no improved locations found -> stop the optimization routine
                break

            # check if solution is stable -> There was no improvement compare to the last iteration
            # If it is stop the algorithm
            if self.check_stable(epsilon=epsilon_stable, warmstart=mip_start):
                logger.info("Solution is stable -> stopping the optimization routine.")
                # set solution to use tha mip start solution
                solution = LocationSolution(v=self.v, w=self.w, u=self.u, sol=sol, sol_det=solve_details, S=self.S, m=self.m)
                break

            # Add mipstart to model
            self.m.add_mip_start(mip_start, complete_vars=True, effort_level=4, write_level=3)

        # Optimization finished
        end_time = time.time()

        # Always return the solution with the best locations
        # That means the returned solution has no filtering applied
        final_solution_start = self.apply_improvement_heuristic(solution=solution, filter_locations=False)
        if not isinstance(final_solution_start, SolveSolution):
            raise ValueError("No new locations found, which should not have happened. Please check code.")

        final_solution = LocationSolution(
            v=self.v, w=self.w, u=self.u, sol=final_solution_start, sol_det=solve_details, S=self.S, m=self.m
        )

        # add to solutions
        self.solutions.append(final_solution)

        # clear model to free resources
        self.m.end()

        logger.info(f"Optimization finished in {round(end_time - start_time, 2)} seconds.")

        return self.solutions

    def apply_improvement_heuristic(
        self,
        solution: LocationSolution,
        min_distance: float | None = None,
        counting_radius: float | None = None,
        filter_locations: bool = False,
    ) -> None | SolveSolution:
        """Apply the improvement heuristic to the current solution.

        Args:
            solution (LocationSolution): current solution to the location improvement problem
            min_distance (float|None, optional): minimum distance for filtering. Defaults to None.
            counting_radius (float|None, optional): counting radius for filtering. Defaults to None.
            filter_locations (bool, optional): whether to filter locations or apply improvement to all built locations. Defaults to False.

        Raises:
            ValueError: min distance not set if filtering is applied
            ValueError: counting radius not set if filtering is applied

        Returns:
            None| SolveSolution]: if no new locations are found, return None. Otherwise, return the mip start with the new locations.
        """
        # compute for every built location its best location. Return that location and its indice
        new_potential_cl, relating_old_potential_cl_indices, cl_built_no_all_indices = self.find_improved_locations(
            built_indices=solution.cl_built_indices, u_sol=solution.u_sol
        )

        if filter_locations:
            # make sure counting and min dsitance are set
            if min_distance is None:
                raise ValueError("Set min distance if applying the filtering")
            if counting_radius is None:
                raise ValueError("Set counting_radius if applying the filtering")

            # filter locations that are built within a distance of a not built location
            new_potential_cl, relating_old_potential_cl_indices = self.filter_locations(
                improved_locations=new_potential_cl,
                old_location_indices=relating_old_potential_cl_indices,
                min_distance=min_distance,
                counting_radius=counting_radius,
            )

        # if no new locations found end the optimisation routine
        n_new_potential_cl = len(new_potential_cl)
        if n_new_potential_cl == 0:
            logger.info("No new locations found -> stopping the optimization routine.")
            return None

        else:  # add improved locations to solution
            self.added_locations.append(new_potential_cl)

        # update problem with new potential charging locations
        # set range for new potential charging locations
        K = range(self.n_potential_cl, self.n_potential_cl + n_new_potential_cl)

        # update locations
        self.coordinates_potential_cl = np.concatenate((self.coordinates_potential_cl, new_potential_cl))
        # update distances and reachable
        self.update_distances_reachable(n_new_cl=n_new_potential_cl, improved_locations=new_potential_cl, K=K)

        # update the model with new locations
        self.add_new_decision_variables(K=K)
        self.update_constraints(K=K)
        self.update_objective(K=K)
        self.update_kpis()

        # Update number of locations and location range
        self.n_potential_cl += n_new_potential_cl
        self.J = range(self.n_potential_cl)
        logger.info(f"{len(new_potential_cl)} improved new locations found. There are now {self.n_potential_cl} locations.")

        # mip start with new locations (allocate to improved)
        # generate new mip start
        mip_start = self.get_mip_start(
            u_sol=solution.u_sol,
            v_sol=solution.v_sol,
            w_sol=solution.w_sol,
            old_potential_cl_indices=relating_old_potential_cl_indices,
            cl_built_no_all_indices=cl_built_no_all_indices,
            n_new_potential_cl=n_new_potential_cl,
            K=K,
        )

        return mip_start

    def add_new_decision_variables(self, K: range):
        """Add new decision variables for new locations.

        Args:
            K (range): new locations added to the problem
        """
        logger.info("Adding new decision variables...")

        self.add_new_dv_v(K=K)
        self.add_new_dv_w(K=K)

        for s in self.S:
            self.add_new_dv_u_s(s=s, K=K)

    def add_new_dv_v(self, K: range) -> None:
        """Add binary variables v_k for each location k in K.

        Args:
            K (range): new locations added to the problem
        """

        self.v = np.append(self.v, np.array([self.m.binary_var(name=f"v_{k}") for k in K]))

    def add_new_dv_w(self, K: range) -> None:
        """Add integer variables w_k for each location k in K.

        Args:
            K (range): new locations added to the problem
        """
        self.w = np.append(self.w, np.array([self.m.integer_var(name=f"w_{k}") for k in K]))

    def add_new_dv_u_s(self, s: Sample, K: range) -> None:
        """Add binary variables u_{s, i, k} for each sample s, vehicle i and location k in K.

        Args:
            s (Sample): current sample
            K (range): new locations added to the problem
        """

        created_u_s = np.array([self.m.binary_var(name=f"u_{s}_{i}_{k}") if s.reachable[i, k] else 0 for i in s.I for k in K])
        created_u_s = created_u_s.reshape(s.n_vehicles, len(K))
        self.u[s.index] = np.concatenate((self.u[s.index], created_u_s), axis=1)

    def update_constraints(self, K: range) -> None:
        """Update constraints for new locations.

        Args:
            K (range): new locations added to the problem
        """
        logger.info("Updating constraints...")
        if self.fixed_station_number is not None:
            self.update_fixed_station_number_constraint(K=K)

        self.add_w_lt_mv_constraints(K=K)
        self.add_v_lt_w_constraints(K=K)

        for s in self.S:
            self.add_max_queue_constraints(s=s, K=K, queue_size=self.queue_size)
            self.update_service_constraint(s=s, K=K)
            self.update_allocation_constraints(s, K=K)

    def update_fixed_station_number_constraint(self, K: range) -> None:
        """Update the fixed station number constraint (include the new locations).

        Args:
            K (range): new locations added to the problem
        """
        # try to get the constraint by name
        constraint = self.m.get_constraint_by_name("fixed_station_number")
        left_sum_K = self.m.sum(self.v[k] for k in K)

        if constraint is None:
            # if not found then add it
            self.fixed_station_number_constraint = self.m.add_constraint(
                left_sum_K == self.fixed_station_number, ctname="fixed_station_number"
            )
        else:
            self.fixed_station_number_constraint = constraint.left_expr.add(left_sum_K)

    def add_w_lt_mv_constraints(self, K: range) -> None:
        """Adding the 'w <= m * v' constraints for new locations, where m is the upper bound of stations per station.
        This ensures that w is only positive if v is positive (if stations are built, location is built).

        Args:
            K (range): new locations added to the problem
        """
        new_w_lt_mv_constraints = self.m.add_constraints(
            (self.w[k] <= self.v[k] * self.station_ub for k in K), names=(f"number_w_{k}" for k in K)
        )
        self.w_lt_mv_constraints += new_w_lt_mv_constraints

    def add_v_lt_w_constraints(self, K: range) -> None:
        """Adding the 'v <= w' constraints for new locations. This ensures that v is only positive if w is positive (if location is built, stations are built).

        Args:
            K (range): new locations added to the problem
        """
        new_v_lt_mv_constraints = self.m.add_constraints((self.v[k] <= self.w[k] for k in K), names=(f"number_v_{k}" for k in K))
        self.v_lt_w_constraints += new_v_lt_mv_constraints

    def add_max_queue_constraints(self, s: Sample, K: range, queue_size: int) -> None:
        """Adding the 'allocated vehicles <= max queue' constraints for new locations.
        Vehicles are only allocated to a station if the station is built and max queue is not exceeded.

        Args:
            s (Sample): sample
            K (range): new locations added to the problem
            queue_size (int): allowed queue size at cl
        """
        new_max_queue_constraints = self.m.add_constraints(
            (self.m.sum(self.u[s.index][i, k] for i in s.I if s.reachable[i, k]) <= queue_size * self.w[k] for k in K),
            names=(f"allocation_qw_{s}_{k}" for k in K),
        )
        self.max_queue_constraints += new_max_queue_constraints

    def update_service_constraint(self, s: Sample, K: range) -> None:
        """Update the service constraint for the sample s (at least XX% of vehicles are allocated to a station).

        Args:
            s (Sample): sample
            K (range): new locations added to the problem
        """
        logger.debug(f"Adding service constraint (min. {self.service_level * 100}% of vehicles are allocated).")

        constraint = self.m.get_constraint_by_name(f"service_level_{s}")
        left_sum_K = self.m.sum(self.u[s.index][i, k] for i in s.I for k in K if s.reachable[i, k])

        if constraint is None:
            # does not exist yet
            self.service_constraints.append(
                self.m.add_constraint((left_sum_K >= self.service_level * s.n_vehicles), ctname=f"service_level_{s}"),
            )
        else:
            self.service_constraints[s.index] = constraint.left_expr.add(left_sum_K)

    def update_allocation_constraints(self, s: Sample, K: range) -> None:
        """Update the allocation constraints for the sample s (every vehicle is allocated to at most one station).

        Args:
            s (Sample): sample
            K (range): new locations added to the problem
        """
        logger.debug("Adding allocation constraints (every vehicle is allocated to at mosts one station).")
        for i in s.I:
            constraint = self.m.get_constraint_by_name(f"charger_allocation_{s}_{i}")
            left_sum_K = self.m.sum(self.u[s.index][i, k] for k in K if s.reachable[i, k])

            if constraint is None:
                self.allocation_constraints[s.index].append(
                    self.m.add_constraint((left_sum_K <= 1), ctname=(f"charger_allocation_{s}_{i}"))
                )

            else:
                self.allocation_constraints[s.index][i] = constraint.left_expr.add(left_sum_K)

    def update_objective(self, K: range) -> None:
        """Update the objective function with the new locations.

        Args:
            K (range): new locations added to the problem
        """
        self.add_to_build_cost(K=K)
        self.add_to_maintenance_cost(K=K)
        self.add_to_drive_charge_cost(K=K)

        # update total cost
        self.total_cost = (
            self.build_cost
            + self.maintenance_cost
            + 365 / len(self.S) * self.drive_charge_cost
            + 365 / len(self.S) * self.fixed_charge_cost
        )

        self.m.minimize(self.total_cost)
        logger.debug("Objective set.")

    def set_fixed_charge_cost(self):
        """Set the fixed charge cost summed across all samples"""
        # independent of K
        self.fixed_charge_cost = sum(s.get_fixed_charge_cost(charge_cost_param=self.charge_cost_param) for s in self.S)

    def add_to_build_cost(self, K: range):
        """Add the build cost for the new locations to the total build cost.

        Args:
            K (range): new locations added to the problem
        """
        self.build_cost += self.build_cost_param * self.m.sum(self.v[k] for k in K)

    def add_to_maintenance_cost(self, K: range):
        """Add the maintenance cost for the new locations to the total maintenance cost.

        Args:
            K (range): new locations added to the problem
        """
        self.maintenance_cost += self.maintenance_cost_param * self.m.sum(self.w[k] for k in K)

    def add_to_drive_charge_cost(self, K: range):
        """Add the drive charge cost (driving to the allocated charging locatin and charging for that amount).
        The amount to fill up to full range is added via add_fixed_charge_cost

        Args:
            K (range): new locations added to the problem
        """
        self.drive_charge_cost += sum(self.get_drive_charge_cost(s=s, K=K) for s in self.S)

    def get_drive_charge_cost(self, s: Sample, K: range):
        """Get the drive charge cost for the new locations for the current sample

        Args:
            s (Sample): sample
            K (range): new locations added to the problem

        Returns:
            np.ndarray: the drive_charge cost to the new locations
        """
        return self.drive_charge_cost_param * self.m.sum(
            self.u[s.index][i, k] * s.distance_matrix[i, k] for i in s.I for k in K if s.reachable[i, k]
        )

    def update_kpis(self):
        """Update the key performance indicators (KPIs) for the optimization problem with the most recent values."""
        # clear all kpis
        self.m.clear_kpis()

        # add new kpis
        self.kpi_total = self.m.add_kpi(self.total_cost, "total_cost")
        self.kpi_build = self.m.add_kpi(self.build_cost, "build_cost")
        self.kpi_maintenance = self.m.add_kpi(self.maintenance_cost, "maintenance_cost")
        self.kpi_drive_charge = self.m.add_kpi(365 / len(self.S) * self.drive_charge_cost, "drive_charge_cost")
        self.kpi_fixed_charge = self.m.add_kpi(365 / len(self.S) * self.fixed_charge_cost, "fixed_charge_cost")

        logger.debug("KPIs set.")

    def filter_locations(
        self,
        improved_locations: np.ndarray,
        old_location_indices: np.ndarray,
        min_distance: float = MOPTA_CONSTANTS["min_distance"],
        counting_radius: float = MOPTA_CONSTANTS["counting_radius"],
    ):
        """Filter locations that are too close to other locations.

        Args:
            improved_locations (np.ndarray): complete list of improved locations
            old_location_indices (np.ndarray): indices of old locations that are built
            min_distance (float, optional): min distance to existing cl. Defaults to MOPTA_CONSTANTS["min_distance"].
            counting_radius (float, optional): counting radius to count existing stations in. Defaults to MOPTA_CONSTANTS["counting_radius"].

        Returns:
            tuple[np.ndarray, np.ndarray]: improved locations and their old indices
        """
        distances = get_distance_matrix(improved_locations, self.coordinates_potential_cl).min(axis=1)
        build_mask = distances > min_distance
        too_close = np.argwhere(~build_mask).flatten()

        if len(too_close) == 0:
            logger.debug("No locations are too close to other locations. No filtering needed.")
            return improved_locations, old_location_indices
        else:
            # compute distances to all vehicles and compute how many are in radius
            distances_vehicles = get_distance_matrix(improved_locations[too_close], self.vehicle_locations)
            number_vehicles_in_radius = (distances_vehicles < counting_radius).sum(axis=1) * MOPTA_CONSTANTS[
                "mu_charging"
            ]  # multiply by expected charging prob
            # compute number of chargers in radius and how many are in radius
            distances_chargers = get_distance_matrix(improved_locations[too_close], self.coordinates_potential_cl)
            number_locations_radius = (distances_chargers < counting_radius).sum(axis=1) * 2 * self.station_ub

            # compute probability of adding a new one location
            # print(number_locations_radius)
            prob = np.zeros(len(number_locations_radius))
            for i in range(len(number_locations_radius)):
                if number_locations_radius[i] == 0:
                    prob[i] = 1
                else:
                    prob[i] = number_vehicles_in_radius[i] / number_locations_radius[i]

            build_mask[too_close] = np.random.uniform(size=len(too_close)) < prob
            logger.debug(f"The probabilities for building of chargers that are too close to others are {prob}.")
            return improved_locations[build_mask], old_location_indices[build_mask]

    def find_improved_locations(self, built_indices: np.ndarray, u_sol: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find improved locations for built chargers.

        Args:
            built_indices (np.ndarray): built indices of chargers
            u_sol (list): allocation solution

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: new_potential_cl, relating_old_potential_cl_indices, cl_built_no_all_indices
        """
        # create lists for improved locations and their old indices (used for warmstart)
        new_potential_cl = []
        relating_old_potential_cl_indices = []  # the list of cl that are built and have vehicles allocated
        cl_built_no_all_indices = []  # indices of cl that are built but have no vehicles allocated

        for j in tqdm(built_indices, desc="Finding improved locations for built chargers"):
            # find allocated vehicles and their ranges
            X_allocated = []
            ranges_allocated = []
            for s in self.S:
                indices_vehicles_s = np.argwhere(
                    u_sol[s.index][:, j] == 1
                ).flatten()  # indices of allocated vehicles to specific charger
                X_allocated.append(s.vehicle_locations[indices_vehicles_s])
                ranges_allocated.append(s.ranges[indices_vehicles_s])

            # combine them
            X_allocated = np.vstack(X_allocated)  # combine all vehicle locations from the different samples
            ranges_allocated = np.hstack(ranges_allocated)  # same for the ranges

            if len(X_allocated) != 0:  # if more than zero vehicles allocated to built charger
                # append new locations
                optimal_location = find_optimal_location(allocated_locations=X_allocated, allocated_ranges=ranges_allocated)
                distance_old = np.linalg.norm(optimal_location - self.coordinates_potential_cl[j])
                # move slightly if really close to old chager
                if distance_old < 10e-2:
                    optimal_location += np.random.normal(scale=0.3, size=2)

                new_potential_cl.append(optimal_location)
                relating_old_potential_cl_indices.append(j)
            else:
                # charger is built bot no vehicles are allocated
                cl_built_no_all_indices.append(j)

        # convert lists to numpy arrays
        new_potential_cl = np.array(new_potential_cl)
        relating_old_potential_cl_indices = np.array(relating_old_potential_cl_indices)
        cl_built_no_all_indices = np.array(cl_built_no_all_indices)

        return new_potential_cl, relating_old_potential_cl_indices, cl_built_no_all_indices

    def check_stable(self, warmstart: SolveSolution, epsilon: float = 10e-2) -> bool:
        """Check if the solution is stable.

        Args:
            warmstart (SolveSolution): the warmstart solution (the created solution with improved locations)
            epsilon (float, optional): The epsilon in objective value. Defaults to 10e-2.

        Returns:
            bool: whether the solution is stable or not
        """
        objective_warmstart = self.m.kpi_value_by_name(name="total_cost", solution=warmstart)
        if abs(self.solutions[-1].kpis["total_cost"] - objective_warmstart) <= epsilon:
            return True
        else:
            return False

    def update_distances_reachable(self, n_new_cl: int, improved_locations: np.ndarray, K: range):
        """Update the distances and reachable matrices for the new locations

        Args:
            n_new_cl (int): number of new locations
            improved_locations (np.ndarray): indices of new locations
            K (range): new locations added to the problem
        """
        for s in self.S:
            # add new distances
            s.distance_matrix = np.concatenate(
                (s.distance_matrix, get_distance_matrix(s.vehicle_locations, improved_locations)), axis=1
            )
            new_reachable = np.array([s.distance_matrix[i, k] <= s.ranges[i] for i in s.I for k in K]).reshape(
                s.n_vehicles, n_new_cl
            )
            s.reachable = np.concatenate((s.reachable, new_reachable), axis=1)

    def get_mip_start(
        self,
        u_sol: list,
        v_sol: np.ndarray,
        w_sol: np.ndarray,
        old_potential_cl_indices: np.ndarray,
        cl_built_no_all_indices: np.ndarray,
        n_new_potential_cl: int,
        K: range,
    ) -> SolveSolution:
        """Obtain a MIP start for the optimization problem.

        Args:
            u_sol (list): allocation solution
            v_sol (np.ndarray): v solution
            w_sol (np.ndarray): w solution
            old_potential_cl_indices (np.ndarray): old potential cl indices
            cl_built_no_all_indices (np.ndarray): cl built no all indices (built but no vehicles allocated)
            n_new_potential_cl (int): number of new potential cl
            K (range): new locations added to the problem

        Returns:
            SolveSolution: warmstart solution
        """
        # create start arrays with zeros for new locations
        v_start, w_start, u_start = self.create_mip_arrays(v_sol, w_sol, u_sol, n_new_potential_cl)

        # set new locations to built and copy their old n value
        v_start, w_start, u_start = self.set_mip_array_new_locations(
            v_start=v_start,
            w_start=w_start,
            u_start=u_start,
            w_sol=w_sol,
            u_sol=u_sol,
            old_potential_cl_indices=old_potential_cl_indices,
            K=K,
        )

        # check whether there are built locations that are empty
        v_start, w_start = self.set_built_but_empty_zero(
            v_start=v_start, w_start=w_start, cl_built_no_all_indices=cl_built_no_all_indices
        )

        # create mip solution from start arrays
        mip_start = self.create_mip_solution(v_start, w_start, u_start)

        return mip_start

    def create_mip_arrays(
        self, v_sol: np.ndarray, w_sol: np.ndarray, u_sol: list[np.ndarray], n_new_potential_cl: int
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """Create MIP arrays for the optimization problem for the decision variables.

        Args:
            v_sol (np.ndarray): v solution
            w_sol (np.ndarray): w solution
            u_sol (list[np.ndarray]): u solution
            n_new_potential_cl (int): number of new potential cl

        Returns:
            tuple[np.ndarray, np.ndarray, list[np.ndarray]]: v_start, w_start, u_start
        """
        v_start = np.concatenate((v_sol, np.zeros(n_new_potential_cl, dtype=float)))
        w_start = np.concatenate((w_sol, np.zeros(n_new_potential_cl, dtype=float)))
        u_start = []
        for s in self.S:
            u_start.append(
                np.concatenate((u_sol[s.index], np.zeros((s.n_vehicles, n_new_potential_cl), dtype=float)), axis=1, dtype=float)
            )
        return v_start, w_start, u_start

    def set_mip_array_new_locations(
        self,
        v_start: np.ndarray,
        w_start: np.ndarray,
        u_start: list[np.ndarray],
        w_sol: np.ndarray,
        u_sol: list[np.ndarray],
        old_potential_cl_indices: np.ndarray,
        K: range,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """In the MIP start, set the new locations to built and copy their old n value.

        Args:
            v_start (np.ndarray): v start
            w_start (np.ndarray): w start
            u_start (list[np.ndarray]): u start
            w_sol (np.ndarray): w solution for the old locations
            u_sol (list[np.ndarray]): u solution for the old locations
            old_potential_cl_indices (np.ndarray): indices of old potential cl
            K (range): range of new locations

        Returns:
            tuple[np.ndarray, np.ndarray, list[np.ndarray]]: v_start, w_start, u_start
        """
        v_start[K] = 1
        w_start[K] = w_sol[old_potential_cl_indices]
        # set old locations to not built
        v_start[old_potential_cl_indices] = 0
        w_start[old_potential_cl_indices] = 0
        # update u
        for s in self.S:
            for k, j in enumerate(old_potential_cl_indices):
                indices_vehicles = np.argwhere(u_sol[s.index][:, j] == 1).flatten()
                for i in indices_vehicles:
                    u_start[s.index][i, j] = 0
                    u_start[s.index][i, K[k]] = 1
                    if not s.reachable[i, K[k]]:
                        logger.warning(f"Vehicle {i} cannot reach location {K[k]}")
        return v_start, w_start, u_start

    def set_built_but_empty_zero(
        self, v_start: np.ndarray, w_start: np.ndarray, cl_built_no_all_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Set built locations with no vehicles allocated to 0, i.e. not built (v and w).

        Args:
            v_start (np.ndarray): v start
            w_start (np.ndarray): w start
            cl_built_no_all_indices (np.ndarray): old potential cl indices

        Returns:
            tuple[np.ndarray, np.ndarray]: v_start, w_start
        """
        if len(cl_built_no_all_indices) > 0:
            logger.info(f"Found {len(cl_built_no_all_indices)} built locations with no vehicles allocated -> set them to 0.")
            for i in cl_built_no_all_indices:
                v_start[i] = 0
                w_start[i] = 0

        return v_start, w_start

    def create_mip_solution(self, v_start: np.ndarray, w_start: np.ndarray, u_start: list[np.ndarray]) -> SolveSolution:
        """Create a MIP solution for the optimization problem from the start arrays.

        Args:
            v_start (np.ndarray): v start
            w_start (np.ndarray): w start
            u_start (list[np.ndarray]): u start

        Returns:
            SolveSolution: mip start solution for the optimization problem from the start arrays
        """
        # construct the MIP start with the arrays computed above
        mip_start = self.m.new_solution()
        # name solution
        mip_start.name = "Improvement Heuristic"
        for j in self.J:
            if v_start[j] == 1:
                if w_start[j] == 0:
                    logger.warning("Built location with n=0.")
                    continue  # skip built locations with n=0, because b should be set to 0 then
                else:
                    mip_start.add_var_value(self.v[j], v_start[j])
                    mip_start.add_var_value(self.w[j], w_start[j])

        for s in self.S:
            for u_dv, u_val in zip(self.u[s.index][s.reachable], u_start[s.index][s.reachable]):
                if u_val == 1:
                    mip_start.add_var_value(u_dv, u_val)

        return mip_start
