import numpy as np
from src.ev_station_solver.logging import get_logger
from src.ev_station_solver.solving.sample import Sample
from src.ev_station_solver.solving.solution import LocationSolution_linear
from src.ev_station_solver.solving.solver import Solver

# create logger
logger = get_logger(__name__)


class LinearSolver(Solver):
    def add_new_dv_y_s(self, s: Sample, K: range):
        """Add binary variables u_{s, i, k} for each sample s, vehicle i and location k in K.

        Args:
            s (Sample): current sample
            K (range): new locations added to the problem
        """

        created_y_s = np.array(
            [self.m.continuous_var(name=f"y_{s}_{i}_{k}", lb=0) for i in range(self.n_vehicles) if s.charging[i] for k in K]
        )
        created_y_s = created_y_s.reshape(s.n_vehicles, len(K))
        self.y[s.index] = np.concatenate((self.y[s.index], created_y_s), axis=1)

    def add_new_dv_u_s(self, s: Sample, K: range):
        """Add binary variables u_{s, i, k} for each sample s, vehicle i and location k in K.

        Args:
            s (Sample): current sample
            K (range): new locations added to the problem
        """
        #
        created_u_s = np.array(
            [self.m.binary_var(name=f"u_{s}_{i}_{k}") for i in range(self.n_vehicles) if s.charging[i] for k in K]
        )
        created_u_s = created_u_s.reshape(s.n_vehicles, len(K))
        self.u[s.index] = np.concatenate((self.u[s.index], created_u_s), axis=1)

    def add_new_dv_d(self, I: range, K: range):
        """Add binary variables u_{s, i, k} for each sample s, vehicle i and location k in K.

        Args:
            s (Sample): current sample
            K (range): new locations added to the problem
        """

        created_d = np.array([self.m.continuous_var(name=f"d_{i}_{k}", lb=0) for i in I for k in K])
        self.d = created_d.reshape(len(I), len(K))

    def add_new_dv_d_root(self, I: range, K: range):
        """Add binary variables u_{s, i, k} for each sample s, vehicle i and location k in K.

        Args:
            s (Sample): current sample
            K (range): new locations added to the problem
        """

        created_d_root = np.array([self.m.continuous_var(name=f"droot_{i}_{k}", lb=0) for i in I for k in K])
        self.d_root = created_d_root.reshape(len(I), len(K))

    def add_new_dv_x(self, K: range):
        """Add binary variables u_{s, i, k} for each sample s, vehicle i and location k in K.

        Args:
            s (Sample): current sample
            K (range): new locations added to the problem
        """
        max_loc = [self.x_max, self.y_max]
        min_loc = [self.x_min, self.y_min]

        created_x = np.array(
            [self.m.continuous_var(name=f"x_{k}_{l}", lb=min_loc[l], ub=max_loc[l]) for k in K for l in range(2)]
        )
        self.x = created_x.reshape(len(K), 2)

    def add_max_queue_constraints(self, s: Sample, K: range, queue_size: int) -> None:
        """Adding the 'allocated vehicles <= max queue' constraints for new locations.
        Vehicles are only allocated to a station if the station is built and max queue is not exceeded.

        Args:
            s (Sample): sample
            K (range): new locations added to the problem
            queue_size (int): allowed queue size at cl
        """
        new_max_queue_constraints = self.m.add_constraints(
            (self.m.sum(self.u[s.index][i, k] for i in s.I) <= queue_size * self.w[k] for k in K),
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
        left_sum_K = self.m.sum(self.u[s.index][i, k] for i in s.I for k in K)

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
            # constraint = self.m.get_constraint_by_name(f"charger_allocation_{s}_{s.I_s[i]}")
            left_sum_K = self.m.sum(self.u[s.index][i, k] for k in K)

            # if constraint is None:
            # self.allocation_constraints[s.index].append(
            self.m.add_constraint((left_sum_K <= 1), ctname=(f"charger_allocation_{s}_{s.I_s[i]}"))
            # )

            # else:
            # self.allocation_constraints[s.index][i] = constraint.left_expr.add(left_sum_K)

    def add_range_constraints(self, s: Sample, K: range) -> None:
        """Adding the 'distance <= range*allocation' constraints .
        Vehicles are only allocated to a station if they are within range.

        Args:
            s (Sample): sample
            K (range): new locations added to the problem

        """
        new_range_constraints = self.m.add_constraints(
            (self.y[s.index][i, k] <= (s.ranges[i] ** 2) * self.u[s.index][i, k] for i in s.I for k in K),
            names=(f"range_{s}_{i}_{k}" for i in s.I_s for k in K),
        )

    def add_distance_range_constraints(self, s: Sample, K: range) -> None:
        """Adding the 'distance <= range*allocation' constraints .
        Vehicles are only allocated to a station if they are within range.

        Args:
            s (Sample): sample
            K (range): new locations added to the problem

        """
        new_distance_range_constraints = self.m.add_constraints(
            (
                (s.ranges[i] ** 2) * self.u[s.index][i, k] + self.d_root[s.I_s[i], k] - s.ranges[i] ** 2 <= self.y[s.index][i, k]
                for i in s.I
                for k in K
            ),
            names=(f"distance_range_{s}_{i}_{k}" for i in s.I_s for k in K),
        )

    def add_distance_uperbound_constraints(self, s: Sample, K: range) -> None:
        """Adding the 'distance <= range*allocation' constraints .
        Vehicles are only allocated to a station if they are within range.

        Args:
            s (Sample): sample
            K (range): new locations added to the problem

        """
        new_distance_range_constraints = self.m.add_constraints(
            (self.y[s.index][i, k] <= self.d_root[s.I_s[i], k] for i in s.I for k in K),
            names=(f"distance_uperbound_{s}_{i}_{k}" for i in s.I_s for k in K),
        )

    def add_euclidian_distance_constraints(self, I: range, K: range) -> None:
        """Adding the 'distance <= range*allocation' constraints .
        Vehicles are only allocated to a station if they are within range.

        Args:
            s (Sample): sample
            K (range): new locations added to the problem

        """

        self.m.add_quadratic_constraints(
            self.d_root[i, k]
            >= (self.x[k, 0] - self.vehicle_locations[i, 0]) ** 2 + (self.x[k, 1] - self.vehicle_locations[i, 1]) ** 2
            for i in I
            for k in K
        )

        # self.m.add_quadratic_constraints(
        #      self.d[i, k]**2 <= self.d_root[i, k] for i in I for k in K
        # )

    def add_new_decision_variables(self):
        ## initalise decision variables not inherited
        self.y = []
        self.u = []
        self.d = []
        self.d_root = []
        self.x = []

        ### add decisdion variables
        self.add_new_dv_v(K=self.J)
        self.add_new_dv_w(K=self.J)
        self.add_new_dv_d(I=range(self.n_vehicles), K=self.J)
        self.add_new_dv_d_root(I=range(self.n_vehicles), K=self.J)
        self.add_new_dv_x(K=self.J)

        for s in self.S:
            self.u.append(np.empty((s.n_vehicles, 0)))
            self.y.append(np.empty((s.n_vehicles, 0)))
            self.add_new_dv_u_s(s=s, K=self.J)
            self.add_new_dv_y_s(s=s, K=self.J)

    def update_constraints(self):
        self.add_w_lt_mv_constraints(K=self.J)
        self.add_v_lt_w_constraints(K=self.J)

        for s in self.S:
            self.add_max_queue_constraints(s=s, K=self.J, queue_size=self.queue_size)
            self.update_service_constraint(s=s, K=self.J)
            self.update_allocation_constraints(s=s, K=self.J)
            self.add_range_constraints(s=s, K=self.J)
            self.add_distance_range_constraints(s=s, K=self.J)
            self.add_distance_uperbound_constraints(s=s, K=self.J)

        self.add_euclidian_distance_constraints(I=range(self.n_vehicles), K=self.J)

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
        return self.drive_charge_cost_param * self.m.sum(self.m.sum(self.y[s.index][i, j] for i in s.I for j in self.J))

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

    def solve(self):
        # self.m = CpoModel(name='buses')
        # intialize the model with inital locations
        self.m.parameters.timelimit.set(60)
        self.add_new_decision_variables()

        self.update_constraints()
        self.update_objective(K=self.J)
        # set fixed charge cost()
        self.set_fixed_charge_cost()
        # update kpis
        self.update_kpis()

        self.sol = self.m.solve(log_output=True)
        solve_details = self.m.solve_details
        solution = LocationSolution_linear(v=self.v, w=self.w, u=self.u, sol=self.sol, sol_det=solve_details, S=self.S, m=self.m)
        return solution
