import logging

from src.ev_station_solver.solving.linear_solver import LinearSolver
from src.ev_station_solver.loading import load_locations
from src.ev_station_solver.solving.solver import Solver
from src.ev_station_solver.logging import get_logger
from src.ev_station_solver.constants import CONSTANTS
import time
import numpy as np
import pandas as pd
logger = get_logger(__name__)

locations = load_locations("large").values
print('open')
percentages = [0.5, .75, 1]
values_t = np.array([1, 5, 15]) * 60 #np.array([1, 5, 15, 30, 60]) * 60  # in minutes
# values_locations = np.array([1, 1.5, 2, 5])  # percentage to multiply initial locations with
# num_samples = [1,2]#[1,2,3,4,5,6,7,8,9,10]
df = pd.DataFrame(
    columns=['T', 'num_vehicales', 'Objective_Value', 'Solve_Time', 'MIP_Gap', 'MIP_Gap_Relative', 'Iterations', 'build_cost',
    'distance_cost', 'Number_End_Locations']
)

df_exact = pd.DataFrame(
    columns=['T', 'num_vehicales', 'Objective_Value', 'Solve_Time', 'MIP_Gap']
)

# df_val = pd.DataFrame(columns=[
#     'sample',
#     'n_iter_sample',
#     'objective_values',
#     'build_cost',
#     'distance_cost',
#     'service_levels',
#     'mip_gap',
# ])
percentage = 1
value_location = 1.1
sample = 5

for percentage in percentages:
    max_time = 0
    for t in values_t:
        print(f'T = {t}')
        np.random.seed(0)
        selected_locations = locations[np.random.randint(0, len(locations), size=int(percentage * len(locations)))]
        n_clusters = np.ceil(len(selected_locations) * CONSTANTS['mu_charging'] /
                             (2 * CONSTANTS['station_ub']) * 2 * value_location)
        
        mopta_solver = Solver(
            vehicle_locations=selected_locations,
            loglevel=logging.INFO,
            service_level=.95,
        )
        # compute number of initial locations
        mopta_solver.add_initial_locations(int(n_clusters), mode='k-means', seed=0)
        mopta_solver.add_samples(num=sample)

        start = time.time()
        n, L, mip_gap, mip_gap_relative, iterations = mopta_solver.solve(
            verbose=False,
            timelimit=t,
            epsilon_stable=100,
        )
        end = time.time()
        if end -  start > max_time:
            max_time = end-start

        # objective_values, build_cost, distance_cost, service_levels, mip_gaps = mopta_solver.allocation_problem(L_sol=L, n_sol=n, n_iter=100)
        # new_df = pd.DataFrame({
        #     'sample': sample,
        #     'T': t,
        #     'objective_values': objective_values,
        #     'build_cost': build_cost,
        #     'distance_cost': distance_cost,
        #     'service_levels': service_levels,
        #     'mip_gap': mip_gaps,
        # })
        
        # df_val = pd.concat([df_val, new_df], ignore_index=True)
        # add values to df
        df.loc[len(df)] = \
            [t,len(selected_locations), mopta_solver.objective_values[-1], end - start, mip_gap, mip_gap_relative, iterations,mopta_solver.build_cost_sol, mopta_solver.drive_cost_sol,  mopta_solver.L.shape[0]]
        # save in every iteration
        
        df.to_csv('comparison_t_samples.csv', index=False)
        # df_val.to_csv('val_t_samples.csv', index=False)
    ls = LinearSolver(vehicle_locations=locations, loglevel=logging.DEBUG, service_level=0.95)
    ls.J = range(ls.n_vehicles)
    ls.S = mopta_solver.S
    # ls.add_samples(num=2)
    start = time.time()
    obj_ls, mip_gap_ls = ls.build_xpress_model(max_time)
    end = time.time()
    df_exact.loc[len(df_exact)] = \
            [t,len(selected_locations), obj_ls, end - start, mip_gap_ls]
