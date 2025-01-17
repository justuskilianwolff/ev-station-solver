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

locations = load_locations("medium").values
print('open')
percentages = [0.01, 0.02, 0.04, 0.05, 0.1, 0.15]
values_t = np.array([5]) * 60#np.array([1, 5, 15, 30, 60]) * 60  # in minutes
value_locations = [10]

df = pd.DataFrame(
    columns=['T', 'S', 'Objective_Value', 'num_vehicales', 'Solve_Time', 'MIP_Gap', 'Iterations','n', 'Number_End_Locations']
)

df_exact = pd.DataFrame(
    columns=['T', 'S', 'num_vehicales', 'Objective_Value', 'Solve_Time', 'MIP_Gap', 'solstatus', 'solvestatus','lower_bound']
)

sampleples = [1,3,5]
value_location = 10
t = 300
for percentage in percentages:
    max_time = 0
    print(f'num_cars = {percentage * len(locations)}')
    for sample in sampleples:
        print(f'sample = {sample}')
        np.random.seed(0)
        selected_locations = locations[np.random.randint(0, len(locations), size=int(percentage * len(locations)))]
        n_clusters = min(np.ceil(len(selected_locations) * CONSTANTS['mu_charging']*value_location /
                            (CONSTANTS['station_ub']*CONSTANTS['queue_size'])),percentage * len(locations))
        print(f'starting n = {n_clusters}')
        
        mopta_solver = Solver(
            vehicle_locations=selected_locations,
            loglevel=logging.INFO,
            service_level=.95,
        )
        
        # # compute number of initial locations
        mopta_solver.add_initial_locations(int(n_clusters), mode='k-means', seed=0)
        mopta_solver.add_samples(num=sample)

        start = time.time()
        solution = mopta_solver.solve(
            verbose=False,
            timelimit=300,
            epsilon_stable=100,
        )
        end = time.time()
        if end -  start > max_time:
            max_time = end-start

        # add values to df
        best_sol = solution[-1]
    #     []'T', 'S', 'Objective_Value', 'num_vehicales', 'Solve_Time', 'MIP_Gap', 'Iterations','n', 'Number_End_Locations']        
        df.loc[len(df)] = \
            [t, sample, best_sol.kpis['total_cost'],np.ceil(len(selected_locations)), end - start, best_sol.mip_gap, len(solution), int(n_clusters),  len(mopta_solver.J)]
        #     # save in every iteration
            
        df.to_csv('comparison_t_samples.csv', index=False)
        # # df_val.to_csv('val_t_samples.csv', index=False)
        ls = LinearSolver(vehicle_locations=selected_locations, loglevel=logging.DEBUG, service_level=0.95)
        ls.J = range(ls.n_vehicles)
        print(ls.n_vehicles)
        ls.S = mopta_solver.S
        # ls.add_samples(num=2)
        start = time.time()
        obj_ls, mip_gap_ls, solstatus, solvestatus, lower_bound = ls.build_xpress_model(max(int(np.ceil(max_time)),3600))
        end = time.time()
        df_exact.loc[len(df_exact)] = \
                [t,sample,len(selected_locations), obj_ls, end - start, mip_gap_ls,solstatus, solvestatus,lower_bound]
        df_exact.to_csv('comparison_t_samples_exact.csv', index=False)