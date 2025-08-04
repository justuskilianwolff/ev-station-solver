import logging

from src.ev_station_solver.solving.solver import Solver
from src.ev_station_solver.solving.sample import Sample
from src.ev_station_solver.logging import get_logger
from src.ev_station_solver.solving.solution import LocationSolution_linear

import xpress as xp
import numpy as np
from src.ev_station_solver.helper_functions import get_distance_matrix
xp.init('c:/xpressmp/bin/xpauth.xpr')

# create logger
logger = get_logger(__name__)

class LinearSolver(Solver):
    def build_xpress_model(self, solve_time):
        p = xp.problem()

        ### Add decicion variables
        v = {j: xp.var(vartype=xp.binary, name='v_{0}'.format(j))
        for j in self.J}
        w = {j: xp.var(vartype=xp.integer, name='w_{0}'.format(j))
        for j in self.J}
        u = {(i,j,s.index): xp.var(vartype=xp.binary, name='u_{0}_{1}_{2}'.format(i,j,s.index))
        for s in self.S for i in s.I_s for j in self.J}
        #vartype=xp.binary, vartype=xp.integer,
        #y = {(i,j,s.index): xp.var(name='y_{0}_{1}_{2}'.format(i,j,s.index),lb = 0)
        y = {(i,s.index): xp.var(name='y_{0}_{1}'.format(i,s.index),lb = 0)
        for s in self.S for i in s.I_s for j in self.J}
        max_loc = [self.x_max,self.y_max]
        min_loc = [self.x_min,self.y_min]
        x = {(j,l): xp.var(name='x_{0}_{1}'.format(j,l), lb = min_loc[l], ub = max_loc[l]) for j in self.J for l in range(2)}
        d_tilde = {(i,j): xp.var(name='d_tilde_{0}_{1}'.format(i,j), lb = 0) for j in self.J for i in range(self.n_vehicles)}
        # vartype=xp.binary, vartype=xp.integer,
        p.addVariable(v,w,u,y,x,d_tilde)
        #### add constraints
        build_1 = [v[j] <= w[j] for j in self.J]
        build_2 = [w[j] <= self.station_ub*v[j] for j in self.J]
        capacity = [xp.Sum(u[i,j,s.index] for i in s.I_s)<= self.queue_size*w[j] for j in self.J for s in self.S]
        assignment = [xp.Sum(u[i,j,s.index] for j in self.J) <= 1 for s in self.S for i in s.I_s]
        servive_level = [xp.Sum(u[i,j,s.index] for j in self.J for i in s.I_s) >= self.service_level * s.n_vehicles for s in self.S]
        #~~~~~~~ old linearisatuion 

        # linearise_1 = [y[s.I_s[i],j,s.index] <= np.ceil((s.ranges[i]))*u[s.I_s[i],j,s.index]  for j in self.J for s in self.S for i in s.I]
        # linearise_2 = [np.ceil((s.ranges[i]))*u[s.I_s[i],j,s.index]*100 + d_tilde[s.I_s[i],j] - np.ceil((s.ranges[i]))*100 <= y[s.I_s[i],j,s.index] for j in self.J for s in self.S for i in s.I]
        # linearise_3 = [y[s.I_s[i],j,s.index]  <= d_tilde[s.I_s[i],j]   for j in self.J for s in self.S for i in s.I]
        
        #~~~~~~~ new linearisatuion
        linearise_1 = [y[s.I_s[i],s.index] <= np.ceil((s.ranges[i]))*xp.Sum(u[s.I_s[i],j,s.index]  for j in self.J) for s in self.S for i in s.I]
        distances = get_distance_matrix(self.vehicle_locations, self.vehicle_locations)
        linearise_2 = [y[s.I_s[i],s.index] >= d_tilde[s.I_s[i],j] - distances[s.I_s[i], :].max()*(1- u[s.I_s[i],j,s.index]) for j in self.J for s in self.S for i in s.I]
        linearise_3 = [y[s.I_s[i],s.index]  <= d_tilde[s.I_s[i],j]   for j in self.J for s in self.S for i in s.I]
        two_norm = [d_tilde[i,j]**2 >= (x[j,0] - self.vehicle_locations[i,0])**2 + (x[i,1] - self.vehicle_locations[i,1])**2 for j in self.J for i in range(self.n_vehicles)]
        
        p.addConstraint(build_1,build_2,capacity,assignment,servive_level,linearise_1,linearise_2,linearise_3)
        # ,linearise_2,linearise_3
        p.addConstraint(two_norm)
        self.set_fixed_charge_cost()
        obj = self.build_cost_param * xp.Sum(v[j] for j in self.J)
        obj += self.maintenance_cost_param * xp.Sum(w[j] for j in self.J)
        #~~~~ old obj ~~
        # obj += 365 / len(self.S) * self.drive_charge_cost_param * xp.Sum(
        #     y[i,j,s.index]   for j in self.J for s in self.S for i in s.I_s)

        #~~~~ new obj ~~
        obj += 365 / len(self.S) * self.drive_charge_cost_param * xp.Sum(
            y[i,s.index] for s in self.S for i in s.I_s)
        obj +=  365 / len(self.S) * self.fixed_charge_cost
        #xp.sqrt()
        
        p.setObjective(obj, 
                  sense = xp.minimize)
        p.write("problem","lp")
        # max_time = -300 # time in seconds

        p.setControl('maxtime', -solve_time )
        p.solve()
        print(p.attributes.solvestatus)
        print(p.attributes.solstatus)
        # u_sol = p.getSolution(u)
        # v_sol = p.getSolution(v)
        # w_sol = p.getSolution(w)
        # y_sol = p.getSolution(y)
        # d_sol = p.getSolution(d_tilde)
        # print('v')
        # for key, value in v_sol.items():
        #     if value > 0:
        #         print(f'{key} = {value}')
        
        # print('w')
        # for key, value in w_sol.items():
        #     if value > 0:
        #         print(f'{key} = {value}')
        
        # print('u')
        # for key, value in u_sol.items():
        #     if value > 0:
        #         print(f'{key} = {value}')

        # print('y')
        # for key, value in y_sol.items():
        #     if value > 0:
        #         print(f'{key} = {value}')
        
        # print('d')
        # for key, value in d_sol.items():
        #     if value > 0:
        #         print(f'{key} = {value}')

        best_bound = p.getAttrib('bestbound')

        mip_gap = 100*((p.getObjVal()-best_bound)/p.getObjVal())
        print(f'The objective function value is {p.getObjVal()}') 
        print(f'The mip gap is {mip_gap}') 
        return p.getObjVal(), mip_gap,p.attributes.solstatus, p.attributes.solvestatus, best_bound