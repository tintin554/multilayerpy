# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 14:08:04 2021

@author: Adam
"""


import numpy as np
import simulate
import scipy.integrate as integrate
import kmsub_model_build
import importlib


class Optimizer():
    '''
    An object which will take in a Simulate, Model and Data object in order to 
    fit the model to the data with various methods
    '''
    
    def __init__(self,simulate_object,data,cost='MSE'):
        
        self.simulate = simulate_object
        self.model = simulate_object.model
        
        # make data into a Data object if not already
        if type(data) != simulate.Data:
            self.data = simulate.Data(data)
        else:
            self.data = data
        
        self.cost = cost
        self.cost_func_val = None
        
    def cost_func(self,model_y):
        
        cost = self.cost
        expt = self.data
        expt_y = expt[:,1]
        
        if cost == 'MSE':
            
            val = (np.square(expt_y-model_y) ** 2)/len(expt_y)
        
        self.cost_func_val = val
        return val
    
    def fit(self,method='least_squares',component_no='1'):

        sim = self.simulate_object
        
        # identify varying params, append to varying_params list and record order
        varying_params = []
        varying_param_keys = []
        
        for param in sim.parameters:
            if sim.parameters[param].vary == True:
                varying_params.append(sim.parameters[param].value)
                varying_param_keys.append(param)
                
        
        
        def minimize_me(varying_param_vals,varying_param_keys,sim,component_no):
            
            # unpack required params to run the model
            n_layers = sim.run_params['n_layers']
            V = sim.run_params['V']
            A = sim.run_params['A']
            n_time = sim.run_params['n_time']
            rp = sim.run_params['rp']
            time_span = sim.run_params['time_span']
            ode_integrate_method = sim.run_params['ode_integrate_method']
            layer_thick = sim.run_params['layer_thick']
            Y0 = sim.run_params['Y0']
            
            for ind, param in enumerate(varying_param_keys):
                sim.parameters[param].value = varying_param_vals[ind]
            
            # import the model from the .py file created in the model building
            # process
            model_import = importlib.import_module(f'{self.model.filename[:-3]}')
                
            # define time interval
            tspan = np.linspace(min(time_span),max(time_span),n_time)
            
            # t_eval for comparing mod + expt at the same timepoints
            # assuming expt time axis is in s
            t_eval = self.data[:,0]
            
          
            model_output = integrate.solve_ivp(lambda t, y:model_import.dydt(t,y,sim.parameters,V,A,n_layers,layer_thick),
                                                     (min(time_span),max(time_span)),
                                                     Y0,t_eval=t_eval,method=ode_integrate_method)
                
            # collect surface concentrations
            surf_conc_inds = [0]
            for i in range(1,len(self.model.model_components)):
                ind = i * n_layers + i
                surf_conc_inds.append(ind)
                
            surf_concs = {}
            # append to surf concs dict for each component
            for ind, i in enumerate(surf_conc_inds):
                surf_concs[f'{ind+1}'] = model_output.y.T[:,i] 
                
            # bulk concentrations
            bulk_concs = {}
            for i in range(len(self.model.model_components)):
                conc_output = model_output.y.T[:,i*n_layers+1+i:(i+1)*n_layers+i+1]
                
                bulk_concs[f'{i+1}'] = conc_output   
                
            # get total no of molecules of component of interest 
            
            bulk_num = bulk_concs[f'{component_no}'] * V
            surf_num = surf_concs[f'{component_no}'] * A[0]
            
            bulk_total_num = np.sum(bulk_num,axis=1)
            total_number_molecules = bulk_total_num + surf_num
            
            # cost func normalises the data, so model output will be normalised 
            norm_number_molecules = total_number_molecules / total_number_molecules[0]
            # CARRY ON COST FUNC CALC ETC.
            
            
            
            
            
            
            
            
            
