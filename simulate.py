# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:47:48 2021

@author: Adam
"""


'''
The Simulate module of multilayerPy

'''

import numpy as np
import importlib
import time
import scipy
from scipy import integrate
import matplotlib.pyplot as plt


class Simulate():
    '''
    A class which takes in a ModelBuilder object and (optionally) some data
    to fit to.
    
    '''
    
    def __init__(self, model, params_dict):
        
        # if Y0 == no_components, assume same for all
        # elif it is == Lorg * n_comp + n_comp, use as-is
        # else use Y0_surf and Y0_bulk

        self.parameters = params_dict
        
        self.model = model
        
        self.model_output = None
        
        self.run_params = {'n_layers':None,
                           'rp': None,
                           'time_span': None,
                           'n_time': None,
                           'V': None,
                           'A': None,
                           'layer_thick': None,
                           'dense_output': None,
                           'Y0': None,
                           'ode_integrator': None,
                           'ode_integrate_method': None}
        
        self.surf_concs = {}
        self.bulk_concs = {}
        
        

    def run(self,n_layers, rp, time_span, n_time, V, A, layer_thick, dense_output=False,
                 Y0=None, ode_integrator='scipy',
                 ode_integrate_method='BDF'):
        '''
        Runs the simulation with the input parameters provided. 
        Model output is a scipy OdeResult object
        Updates the model_output, which includes an array of shape = (n_time,Lorg*n_components + n_components)

        '''
        
        # record run parameters
        self.run_params['n_layers'] = n_layers
        self.run_params['rp'] = rp
        self.run_params['time_span'] = time_span
        self.run_params['n_time'] = n_time
        self.run_params['V'] = V
        self.run_params['A'] = A
        self.run_params['layer_thick'] = layer_thick
        self.run_params['dense_output'] = dense_output
        self.run_params['Y0'] = Y0
        self.run_params['ode_integrator'] = ode_integrator
        self.run_params['ode_integrate_method'] = ode_integrate_method
            
        
        assert len(time_span) == 2, "time_span needs to be a sequence of 2 numbers"
        assert type(Y0) != None, "Need to supply initial concentrations (Y0)"

        params = self.parameters
        
        # import the model from the .py file created in the model building
        # process
        model_import = importlib.import_module(f'{self.model.filename[:-3]}')
            
        # define time interval
        tspan = np.linspace(min(time_span),max(time_span),n_time)
        
        start_int = time.time()
        model_output = integrate.solve_ivp(lambda t, y:model_import.dydt(t,y,params,V,A,n_layers,layer_thick),
                                                 (min(time_span),max(time_span)),
                                                 Y0,t_eval=tspan,method=ode_integrate_method)
        end_int = time.time()
                
        print(f'Model run took {end_int-start_int:.2f} s')
        
        # return model output and assign dicts of surf + bulk concs
        
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
        
        self.surf_concs = surf_concs
        self.bulk_concs = bulk_concs
        self.model_output = model_output
        
        
        return model_output

        # function to plot output
        
    def plot(self,norm=False):
        model_output = self.model_output.y.T 
        
        n_layers = self.run_params['n_layers']
        n_comps = len(self.model.model_components)
        mod_comps = self.model.model_components
        
        # plot surface concentrations
        plt.figure()
        plt.title('Surface concentrations',fontsize='large')
        
        for i in range(n_comps):
            comp_name = mod_comps[f'{i+1}'].name
            plt.plot(self.model_output.t,self.surf_concs[f'{i+1}'],label=comp_name)
        
        plt.ylabel('Surface conc. / cm$^{-2}$',fontsize='large')
        plt.xlabel('Time / s',fontsize='large')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        
        # plot bulk concentrations    
        plt.figure()
        if norm:
            plt.title('Normalised amount of each component',fontsize='large')
            
        else:
            plt.title('Total number of molecules',fontsize='large')
        
        # make a matrix of volume values
        #vol_matrix = np.ones(bulk_concs[0].shape) * self.run_params['V']
        
        for i in range(n_comps):
            comp_name = mod_comps[f'{i+1}'].name
            
            surf_no = self.surf_concs[f'{i+1}'] * self.run_params['A'][0]
            bulk_no = self.bulk_concs[f'{i+1}'] * self.run_params['V']
            tot_bulk_no = np.sum(bulk_no,axis=1)
            total_no = surf_no + tot_bulk_no
            
            if norm:
                plt.plot(self.model_output.t,total_no/max(total_no),label=comp_name)
            else:
                plt.plot(self.model_output.t,total_no,label=comp_name)
        if norm:
            plt.ylabel('[component] / [component]$_{max}$',fontsize='large')
        else:
            plt.ylabel('N$_{component}$ / molec.',fontsize='large')
        plt.xlabel('Time')
        plt.legend()
        plt.tight_layout()
        plt.show()
        






# function to make Y0?

# function to make V and A arrays - makelayers