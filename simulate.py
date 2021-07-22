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
                
        #print(f'Model run took {end_int-start_int:.2f} s')
        
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
        
    def plot(self,norm=False,data=None):
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
        
        # plot data if provided, column 0 = time, column 1 = value
        # optional column 3 = uncertainty

        try:
            rows, cols = data.shape
            if cols == 2:
                plt.scatter(data[:,0],data[:,1],facecolors='none',edgecolors='k',label='data')
            else:
                plt.errorbar(data[:,0],data[:,1],yerr=data[:,2],mfc='none',
                             mec='k',linestyle='none',label='data',marker='o',color='k')
        
            plt.xlabel('Time')
            plt.legend()
            plt.tight_layout()
            plt.show()
        except:
            
            plt.xlabel('Time')
            plt.legend()
            plt.tight_layout()
            plt.show()
            
        
    def plot_bulk_concs(self,cmap='viridis'):
        
        n_comps = len(self.bulk_concs)
        
        # for each component, plot a heatmap plot
        
        for i in range(n_comps):
            comp_name = self.model.model_components[f'{i+1}'].name
            comp_bulk_conc = self.bulk_concs[f'{i+1}'].T
            
            plt.figure()
            plt.title(comp_name,fontsize='large')
            plt.pcolormesh(comp_bulk_conc,cmap=cmap)
            plt.xlabel('Time points',fontsize='large')
            plt.ylabel('Layer number',fontsize='large')
            
            # invert y-axis so that layer 0 is at the top of the plot
            plt.gca().invert_yaxis()
            cb=plt.colorbar()
            cb.set_label(label='conc. / cm$^{-3}$',fontsize='large',size='large')
            plt.tight_layout()
            plt.show()


def initial_concentrations(bulk_conc_dict,surf_conc_dict,n_layers):
    '''
    Returns an array of initial bulk and surface concentrations (Y0)
    
    bulk_conc: dict of initial bulk concentration of each component (key = component number)
    surf_conc: dict of initial surf concentration of each component (key = component number)
    n_layers: number of model layers
    '''
    
    n_comps = len(bulk_conc_dict)
    
    # initialise the Y0 array
    Y0 = np.zeros(n_layers * n_comps + n_comps)
    
    # for each model component 
    for i in range(n_comps):
        
        bulk_conc_val = bulk_conc_dict[f'{i+1}']
        surf_conc_val = surf_conc_dict[f'{i+1}']
        
        # define surface conc
        Y0[i*n_layers+i] = surf_conc_val
        
        # define bulk concs
        for k in np.arange(n_layers*i+1+i,(i+1)*n_layers+i+1):
            Y0[k] = bulk_conc_val
        
    
    return Y0
        

def make_layers(n_layers,bulk_radius,geometry):
    '''
    defines the volume, surface area and layer thickness for each model layer
    
    Returns a tuple of V, A and layer_thick arrays
    
    bulk radius is normally defined as the particle radius - molecular diameter
    
    '''
    
        
    delta = bulk_radius/n_layers
    
    if geometry == 'spherical':
    
        V = np.zeros(n_layers)
        A = np.zeros(n_layers)
        for i in np.arange(n_layers):
            layerno = i + 1
            V[i] = (4/3) * np.pi * ((bulk_radius-(layerno-1)*delta)**3 - (bulk_radius-layerno*delta)**3)    
            A[i] = 4 * np.pi * (bulk_radius-(layerno-1)*delta)**2
        
        layer_thick = np.ones(n_layers) * delta
        
    elif geometry == 'film':
        
        # define the square cross-section of the film to model
        square_length = 1e-4 # cm (1 µm)
        
        V = np.ones(n_layers) * square_length * square_length * delta
        A = np.ones(n_layers) * square_length * square_length

        layer_thick = np.ones(n_layers) * delta
        
    return (V, A, layer_thick)
 
    
class Data():
    
    def __init__(self,data,n_skipped_rows=0,norm=False,norm_index=0):
        
        # if a filename string is supplied, read in the data as an array
        if type(data) == str:
            data = np.genfromtxt(data,skip_header=n_skipped_rows)
            
        self.x = data[:,0]
        self.y = data[:,1]
        self.y_err = np.zeros(len(self.y))
        
        nrows, ncols = data.shape
        if ncols == 3:
            self.y_err = data[:,2]
            
        if norm == True:
            self.y = self.y / self.y[norm_index]
            self.y_err = self.y_err / self.y[norm_index]
        


# function to make Y0?

# function to make V and A arrays - makelayers
