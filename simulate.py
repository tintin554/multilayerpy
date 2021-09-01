# -*- coding: utf-8 -*-
"""
@author: Adam Milsom

    MultilayerPy - build, run and optimise kinetic multi-layer models for 
    aerosol particles and films.
    
    Copyright (C) 2021  Adam Milsom

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    
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
from kmsub_model_build import Parameter


class Simulate():
    '''
    A class which takes in a ModelBuilder object and (optionally) some data
    to fit to.
    
    '''
    
    def __init__(self, model, params_dict, data=None):
        
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
                           'ode_integrate_method': None,
                           'atol':None,
                           'rtol':None}
        
        self.surf_concs = {}
        self.bulk_concs = {}
        self.static_surf_concs = {}
        self.optimisation_result = None
        
        # convert parameter dictionary values to Parameter objects if they are not
        # this will help with using the Optimizer
        for par, value in self.parameters.items():
            # if not Parameter object, an attribute error would be raised
            try:
                val = value.value
            except AttributeError:
                self.parameters[par] = Parameter(value)
        
        self.data = data
        if type(self.data) != type(None):
            self.data = Data(data,norm=True)
        
        
    def calc_Vt_At_layer_thick(self):

        '''
        calculate V and A of each layer at each timepoint in the simulation
        '''
        



        output = self.model_output.y.T

        n_comps = len(self.model.model_components)
        n_layers = self.run_params['n_layers']
        
        Vtot = np.array([])
        for i in range(n_comps):
            cn = i + 1
            # volume
            N_bulk = output[:,(cn-1)*n_layers+2*(cn-1)+2:(cn)*n_layers+(cn)+(cn-1)+1]

            #print('len N_bulk= ',len(N_bulk))

            v_molec = self.parameters[f'delta_{cn}'].value ** 3
            V_tot_comp = N_bulk * v_molec
            if i == 0:
                Vtot = V_tot_comp
            else:
                Vtot = Vtot + V_tot_comp
                
        #print('shape Vtot= ',Vtot.shape)
        # area (spherical geom)
        sum_V = np.sum(Vtot,axis=1)
        cumsum_V = np.cumsum(np.flip(Vtot,axis=1),axis=1)
        
        layer_thick = []
        if self.model.geometry == 'spherical':
            # calc layer thick
            for ind, val in enumerate(cumsum_V):
                t_slice_v_vals = np.flip(cumsum_V[ind,:])
                # if ind == 0:
                    #print('t_slice_v_vals ',t_slice_v_vals)
                thick_vals_t = []
                for i, v in enumerate(t_slice_v_vals):
                    if i != len(t_slice_v_vals) - 1:
                        v_next_shell = t_slice_v_vals[i+1]
                    else: 
                        v_next_shell = 0.0
                    r_shell = np.cbrt((3*v)/(4*np.pi))
                    r_next_shell = np.cbrt((3*v_next_shell)/(4*np.pi))
                    
                    thick = r_shell - r_next_shell
                    if i == len(t_slice_v_vals) - 1:
                        thick = r_shell
                    thick_vals_t.append(thick)
                if ind == 0:
                    layer_thick = np.array(thick_vals_t)
                else:
                    layer_thick = np.vstack((layer_thick,thick_vals_t))
            
        
            #print('shape cumsum_V= ',cumsum_V.shape)
            r_pos = np.cbrt((3.0* np.flip(cumsum_V))/(4*np.pi))
            A = 4 * np.pi * r_pos**2
        
        elif self.model.geometry == 'film':
            square_length = 1e-4 # length of square cross-section of the film (1 µm), arbitrary
            A = square_length ** 2 # same for all layers
            layer_thick = Vtot / A
            
        # layer thickness
        #layer_thick = r_pos[:,:-1] - r_pos[:,1:]
        #print('shape layer thick ', layer_thick.shape)
        #print('shape rpos ', r_pos.shape)
        #layer_thick = np.column_stack((layer_thick,r_pos[:,-1]))
        
        
        #print('shape sumV= ',sum_V.shape,'shape A= ',A.shape, 'shape layer_thick= ',layer_thick.shape)

        return Vtot, A, layer_thick
        

    def run(self,n_layers, rp, time_span, n_time, V, A, layer_thick, dense_output=False,
                 Y0=None, ode_integrator='scipy',
                 ode_integrate_method='BDF', rtol=1e-3, atol=1e-6):
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
        self.run_params['rtol'] = rtol
        self.run_params['atol'] = atol
        self.run_params['geometry'] = self.model.reaction_scheme.model_type.geometry

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
                                                 Y0,t_eval=tspan,method=ode_integrate_method,
                                                 rtol=rtol,atol=atol,dense_output=dense_output)
        self.model_output = model_output
        
        end_int = time.time()
                
        #print(f'Model run took {end_int-start_int:.2f} s')
        
        # return model output and assign dicts of surf + bulk concs
        if self.model.model_type.lower() == 'km-sub':
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
        
        elif self.model.model_type.lower() == 'km-gap':
            # REMEMBER division by A or V to get molec. cm-2 or cm-3 (km-gap)
            
            # calculate V_t and A_t at each time point

            V_t, A_t, layer_thick = self.calc_Vt_At_layer_thick()
            
            # collect surface concentrations
            surf_conc_inds = []
            for i in range(len(self.model.model_components)):
                cn = i + 1
                ind = (cn-1) * n_layers + 2 * (cn-1)
                surf_conc_inds.append(ind)
                
            surf_concs = {}
            # append to surf concs dict for each component
            for ind, i in enumerate(surf_conc_inds):
                surf_concs[f'{ind+1}'] = model_output.y.T[:,i] / A_t[:,0]
                
            # collect static surface concentrations
            static_surf_conc_inds = []
            for i in range(len(self.model.model_components)):
                cn = i + 1
                ind = (cn-1)*n_layers+2*(cn-1)+1
                static_surf_conc_inds.append(ind)
                
            static_surf_concs = {}
            # append to surf concs dict for each component
            for ind, i in enumerate(surf_conc_inds):
                static_surf_concs[f'{ind+1}'] = model_output.y.T[:,i+1] / A_t[:,0]
                
            # get bulk concentrations
            bulk_concs = {}
            for i in range(len(self.model.model_components)):
                cn = i + 1
                conc_output = model_output.y.T[:,(cn-1)*n_layers+2*(cn-1)+2:cn*n_layers+cn+(cn-1)+1] / V_t
                
                bulk_concs[f'{i+1}'] = conc_output
                
            self.surf_concs = surf_concs
            self.static_surf_concs = static_surf_concs
            self.bulk_concs = bulk_concs
            

            
        
        return model_output

        # function to plot output
        
            
        
    def plot(self,norm=False,data=None,comp_number='all'):
        
        # if data not supplied with func call, use self.data
        if type(data) == type(None):
            data = self.data
        
        
        mod_type = self.model.model_type.lower()
        # km-gap: V, A and layer thickness over time
        if  mod_type == 'km-gap':
            Vt, At, thick_t = self.calc_Vt_At_layer_thick()
        

        model_output = self.model_output.y.T 
        
        n_layers = self.run_params['n_layers']
        n_comps = len(self.model.model_components)
        mod_comps = self.model.model_components
        
        # plot surface concentrations
        plt.figure()
        
        if mod_type == 'km-sub':
            plt.title('Surface concentrations',fontsize='large')
            
        elif mod_type == 'km-gap':
            plt.title('Static surface layer concentrations',fontsize='large')
        
        for i in range(n_comps):
            if comp_number == 'all' or i+1 == comp_number:
                comp_name = mod_comps[f'{i+1}'].name
                if mod_type == 'km-sub':
                    plt.plot(self.model_output.t,self.surf_concs[f'{i+1}'],label=comp_name)
                elif mod_type == 'km-gap':
                    plt.plot(self.model_output.t,self.static_surf_concs[f'{i+1}'],label=comp_name)
                    
                
        
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
            if comp_number == 'all' or i+1 == comp_number:
                comp_name = mod_comps[f'{i+1}'].name
                if mod_type == 'km-sub':
                    surf_no = self.surf_concs[f'{i+1}'] * self.run_params['A'][0]
                    bulk_no = self.bulk_concs[f'{i+1}'] * self.run_params['V']
                    stat_surf_no = np.zeros(len(surf_no))
                elif mod_type == 'km-gap':
                    surf_no = self.surf_concs[f'{i+1}'] * At[:,0]
                    bulk_no = self.bulk_concs[f'{i+1}'] * Vt
                    stat_surf_no = self.static_surf_concs[f'{i+1}'] * At[:,0]
                    
                tot_bulk_no = np.sum(bulk_no,axis=1)
                total_no = surf_no + tot_bulk_no + stat_surf_no                  
                
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
                if norm:
                    plt.errorbar(data[:,0],data[:,1],yerr=data[:,2]/max(data[:,1]),mfc='none',
                                 mec='k',linestyle='none',label='data',marker='o',color='k')
                else:
                    plt.errorbar(data[:,0],data[:,1],yerr=data[:,2],mfc='none',
                             mec='k',linestyle='none',label='data',marker='o',color='k')
        
            plt.xlabel('Time')
            plt.legend()
            plt.tight_layout()
            plt.show()
        except:
            try: 
                if norm:
                    
                    plt.errorbar(data.x,data.y,yerr=data.y_err/max(data.y),mfc='none',
                                 mec='k',linestyle='none',label='data',marker='o',color='k')
                    
                else:
                    plt.errorbar(data.x,data.y,yerr=data.y_err,mfc='none',
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

    def xy_data_total_number(self,components='all'):
        '''
        Returns the x-y data from the model. 
        Either for all components or selected component(s).
        Components are in order of component number.
        
        if selected components desired, supply a list of integers. 
        if one component, supply an int (component number)
        '''
        
        A = self.run_params['A']
        V = self.run_params['V']
        
        mod_type = self.model.model_type.lower()
        # km-gap: V, A and layer thickness over time
        if  mod_type == 'km-gap':
            Vt, At, thick_t = self.calc_Vt_At_layer_thick()
            
        model_output = self.model_output.y.T 
        
        n_layers = self.run_params['n_layers']
        n_comps = len(self.model.model_components)
        mod_comps = self.model.model_components
        
        # xy data first column is time
        xy_output = self.model_output.t
        for i in range(n_comps):
            # all components outputted
            if components == 'all':
                if mod_type == 'km-sub':
                    surf_no = self.surf_concs[f'{i+1}'] * A[0]
                    bulk_no = self.bulk_concs[f'{i+1}'] * V
                    stat_surf_no = np.zeros(len(surf_no))
                elif mod_type == 'km-gap':
                    surf_no = self.surf_concs[f'{i+1}'] * At[:,0]
                    stat_surf_no = self.static_surf_concs[f'{i+1}'] * At[:,0]
                    bulk_no = self.bulk_concs[f'{i+1}'] * Vt
                    
                tot_bulk_no = np.sum(bulk_no,axis=1)
                total_no = surf_no + tot_bulk_no + stat_surf_no
                
                xy_output = np.column_stack((xy_output,total_no))
            
            # selected components outputted
            elif type(components) == type(list):
                if i in components:
                    if mod_type == 'km-sub':
                        surf_no = self.surf_concs[f'{i+1}'] * A[0]
                        bulk_no = self.bulk_concs[f'{i+1}'] * V
                        stat_surf_no = np.zeros(len(surf_no))
                    elif mod_type == 'km-gap':
                        surf_no = self.surf_concs[f'{i+1}'] * At[:,0]
                        stat_surf_no = self.static_surf_concs[f'{i+1}'] * At[:,0]
                        bulk_no = self.bulk_concs[f'{i+1}'] * Vt
                    
                    tot_bulk_no = np.sum(bulk_no,axis=1)
                    total_no = surf_no + tot_bulk_no + stat_surf_no
                    
                    xy_output = np.column_stack((xy_output,total_no))
            
            # one component outputted
            elif type(components) == type(int):
                if i == components:
                    if mod_type == 'km-sub':
                        surf_no = self.surf_concs[f'{i+1}'] * A[0]
                        bulk_no = self.bulk_concs[f'{i+1}'] * V
                        stat_surf_no = np.zeros(len(surf_no))
                    elif mod_type == 'km-gap':
                        surf_no = self.surf_concs[f'{i+1}'] * At[:,0]
                        stat_surf_no = self.static_surf_concs[f'{i+1}'] * At[:,0]
                        bulk_no = self.bulk_concs[f'{i+1}'] * Vt
                    
                    tot_bulk_no = np.sum(bulk_no,axis=1)
                    total_no = surf_no + tot_bulk_no + stat_surf_no
                    
                    xy_output = np.column_stack((xy_output,total_no))

        return xy_output

    def save_params_csv(self,filename='model_parameters.csv'):
        '''
        Saves model parameters (name, value and bounds - if available) to a .csv file
        '''
        
        params = self.parameters
        
        output_array = np.array(['Parameter_name','value','lower_bound','upper_bound'])
        
        # for each param in the param dict, append info to the output array
        for name, param in params.items():
            
            lower_bound = 'n/a'
            upper_bound = 'n/a'
            
            if type(param.bounds) != type(None):
                lb = min(param.bounds)
                ub = max(param.bounds)
                
                lower_bound = '{:.2e}'.format(lb)
                upper_bound = '{:.2e}'.format(ub)
    
            val = float(param.value)
            
            arr = np.array([name,val,lower_bound,upper_bound])
            
            output_array = np.vstack((output_array,arr))
            
        # make sure the filename has an extension
        if '.' not in filename:
            filename += '.csv'
        
        # save the file
        np.savetxt(filename,output_array,delimiter=',',fmt='%s')
        
    def rp_vs_t(self):
        '''
        Calculate the radius of the particle/thickness of the film at each 
        timepoint. Returns None if not KM-GAP.

        '''
        if self.model.reaction_scheme.model_type.model_type.lower() == 'km-gap':
            Vt, At, layer_thick = self.calc_Vt_At_layer_thick()
            
            rp = np.sum(layer_thick,axis=1)
        else:
            rp = None
        return rp
        
    
    

def initial_concentrations(model_type,bulk_conc_dict,surf_conc_dict,n_layers,
                           static_surf_conc_dict=None,V=None,A=None,parameter_dict=None,vol_frac=1.0):

    '''
    Returns an array of initial bulk and surface concentrations (Y0)
    
    bulk_conc: dict of initial bulk concentration of each component (key = component number)
    surf_conc: dict of initial surf concentration of each component (key = component number)
    n_layers: number of model layers
    '''
    
    n_comps = len(bulk_conc_dict)
    
    # initialise the Y0 array
    Y0 = np.zeros(n_layers * n_comps + n_comps)
    
    if model_type.model_type.lower() == 'km-gap':
        Y0 = np.zeros(n_layers * n_comps + 2 * n_comps)
    
    # for each model component 
    for i in range(n_comps):
        
        bulk_conc_val = bulk_conc_dict[f'{i+1}']
        surf_conc_val = surf_conc_dict[f'{i+1}']
        
        if model_type.model_type.lower() == 'km-sub':
            # define surface conc
            Y0[i*n_layers+i] = surf_conc_val
            
            # define bulk concs
            for k in np.arange(n_layers*i+1+i,(i+1)*n_layers+i+1):
                Y0[k] = bulk_conc_val
                
        elif model_type.model_type.lower() == 'km-gap':

            assert type(V) != None, "supply Vol. array for calculation of initial number of molecules in each layer (KM-GAP)"
            assert type(A) != None, "supply Area array for calculation of initial number of molecules in surface layers (KM-GAP)"
            assert type(parameter_dict) != None, "supply Model Comonents dictionary for calculation of initial number of molecules in each layer (KM-GAP)"
            
            if float(bulk_conc_val) > 1.0:
            
                delta = parameter_dict[f'delta_{i+1}'].value
                v_molec = delta ** 3
                
                static_surf_conc = static_surf_conc_dict[f'{i+1}']
                # define surface conc
                Y0[i*n_layers+2*i] = surf_conc_val * A[0]
                
                # define static surface conc
                Y0[i*n_layers+2*i+1] = static_surf_conc * A[0]
                
                # define bulk concs
                for ind,k in enumerate(np.arange(i*n_layers+2*i+2,(i+1)*n_layers+(i+1)+i+1)):
                    # accounting for volume fraction if particle starts as a mixture
                    if type(vol_frac) != int and type(vol_frac) != float:
                        volume_fraction = vol_frac[i]
                    else:
                        volume_fraction = vol_frac
                    
                    #Y0[k] = bulk_conc_val * V[ind] 
                    Y0[k] = (V[ind] * volume_fraction) / v_molec
               
    return Y0
        




def make_layers(model_type,n_layers,bulk_radius,n_components=None,y0=None,parameter_dict=None):

    '''
    defines the volume, surface area and layer thickness for each model layer
    
    Returns a tuple of V, A and layer_thick arrays
    
    bulk radius is normally defined as the particle radius - molecular diameter
    
    '''
    # actually, get y0 from initial conc...
    # if model_type.mod_type.lower() == 'km-gap':
    #     assert type(y0) != None, "y0 needs to be supplied to calculate V from number of molecules"
    #     assert type(n_components) == int, "n_components needs to be supplied as an integer"
    #     assert type(parameter_dict) == dict, "parameter_dict needs to be supplied to calculate V from number of molecules and molecular volume"
        
    #     V = np.zeros(n_layers)
        
    #     for i in range(n_components):
    #         cn = i + 1
    #         delta_comp = parameter_dict[f'delta_{cn}']
    #         N_bulk_comp = y0[(cn-1)*n_layers+2*(cn-1)+2:cn*n_layers+cn+(cn-1)+1]
    #         v_comp = delta_comp**3
    #         Vtot_comp = N_bulk_comp * v_comp
    #         V = V + Vtot_comp
        
    delta = bulk_radius/n_layers
    
    geometry = model_type.geometry
    
    if geometry == 'spherical':
    
        V = np.zeros(n_layers)
        A = np.zeros(n_layers)
        
        cumulative_V = []
        for i in np.arange(n_layers):
            
            # V[i] = (4/3) * np.pi * ((bulk_radius-(layerno-1)*delta)**3 - (bulk_radius-layerno*delta)**3)    
            # A[i] = 4 * np.pi * (bulk_radius-(layerno-1)*delta)**2
            
            # working from core shell outwards
            v = (4/3) * np.pi * (delta*(i+1))**3
            cumulative_V.append(v)
        
        # now add V of each layer to V array
        for i, v in enumerate(cumulative_V):
            layerno = n_layers - i - 1 
            if i == 0:
                V[layerno] = v 
            else:
                V[layerno] = v - cumulative_V[i-1]
            
        
        layer_thick = np.ones(n_layers) * delta
        
        # calculate A at each shell radius (from inside out)
        r = 0.0
        for i, thick in enumerate(layer_thick):
            layerno = n_layers - i - 1
            r += thick
            surf_area = 4 * np.pi * r**2
            A[layerno] = surf_area
            
        
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
            self.y_err = self.y_err / data[:,1][norm_index]
        


# function to make Y0?

# function to make V and A arrays - makelayers
