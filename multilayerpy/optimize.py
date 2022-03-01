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


import numpy as np
import multilayerpy.simulate as simulate
import scipy.integrate as integrate
import multilayerpy.build
from multilayerpy.build import Parameter
import importlib
from scipy.optimize import differential_evolution, minimize
from multilayerpy.simulate import Data
import emcee
import matplotlib.pyplot as plt
import copy

class Optimizer():
    '''
    An object which will take in a Simulate, Model and Data object in order to 
    fit the model to the data with various methods.
    
    Parameters
    ----------
    simulate_object : multilayerpy.simulate.Simulate 
        A Simulate object created in the model building process.

    cost : str, optional
        The cost function to use in the optimisation process.
        Current option is mean-squared error 'MSE'.

    cfunc : func, optional
        A function which takes in data (in the Data object format) and model_y
        values supplied to the cost_func method. 
        Returns the value of the cost function. Lower values = better fit.

    param_evolution_func : func, optional
        A function which is called f(t,y,param_dict,param_evolution_func_extra_vary_params). 

        This is taken from the supplied Simulate object.

        It returns the param_dict and allows for model parameters to evolve
        over time, values of y (array of ODE solutions evaluated at time t).
        Parameters can be parameterised with extra parameters which themselves can
        vary. See param_evolution_func_extra_vary_params below. 

    param_evolution_func_extra_vary_params : list, optional
        List of Parameter objects which represent additional parameters used
        to evolve/parameterise model input parameters. These values can be optimised
        during model optimisation. Supplied to the param_evolution_func function.

        This is taken from the supplied Simulate object.

    '''
    
    def __init__(self,simulate_object,cost='MSE',cfunc=None,param_evolution_func=None,
                 param_evolution_func_extra_vary_params=None,custom_model_y_func=None):
        
        
        self.simulate = simulate_object
        self.model = simulate_object.model
        
        # make data into a Data object if not already
        data = simulate_object.data
        if isinstance(data,Data):
            self.data = simulate_object.data
        else:
            raise RuntimeError("Simulate.data needs to be an instance of the Data class.")
        
        self.cost = cost
        self.cfunc = cfunc
        self.cost_func_val = None
        self.param_evolution_func = copy.deepcopy(self.simulate.param_evo_func)
        self.param_evolution_func_extra_vary_params = self.simulate.param_evo_additional_params
        self.custom_model_y_func = simulate_object.custom_model_y_func
        self._sampled_xy_data = []
        
        self._vary_param_keys = None
        self._extra_vary_params_start_ind = None
        self._fitting_component_no = None
        self._vary_param_bounds = None
        self._emcee_sampler = None
        self._sampling_component_number = None
        
    def cost_func(self,model_y,weighted=False):
        '''
        Will calculate the cost function used in the optimisation process. 
        A custom cost function will be used if suppled via the cfunc attribute 
        of the Optimizer object. 

        Parameters
        ----------
        model_y : np.ndarray
            Model y values.  

        Returns
        -------
        float
            Cost function value.

        '''
        # use user-supplied cost function if supplied
        if type(self.cfunc) != type(None):
            val = self.cfunc(self.data,model_y)
        
        # use built-in cost function (MSE), *future development*
        elif type(model_y) != type(None):
            cost = self.cost
            expt = self.data
            
            # normalise the expt data if not already
            if expt._normed == False:
                expt.norm(expt.norm_index)
            
            # extract data from expt
            expt_y = expt.y
            expt_y_err = expt.y_err
            
            
            if cost == 'MSE':
                # non-weighted cost function
                if np.any(np.isnan(expt_y_err)):
                    val = (np.square(expt_y-model_y)).mean()
                                        
                # weighted cost function
                elif weighted == True:
                    val = ((1.0/expt_y_err**2) * np.square(expt_y-model_y)).mean()
                                        
                # ignore weightings even though possible (non-weighted)
                else:
                    val = (np.square(expt_y-model_y)).mean()
                                    
        else:
            val = 0.0
        
        # fitting to rp PARKED for the moment          
        # if type(rp) != type(None):
        #     rp_expt = expt.rp 
        #     rp_cost_val = (np.square(rp_expt-rp)).mean(axis=None)
            
        #     # return an average cost function value accounting for fit to rp
        #     # and data
        #     if val != 0.0:
        #         self.cost_func_val = (val + rp_cost_val) / 2
        #         return (val + rp_cost_val) / 2
            
        #     # only fitting to rp
        #     else:
        #         self.cost_func_val = rp_cost_val
        #         return rp_cost_val
            
        self.cost_func_val = val
        return val
    
    def lnlike(self,vary_params):
        '''
        Calculates the log-likelihood of the model-data fit for MCMC sampling.
        
        Parameters
        ----------
        vary_params : np.ndarray
            An array of the varying model parameters.

        Returns
        -------
        loglike : float
            The log-likelihood of the model fit.

        '''

        data = self.data
        sim = self.simulate
        # check if data have yerrs
        nan_count = 0
        for err in data.y_err:
            if float(err) == np.nan:
                nan_count += 1
        
        # run the model to get model_y
        vary_param_keys = self._vary_param_keys
        
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
        
        if type(self._extra_vary_params_start_ind) == int:
            additional_params = vary_params[self._extra_vary_params_start_ind:]
        else:
            additional_params = None


        # update the simulate object parameters 
        if type(self._vary_param_keys) != type(None):
            for ind, param in enumerate(self._vary_param_keys):
                sim.parameters[param].value = vary_params[ind]
            
         
        # define time interval
        tspan = np.linspace(min(time_span),max(time_span),n_time)
        
        # t_eval for comparing mod + expt at the same timepoints
        # assuming expt time axis is in s
        t_eval = self.data.x
        
        
        model_output = integrate.solve_ivp(lambda t, y:sim._dydt(t,y,sim.parameters,V,A,n_layers,layer_thick,
                                                                         param_evolution_func=sim.param_evo_func,
                                                                         additional_params=additional_params),
                                                 (min(time_span),max(time_span)),
                                                 Y0,t_eval=t_eval,method=ode_integrate_method)
        
        sim.model_output = model_output
        
        if sim.model.model_type.lower() == 'km-sub':
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
                
            bulk_num = bulk_concs[f'{self._fitting_component_no}'] * V
            surf_num = surf_concs[f'{self._fitting_component_no}'] * A[0]
            static_surf_num = np.zeros(len(surf_concs[f'{self._fitting_component_no}'])) # only applicable to km-gap (surf is static surf and sorption layer in km-sub)
            
            bulk_total_num = np.sum(bulk_num,axis=1)
            total_number_molecules = bulk_total_num + surf_num + static_surf_num
            
            # use custom model y function if supplied
            if type(self.custom_model_y_func) != type(None):
                model_y = self.custom_model_y_func(bulk_concs,surf_concs,V,A)
            else:
                # the data is normalised, so model output will be normalised 
                model_y = total_number_molecules / total_number_molecules[0]
            
        elif sim.model.model_type.lower() == 'km-gap':
            if type(self._fitting_component_no) != type(None):
                # REMEMBER division by A or V to get molec. cm-2 or cm-3 (km-gap)
                
                # calculate V_t and A_t at each time point
    
                V_t, A_t, layer_thick = sim.calc_Vt_At_layer_thick()
                
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
                    static_surf_concs[f'{ind+1}'] = model_output.y.T[:,i] / A_t[:,0]
                    
                # get bulk concentrations
                bulk_concs = {}
                for i in range(len(self.model.model_components)):
                    cn = i + 1
                    conc_output = model_output.y.T[:,(cn-1)*n_layers+2*(cn-1)+2:cn*n_layers+cn+(cn-1)+1] / V_t
                    
                    bulk_concs[f'{i+1}'] = conc_output
                    
                self.surf_concs = surf_concs
                self.static_surf_concs = static_surf_concs
                self.bulk_concs = bulk_concs
            
                
                # get total no of molecules of component of interest 
                
                bulk_num = bulk_concs[f'{self._fitting_component_no}'] * V_t
                surf_num = surf_concs[f'{self._fitting_component_no}'] * A_t[:,0]
                static_surf_num = static_surf_concs[f'{self._fitting_component_no}'] * A_t[:,0]
            
                bulk_total_num = np.sum(bulk_num,axis=1)
                total_number_molecules = bulk_total_num + surf_num + static_surf_num
                
                # use custom model y function if supplied
                if type(self.custom_model_y_func) != type(None):
                    model_y = self.custom_model_y_func(bulk_concs,surf_concs,static_surf_concs,V_t,A_t)
                else:
                    # the data is normalised, so model output will be normalised 
                    model_y = total_number_molecules / total_number_molecules[0]
        
        #assert model_output.t == self.data.x, "model and experimental datapoints not equivalent"
        
        # if there are errors, account for them
        if not np.any(np.isnan(data.y_err)):

            # make sure data are normalised
            data.norm(data.norm_index)
            numerator = np.square(data.y - model_y)
            denominator = data.y_err ** 2

            loglike = - 0.5 * np.sum((numerator / denominator) + np.log(2*np.pi*denominator)) 
        else:
            raise RuntimeError("Log-likelihood calcuation requires y uncertainty.") 
        
        return loglike
    
    def lnprior(self,vary_params):
        '''
        Calculate the log-prior of the parameter set for MCMC sampling.

        Parameters
        ----------
        vary_params : np.ndarray
            An array of the varying model parameters.

        Returns
        -------
        float
            The log-prior probability of the selected parameter set.

        '''
        vary_param_bounds = self._vary_param_bounds
        violation = False
        # make sure all parameters are within their bounds
        for i, par in enumerate(vary_params):
            lb, ub = min(vary_param_bounds[i]), max(vary_param_bounds[i])
            if par < lb or par > ub:
                violation = True
        if violation == True:
            return -np.inf
        else:
            return 0.0 # uniform prior
        
    def lnprob(self,vary_params):
        '''
        Calculate the log-probability for a parameter set, considering the priors. 

        Parameters
        ----------
        vary_params : np.ndarray
            An array of the varying model parameters..

        Returns
        -------
        float
            The log-probability for the model-data system, considering priors.

        '''
        lp = self.lnprior(vary_params)
        if not np.isfinite(lp):
            return -np.inf
        
        return lp + self.lnlike(vary_params)
        
    
    def fit(self,weighted=False,method='least_squares',component_no='1',n_workers=1,
            popsize=15):
        '''
        Use either a local or global optimisation algorithm to fit the model
        to the data. 

        Parameters
        ----------
        weighted : bool, optional
            Whether to weight the fit to datapoint uncertainties (requires the data to have uncertainties)

        method : str, optional
            Algorithm to be used. 'least_squares' (local) or 'differential_evolution' (global). 
            The default is 'least_squares' Nelder-Mead (simplex).

        component_no : int, optional
            The model component to fit to the data. The default is '1'.

        n_workers : int, optional
            Number of cores to use (only for 'differential_evolution' method). 
            The default is 1.

        popsize : int, optional
            The multiplyer used by the differential_evolution implementation in SciPy. 
            The total population size of parameter sets in the algorithm is len(varying_parameters) * popsize. 
            
        returns
        ----------
        optimize_result : scipy.optimize.OptimizeResult
            The optimization result represented as a OptimizeResult object.
            Important attributes are: x the solution array, success a Boolean
            flag indicating if the optimizer exited successfully and message
            which describes the cause of the termination. 

            See the documentationfor more details: 
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
        
        '''

        self._fitting_component_no = component_no
        sim = self.simulate
        param_evolution_func_extra_vary_params = self.param_evolution_func_extra_vary_params

        # check that there are data to fit to
        if sim.data is None:
            raise RuntimeError("There are no data associated with the Simulate object.")
        
        # identify varying params, append to varying_params list and record order
        varying_params = []
        varying_param_keys = []
        param_bounds = []
        add_param_names = []
        add_param_bounds = []
        
        for param in sim.parameters:
            if sim.parameters[param].vary == True:
                varying_params.append(sim.parameters[param].value)
                varying_param_keys.append(param)
                param_bounds.append(sim.parameters[param].bounds)
                
        # account for extra varying params supplied to param_evolution_func
        # if they are required
        extra_vary_params_start_ind = None
        if type(self.param_evolution_func) != type(None):
            # get the starting index of the varying_param_vals list supplied to 
            # minimize_me
            if type(param_evolution_func_extra_vary_params) != type(None):
                extra_vary_params_start_ind = len(varying_params)
                
                # now append the param_evolution_func_extra_params to varying_params
                # only used in least_squares optimisation (requires an initial guess)
                
                for par in param_evolution_func_extra_vary_params:  
                    varying_params.append(par.value)
                    param_bounds.append(par.bounds)
                    add_param_bounds.append(par.bounds) # for recreating Parameter object after optimisation
                    add_param_names.append(par.name)
            
        def minimize_me(varying_param_vals,varying_param_keys,sim,component_no=component_no,
                        extra_vary_params_start_ind=extra_vary_params_start_ind,
                        ):
            '''
            The function to minimise during the fitting process. 

            Parameters
            ----------
            varying_param_vals : array-type
                An array of varying model parameters to be optimised.

            varying_param_keys : list
                List of parameter dictionary keys corresponding to the parameter
                represented in varying_param_vals.

            sim : multilayerpy.simulate.Simulate 
                The simulate object to be optimised.

            component_no : int, optional
                The component number of the model component to fit to. 
                i.e. If the experimental data corresponds to component number 4,
                fit to component_no = 4.

            extra_vary_params_start_ind : int, optional
                The starting index for the 'extra' varying parameters 
                (not included in the parameter dictionary) used in the parameter
                evolution function. 

            fit_particle_radius : bool, optional
                Whether or not to fit to particle radius data. 

            Returns
            -------
            cost_val : float
                Value of the cost function.

            '''
            
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
            
            if type(extra_vary_params_start_ind) != None:
                additional_params = varying_param_vals[extra_vary_params_start_ind:]
            else:
                additional_params = None
            
            # update the simulate object parameters 
            for ind, param in enumerate(varying_param_keys):
                sim.parameters[param].value = varying_param_vals[ind]
            
            # import the model from the .py file created in the model building
            # process
            #model_import = importlib.import_module(f'{self.model.filename[:-3]}')
                
            # define time interval
            tspan = np.linspace(min(time_span),max(time_span),n_time)
            
            # t_eval for comparing mod + expt at the same timepoints
            # assuming expt time axis is in s
            t_eval = self.data.x
            
          
            model_output = integrate.solve_ivp(lambda t, y:sim._dydt(t,y,sim.parameters,V,A,n_layers,layer_thick,
                                                                             param_evolution_func=self.param_evolution_func,
                                                                             additional_params=additional_params),
                                                     (min(time_span),max(time_span)),
                                                     Y0,t_eval=t_eval,method=ode_integrate_method)
            
            sim.model_output = model_output
            
            if sim.model.model_type.lower() == 'km-sub':
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
                static_surf_num = np.zeros(len(surf_concs[f'{component_no}'])) # only applicable to km-gap (surf is static surf and sorption layer in km-sub)
                
                bulk_total_num = np.sum(bulk_num,axis=1)
                total_number_molecules = bulk_total_num + surf_num + static_surf_num
                
                # use custom model y function if supplied
                if type(self.custom_model_y_func) != type(None):
                    model_y = self.custom_model_y_func(bulk_concs,surf_concs,V,A)
                else:
                    # the data is normalised, so model output will be normalised 
                    model_y = total_number_molecules / total_number_molecules[0]
                
                # calculate the cost function
                
                cost_val = self.cost_func(model_y,weighted=weighted)
            
            elif sim.model.model_type.lower() == 'km-gap':
                if type(component_no) != type(None):
                    # REMEMBER division by A or V to get molec. cm-2 or cm-3 (km-gap)
                    
                    # calculate V_t and A_t at each time point
        
                    V_t, A_t, layer_thick = sim.calc_Vt_At_layer_thick()
                    
                    # get radius of the particle as fn of time
                    rp_t = np.sum(layer_thick,axis=1)
                    
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
                        static_surf_concs[f'{ind+1}'] = model_output.y.T[:,i] / A_t[:,0]
                        
                    # get bulk concentrations
                    bulk_concs = {}
                    for i in range(len(self.model.model_components)):
                        cn = i + 1
                        conc_output = model_output.y.T[:,(cn-1)*n_layers+2*(cn-1)+2:cn*n_layers+cn+(cn-1)+1] / V_t
                        
                        bulk_concs[f'{i+1}'] = conc_output
                        
                    self.surf_concs = surf_concs
                    self.static_surf_concs = static_surf_concs
                    self.bulk_concs = bulk_concs
                
                    
                    # get total no of molecules of component of interest 
                    
                    bulk_num = bulk_concs[f'{component_no}'] * V_t
                    surf_num = surf_concs[f'{component_no}'] * A_t[:,0]
                    static_surf_num = static_surf_concs[f'{component_no}'] * A_t[:,0]
                
                    bulk_total_num = np.sum(bulk_num,axis=1)
                    total_number_molecules = bulk_total_num + surf_num + static_surf_num
                    
                    # use custom model y function if supplied
                    if type(self.custom_model_y_func) != type(None):
                        model_y = self.custom_model_y_func(bulk_concs,surf_concs,static_surf_concs,V_t,A_t)
                    else:
                        # the data is normalised, so model output will be normalised 
                        model_y = total_number_molecules / total_number_molecules[0]
                        
                    # calculate the cost function
                    
                    cost_val = self.cost_func(model_y,weighted=weighted)
                    
                # PARKED fitting to rp for now
                # also fit the particle radius with custom cost function (cfunc)
                # if fit_particle_radius:
                #     print('\nFitting to particle radius or film thickness **CHECK UNITS**\n(should be in cm)\n')
                #     # option to only fit to rp and not number of molecules
                #     if type(component_no) == type(None):
                #         norm_number_molecules = None
                #     cost_val = self.cost_func(norm_number_molecules,rp_t)
            
            
            #print(cost_val)
            return cost_val

        if method == 'differential_evolution':
            print('\nOptimising using differential_evolution algorithm...\n')
            result = differential_evolution(minimize_me,param_bounds,
                                        (varying_param_keys,sim,component_no),
                                        disp=True,workers=n_workers,popsize=popsize)
        elif method == 'least_squares':
            #print(varying_params)
            print('\nOptimising using least_squares Nelder-Mead algorithm...\n')
            result = minimize(minimize_me,varying_params,args=(varying_param_keys,sim,component_no),
                          method='Nelder-Mead',bounds=param_bounds,options={'disp':True, 'return_all':True})
            
        # collect results into a result dict for printing
        res_dict = {}

        if len(varying_param_keys) != 0:
            for i in range(len(varying_param_keys)):
                key = varying_param_keys[i]
                res_dict[key] = result.x[i]

        if len(add_param_names) != 0:
            optimised_extra_vary_params_list = []
            for i in range(len(add_param_names)):
                key = add_param_names[i]
                res_dict[key] = result.x[i+len(varying_param_keys)]
                
                optimised_param_obj = Parameter(result.x[i+len(varying_param_keys)],name=key,vary=True,bounds=add_param_bounds[i])
                optimised_extra_vary_params_list.append(optimised_param_obj)

            sim.param_evo_additional_params = optimised_extra_vary_params_list
        
        # print out results
        
        print('optimised params:\n')
        print(res_dict)
        print('\nSuccess=:',result.success,',termination message:',result.message)
        print('number of iters:',result.nit)
        print('final cost function value = ')
        print(result.fun)
        
        # run the simulation once more to update the Simulate object with
        # optimised parameters 
        sim.run(sim.run_params['n_layers'],sim.run_params['rp'],
                sim.run_params['time_span'],sim.run_params['n_time'],
                sim.run_params['V'],sim.run_params['A'],
                sim.run_params['layer_thick'],Y0=sim.run_params['Y0'])
        
        sim.optimisation_result = res_dict
        
        return result
        
    def sample(self,samples,n_walkers=100, n_burn=0,pool=None,**kwargs):
        '''
        Runs the MCMC sampling procedure using the emcee package. 

        Parameters
        ----------
        samples : int
            The number of MCMC samples (steps) to take.

        n_walkers : int, optional
            The number of walkers to initialise for MCMC sampling.

        n_burn : int, optional
            Number of burn-in steps before the production MCMC run. 

        pool : optional
            An object with a map method that follows the same calling sequence
            as the built-in map function. This is generally used to compute the
            log-probabilities for the ensemble in parallel.
        
        Further details about the emcee EnsembleSampler object are available 
        in its documentation: https://emcee.readthedocs.io/en/stable/user/sampler/

        Returns
        -------
        sampler : emcee.EnsembleSampler 
            EnsembleSampler object.

        '''
        
        sim = self.simulate
        param_evolution_func_extra_vary_params = sim.param_evo_additional_params
        
        # identify varying params, append to varying_params list and record order
        varying_params = []
        varying_param_keys = []
        param_bounds = []
        add_param_names = []
        
        for param in sim.parameters:
            if sim.parameters[param].vary == True:
                varying_params.append(sim.parameters[param].value)
                varying_param_keys.append(param)
                param_bounds.append(sim.parameters[param].bounds)
                
        # account for extra varying params supplied to param_evolution_func
        # if they are required
        extra_vary_params_start_ind = None
        if type(self.param_evolution_func) != type(None):
            # get the starting index of the varying_param_vals list supplied to 
            # minimize_me
            if type(param_evolution_func_extra_vary_params) != type(None):
                extra_vary_params_start_ind = len(varying_params)
            
                # now append the param_evolution_func_extra_params to varying_params
                
                for par in param_evolution_func_extra_vary_params:  
                    varying_params.append(par.value)
                    param_bounds.append(par.bounds)
                    add_param_names.append(par.name)
                    
        self._vary_param_keys = varying_param_keys
        self._extra_vary_params_start_ind = extra_vary_params_start_ind
        self._vary_param_bounds = param_bounds
        
        # make initial guess
        
        ndim = len(varying_params)

        # array of initialised chains centred around the initial guess (small gaussian distribution)
        p0 = [np.array(varying_params) * np.random.normal(1.0,1e-9,len(varying_params)) for i in range(n_walkers)]


        sampler = emcee.EnsembleSampler(nwalkers=n_walkers, ndim=ndim,
                                            log_prob_fn=self.lnprob, pool=pool)

        if n_burn != 0:
            print("Running MCMC burn-in step...")
            p0, _, _ = sampler.run_mcmc(p0, n_burn,progress=True)
            sampler.reset()
            
            print("Running MCMC production run...")
            pos, prob, state = sampler.run_mcmc(p0, samples,progress=True)
            
        else:
            
            print("Running MCMC production run...")
            pos, prob, state = sampler.run_mcmc(p0, samples,progress=True)
                
        self._emcee_sampler = sampler

        # update the parameter objects with the statistics from MCMC sampling (including burn-in and thinning)
        self._update_param_stats(self._emcee_sampler,n_burn=n_burn,thin=1)

        return sampler
    
    def plot_chains(self,discard=0,thin=1):
        '''
        Plots the MCMC chains saved during the sampling run. Each plot is the 
        parameter value at each sampling iteration.
        
        Parameters
        ----------
        discard : int, optional
            The number of setps to burn-in (discard) from the chains before plotting.

        thin : int, optional
            Only take every 'thin' number of samples from the chain (called thinning).

        Returns
        -------
        A multi-panel plot of parameter values vs step number

        '''
        
        sampler = self._emcee_sampler
        
        # get the chains
        chains = sampler.get_chain(thin=thin,discard=discard)
        
        n_samples, n_walkers, n_dim = chains.shape
        
        n_rows = int(np.ceil(n_dim / 3.0))
        
        # make the figure
        
        fig, ax = plt.subplots(nrows=n_rows,ncols=3,figsize=(3*2.5,n_rows*2.5))
        
        ax = ax.ravel()
        
        for i in range(n_dim):
            ylabel = self._vary_param_keys[i]
            ax[i].plot(chains[:,:,i])
            ax[i].set_ylabel(ylabel)
            ax[i].set_xlabel('iteration number')
        
        fig.tight_layout()
        
        plt.show()
        
    def plot_chain_outputs(self):
        '''
        Plots the experimental data with n_samples number of model runs from MCMC sampling. 
        Requires that a sample of model outputs from the MCMC chains has been taken using Optimizer.get_chain_outputs.

        Returns
        -------
        Plot of experimental data with normalised model output vs time for a number of MCMC samples. 

        '''

        # make sure there are model outputs to sample from
        if self._sampled_xy_data == []:
            raise RuntimeError("There are no model runs from the sampled chains to plot. Please call Optimizer.get_chain_outputs() to run the model with parameters from the MCMC sampling procedure")

        # normalise experimental data
        self.simulate.data.norm(self.simulate.data.norm_index)

        time = self.simulate.data.x
        data_y = self.simulate.data.y
        data_y_err = self.simulate.data.y_err
        #print(data_y)
        sampled_data = self._sampled_xy_data

        plt.figure(figsize=(3.5,3.5))

        # plot data
        plt.errorbar(time,data_y,yerr=data_y_err,mfc='none',
                                     mec='k',linestyle='none',label='data',marker='o',color='k',
                                     )

        # unnormalise the data
        self.simulate.data.unnorm()

        # plot each sample output
        for i, sample_output in enumerate(sampled_data):
            t = sample_output[:,0]
            y = sample_output[:,1]
            if i == 0:
                plt.plot(t,y/y[0],ls='-',color=(1,0,0,0.1),label='sampled model output',lw=1)
            else:
                plt.plot(t,y/y[0],ls='-',color=(1,0,0,0.1),lw=2)

        plt.xlabel('Time / s')
        plt.ylabel('Normalised amount of component')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def get_chain_outputs(self, n_samples='all', parallel=False,component_number=1,n_burn=0,thin=1,
                            override_stop_run=False):
        '''
        Runs the model with parameters randomly sampled from the MCMC chains created during the MCMC sampling procedure. 
        Outputs are returned as a numpy array. 

        Parameters
        ----------
        n_samples : int or 'all', optional
            The number of random samples to take from the MCMC chains. Default is 'all' of the samples.

        parallel : bool, optional
            Run each sample in parallel across all processors. 
            WARNING: if a parameter evolution function is used by the Simulate object, set this to False. Otherwise the sampling freezes. 

        component_number : int, optional
            The component number of the model component to be used as the model output. 
        
        n_burn : int, optional
            The number of initial steps to burn (discard) in the MCMC chain.

        thin : int, optional
            Use every 'thin' number of samples along the MCMC chains (called thinning). 

        override_stop_run : bool, optional
            Overrides stopping running this function if there is already a set of model runs sampled from the MCMC procedure.

        Returns
        -------
        xy_array_output : list
            A list of model xy outputs (first column = time, second column = model output). 
        '''


        # do an initial check for chains that have already been sampled
        if self._sampled_xy_data != [] and override_stop_run == False:
            print('get_chain_outputs(): ')
            print('RUN NOT STARTED: there are already model outputs stored by the optimiser.\nTo override and aquire model outputs from a new sample, set "override_stop_run" to True.')
            print('Sampled data from previous run returned.')
            return self._sampled_xy_data

        if n_samples == 'all':
            n_samples = self._emcee_sampler.get_chain().shape[0]

        self._sampling_component_number = component_number

        print(f'Getting {n_samples} sampled model outputs from MCMC chain with {n_burn} discarded steps and {thin} thinning steps...')
        chains = self._emcee_sampler.get_chain(discard=n_burn,thin=thin)

        # select n_samples number of random samples from the MCMC chains

        random_sample_numbers = set([])
        while len(random_sample_numbers) < n_samples:
            randint = np.random.choice(n_samples)
            random_sample_numbers.add(int(randint))

        # make a list of simulate objects (deepcopies)
        sim_obj_list = []
        for i in random_sample_numbers:
            sim_obj = copy.deepcopy(self.simulate)

            # edit the parameter dictionary with sampled parameter values
            for k, key in enumerate(self._vary_param_keys):

                # chains.shape = (n_samples, n_walkers, n_dim)
                # pick a random chain to sample from
                rand_chain_no = np.random.choice(chains.shape[1])

                # the kth varying parameter from a random chain in the ith sample 
                sim_obj.parameters[key].value = chains[i,int(rand_chain_no),k]

            sim_obj_list.append(sim_obj)

        # update the parameter objects with the statistics from MCMC sampling (including burn-in and thinning)
        self._update_param_stats(self._emcee_sampler,n_burn=n_burn,thin=thin)

        # do the model runs in parallel (default)
        if parallel:
            print('Running sampled model simulations in parallel...')
            from multiprocessing import Pool

            # carrying out paralell things in an IDE on Windows needs this line below
            if __name__ == "__main__":
                
                with Pool() as p:
                    xy_array_output = p.map(self._get_xy_data_sample,sim_obj_list)

                    print('xy_array_output',xy_array_output)
                #return xy_array_output

            # otherwise (Jupyter notebook) we can skip that
            else: 
                with Pool() as p:
                    xy_array_output = p.map(self._get_xy_data_sample,sim_obj_list)

                    #print(len(xy_array_output))
                #return xy_array_output

        # do the model runs in series
        else:
            print('Running sampled model simulations in series...')
            xy_array_output = []
            for sim in sim_obj_list:
                xy_array = self._get_xy_data_sample(sim)
                xy_array_output.append(xy_array)

        self._sampled_xy_data = xy_array_output
        return xy_array_output



    def _get_xy_data_sample(self, sim):

        component_number = self._sampling_component_number
        
        sim.run(sim.run_params['n_layers'],sim.run_params['rp'],sim.run_params['time_span'],
            sim.run_params['n_time'],sim.run_params['V'],sim.run_params['A'],
            sim.run_params['layer_thick'],
            sim.run_params['Y0'],ode_integrator=sim.run_params['ode_integrator'],
            ode_integrate_method=sim.run_params['ode_integrate_method'],
            rtol=sim.run_params['rtol'], atol=sim.run_params['atol'])

        xy_data = sim.xy_data_total_number(components=int(component_number))

        
        xy_array = xy_data
        #print(xy_array.shape)
        return xy_array

    def _update_param_stats(self,sampler,n_burn,thin):

        '''
        Updates the Parameter.stats dictionary after MCMC sampling.
        '''

        sim = self.simulate
        
        chains = sampler.get_chain(discard=n_burn,thin=thin)
        _,nwalkers,_ = chains.shape

        # get the flattened chains
        flat_chains = sampler.get_chain(flat=True,discard=n_burn,thin=thin)

        nsamples, ndim = flat_chains.shape

        vary_param_keys = self._vary_param_keys

        # make mcmc stats dict for each param

        # for each varying param, calculate stats and update mcmc stats dict

        for i, key in enumerate(vary_param_keys):

            # calc stats
            param_chain = flat_chains[:,i]
            
            mean = np.mean(param_chain)
            percentile_2point5 = np.percentile(param_chain,2.5)
            percentile_97point5 = np.percentile(param_chain,97.5)
            percentile_25 = np.percentile(param_chain,25.0)
            percentile_75 = np.percentile(param_chain,75.0)
            std = np.std(param_chain)
            stats_dict = {'mean_mcmc': mean,
                          '2.5th percentile': percentile_2point5,
                          '25th percentile': percentile_25,
                          '75th percentile': percentile_75,
                          '97.5th percentile': percentile_97point5,
                          'std_mcmc': std,
                          'n_samples':nsamples,
                          'n_walkers': nwalkers,
                          'n_burn': n_burn,
                          'n_thin': thin,
                          'original_value': sim.parameters[key].value}

            sim.parameters[key].stats = stats_dict





