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
import importlib
from scipy.optimize import differential_evolution, minimize
from multilayerpy.simulate import Data
import emcee


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
        A function which takes in data (in the Data object format), model_y
        values and the rp parameter supplied to the cost_func method. 
        Returns the value of the cost function. Lower values = better fit.
    param_evolution_func : func, optional
        A function which is called f(t,y,param_dict,param_evolution_func_extra_vary_params).
        It returns the param_dict and allows for model parameters to evolve
        over time, values of y (array of ODE solutions evaluated at time t).
        Parameters can be parameterised with extra parameters which themselves can
        vary. See param_evolution_func_extra_vary_params below. 
    param_evolution_func_extra_vary_params : list, optional
        List of Parameter objects which represent additional parameters used
        to evolve/parameterise model input parameters. These values can be optimised
        during model optimisation. Supplied to the param_evolution_func function.
    lnprior_func : func, optional
        A custom log-prior function for model parameters. Takes in self and array
        of varying parameters as arugments. 
    lnlike_func : func, optional
        A custom log-likelihood function which takes y_experiment, y_model and 
        (optionally) y_err and returns the log-likelihood of the fit. 
    '''
    
    def __init__(self,simulate_object,cost='MSE',cfunc=None,param_evolution_func=None,
                 param_evolution_func_extra_vary_params=None,lnprior_func=None,
                 lnlike_func=None,custom_model_y_func=None):
        
        
        self.simulate = simulate_object
        self.model = simulate_object.model
        
        # make data into a Data object if not already
        data = simulate_object.data
        if type(data) != Data:
            self.data = Data(data,norm=True)
        else:
            self.data = simulate_object.data
        
        self.cost = cost
        self.cfunc = cfunc
        self.cost_func_val = None
        self.param_evolution_func = param_evolution_func
        self.param_evolution_func_extra_vary_params = param_evolution_func_extra_vary_params
        self.lnprior_func = lnlike_func
        self.lnlike_func = lnlike_func
        self.custom_model_y_func = simulate_object.custom_model_y_func
        
        self._vary_param_keys = None
        self._extra_vary_params_start_ind = None
        self._fitting_component_no = None
        self._vary_param_bounds = None
        self._emcee_sampler = None
        
    def cost_func(self,model_y,rp=None,weighted=False):
        '''
        Will calculate the cost function used in the optimisation process. 
        A custom cost function will be used if suppled via the cfunc attribute 
        of the Optimizer object. 
        
        If rp and model_y provided, will calculate a weighted cost function. 

        Parameters
        ----------
        model_y : np.ndarray
            Model y values. If None, assumes only rp data will be fitted to.
        rp : bool, optional
            Whether to fit to particle radius/film thickness data.
            This data should be the final column in the Data object. 

        Returns
        -------
        float
            Cost function value.

        '''
        # use user-supplied cost function if supplied
        if type(self.cfunc) != type(None):
            val = self.cfunc(self.data,model_y,rp=rp)
        
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
        
        # fitting to rp            
        if type(rp) != type(None):
            rp_expt = expt.rp 
            rp_cost_val = (np.square(rp_expt-rp)).mean(axis=None)
            
            # return an average cost function value accounting for fit to rp
            # and data
            if val != 0.0:
                self.cost_func_val = (val + rp_cost_val) / 2
                return (val + rp_cost_val) / 2
            
            # only fitting to rp
            else:
                self.cost_func_val = rp_cost_val
                return rp_cost_val
            
        self.cost_func_val = val
        return val
    
    def lnlike(self,vary_params):
        '''
        Calculate the log-likelihood of the model-data fit for MCMC sampling.
        
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
        
        if type(self._extra_vary_params_start_ind) != None:
            additional_params = vary_params[self._extra_vary_params_start_ind:]
        else:
            additional_params = None
        
        # update the simulate object parameters 
        for ind, param in enumerate(self._varying_param_keys):
            sim.parameters[param].value = vary_params[ind]
        
        # import the model from the .py file created in the model building
        # process
        model_import = importlib.import_module(f'{self.model.filename[:-3]}') #XXX
            
        # define time interval
        tspan = np.linspace(min(time_span),max(time_span),n_time)
        
        # t_eval for comparing mod + expt at the same timepoints
        # assuming expt time axis is in s
        t_eval = self.data.x
        
      
        model_output = integrate.solve_ivp(lambda t, y:model_import.dydt(t,y,sim.parameters,V,A,n_layers,layer_thick,
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
            loglike = -np.sum(np.square(((data.y - model_y) / data.y_err))) / 2.0
        else:
            loglike = -np.sum(np.square(data.y - model_y)) / 2.0
        
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
        
    
    def fit(self,weighted=False,method='least_squares',component_no='1',n_workers=1,fit_particle_radius=False):
        '''
        Use either a local or global optimisation algorithm to fit the model
        to the data. 

        Parameters
        ----------
        weighted : bool, optional
            Whether to weight the fit to datapoint uncertainties (requires the data to have uncertainties)
        method : str, optional
            Algorithm to be used. 'least_squares' (local) or 'differential_evolution' (global). 
            The default is 'least_squares'.
        component_no : int, optional
            The model component to fit to the data. The default is '1'.
        n_workers : int, optional
            Number of cores to use (only for 'differential_evolution' method). 
            The default is 1.
        fit_particle_radius : bool, optional
            Whether to fit to particle radius data. The default is False.
            
        returns
        ----------
        optimize_result : scipy.optimize.OptimizeResult
            The optimization result represented as a OptimizeResult object.
            Important attributes are: x the solution array, success a Boolean
            flag indicating if the optimizer exited successfully and message
            which describes the cause of the termination. See the documentation
            for more details: 
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
        
        '''
        self._fitting_component_no = component_no
        sim = self.simulate
        param_evolution_func_extra_vary_params = self.param_evolution_func_extra_vary_params
        
        # identify varying params, append to varying_params list and record order
        varying_params = []
        varying_param_keys = []
        param_bounds = []
        
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
                # only for least_squares optimisation (requires an initial guess)
                if method == 'least_squares':
                    for par in param_evolution_func_extra_vary_params:  
                        varying_params.append(par.value)
                        param_bounds.append(par.bounds)
            
        def minimize_me(varying_param_vals,varying_param_keys,sim,component_no=component_no,
                        extra_vary_params_start_ind=extra_vary_params_start_ind,
                        fit_particle_radius=fit_particle_radius):
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
            model_import = importlib.import_module(f'{self.model.filename[:-3]}')
                
            # define time interval
            tspan = np.linspace(min(time_span),max(time_span),n_time)
            
            # t_eval for comparing mod + expt at the same timepoints
            # assuming expt time axis is in s
            t_eval = self.data.x
            
          
            model_output = integrate.solve_ivp(lambda t, y:model_import.dydt(t,y,sim.parameters,V,A,n_layers,layer_thick,
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
            
                # also fit the particle radius with custom cost function (cfunc)
                if fit_particle_radius:
                    print('\nFitting to particle radius or film thickness **CHECK UNITS**\n(should be in cm)\n')
                    # option to only fit to rp and not number of molecules
                    if type(component_no) == type(None):
                        norm_number_molecules = None
                    cost_val = self.cost_func(norm_number_molecules,rp_t)
            
            
            #print(cost_val)
            return cost_val
        
        if method == 'differential_evolution':
            print('\nOptimising using differential_evolution algorithm...\n')
            result = differential_evolution(minimize_me,param_bounds,
                                        (varying_param_keys,sim,component_no),
                                        disp=True,workers=n_workers)
        elif method == 'least_squares':
            #print(varying_params)
            print('\nOptimising using least_squares Nelder-Mead algorithm...\n')
            result = minimize(minimize_me,varying_params,args=(varying_param_keys,sim,component_no),
                          method='Nelder-Mead',bounds=param_bounds,options={'disp':True, 'return_all':True})
            
        # collect results into a result dict for printing
        res_dict = {}
        for i in range(len(varying_param_keys)):
            key = varying_param_keys[i]
            res_dict[key] = result.x[i]
            
        
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
        param_evolution_func_extra_vary_params = self.param_evolution_func_extra_vary_params
        
        # identify varying params, append to varying_params list and record order
        varying_params = []
        varying_param_keys = []
        param_bounds = []
        
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
                # only for least_squares optimisation (requires an initial guess)
                
                for par in param_evolution_func_extra_vary_params:  
                    varying_params.append(par.value)
                    param_bounds.append(par.bounds)
                    
        self._varying_param_keys = varying_param_keys
        self._extra_vary_params_start_ind = extra_vary_params_start_ind
        self._vary_param_bounds = param_bounds
        
        # make initial guess
        
        ndim = len(varying_params)
        print('Order of varying params:')
        print(varying_param_keys)
        
        # array of initialised chains centred around the initial guess (small gaussian distribution)
        p0 = [np.array(varying_params) * np.random.normal(1.0,1e-9,len(varying_params)) for i in range(n_walkers)]

        
        # run the model with 
        
        sampler = emcee.EnsembleSampler(nwalkers=n_walkers, ndim=ndim,
                                        log_prob_fn=self.lnprob, pool=pool)

        if n_burn != 0:
            print("Running burn-in step...")
            p0, _, _ = sampler.run_mcmc(p0, n_burn,progress=True)
            sampler.reset()
            
            print("Running production run...")
            pos, prob, state = sampler.run_mcmc(p0, samples,progress=True)
            
        else:
            
            print("Running production run without burn-in...")
            pos, prob, state = sampler.run_mcmc(p0, samples,progress=True)
            
        self._emcee_sampler = sampler
        return sampler