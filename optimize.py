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
import simulate
import scipy.integrate as integrate
import kmsub_model_build
import importlib
from scipy.optimize import differential_evolution, minimize


class Optimizer():
    '''
    An object which will take in a Simulate, Model and Data object in order to 
    fit the model to the data with various methods
    '''
    
    def __init__(self,simulate_object,cost='MSE',cfunc=None,param_evolution_func=None,
                 param_evolution_func_extra_vary_params=None,lnprior_func=None,
                 lnlike_func=None):
        
        self.simulate = simulate_object
        self.model = simulate_object.model
        
        # make data into a Data object if not already
        data = simulate_object.data
        if type(data) != simulate.Data:
            self.data = simulate.Data(data,norm=True)
        else:
            self.data = simulate_object.data
        
        self.cost = cost
        self.cfunc = cfunc
        self.cost_func_val = None
        self.param_evolution_func = param_evolution_func
        self.param_evolution_func_extra_vary_params = param_evolution_func_extra_vary_params
        self.lnprior_func = lnlike_func
        self.lnlike_func = lnlike_func
        
        self._vary_param_keys = None
        self._extra_vary_params_start_ind = None
        self._fitting_component_no = None
        self._vary_param_bounds = None
        self._emcee_sampler = None
        
    def cost_func(self,model_y,rp=None):
        
        # use user-supplied cost function if supplied
        if type(self.cfunc) != type(None):
            val = self.cfunc(self.data,model_y,rp=rp)
        
        # use built-in cost function
        elif type(model_y) != type(None):
            cost = self.cost
            expt = self.data
            expt_y = expt.y
        
            if cost == 'MSE':
                    
                val = np.sum((np.square(expt_y-model_y) ** 2)/len(expt_y))
        else:
            val = 0.0
                    
        if type(rp) != type(None):
            rp_expt = expt[:,-1] # experimental rp should be final column (in cm)
            rp_cost_val = np.sum((np.square(rp_expt-rp) ** 2)/len(rp))
            
            if val != 0.0:
                self.cost_func_val = (val + rp_cost_val) / 2
                return (val + rp_cost_val) / 2
            else:
                self.cost_func_val = rp_cost_val
                return rp_cost_val
            
        self.cost_func_val = val
        return val
    
    def lnlike(self,vary_params):
        data = self.data
        sim = self.simulate
        # check if data have yerrs
        zero_count = 0
        for err in data.y_err:
            if float(err) == 0.0:
                zero_count += 1
        
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
                
            bulk_num = bulk_concs[f'{self._fitting_component_no}'] * V
            surf_num = surf_concs[f'{self._fitting_component_no}'] * A[0]
            static_surf_num = np.zeros(len(surf_concs[f'{self._fitting_component_no}'])) # only applicable to km-gap (surf is static surf and sorption layer in km-sub)
            
            bulk_total_num = np.sum(bulk_num,axis=1)
            total_number_molecules = bulk_total_num + surf_num + static_surf_num
            
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
                
                # the data is normalised, so model output will be normalised 
                model_y = total_number_molecules / total_number_molecules[0]
        
        #assert model_output.t == self.data.x, "model and experimental datapoints not equivalent"
        
        # if there are errors, account for them
        if zero_count == 0:
            loglike = -np.sum(np.square(((data.y - model_y) / data.y_err))) / 2.0
        else:
            loglike = -np.sum(np.square(data.y - model_y)) / 2.0
        
        return loglike
    
    def lnprior(self,vary_params):
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
        lp = self.lnprior(vary_params)
        if not np.isfinite(lp):
            return -np.inf
        
        return lp + self.lnlike(vary_params)
        
    
    def fit(self,method='least_squares',component_no='1',n_workers=1,fit_particle_radius=False):
        
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
                        extra_vary_params_start_ind=extra_vary_params_start_ind,fit_particle_radius=fit_particle_radius):
            
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
                
                # the data is normalised, so model output will be normalised 
                norm_number_molecules = total_number_molecules / total_number_molecules[0]
                
                # calculate the cost function
                
                cost_val = self.cost_func(norm_number_molecules)
            
            elif sim.model.model_type.lower() == 'km-gap':
                if type(component_no) != type(None):
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
                    
                    bulk_num = bulk_concs[f'{component_no}'] * V_t
                    surf_num = surf_concs[f'{component_no}'] * A_t[:,0]
                    static_surf_num = static_surf_concs[f'{component_no}'] * A_t[:,0]
                
                    bulk_total_num = np.sum(bulk_num,axis=1)
                    total_number_molecules = bulk_total_num + surf_num + static_surf_num
                    
                    # the data is normalised, so model output will be normalised 
                    norm_number_molecules = total_number_molecules / total_number_molecules[0]
                    
                    # calculate the cost function
                    
                    cost_val = self.cost_func(norm_number_molecules)
            
                # also fit the particle radius with custom cost function (cfunc)
                if fit_particle_radius:
                    print('\nFitting to particle radius **CHECK UNITS**\n(should be in cm)\n')
                    # option to only fit to rp and not number of molecules
                    if type(component_no) == type(None):
                        norm_number_molecules = None
                    cost_val = self.cost_func(norm_number_molecules,rp)
            
            
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
        p0 = [np.array(varying_params) * np.random.normal(1.0,0.4,len(varying_params)) for i in range(n_walkers)]

        import emcee
        
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