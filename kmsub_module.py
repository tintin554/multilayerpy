# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 16:31:36 2021

@author: Adam
"""

'''
Adam Milsom, University of Birmingham Jan 2021

KM-SUB multi-layer model module

'''
import numpy as np


class KMSUB:
    '''
    An object containing the KM-SUB code and associated parameters.
    There is the facility to run the code with pre-defined parameters.
    The parameters can be optimised using SciPy odeint differential_evolution.
    Different iterations of the oleic acid-ozone reaction scheme are available.
    You can choose between a spherical or planar geometry.
    
    *future iterations will allow the user to create a custom reaction scheme
    with input values for the number of model components, whether they are a gas etc.
    '''
    
    def __init__(self,param_dict=None,n_layers=100,n_time=500):
        '''
        Initiallise the object with a dictionary of parameters (a default set is applied if None)
        The keys of the param_dict must be the same as in the default case.
        '''
        
        # initiallise the params dictionary
        
        # provide default params if no param_dict provided
        if param_dict == None:
            self.params = {'alpha_s0' : 1,
                           'T' : 298,
                           'X_gs' : 1.3052e15,
                           'Hcp' : 4.80e-4,
                           'scale_bulktosurf' : 4.41e6,
                           'k_bulk' : 1.13e-18,
                           'k_surf' : 6e-12,
                           'k_CI' : 1.44,
                           'k_CI_CI' : 5.06e-18,
                           'k_CI_C9' : 1.86e-17,
                           'k_CI_di' : 1.99e-16,
                           'Td' : 1e-9,
                           'delta_O3' : 0.4e-7,
                           'Dx' : 1e-5,
                           'Dx_di' : 1e-8,
                           'Dx_tri' : 1e-9,
                           'delta_ole' : 0.8e-7,
                           'Dy' : 1e-10,
                           'Dy_di' : 1e-11,
                           'Dy_tri' : 1e-12,
                           'D_di' : 1e-9,
                           'D_tri' : 1e-9,
                           'rp' : 1,
                           'Lorg' : self.n_layers,
                           'c' : 0.454,
                           'kloss' : 1.4,
                           'R' : 82.0578,
                           'W' : 3.6e4}
        else:
            #*raise an error if not dict
            self.params = param_dict
        
        # number of timepoints for integration
        self.n_time = n_time
        
        # bounds used for differential evolution optimisation of model params
        self.bounds = None
        
        # name of the model system (if desired)
        self.name = 'Default oleic acid-ozone system'
        
        
    # check all required parameters included
    
    # run the model
    
    






