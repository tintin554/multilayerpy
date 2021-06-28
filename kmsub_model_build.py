# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 19:19:11 2021

@author: Adam
"""

'''
Writing a custom KM-SUB model
'''

'''
input = dict with each col representing a component (starting from comp 0)
numer of reactions


rows:

define strings for each dydt for: 1. surface, 2. sub-surface, 3. bulk, 4. core

'''

import numpy as np

class ReactionScheme:
    '''
    Defining the reaction scheme (what reacts with what)
    '''
    
    def __init__(self,name='rxn scheme',n_components=None,
                 reaction_tuple_list=None,products_of_reactions_list=None,
                 component_names=[],reactant_stoich=None, product_stoich=None):
    
        # name of reaction scheme
        self.name = name # *error if not a string
        
        # number of components
        self.n_components = n_components # *error if None
    
        # tuple of component reactions e.g. (1,2) means component 1 reacts with 2
        # for first order decay, e.g. (1,0) means component 1 decays first order
        self.reaction_tuples = reaction_tuple_list #*error if: not tuple list, same rxn more than once and len != n_components
        
        # check same reaction is not repeated
        # tuple list of which components are produced from each reaction
        self.reaction_products = products_of_reactions_list # * error if None and len != n_components
        
        self.reactant_stoich = reactant_stoich
        
        self.product_stoich = product_stoich
        
        # list of component names, defaults to empty list if nothing supplied
        self.comp_names = component_names # *error if not list of strings with len = n_components
        
        # make a "checked" state, this would need to be True to be used in the model builder
        self.checked = False

        
    def validate_reaction_scheme(self):
        '''
        XXX
        '''
        self.name
        self.n_components
        self.reaction_tuples
        self.reaction_products
        self.comp_names
        self.checked
        # check name value is a string 
        try:
            type(self.name) == str
            
        except TypeError:
            print('The name needs to be a string')
            
        # check n_components = number of unique numbers in rxn & prod. tuples
        unique_comps = set([])
        for tup in self.reaction_tuples:
            x, y = tup
            # add component numbers to unique comps set
            unique_comps.add(x)
            unique_comps.add(y)
        for tup in self.reaction_products:
            if len(tup) == 1:
                # add component number to unique comps set
                # the add() method of a set will only add the number
                # if it is unique
                x = tup
                unique_comps.add(x)
                
            else:
                x, y = tup
                # add component numbers to unique comps set
                unique_comps.add(x)
                unique_comps.add(y)
            
        # if the number of unique components != len(unique_comps), 
        # error - either too many or too few components, ask to check if this is
        # correct
        try:
            assert (self.n_components == len(unique_comps)) == True 
            ##** sort this out so that the error message works**
        
        except AssertionError:
            if float(self.n_components) > float(len(unique_comps)):
                diff = float(self.n_components) - float(len(unique_comps))
                print(f'n_components ({self.n_components}) is {diff} more than the unique reactants + products provided in reaction and product list of tuples')
            
            if float(self.n_components) < float(len(unique_comps)):
                diff = float(len(unique_comps)) - float(self.n_components)  
                print(f'n_components ({self.n_components}) is {diff} less than the unique reactants + products provided in reaction and product list of tuples')
            
        # comp names should be strings
        isstring_bool_list = []
        for name in self.comp_names:
            string_bool = name == str
            isstring_bool_list.append(string_bool)
            
        try:
            if self.comp_names != []:
                False not in isstring_bool_list
                
        except TypeError as err:
            print(err,'...All component names need to be strings')
            
        self.checked = True
        
        
    
class ModelComponent:
    '''
    Model component class with information about it's role in the reaction scheme
    '''
    
    def __init__(self,diff_coeff,reaction_scheme,component_number=None,
                 name=None,gas=False,compartmental=False,diameter=None):
        # make strings representing surf, sub-surf, bulk and core dydt
        # dependent on reaction scheme
        # initially account for diffusion
        
        # optional name for the component
        self.name = name
        if name == None and gas != True:
            self.name = 'Y{}'.format(component_number)
            
        elif name == None and gas == True:
            self.name = 'X{}'.format(component_number)
            
        self.w = None # mean molecular thermal velocity of component in gas phase cm s-1
        
        self.diff_coeff = diff_coeff # cm2 s-1
        
        self.component_number = component_number
        # if component_number == None or 0, error
        
        # the effective molecular diameter 
        self.diameter = diameter # cm
        
        self.reaction_scheme = reaction_scheme
        
        self.compartmental = compartmental
        
        self.param_dict = None
        
        self.gas = gas
        
        # define initial strings for each differential equation to be solved
        # remembering Python counts from 0
        if component_number == 1:
            self.surf_string = 'dydt[0] = '  
            self.firstbulk_string = 'dydt[1] = '
            self.bulk_string = 'dydt[{}*Lorg*{}+i] = '.format(component_number-1,component_number)
            self.core_string = 'dydt[{}*Lorg+{}] = '.format(component_number,component_number-1)
        else:
            self.surf_string = 'dydt[{}*Lorg+{}] = '.format(component_number-1,component_number-1)
            self.firstbulk_string = 'dydt[{}*Lorg+{}] = '.format(component_number-1,component_number)
            self.bulk_string = 'dydt[{}*Lorg*{}+i] = '.format(component_number-1,component_number)
            self.core_string = 'dydt[{}*Lorg+{}] = '.format(component_number,component_number-1)
        
        # add to initial strings later with info from ReactionScheme
        
        # add mass transport terms to each string
        
        #surf transport different for gases and non-volatiles
        if self.gas == True:
            if component_number == 1:
                self.surf_string += 'kbs_1 * y[1] - ksb_1 * y[0] '
                self.firstbulk_string += '(ksb_1 * y[0] - kbs_1 * y[1]) * (A[0]/V[0]) + kbbx_1[0] * (y[2] - y[1]) * (A[1]/V[0]) '
                self.bulk_string += 'kbbx_1[i] * (y[{}*Lorg*{}+(i-1)] - y[{}*Lorg*{}+i]) * (A[i]/V[i]) + kbbx_1[i+1] * (y[{}*Lorg*{}+(i+1)] - y[{}*Lorg*{}+i]) * (A[i+1]/V[i]) '.format(component_number-1,component_number,
                                           component_number-1,component_number,component_number-1,component_number,component_number-1,component_number)
                self.core_string += 'kbbx_1[-1] * (y[{}*Lorg+{}-1] - y[{}*Lorg+{}]) * (A[-1]/V[-1]) '.format(component_number,component_number-1,component_number,component_number-1)
            else:
                self.surf_string += 'kbs_{} * y[{}*Lorg+{}]] - ksb_{} * y[{}*Lorg+{}] '.format(component_number,component_number-1,component_number,component_number,component_number-1,component_number-1)
                self.firstbulk_string += '(ksb_{} * y[{}*Lorg+{}] - kbs_{} * y[{}*Lorg+{}]) * (A[0]/V[0]) + kbbx_{}[0] * (y[{}*Lorg+{}] - y[{}*Lorg+{}]) * (A[1]/V[0]) '.format(component_number,component_number-1,component_number-1,component_number,
                                          component_number-1,component_number,component_number,component_number-1,component_number+1,component_number-1,component_number)
                self.bulk_string += 'kbbx_{}[i] * (y[{}*Lorg*{}+(i-1)] - y[{}*Lorg*{}+i]) * (A[i]/V[i]) + kbbx_{}[i+1] * (y[{}*Lorg*{}+(i+1)] - y[{}*Lorg*{}+i]) * (A[i+1]/V[i]) '.format(component_number,component_number-1,component_number,
                                           component_number-1,component_number,component_number,component_number-1,component_number,component_number-1,component_number)
                self.core_string += 'kbbx_{}[-1] * (y[{}*Lorg+{}-1] - y[{}*Lorg+{}]) * (A[-1]/V[-1]) '.format(component_number,component_number,component_number-1,component_number,component_number-1)
        else:
            if component_number == 1:
                self.surf_string += 'kbss_1 * y[1] - kssb_1 * y[0] '
                self.firstbulk_string += '(kssb_1 * y[0] - kbss_1 * y[1]) * (A[0]/V[0]) + kbby_1[0] * (y[2] - y[1]) * (A[1]/V[0]) '
                self.bulk_string += 'kbby_1[i] * (y[{}*Lorg*{}+(i-1)] - y[{}*Lorg*{}+i]) * (A[i]/V[i]) + kbby_1[i+1] * (y[{}*Lorg*{}+(i+1)] - y[{}*Lorg*{}+i]) * (A[i+1]/V[i]) '.format(component_number-1,component_number,
                                           component_number-1,component_number,component_number-1,component_number,component_number-1,component_number)
                self.core_string += 'kbby_1[-1] * (y[{}*Lorg+{}-1] - y[{}*Lorg+{}]) * (A[-1]/V[-1]) '.format(component_number,component_number-1,component_number,component_number-1)
                
            else:
                self.surf_string += 'kbss_{} * y[{}*Lorg+{}]] - kssb_{} * y[{}*Lorg+{}] '.format(component_number,component_number-1,component_number,component_number,component_number-1,component_number-1)
                self.firstbulk_string += '(kssb_{} * y[{}*Lorg+{}] - kbss_{} * y[{}*Lorg+{}]) * (A[0]/V[0]) + kbby_{}[0] * (y[{}*Lorg+{}] - y[{}*Lorg+{}]) * (A[1]/V[0]) '.format(component_number,component_number-1,component_number-1,component_number,
                                          component_number-1,component_number,component_number,component_number-1,component_number+1,component_number-1,component_number)
                self.bulk_string += 'kbby_{}[i] * (y[{}*Lorg*{}+(i-1)] - y[{}*Lorg*{}+i]) * (A[i]/V[i]) + kbby_{}[i+1] * (y[{}*Lorg*{}+(i+1)] - y[{}*Lorg*{}+i]) * (A[i+1]/V[i]) '.format(component_number,component_number-1,component_number,
                                           component_number-1,component_number,component_number,component_number-1,component_number,component_number-1,component_number)
                self.core_string += 'kbby_{}[-1] * (y[{}*Lorg+{}-1] - y[{}*Lorg+{}]) * (A[-1]/V[-1]) '.format(component_number,component_number,component_number-1,component_number,component_number-1)
                
                
        # add reactions
    
class DiffusionRegime():
    '''
    XXX
    Calling this will build the diffusion regime and return a list of strings
    defining the composition-dependent (or not) diffusion evolution for each 
    component and the definition of kbb for each component.
    '''
    # vignes, obstruction, no compositional dependence
        
    # which components have a comp dependence?
    # for each component write definition of D as a string
    # ouput a string which defines kbb, ksb etc (?)
    def __init__(self,model_components_dict,regime='vignes',diff_dict=None):
        
        # string defining the diffusion regime to use
        self.regime = regime
        
        # dictionary describing composition-dependent diffusion for each 
        # model component
        self.diff_dict = diff_dict
        
        # list of strings descibing kbb for each component
        # movement of each component between model layers
        self.kbb_strings = None
        
        # list of strings for kbs/ksb and kbss/kssb
        self.kbs_strings = None
        self.ksb_strings = None
        self.kbss_strings = None
        self.kssb_strings = None
        
        # list of strings describing Db for each model component
        self.Db_strings = None
        
        # list of strings describing Ds for each model component
        self.Ds_strings = None
        
        self.req_diff_params = None

        self.model_components_dict = model_components_dict
    
            
        # A boolean which changes to true once the diffusion regime has been 
        # built
        self.constructed = False
        
        
    def __call__(self):
        
        # for each component write D and kbb, kbs string
        # Db kbb will be an array
        # write f_y also - no, include in ModelBuilder **
        
        diff_dict = self.diff_dict
        Db_definition_string_list = []
        Ds_definition_string_list = []
        kbb_string_list = [] # no need for this, it's the same definition
        ksb_string_list = []
        kbs_string_list = []
        kbss_string_list = []
        kssb_string_list = []
        req_diff_params = set([])
        
        # for each component in the diffuion evolution dictionary
        # this dict should be the same length as the number of components
        # in the model
        #DO THE SAME FOR SURFACE CONCS ETC.
        for i in np.arange(len(diff_dict)):
            # only consider components who's diffusivities are composition-
            # dependent
            if diff_dict[f'{i+1}'] != None:
                
                # composition dependence tuple
                compos_depend_tup = diff_dict[f'{i+1}']
                
                # vignes diffusion
                if self.regime == 'vignes':
                    # initially dependent on D_comp in pure comp
                    # raised to the power of fraction of component
                    Db_string = f'Db_{i+1}_arr = (Db_{i+1}_arr**f_{i+1}_arr) '
                    
                    Ds_string = f'Ds_{i+1} = (Db_{i+1}**fs_{i+1}) '
                    
                    req_diff_params.add(f'Db_{i+1}')
                    
                    # loop over depending components 
                    for comp in compos_depend_tup:
                        Db_string += f'* (Db_{i+1}_{comp}_arr**f_{comp}_arr) '
                        
                        Ds_string += f'* (Db_{i+1}_{comp}**fs_{comp}) '
                        
                        req_diff_params.add(f'Db_{i+1}_{comp}')
                        
                        
                    Db_definition_string_list.append(Db_string)
                    Ds_definition_string_list.append(Ds_string)
                    
                    
                        
                # km-sub combination
                elif self.regime == 'fractional':
                    # initially dependent on D_comp in pure comp
                    # multiply by fraction of component
                    Db_string = f'Db_{i+1}_arr = (Db_{i+1}_arr * f_{i+1}_arr) '
                    
                    Ds_string = f'Ds_{i+1} = (Db_{i+1}**fs_{i+1}) '
                    
                    req_diff_params.add(f'Db_{i+1}')
                    
                    # loop over depending components 
                    for comp in compos_depend_tup:
                        Db_string += f'+ (Db_{i+1}_{comp}_arr * f_{comp}_arr) '
                        
                        Ds_string += f'+ (Db_{i+1}_{comp} * fs_{comp}) '
                        
                        req_diff_params.add(f'Db_{i+1}_{comp}')
                        
                    Db_definition_string_list.append(Db_string)
                    Ds_definition_string_list.append(Ds_string)
                    
                    
                # obstruction theory
                elif self.regime == 'obstruction':
                    # only dependent on total fraction of products
                    sum_product_fractions = 'f_prod = '
                    
                    # add in fraction of each product to sum of product frac.
                    # string
                    for a, comp in enumerate(compos_depend_tup):
                        if a == 0:
                            sum_product_fractions += f'f_{comp} '
                        else:
                            sum_product_fractions += f'+ f_{comp} '
                            
                    # obstruction theory equation (Stroeve 1975)    
                    Db_string = f'Db_{i+1}_arr = (Db_{i+1}_arr * (2 - 2 * f_prod) / (2 + f_prod)'
                    
                    Ds_string = f'Ds_{i+1} = (Db_{i+1} * (2 - 2 * fs_prod) / (2 + fs_prod)'
                    
                    req_diff_params.add(f'Db_{i+1}')
                    
                    Db_definition_string_list.append(Db_string)
                    Ds_definition_string_list.append(Ds_string)
                    
                # Other diffusion regimes may be added in later developments
                
                
            # else there is no evolution and just Db of pure
            # component, end of story
            else:
                Db_string = f'Db_{i+1}_arr = D_{i+1}_arr'
                Ds_string = f'Ds_{i+1} = D_{i+1}'
                
                req_diff_params.add(f'Db_{i+1}')
                
                Db_definition_string_list.append(Db_string)
                Ds_definition_string_list.append(Ds_string)
                
            # define kbb and kbs/ksb strings for each component
            if self.model_components_dict[f'{i+1}'].gas == True:
                ksb_string = 'ksb_{} = H_{} * kbs_{} / Td_{} / (W_{}*alpha_s_{}/4) '.format(i+1,i+1,i+1,i+1,i+1,i+1)
                ksb_string_list.append(ksb_string)
                
                kbs_string = 'kbs_{} = (4/pi) * Ds_{} / delta '.format(i+1,i+1)
                kbs_string_list.append(kbs_string)
                
                kbb_string = 'kbbx_{} = (4/pi) * Db_{}_arr / delta '.format(i+1,i+1)
                kbb_string_list.append(kbb_string)
                
            else:
                kssb_string = 'kssb_{} = kbss_{} / delta_{} '.format(i+1,i+1,i+1)
                kssb_string_list.append(kssb_string)
                
                kbss_string = 'kbss_{} = (8*Db_{}_arr[0])/((delta+delta_{})*pi) '.format(i+1,i+1,i+1)
                kbss_string_list.append(kbss_string)
                
                kbb_string = 'kbby_{} = (4/pi) * Db_{}_arr / delta '.format(i+1,i+1)
                kbb_string_list.append(kbb_string)
        
        # update the Db_strings and Ds_strings attributes
        self.Db_strings = Db_definition_string_list
        
        self.Ds_strings = Ds_definition_string_list
        
        self.kbb_strings = kbb_string_list
        self.kbs_strings = kbs_string_list
        self.ksb_strings = ksb_string_list
        self.kbss_strings = kbss_string_list
        self.kssb_strings = kssb_string_list
        
        
        # this diffusion regime is now constructed
        self.constructed = True        
        
    def __str__(self):
        '''
        This will define what gets printed to the console when
        print(DiffusionRegime) is called.
        
        Returns
        -------
        None.

        '''
    def savetxt(self,filename):
        '''
        Save the diffusion regime as a .txt file with each Db/Ds and kbb
        definition for each component

        Returns
        -------
        saved .txt file in current working directory

        '''

class ModelBuilder():
    '''
    An object which constructs the model file from the reaction scheme, list
    of model components and a diffusion regime
    '''
    
    def __init__(self,reaction_scheme,model_components_dict,diffusion_regime,
                 volume_layers,area_layers,n_layers,
                 model_type='KM-SUB',geometry='spherical'):
       '''
       XXX
       '''

       self.reaction_scheme = reaction_scheme
       
       self.model_components = model_components_dict
       
       self.diffusion_regime = diffusion_regime # error if not dict
       
       self.geometry = geometry
       
       self.volume_layers = volume_layers
       
       self.area_layers = area_layers
       
       self.n_layers = n_layers
       
       self.model_type = model_type # error if not in accepted types
       
       # a SET of strings with names of required parameters
       self.req_params = None
       
       # list of output strings for final file write and coupling to 
       # compartmental models
       self.file_string_list = []
       
       self.constructed = False
       
       def build(self, name_extention=None, date=False):
           '''
           Builds the model and saves it to a separate .py file
           Different for different model types 
           '''
           # the list of strings which will be used with file.writelines()
           # at the end
           master_string_list = []
           four_space = '    '
           
           date = ''
           mod_type = self.model_type
           rxn_name = self.reaction_scheme.rxn_name
           mod_comps = self.model_components
           
           heading_strings = ['###############################################\n',
                              f'#A {mod_type} model constructed using MultilayerPy\n',
                              '\n',
                              f'#Created {date}\n',
                              '\n',
                              f'#Reaction name: {rxn_name}\n',
                              f'#Geometry: {self.geometry}\n',
                              f'#Number of model components: {len(self.model_components)}\n',
                              f'#Diffusion regime: {self.diffusion_regime}\n',
                              '###############################################\n']
           
           # append the heading to the master strings
           for s in heading_strings:
               master_string_list.append(s)
               
           # define dydt function
           func_def_strings = ['\n def dydt(t,y,param_dict,V,A):\n',
                               '    """ Function defining ODEs, returns list of dydt"""\n',
                       ]
           
           for s in func_def_strings:
               master_string_list.append(s)
               
           # initialise dydt output array
           n_comps = len(mod_comps)
           init_dydt = f'\n    dydt = np.zeros(Lorg * {n_comps} + {n_comps})'
           master_string_list.append(init_dydt)
           
           # unpack all params from params dictionary
           master_string_list.append('\n    #--------------Unpack parameters---------------\n')
           
           for param_str in self.req_params:
               param_unpack_str = '\n    ' + param_str + ' = param_dict[' + '"' + param_str + '"]'
               
               master_string_list.append(param_unpack_str)
               
               # convert to surface reaction rates from bulk reaction rates
               if 'k_' in param_str and '_surf' not in param_str:
                   k_surf_string = '\n    ' + param_str + '_surf = ' + param_str + ' * scale_bulk_to_surf'
                   
                   master_string_list.append(k_surf_string)
           
           # surface uptake definition for each gas component
           master_string_list.append('\n    #--------------Define surface uptake parameters for gas components---------------\n')
           
           # calculate surface fraction of each component
           master_string_list.append('\n    #calculate surface fraction of each component\n')
           for comp in mod_comps:
               comp_no = comp.component_number
               fs_string = f'    fs_{comp_no} = y[{comp_no}*Lorg+{comp_no}] / ('
               
               # loop over each other component and add to the fs_string
               for ind, c in enumerate(mod_comps):
                   if c.component_number != comp_no:
                       cn = c.component_number
                       if ind == 0:
                           fs_string += f'y[{cn}*Lorg+{cn}] '
                       else:
                           fs_string += f'+ y[{cn}*Lorg+{cn}]'
                    
             # close the bracket on the demoninator
               fs_string += ')'
               
               master_string_list.append(fs_string)
           
           # for each gaseous model component, calculate surface uptake/loss params
           # first calculate surface coverage separately
           
           surf_cover_str = '\n    surf_cover = '
           counter = 0
           for comp in mod_comps:
               if comp.gas == True:
                   comp_no = comp.component_number
                   if counter == 0:
                       surf_cover_str += f'delta_{comp_no}**2 * y[{comp_no}*Lorg+{comp_no}] '
                       counter += 1
                   else:
                       surf_cover_str += f'+ delta_{comp_no}**2 * y[{comp_no}*Lorg+{comp_no}]'
           
           for comp in mod_comps:
               if comp.gas == True:
                   comp_no = comp.component_number
                   
                   if comp.name != None:
                       master_string_list.append(f'\n    # component {comp_no} surf params\n')
                   else:
                       master_string_list.append(f'\n    # component {comp_no} ({comp.name}) surf params\n')
                   
                   # alpha_s_X
                   alpha_s_str = f'\n    alpha_s_{comp_no} = alpha_s_0_{comp_no} * (1-surf_cover)'
                   master_string_list.append(alpha_s_str)
                   
                   # J_coll_X
                   j_coll_str = f'\n    J_coll_X_{comp_no} = Xgs_{comp_no} * w_{comp_no}/4'
                   master_string_list.append(j_coll_str)
                   
                   # J_ads_X
                   j_ads_str = f'\n    J_ads_X_{comp_no} = alpha_s_{comp_no} * J_coll_X_{comp_no}'
                   master_string_list.append(j_ads_str)
                   
                   # J_des_X
                   j_des_str = f'\n    J_des_X_{comp_no} = kd_X_{comp_no} * y[{comp_no}*Lorg+{comp_no}]'
                   master_string_list.append(j_des_str)
            
           
            # diffusion evolution
           
           master_string_list.append('\n    #--------------Diffusion evolution---------------\n')
           
           # calculate bulk fraction array for each component in each layer
           # DO THIS ADAM - Make Db arrays etc.
           
           diff_regime = self.diffusion_regime
           
           master_string_list.append('\n    # surface diffusion')
           #Ds
           for s in diff_regime.Ds_strings:
               master_string_list.append(four_space+s+'\n')
           # ksb
           for s in diff_regime.ksb_strings:
               master_string_list.append(four_space+s+'\n')
           # kbs
           for s in diff_regime.kbs_strings:
               master_string_list.append(four_space+s+'\n')
           #kssb
           for s in diff_regime.kssb_strings:
               master_string_list.append(four_space+s+'\n')
           #kbss
           for s in diff_regime.kbss_strings:
               master_string_list.append(four_space+s+'\n') 
               
           master_string_list.append('\n    # bulk diffusion')
           #Db
           for s in diff_regime.Db_strings:
               master_string_list.append(four_space+s+'\n')
           # kbb
           for s in diff_regime.kbb_strings:
               master_string_list.append(four_space+s+'\n')
           
            
           '''
           For each component, loop over the reaction scheme and identify which
           rxns the component is involved in. Check if stoichiometry involved.
           Add the reaction as necessary to the relevant string.
           '''
       
    def __call__(self):
        '''
        XXX
     
        Returns
        -------
        None.
     
        '''
        # validate, check req_params are in the param dicts of the corresponding
        # ModelComponent objects
        # create param_dict, user warning to provide param_dict#
        
   
class Parameter():
    '''
    A class which will define a parameter object.
    '''
    def __init__(self, value=np.inf, name=None,bounds=None, vary=False):
        
        self.name = name
        
        # define bounds if given
        if bounds is not None:
            self.bounds = bounds
            
        self.value = value
        self.vary = vary
       
        

    
# testing 123
    
init = ReactionScheme(n_components=3,
                      reaction_tuple_list=[(1,2)],
                      products_of_reactions_list=[(3,)])
                      
# something that will tell me what each component                       
diff_dict = {'1' : None,
             '2': (3,1),
             }                     
                      
                      
                      
                      
                      