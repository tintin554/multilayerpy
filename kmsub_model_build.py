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

class ReactionScheme:
    '''
    Defining the reaction scheme (what reacts with what)
    '''
    
    def __init__(self,name='rxn scheme',n_components=None,reaction_tuple_list=None,products_of_reactions_list=None,component_names=[]):
    
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
        
        # list of component names, defaults to empty list if nothing supplied
        self.comp_names = component_names # *error if not list of strings with len = n_components
        
        # make a "checked" state, this would need to be True to be used in the model builder
        self.checked = False

        
def validate_reaction_scheme(ReactionScheme):
    '''
    XXX
    '''
    ReactionScheme.name
    ReactionScheme.n_components
    ReactionScheme.reaction_tuples
    ReactionScheme.reaction_products
    ReactionScheme.comp_names
    ReactionScheme.checked
    # check name value is a string 
    try:
        type(ReactionScheme.name) == str
        
    except TypeError:
        print('The name needs to be a string')
        
    # check n_components = number of unique numbers in rxn & prod. tuples
    unique_comps = set([])
    for tup in ReactionScheme.reaction_tuples:
        x, y = tup
        # add component numbers to unique comps set
        unique_comps.add(x)
        unique_comps.add(y)
    for tup in ReactionScheme.reaction_products:
        x, y = tup
        # add component numbers to unique comps set
        unique_comps.add(x)
        unique_comps.add(y)
        
    # if the number of unique components != len(unique_comps), 
    # error - either too many or too few components, ask to check if this is
    # correct
    try:
        float(ReactionScheme.n_components) == float(len(unique_comps))
    
    except:
        if float(ReactionScheme.n_components) > float(len(unique_comps)):
            diff = float(ReactionScheme.n_components) - float(len(unique_comps))
            print(f'n_components ({ReactionScheme.n_components}) is {diff} more than the unique reactants + products provided in reaction and product list of tuples')
        
        if float(ReactionScheme.n_components) < float(len(unique_comps)):
            diff = float(len(unique_comps)) - float(ReactionScheme.n_components)  
            print(f'n_components ({ReactionScheme.n_components}) is {diff} less than the unique reactants + products provided in reaction and product list of tuples')
        
    # comp names should be strings
    isstring_bool_list = []
    for name in ReactionScheme.comp_names:
        string_bool = name == str
        isstring_bool_list.append(string_bool)
        
    try:
        if ReactionScheme.comp_names != []:
            False not in isstring_bool_list
            
    except TypeError:
        print('All component names need to be strings')
        
    ReactionScheme.checked = True
        
        
    
class ModelComponent:
    '''
    Model component class with information about it's role in the reaction scheme
    '''
    
    def __init__(self,diff_coeff,reaction_scheme,component_number=None,name=None,react_gas=False,w=None,gas_conc=None):
        pass
        # make strings representing surf, sub-surf, bulk and core dydt
        # dependent on reaction scheme
        # initially account for diffusion
        
        # optional name for the component
        self.name = name
        if name == None and react_gas != True:
            self.name = 'Y{}'.format(component_number)
            
        elif name == None and react_gas == True:
            self.name = 'X{}'.format(component_number)
        
        self.diff_coeff = diff_coeff # cm2 s-1
        if react_gas == True:
            #* error if not a number
            self.w = w # cm s-1, rms speed of the gas molecule
            self.gas_conc = gas_conc
        
        self.component_number = component_number
        # if component_number == None, error
        
        # the effective molecular diameter (delta)
        self.delta = None # cm
        
        self.reaction_scheme = reaction_scheme
        
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
        if react_gas == True:
            if component_number == 1:
                self.surf_string += 'kbs_1 * y[1] - ksb_1 * y[0] '
                self.firstbulk_string += '(ksb_1 * y[0] - kbs_1 * y[1]) * (A[0]/V[0]) + kbbx_1[0] * (y[2] - y[1]) * (A[1]/V[0]) '
                self.bulk_string += 'kbbx_1[i] * (y[{}*Lorg*{}+(i-1)] - y[{}*Lorg*{}+i]) * (A[i]/V[i]) + kbbx_1[i+1] * (y[{}*Lorg*{}+(i+1)] - y[{}*Lorg*{}+i]) * (A[i+1]/V[i]) '.format(component_number-1,component_number,
                                           component_number-1,component_number,component_number-1,component_number,component_number-1,component_number)
            else:
                self.surf_string += 'kbs_{} * y[{}*Lorg+{}]] - ksb_{} * y[{}*Lorg+{}] '.format(component_number,component_number-1,component_number,component_number,component_number-1,component_number-1)
                self.firstbulk_string += '(ksb_{} * y[{}*Lorg+{}] - kbs_{} * y[{}*Lorg+{}]) * (A[0]/V[0]) + kbbx_{}[0] * (y[{}*Lorg+{}] - y[{}*Lorg+{}]) * (A[1]/V[0]) '.format(component_number,component_number-1,component_number-1,component_number,
                                          component_number-1,component_number,component_number,component_number-1,component_number+1,component_number-1,component_number)
                self.bulk_string += 'kbbx_{}[i] * (y[{}*Lorg*{}+(i-1)] - y[{}*Lorg*{}+i]) * (A[i]/V[i]) + kbbx_{}[i+1] * (y[{}*Lorg*{}+(i+1)] - y[{}*Lorg*{}+i]) * (A[i+1]/V[i]) '.format(component_number,component_number-1,component_number,
                                           component_number-1,component_number,component_number,component_number-1,component_number,component_number-1,component_number)
        else:
            if component_number == 1:
                self.surf_string += 'kbss_1 * y[1] - kssb_1 * y[0] '
                self.firstbulk_string += '(kssb_1 * y[0] - kbss_1 * y[1]) * (A[0]/V[0]) + kbby_1[0] * (y[2] - y[1]) * (A[1]/V[0]) '
                self.bulk_string += 'kbby_1[i] * (y[{}*Lorg*{}+(i-1)] - y[{}*Lorg*{}+i]) * (A[i]/V[i]) + kbby_1[i+1] * (y[{}*Lorg*{}+(i+1)] - y[{}*Lorg*{}+i]) * (A[i+1]/V[i]) '.format(component_number-1,component_number,
                                           component_number-1,component_number,component_number-1,component_number,component_number-1,component_number)
                
            else:
                self.surf_string += 'kbss_{} * y[{}*Lorg+{}]] - kssb_{} * y[{}*Lorg+{}] '.format(component_number,component_number-1,component_number,component_number,component_number-1,component_number-1)
                self.firstbulk_string += '(kssb_{} * y[{}*Lorg+{}] - kbss_{} * y[{}*Lorg+{}]) * (A[0]/V[0]) + kbby_{}[0] * (y[{}*Lorg+{}] - y[{}*Lorg+{}]) * (A[1]/V[0]) '.format(component_number,component_number-1,component_number-1,component_number,
                                          component_number-1,component_number,component_number,component_number-1,component_number+1,component_number-1,component_number)
                self.bulk_string += 'kbby_{}[i] * (y[{}*Lorg*{}+(i-1)] - y[{}*Lorg*{}+i]) * (A[i]/V[i]) + kbby_{}[i+1] * (y[{}*Lorg*{}+(i+1)] - y[{}*Lorg*{}+i]) * (A[i+1]/V[i]) '.format(component_number,component_number-1,component_number,
                                           component_number-1,component_number,component_number,component_number-1,component_number,component_number-1,component_number)
        
        
    
class DiffusionRegime:
    pass
    
    