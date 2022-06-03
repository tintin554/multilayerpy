# -*- coding: utf-8 -*-
"""
@author: Adam Milsom

MultilayerPy - build, run and optimise kinetic multi-layer models for
aerosol particles and films.

Copyright (C) 2022 Adam Milsom

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
#import os
from datetime import datetime
import multilayerpy


class ModelType:
    '''
    A class which contains information about the model type

    Parameters
    ----------
    model_type : str
        Defines the model type. Currently allowed types are 'km-sub'
        and 'km-gap'
    geometry : str
        Defines the geometry of the model. Either 'spherical' or 'film'
    '''

    def __init__(self, model_type, geometry):

        # make sure model_type is a string
        assert type(model_type) == str

        # geometry needs to be in list of accepted geometries
        assert geometry.lower() in ['spherical', 'film']

        self.model_type = model_type

        # make sure the model type is in the accepted list
        assert self.model_type.lower() in ['km-sub', 'km-gap']

        self.geometry = geometry


class ReactionScheme:
    '''
    Defining the reaction scheme (what reacts with what?)

    Parameters
    ----------
    model_type_object : multilayerpy.Build.ModelType
        Defines the model type necessary for naming conventions.

    name : str
        Name of the reaction scheme

    reactants : list
        List of tuples defining which components react with which.

        EXAMPLE:
        >>> reactants = [(1,2),(1,3)]

        This states that the first reaction is between component 1 and 2 and
        the second reaction is between component 1 and 3.

    products : list
        List of tuples defining which components are reaction products.

        EXAMPLE:
        >>> products = [(3,),(4,)]

        This states that component 3 is a product of reaction 1 and component 4
        is a product of reaction 2.

    reactant_stoich : list
        List of tuples which define the stoichiometric coefficients (if any)
        applied to each reactant. (Optional).

        EXAMPLE:
        >>> reactants = [(1,2)]
        >>> reactant_stoich = [(0.5,1.0)]

        This states that for reaction 1, reactant 1 reacts with reactant 2 and
        their stoichiometric coefficients are 0.5 and 1.0, respectively.

    product_stoich : list
        List of tuples which define the stoichiometric coefficients (if any)
        applied to each product. (Optional).

        EXAMPLE:
        >>> products = [(3,4)]
        >>> product_stoich = [(0.5,0.5)]

        This states that for reaction 1, the products 3 and 4 have
        stoichiometric coefficients (branching ratios) of 0.5.
    '''

    def __init__(self, model_type_object, name='rxn_scheme',
                 reactants=None, products=None,
                 reactant_stoich=None, product_stoich=None):

        # model type
        self.model_type = model_type_object

        # name of reaction scheme
        self.name = name  # *error if not a string

        # number of components
        self.n_components = None

        # tuple of component reactions e.g. (1,2) means component 1 + 2 react
        # for first order decay, e.g. (1,0) means component 1 decays first ord.
        # *error if: not tuple list,
        # same rxn more than once and len != n_components
        self.reaction_reactants = reactants

        # check same reaction is not repeated
        # tuple list of which components are produced from each reaction
        # * error if None and len != n_components
        self.reaction_products = products
        self.reactant_stoich = reactant_stoich

        self.product_stoich = product_stoich

        # list of component names, defaults to empty list if nothing supplied
        # self.comp_names = component_names # *error if not list of strings
        # with len = n_components

        # =====================================================================
        # checking the properties supplied when instantiating this object
        # =====================================================================

        # make a "checked" state
        self.checked = False

        # check name value is a string
        assert type(self.name) == str, 'The name needs to be a string'

        # check if any of the reaction/product tuple list are not None
        # raise type error if one of them are not supplied
        if None in [self.reaction_reactants, self.reaction_products]:
            raise TypeError("Supply reactant and product tuple lists")

        # check n_components = number of unique numbers in rxn & prod. tuples
        # A set will only accept one of each unique integer
        unique_comps = set([])
        for tup in self.reaction_reactants:
            x, y = tup
            # add component numbers to unique comps set
            unique_comps.add(x)
            unique_comps.add(y)

            assert x != y, "Same component [{x}] appears twice in reaction tuple {tup}"

        for tup in self.reaction_products:

            counted_components = []

            if len(tup) == 1:
                # add component number to unique comps set
                # the add() method of a set will only add the number
                # if it is unique
                x = tup
                unique_comps.add(x)

            else:
                for c in tup:
                    # check same component number not repeate
                    assert c not in counted_components, f"Same component [{c}] appears twice in product tuple {tup}"

                    # add component numbers to unique comps set
                    unique_comps.add(c)

                    counted_components.append(c)

                # x, y = tup
                # # add component numbers to unique comps set
                # unique_comps.add(x)
                # unique_comps.add(y)

        # set the number of components
        self.n_components = len(unique_comps)

        # comp names should be strings
        # isstring_bool_list = []
        # for name in self.comp_names:
        #     string_bool = name == str
        #     isstring_bool_list.append(string_bool)

        # try:
        #     if self.comp_names != []:
        #         False not in isstring_bool_list

        # except TypeError as err:
        #     print(err,'...All component names need to be strings')

        self.checked = True

    def display(self):
        '''
        Function which prints the reaction scheme to the console.
        Not including stoichiometry.

        '''
        strings = ['#########################################################',
                   'Reaction scheme: ' + self.name,
                   'Model type: ' + self.model_type.model_type,
                   '** No stoichiometry shown **',
                   ]
        for i in range(len(self.reaction_reactants)):
            s = f'R{i+1}: '
            for comp_no in self.reaction_reactants[i]:
                if len(s) < 5:
                    s += f'{comp_no} '
                else:
                    s += f'+ {comp_no} '

            s += '-----> '

            for comp_no in self.reaction_products[i]:
                if s[-2] == '>':
                    s += f'{comp_no} '
                else:
                    s += f'+ {comp_no} '

            strings.append(s)
        strings.append('#########################################################')
        for s in strings:
            print(s)


class ModelComponent:
    '''
    Model component class representing a model component (chemical) and key properties.

    Parameters
    ----------
    component_number : int
        The number given to the component. This is how the component is referred
        to in the model building process.

    reaction_scheme : multilayerpy.build.ReactionScheme
        The reaction scheme object created for a model system.

    name : str
        The name of the model component.

    gas : bool
        Whether or not the component is found in the gas phase (volatile)

    comp_dependent_adsorption : bool
        Whether the adsorption of this component is dependent on the composition
        of the particle surface.

    EXAMPLE:
    # ozone as component number 1, no surface composition-dependent adsorption:

    >>> from multilayerpy.build import ModelComponent

    >>> ozone = ModelComponent(1, reaction_scheme, name='ozone', gas=True)

    '''

    def __init__(self, component_number, reaction_scheme,
                 name=None, gas=False, comp_dependent_adsorption=False):
        # make strings representing surf, sub-surf, bulk and core dydt
        # dependent on reaction scheme
        # initially account for diffusion

        # optional name for the component
        self.name = name
        if name is None and gas is not True:
            self.name = 'Y{}'.format(component_number)

        elif name is None and gas is True:
            self.name = 'X{}'.format(component_number)

        self.component_number = component_number
        # if component_number == None or 0, error

        self.reaction_scheme = reaction_scheme

        self.param_dict = None

        self.gas = gas

        self.comp_dependent_adsorption = comp_dependent_adsorption

        self.reactions_added = False

        self.mass_transport_rate_strings = []
        self.mass_transport_rate_inloop_strings = []

        if self.reaction_scheme.model_type.model_type.lower() == 'km-sub':
            # define initial strings for each differential equation to be solved
            # remembering Python counts from 0

            cn = self.component_number

            # mass fluxes
            if self.gas:
                Jsb_str = f'Jsb_{cn} = ksb_{cn} * y[{cn-1}*Lorg+{cn-1}] '
                Jbs_str = f'Jbs_{cn} = kbs_{cn} * y[{cn-1}*Lorg+{cn}] '
                Jb1b2_str = f'Jb1b2_{cn} = kbbx_{cn}[0] * (y[{cn-1}*Lorg+{cn}+1] - y[{cn-1}*Lorg+{cn}]) '
                Jbb_minus1_str = f'Jbb_minus1_{cn} = kbbx_{cn}[i] * (y[{cn-1}*Lorg+{cn}+i-1] - y[{cn-1}*Lorg+{cn}+i]) '
                Jbb_plus1_str = f'Jbb_plus1_{cn} = kbbx_{cn}[i+1] * (y[{cn-1}*Lorg+{cn}+i+1] - y[{cn-1}*Lorg+{cn}+i]) '
                Jbb_core_str = f'Jbb_core_{cn} = kbbx_{cn}[-1] * (y[{cn}*Lorg+{cn-1}-1] - y[{cn}*Lorg+{cn-1}]) '

                collected_mass_transport_strings = [Jsb_str, Jbs_str,
                                                    Jb1b2_str, Jbb_core_str]

                collected_mass_transport_inloop_strings = [Jbb_minus1_str,
                                                           Jbb_plus1_str]
            else:
                Jssb_str = f'Jssb_{cn} = kssb_{cn} * y[{cn-1}*Lorg+{cn-1}] '
                Jbss_str = f'Jbss_{cn} = kbss_{cn} * y[{cn-1}*Lorg+{cn}] '
                Jb1b2_str = f'Jb1b2_{cn} = kbby_{cn}[0] * (y[{cn-1}*Lorg+{cn}+1] - y[{cn-1}*Lorg+{cn}]) '
                Jbb_minus1_str = f'Jbb_minus1_{cn} = kbby_{cn}[i] * (y[{cn-1}*Lorg+{cn}+i-1] - y[{cn-1}*Lorg+{cn}+i]) '
                Jbb_plus1_str = f'Jbb_plus1_{cn} = kbby_{cn}[i+1] * (y[{cn-1}*Lorg+{cn}+i+1] - y[{cn-1}*Lorg+{cn}+i]) '
                Jbb_core_str = f'Jbb_core_{cn} = kbby_{cn}[-1] * (y[{cn}*Lorg+{cn-1}-1] - y[{cn}*Lorg+{cn-1}]) '

                collected_mass_transport_strings = [Jssb_str, Jbss_str,
                                                    Jb1b2_str, Jbb_core_str]

                collected_mass_transport_inloop_strings = [Jbb_minus1_str,
                                                           Jbb_plus1_str]

            for mass_transport_term in collected_mass_transport_strings:
                self.mass_transport_rate_strings.append(mass_transport_term)

            for mass_transport_loop_term in collected_mass_transport_inloop_strings:
                self.mass_transport_rate_inloop_strings.append(mass_transport_loop_term)

            # if component_number == 1:
            #     self.surf_string = 'dydt[0] = '
            #     self.firstbulk_string = 'dydt[1] = '
            #     self.bulk_string = 'dydt[{}*Lorg+{}+i] = '.format(component_number-1,component_number)
            #     self.core_string = 'dydt[{}*Lorg+{}] = '.format(component_number,component_number-1)
            # else:

            # self.surf_string = 'dydt[{}*Lorg+{}] = '.format(component_number-1,component_number-1)
            # self.firstbulk_string = 'dydt[{}*Lorg+{}] = '.format(component_number-1,component_number)
            # self.bulk_string = 'dydt[{}*Lorg+{}+i] = '.format(component_number-1,component_number)
            # self.core_string = 'dydt[{}*Lorg+{}] = '.format(component_number,component_number-1)

            self.surf_string = f'dydt[{cn-1}*Lorg+{cn-1}] = '
            self.firstbulk_string = f'dydt[{cn-1}*Lorg+{cn}] = '
            self.bulk_string = f'dydt[{cn-1}*Lorg+{cn}+i] = '
            self.core_string = f'dydt[{cn}*Lorg+{cn-1}] = '

            # add mass transport to component strings

            if self.gas:
                self.surf_string += f'Jbs_{cn} - Jsb_{cn} '
                self.firstbulk_string += f'(Jsb_{cn} - Jbs_{cn}) * (A[0]/V[0]) + Jb1b2_{cn} * (A[1]/V[0]) '
                self.bulk_string += f'Jbb_minus1_{cn} * (A[i]/V[i]) + Jbb_plus1_{cn} * (A[i+1]/V[i]) '
                self.core_string += f'Jbb_core_{cn} * (A[-1]/V[-1]) '

            else:
                self.surf_string += f'Jbss_{cn} - Jssb_{cn} '
                self.firstbulk_string += f'(Jssb_{cn} - Jbss_{cn}) * (A[0]/V[0]) + Jb1b2_{cn} * (A[1]/V[0]) '
                self.bulk_string += f'Jbb_minus1_{cn} * (A[i]/V[i]) + Jbb_plus1_{cn} * (A[i+1]/V[i]) '
                self.core_string += f'Jbb_core_{cn} * (A[-1]/V[-1]) '

            # if self.gas:
            #     if component_number == 1:
            #         self.surf_string += 'kbs_1 * y[1] - ksb_1 * y[0] '
            #         self.firstbulk_string += '(ksb_1 * y[0] - kbs_1 * y[1]) * (A[0]/V[0]) + kbbx_1[0] * (y[2] - y[1]) * (A[1]/V[0]) '
            #         self.bulk_string += 'kbbx_1[i] * (y[{}*Lorg+{}+(i-1)] - y[{}*Lorg+{}+i]) * (A[i]/V[i]) + kbbx_1[i+1] * (y[{}*Lorg+{}+(i+1)] - y[{}*Lorg+{}+i]) * (A[i+1]/V[i]) '.format(component_number-1,component_number,
            #                                    component_number-1,component_number,component_number-1,component_number,component_number-1,component_number)
            #         self.core_string += 'kbbx_1[-1] * (y[{}*Lorg+{}-1] - y[{}*Lorg+{}]) * (A[-1]/V[-1]) '.format(component_number,component_number-1,component_number,component_number-1)
            #     else:
            #         self.surf_string += 'kbs_{} * y[{}*Lorg+{}] - ksb_{} * y[{}*Lorg+{}] '.format(component_number,component_number-1,component_number,component_number,component_number-1,component_number-1)
            #         self.firstbulk_string += '(ksb_{} * y[{}*Lorg+{}] - kbs_{} * y[{}*Lorg+{}]) * (A[0]/V[0]) + kbbx_{}[0] * (y[{}*Lorg+{}] - y[{}*Lorg+{}]) * (A[1]/V[0]) '.format(component_number,component_number-1,component_number-1,component_number,
            #                                   component_number-1,component_number,component_number,component_number-1,component_number+1,component_number-1,component_number)
            #         self.bulk_string += 'kbbx_{}[i] * (y[{}*Lorg+{}+(i-1)] - y[{}*Lorg+{}+i]) * (A[i]/V[i]) + kbbx_{}[i+1] * (y[{}*Lorg+{}+(i+1)] - y[{}*Lorg+{}+i]) * (A[i+1]/V[i]) '.format(component_number,component_number-1,component_number,
            #                                    component_number-1,component_number,component_number,component_number-1,component_number,component_number-1,component_number)
            #         self.core_string += 'kbbx_{}[-1] * (y[{}*Lorg+{}-1] - y[{}*Lorg+{}]) * (A[-1]/V[-1]) '.format(component_number,component_number,component_number-1,component_number,component_number-1)
            # else:
            #     if component_number == 1:
            #         self.surf_string += 'kbss_1 * y[1] - kssb_1 * y[0] '
            #         self.firstbulk_string += '(kssb_1 * y[0] - kbss_1 * y[1]) * (A[0]/V[0]) + kbby_1[0] * (y[2] - y[1]) * (A[1]/V[0]) '
            #         self.bulk_string += 'kbby_1[i] * (y[{}*Lorg+{}+(i-1)] - y[{}*Lorg+{}+i]) * (A[i]/V[i]) + kbby_1[i+1] * (y[{}*Lorg+{}+(i+1)] - y[{}*Lorg+{}+i]) * (A[i+1]/V[i]) '.format(component_number-1,component_number,
            #                                    component_number-1,component_number,component_number-1,component_number,component_number-1,component_number)
            #         self.core_string += 'kbby_1[-1] * (y[{}*Lorg+{}-1] - y[{}*Lorg+{}]) * (A[-1]/V[-1]) '.format(component_number,component_number-1,component_number,component_number-1)

            #     else:
            #         self.surf_string += 'kbss_{} * y[{}*Lorg+{}] - kssb_{} * y[{}*Lorg+{}] '.format(component_number,component_number-1,component_number,component_number,component_number-1,component_number-1)
            #         self.firstbulk_string += '(kssb_{} * y[{}*Lorg+{}] - kbss_{} * y[{}*Lorg+{}]) * (A[0]/V[0]) + kbby_{}[0] * (y[{}*Lorg+{}] - y[{}*Lorg+{}]) * (A[1]/V[0]) '.format(component_number,component_number-1,component_number-1,component_number,
            #                                   component_number-1,component_number,component_number,component_number-1,component_number+1,component_number-1,component_number)
            #         self.bulk_string += 'kbby_{}[i] * (y[{}*Lorg+{}+(i-1)] - y[{}*Lorg+{}+i]) * (A[i]/V[i]) + kbby_{}[i+1] * (y[{}*Lorg+{}+(i+1)] - y[{}*Lorg+{}+i]) * (A[i+1]/V[i]) '.format(component_number,component_number-1,component_number,
            #                                    component_number-1,component_number,component_number,component_number-1,component_number,component_number-1,component_number)
            #         self.core_string += 'kbby_{}[-1] * (y[{}*Lorg+{}-1] - y[{}*Lorg+{}]) * (A[-1]/V[-1]) '.format(component_number,component_number,component_number-1,component_number,component_number-1)

        elif self.reaction_scheme.model_type.model_type.lower() == 'km-gap':
            # build strings for km-gap, using number of molecules instead of molec cm-3
            # remember to divide by A or V to get molec -3 (e.g. Jss_s calculation)
            cn = self.component_number

            # mass fluxes
            Jss_s_str = f'Jss_s_{cn} = kss_s_{cn} * (y[{cn-1}*Lorg+2*{cn-1}+1]/A[0]) '
            Js_ss_str = f'Js_ss_{cn} = ks_ss_{cn} * (y[{cn-1}*Lorg+2*{cn-1}]/A[0]) '
            Jssb_str = f'Jssb_{cn} = kssb_{cn} * (y[{cn-1}*Lorg+2*{cn-1}+1]/A[0]) '
            Jbss_str = f'Jbss_{cn} = kbss_{cn} * (y[{cn-1}*Lorg+2*{cn-1}+2]/V[0]) '
            Jb1b2_str = f'Jb1b2_{cn} = kbb_{cn}[0] * (y[{cn-1}*Lorg+2*{cn-1}+2+1]/V[1] - y[{cn-1}*Lorg+2*{cn-1}+2]/V[0]) '
            #Jb2b1_str = f'Jb2b1_{cn} = kbb_{cn}[1] * (y[{cn-1}*Lorg+2*{cn-1}+2+1]/V[1] - y[{cn-1}*Lorg+2*{cn-1}+2]/V[0]) '
            Jbb_minus1_str = f'Jbb_minus1_{cn} = kbb_{cn}[i] * (y[{cn-1}*Lorg+{cn}+i+{cn}-1]/V[i-1] - y[{cn-1}*Lorg+{cn}+i+{cn}]/V[i]) '
            Jbb_plus1_str = f'Jbb_plus1_{cn} = kbb_{cn}[i+1] * (y[{cn-1}*Lorg+{cn}+i+{cn}+1]/V[i+1] - y[{cn-1}*Lorg+{cn}+i+{cn}]/V[i])'
            Jbb_core_str = f'Jbb_core_{cn} = kbb_{cn}[-1] * (y[{cn}*Lorg+{cn}+{cn-1}-1]/V[-2] - y[{cn}*Lorg+{cn}+{cn-1}]/V[-1]) '

            collected_mass_transport_strings = [Jss_s_str, Js_ss_str, Jssb_str,
                                                Jbss_str, Jb1b2_str,
                                                Jbb_core_str]
            collected_mass_transport_inloop_strings = [Jbb_minus1_str,
                                                       Jbb_plus1_str]

            for mass_transport_term in collected_mass_transport_strings:
                self.mass_transport_rate_strings.append(mass_transport_term)

            for mass_transport_loop_term in collected_mass_transport_inloop_strings:
                self.mass_transport_rate_inloop_strings.append(mass_transport_loop_term)

            self.surf_string = f'dydt[{cn-1}*Lorg+2*{cn-1}] = '
            self.static_surf_string = f'dydt[{cn-1}*Lorg+2*{cn-1}+1] = '
            self.firstbulk_string = f'dydt[{cn-1}*Lorg+2*{cn-1}+2] = '
            self.bulk_string = f'dydt[{cn-1}*Lorg+{cn}+i+{cn}] = '
            self.core_string = f'dydt[{cn}*Lorg+{cn}+{cn-1}] = '

            # adding transport terms to each string
            self.surf_string += f'Jss_s_{cn} * A[0] - Js_ss_{cn} * A[0] '
            self.static_surf_string += f'Js_ss_{cn} * A[0] - Jss_s_{cn} * A[0] + Jbss_{cn} * A[0] - Jssb_{cn} * A[0] '
            self.firstbulk_string += f'Jssb_{cn} * A[0] - Jbss_{cn} * A[0] + Jb1b2_{cn} * A[1] '

            self.bulk_string += f'Jbb_minus1_{cn} * A[i] + Jbb_plus1_{cn} * A[i+1] '

            self.core_string += f'Jbb_core_{cn} * A[-1] '


class DiffusionRegime():
    '''
    Calling this will build the diffusion regime and return a list of strings
    defining the composition-dependent (or not) diffusion evolution for each
    component and the definition of kbb for each component.

    Parameters
    ----------
    model_type : multilayerpy.build.ModelType
        Defines the model type being considered.
    model_components_dict : dict
        A dictionary of multilayer.build.Parameter objects representing each
        model parameter.
    regime : str
        Which diffusion regime to use.
    diff_dict : dict
        A dictionary defining how each component's diffusivity depends on composition.

        EXAMPLE:
        >>> diff_dict = {'1' : (2,3),
                         '2' : None,
                         '3' : None}

        This states that component 1 diffusivity depends on the amount of
        component 2 and 3 and that the diffusivities of component 2 and 3 are
        not composition dependent.
    '''

    def __init__(self, model_type, model_components_dict, regime='vignes',
                 diff_dict=None):

        # ModelType object
        self.model_type = model_type

        # string defining the diffusion regime to use
        self.regime = regime

        # dictionary describing composition-dependent diffusion for each
        # model component
        self.diff_dict = diff_dict

        # if the diffusion dict is None, make a dict of Nones for each comp

        if diff_dict is None:

            self.diff_dict = {f'{x}': None for x in np.arange(1, len(model_components_dict)+1)}

        # make sure diff_dict includes all model components
        error_msg = f"diffusion dict length = {len(self.diff_dict)}, model components dict length = {len(model_components_dict)}. They need to be the same."
        assert len(self.diff_dict) == len(model_components_dict), error_msg

        # list of strings descibing kbb for each component
        # movement of each component between model layers
        self.kbb_strings = None

        # list of strings for kbs/ksb and kbss/kssb
        self.kbs_strings = None
        self.ksb_strings = None
        self.kbss_strings = None
        self.kssb_strings = None
        # list of strings for kss_s/ks_ss for km-gap
        self.kss_s_strings = None
        self.ks_ss_strings = None

        # list of strings describing Db for each model component
        self.Db_strings = None

        # sum of the fraction of products
        self.sum_product_fractions = None
        self.sum_product_fractions_surf = None
        # list of strings describing Ds for each model component
        self.Ds_strings = None

        self.req_diff_params = None

        self.model_components_dict = model_components_dict

        # A boolean which changes to true once the diffusion regime has been
        # built
        self.constructed = False

        self.req_diff_params = None

        self._equivalent_diffusion_component_numbers = []
        if type(self.diff_dict) == dict:
            for comp in self.diff_dict:
                if type(self.diff_dict[comp]) == int:
                    self._equivalent_diffusion_component_numbers.append(int(comp))

    def __call__(self):
        '''
        Constructs the diffusion regime by making strings representing the rate of diffusion
        of model components between bulk layers and between the bulk and surface layers.

        The diffusion regime will then be ready to be used in multilayerpy.build.ModelBuilder.
        '''

        # for each component write D and kbb, kbs string
        # Db kbb will be an array
        # write f_y also - no, include in ModelBuilder **

        diff_dict = self.diff_dict
        Db_definition_string_list = []
        Ds_definition_string_list = []
        kbb_string_list = []
        ksb_string_list = []
        kbs_string_list = []
        kbss_string_list = []
        kssb_string_list = []
        kss_s_string_list = []
        ks_ss_string_list = []
        req_diff_params = set([])

        # for each component in the diffuion evolution dictionary
        # this dict should be the same length as the number of components
        # in the model

        for i in np.arange(len(diff_dict)):
            # only consider components who's diffusivities are composition-
            # dependent
            if diff_dict[f'{i+1}'] is not None:
                # composition dependence tuple
                compos_depend_tup = diff_dict[f'{i+1}']

                # if D of this component is same as another
                # (i.e. int supplied in diffusion dict), make this so
                if type(compos_depend_tup) == int:
                    copy_diffusion_comp_no = compos_depend_tup
                    Db_string = f'Db_{i+1}_arr = Db_{copy_diffusion_comp_no}_arr'
                    Ds_string = f'Ds_{i+1} = Ds_{copy_diffusion_comp_no}'
                    Db_definition_string_list.append(Db_string)
                    Ds_definition_string_list.append(Ds_string)

                # vignes diffusion
                elif self.regime.lower() == 'vignes':
                    # initially dependent on D_comp in pure comp
                    # raised to the power of fraction of component
                    Db_string = f'Db_{i+1}_arr = (Db_{i+1}_arr**fb_{i+1}) '

                    Ds_string = f'Ds_{i+1} = (Db_{i+1}**fs_{i+1}) '

                    req_diff_params.add(f'Db_{i+1}')

                    # loop over depending components
                    for comp in compos_depend_tup:
                        Db_string += f'* (Db_{i+1}_{comp}_arr**fb_{comp}) '

                        Ds_string += f'* (Db_{i+1}_{comp}**fs_{comp}) '

                        req_diff_params.add(f'Db_{i+1}_{comp}')

                    Db_definition_string_list.append(Db_string)
                    Ds_definition_string_list.append(Ds_string)

                # km-sub linear combination
                elif self.regime.lower() == 'linear':
                    # initially dependent on D_comp in pure comp
                    # multiply by fraction of component
                    Db_string = f'Db_{i+1}_arr = (Db_{i+1}_arr * fb_{i+1}) '

                    Ds_string = f'Ds_{i+1} = (Db_{i+1}**fs_{i+1}) '

                    req_diff_params.add(f'Db_{i+1}')

                    # loop over depending components
                    for comp in compos_depend_tup:
                        Db_string += f'+ (Db_{i+1}_{comp}_arr * fb_{comp}) '

                        Ds_string += f'+ (Db_{i+1}_{comp} * fs_{comp}) '

                        req_diff_params.add(f'Db_{i+1}_{comp}')

                    Db_definition_string_list.append(Db_string)
                    Ds_definition_string_list.append(Ds_string)

                # obstruction theory
                elif self.regime == 'obstruction':
                    # only dependent on total fraction of products
                    sum_product_fractions = 'f_prod = '
                    sum_product_fractions_surf = 'fs_prod = '

                    # add in fraction of each product to sum of product frac.
                    # string
                    for a, comp in enumerate(compos_depend_tup):
                        if a == 0:
                            sum_product_fractions += f'fb_{comp} '
                            sum_product_fractions_surf += f'fs_{comp} '
                        else:
                            sum_product_fractions += f'+ fb_{comp} '
                            sum_product_fractions_surf += f'+ fs_{comp} '

                    # obstruction theory equation (Stroeve 1975)
                    Db_string = f'Db_{i+1}_arr = Db_{i+1}_arr * ((2 - 2 * f_prod) / (2 + f_prod))'

                    Ds_string = f'Ds_{i+1} = Db_{i+1} * ((2 - 2 * fs_prod) / (2 + fs_prod))'

                    req_diff_params.add(f'Db_{i+1}')

                    self.sum_product_fractions = sum_product_fractions
                    self.sum_product_fractions_surf = sum_product_fractions_surf

                    Db_definition_string_list.append(Db_string)
                    Ds_definition_string_list.append(Ds_string)

                # Other diffusion regimes may be added in later developments

            # else there is no evolution and just Db of pure
            # component, end of story
            else:
                Db_string = f'Db_{i+1}_arr = Db_{i+1}_arr'
                Ds_string = f'Ds_{i+1} = Db_{i+1}'

                req_diff_params.add(f'Db_{i+1}')

                Db_definition_string_list.append(Db_string)
                Ds_definition_string_list.append(Ds_string)

            # define kbb and kbs/ksb strings for each component
            if self.model_components_dict[f'{i+1}'].gas and self.model_type.model_type.lower() == 'km-sub':
                Hcc_definition = f'H_cc_{i+1} = H_{i+1} * R * T'
                ksb_string = 'ksb_{} = H_cc_{} * kbs_{} / Td_{} / (w_{}*alpha_s_{}/4) '.format(i+1,i+1,i+1,i+1,i+1,i+1)
                ksb_string_list.append(Hcc_definition)
                ksb_string_list.append(ksb_string)

                req_diff_params.add(f'H_{i+1}')
                req_diff_params.add(f'Td_{i+1}')
                req_diff_params.add(f'w_{i+1}')
                req_diff_params.add('T')

                kbs_string = 'kbs_{} = (8/np.pi) * Ds_{} / layer_thick[0] '.format(i+1,i+1)
                # add the molec. diameters of other components in the surface layer
                for comp in self.model_components_dict.values():
                    kbs_string += f'+ delta_{comp.component_number}'

                kbs_string_list.append(kbs_string)

                kbb_string = 'kbbx_{} = (4/np.pi) * Db_{}_arr / layer_thick '.format(i+1,i+1)
                kbb_string_list.append(kbb_string)

            elif self.model_type.model_type.lower() == 'km-sub':
                kssb_string = 'kssb_{} = kbss_{} / delta_{} '.format(i+1,i+1,i+1)
                kssb_string_list.append(kssb_string)

                kbss_string = 'kbss_{} = (8*Db_{}_arr[0])/((layer_thick[0]+delta_{})*np.pi) '.format(i+1,i+1,i+1)
                kbss_string_list.append(kbss_string)

                kbb_string = 'kbby_{} = (4/np.pi) * Db_{}_arr / layer_thick '.format(i+1,i+1)
                kbb_string_list.append(kbb_string)

            elif self.model_type.model_type.lower() == 'km-gap':
                kss_s_string = f'kss_s_{i+1} = Db_{i+1} / delta_{i+1}**2'
                # turn off movement between ss an s for non-volatile components
                if not self.model_components_dict[f'{i+1}'].gas:
                    kss_s_string = f'kss_s_{i+1} = 0.0 '
                kss_s_string_list.append(kss_s_string)

                ks_ss_string = f'ks_ss_{i+1} = kss_s_{i+1} * ((kd_{i+1} * Zss_eq_{i+1}) / (ka_{i+1} * Zg_eq_{i+1}))'
                # turn off movement between ss an s for non-volatile components
                if not self.model_components_dict[f'{i+1}'].gas:
                    ks_ss_string = f'ks_ss_{i+1} = 0.0'
                ks_ss_string_list.append(ks_ss_string)

                kbss_string = f'kbss_{i+1} = (2 * Db_{i+1}) / (delta_{i+1} + layer_thick[0])'
                kbss_string_list.append(kbss_string)

                kssb_string = f'kssb_{i+1} = kbss_{i+1} / delta_{i+1}'
                kssb_string_list.append(kssb_string)

                # kbb slightly different definition in km-gap
                kbb_string = f'kbb_{i+1} = (2 * Db_{i+1}_arr[:-1]) / (layer_thick[:-1] + layer_thick[1:])'
                # tag on kbb for the core layer
                core_kbb_string = f'kbb_core_{i+1} = (2 * Db_{i+1}_arr[-1]) / layer_thick[-1] '
                kbb_core_add = f'kbb_{i+1} = np.append(kbb_{i+1},kbb_core_{i+1})'
                kbb_string_list.append(kbb_string)
                kbb_string_list.append(core_kbb_string)
                kbb_string_list.append(kbb_core_add)

        # update the Db_strings and Ds_strings attributes
        self.Db_strings = Db_definition_string_list

        self.Ds_strings = Ds_definition_string_list

        self.kbb_strings = kbb_string_list
        self.kbs_strings = kbs_string_list
        self.ksb_strings = ksb_string_list
        self.kbss_strings = kbss_string_list
        self.kssb_strings = kssb_string_list
        self.kss_s_strings = kss_s_string_list
        self.ks_ss_strings = ks_ss_string_list

        self.req_diff_params = req_diff_params

        # this diffusion regime is now constructed
        self.constructed = True


class ModelBuilder():
    '''
    An object which constructs the model file from the reaction scheme, list
    of model components and a diffusion regime.

    Parameters
    ----------
    reaction_scheme : multilayerpy.build.ReactionScheme
        The reaction scheme used in the model.

    model_components_dict : dict
        A dictionary of multilayer.build.Parameter objects representing each
        model parameter.

    diffusion_regime : multilayerpy.build.DiffusionRegime
        The diffusion regime used in the model.
    '''

    def __init__(self, reaction_scheme, model_components_dict,
                 diffusion_regime):

        self.reaction_scheme = reaction_scheme

        self.model_components = model_components_dict
        # make sure the model components dictionary is actually a dictionary
        assert type(model_components_dict) == dict, "The model_components_dict is not a dictionary."

        self.diffusion_regime = diffusion_regime

        self.geometry = reaction_scheme.model_type.geometry

        self.model_type = reaction_scheme.model_type.model_type
        assert self.model_type.lower() in ['km-sub','km-gap'], f"The model type '{self.model_type.lower()}'' is not in the accepted model types:\n['km-sub','km-gap']"

        # a SET of strings with names of required parameters
        self.req_params = set([])

        for s in diffusion_regime.req_diff_params:
            self.req_params.add(s)

        # list of output strings for final file write and coupling to
        # compartmental models (future dev)
        self.file_string_list = []

        self.filename = None

        self.constructed = False

    def build(self, name_extention='', date_tag=False,
              use_scaled_k_surf=False, **kwargs):
        '''
        Builds the model and saves it to a separate .py file
        Different for different model types

        Parameters
        ----------
        name_extention : str
            An extra tag added to the model filename.

        date_tag : bool
            Whether to add a date tag to the filename with today's date.

        use_scaled_k_surf : bool
            Whether to use a scale factor to convert from bulk second order rate
            constants to surface second order rate constants.

        returns
        ----------
        Saves a .py file in the current working directory containing the model
        code defining the system of ODEs to be solved.
        '''

        # check that the diffusion regime has been build before build starts
        assert self.diffusion_regime.constructed == True, "The DiffusionRegime was not called prior to model construction.\nCall the DiffusionRegime object in order to build it."

        # the list of strings which will be used with file.writelines()
        # at the end
        master_string_list = []
        four_space = '    '
        extention = name_extention

        now = datetime.now()
        date = now.strftime('%d-%m-%Y (day-month-year)')

        mod_type = self.model_type
        rxn_name = self.reaction_scheme.name
        mod_comps = self.model_components

        regime = self.diffusion_regime.regime
        if self.diffusion_regime.diff_dict is None:
            regime = 'None'
        else:
            diff_evo = False
            for k in self.diffusion_regime.diff_dict.keys():
                if self.diffusion_regime.diff_dict[k] is not None:
                    diff_evo = True
            if not diff_evo:
                regime = 'No composition-dependent diffusion'

        # make T a required param
        self.req_params.add('T')

        heading_strings = ['#================================================\n',
                           f'# A {mod_type} model constructed using MultilayerPy version {multilayerpy.__version__}\n',
                           '# MultilayerPy - build, run and optimise kinetic multi-layer models for\n# aerosol particles and films.\n\n',
                           '# MultilayerPy is released under the GNU General Public License v3.0\n',
                           '\n',
                           f'# Created {date}\n',
                           '\n',
                           f'# Reaction name: {rxn_name}\n',
                           f'# Geometry: {self.geometry}\n',
                           f'# Number of model components: {len(self.model_components)}\n',
                           f'# Diffusion regime: {regime}\n',
                           '#================================================\n',
                           '\n',
                           'import numpy as np']

        # append the heading to the master strings
        # for s in heading_strings:
        #     master_string_list.append(s)

        # define dydt function
        func_def_strings = ['\ndef dydt(t,y,param_dict,V,A,Lorg,layer_thick,param_evolution_func=None,additional_params=None):\n',
                            '    """ Function defining ODEs, returns list of dydt"""\n\n',
                            '    if type(param_evolution_func) != type(None):\n',
                            '        param_dict = param_evolution_func(t,y,param_dict,additional_params=additional_params)\n',
                    ]

        # for s in func_def_strings:
        #     master_string_list.append(s)

        # initialise dydt output array
        n_comps = len(mod_comps)
        init_dydt = f'\n\n    # init dydt array\n    dydt = np.zeros(Lorg * {n_comps} + {n_comps})\n'
        if mod_type.lower() == 'km-gap':
            init_dydt = f'\n\n    # init dydt array\n    dydt = np.zeros(Lorg * {n_comps} + 2 * {n_comps})\n'

        master_string_list.append(init_dydt)

        # calculate each bulk layer volume, area and thickness as a function of
        # number of molecules in that layer (KM-GAP only)
        if mod_type.lower() == 'km-gap':
            master_string_list.append('\n    #--------------Define V, A and layer thickness as function of Number of molecules---------------\n')

            comp_V_list = []  # list of V contribution from each component
            for comp in mod_comps.values():
                cn = comp.component_number
                master_string_list.append(f'\n\n    #--------------component {cn} V---------------\n')
                N_bulk_comp_str = f'\n    N_bulk_{cn} = y[{cn-1}*Lorg+2*{cn-1}+2:{cn}*Lorg+{cn}+{cn-1}+1]'
                v_comp_str = f'\n    v_{cn} = delta_{cn}**3'
                Vtot_comp_str = f'\n    Vtot_{cn} = N_bulk_{cn} * v_{cn}'
                # add Vtot_comp to comp_V_list
                comp_V_list.append(f'Vtot_{cn}')

                master_string_list.append(N_bulk_comp_str)
                master_string_list.append(v_comp_str)
                master_string_list.append(Vtot_comp_str)

            # now calculate total V, A and layer thick
            master_string_list.append('\n\n    # calc total V, A, layer thick\n')
            Vtot_str = '\n    Vtot = '
            for vtot in comp_V_list:
                Vtot_str += '+ ' + vtot + ' '

            if self.geometry == 'spherical':
                sum_V_str = '\n    sum_V = np.cumsum(np.flip(Vtot))'
                r_pos_str = '\n    r_pos = np.cbrt((3.0* np.flip(sum_V))/(4*np.pi))' # different for planar SORT THIS OUT
                A_str = '\n    A = 4 * np.pi * r_pos**2'

                layer_thick_str = '\n    layer_thick = r_pos[:-1] - r_pos[1:]'
                layer_thick_core_str = '\n    layer_thick = np.append(layer_thick,r_pos[-1])'

            elif self.geometry == 'film':
                A_str = '\n    A = np.ones(len(Vtot)) * (1e-4 ** 2) # 1e-4 cm is nominal square cross-section length'
                layer_thick_str = '\n    layer_thick = Vtot / A'

            # append relevant strings to master string list
            master_string_list.append(Vtot_str)
            if self.geometry == 'spherical':
                master_string_list.append(sum_V_str)
                master_string_list.append(r_pos_str)

            master_string_list.append(A_str)
            master_string_list.append(layer_thick_str)

            if self.geometry == 'spherical':
                master_string_list.append(layer_thick_core_str)
            master_string_list.append('\n    V = Vtot')  # V now re-defined

        # surface uptake definition for each gas component
        master_string_list.append('\n    #--------------Define surface uptake parameters for gas components---------------\n')

        # calculate surface fraction of each component
        master_string_list.append('\n    #calculate surface fraction of each component\n')
        for comp in mod_comps.values():
            comp_no = comp.component_number
            if mod_type.lower() == 'km-sub':
                fs_string = f'    fs_{comp_no} = y[{comp_no-1}*Lorg+{comp_no-1}] / ('

            # frac of static surface layer for km-gap
            elif mod_type.lower() == 'km-gap':
                fs_string = f'    fss_{comp_no} = y[{comp_no-1}*Lorg+2*{comp_no-1}+1] / ('

            # loop over each other component and add to the fs_string
            for ind, c in enumerate(mod_comps.values()):
                cn = c.component_number

                if ind == 0:
                    if mod_type.lower() == 'km-sub':
                        fs_string += f'y[{cn-1}*Lorg+{cn-1}] '
                    elif mod_type.lower() == 'km-gap':
                        fs_string += f'y[{cn-1}*Lorg+2*{cn-1}+1] '

                else:
                    if mod_type.lower() == 'km-sub':
                        fs_string += f'+ y[{cn-1}*Lorg+{cn-1}]'
                    elif mod_type.lower() == 'km-gap':
                        fs_string += f'+ y[{cn-1}*Lorg+2*{cn-1}+1]'

            # close the bracket on the demoninator
            fs_string += ')\n'

            master_string_list.append(fs_string)

        # for each gaseous model component, calculate surface uptake/loss params
        # first calculate surface coverage separately

        surf_cover_str = '\n    surf_cover = '
        # static_surf_cover_str = '\n    static_surf_cover = '
        counter = 0
        for comp in mod_comps.values():
            if mod_type.lower() == 'km-sub':
                if comp.gas:
                    comp_no = comp.component_number

                    if counter == 0:
                        surf_cover_str += f'delta_{comp_no}**2 * y[{comp_no-1}*Lorg+{comp_no-1}] '
                        counter += 1

                    else:
                        surf_cover_str += f'+ delta_{comp_no}**2 * y[{comp_no-1}*Lorg+{comp_no-1}]'
            
            # km-gap considers all components, not just volatiles
            elif mod_type.lower() == 'km-gap':
                comp_no = comp.component_number

                if counter == 0:
                    surf_cover_str += f'delta_{comp_no}**2 * (y[{comp_no-1}*Lorg+2*{comp_no-1}] / A[0])'
                    # static_surf_cover_str += f'delta_{comp_no}**2 * (y[{comp_no-1}*Lorg+2*{comp_no-1}+1] / A[0])'
                    counter += 1

                else:
                    surf_cover_str += f'+ delta_{comp_no}**2 * (y[{comp_no-1}*Lorg+2*{comp_no-1}] / A[0])'
                    #static_surf_cover_str += f'+ delta_{comp_no}**2 * (y[{comp_no-1}*Lorg+2*{comp_no-1}+1] / A[0])'

            self.req_params.add(f'delta_{comp.component_number}')

        for comp in mod_comps.values():
            if comp.gas or mod_type.lower() == 'km-gap':
                comp_no = comp.component_number

                if comp.name is not None:
                    master_string_list.append(f'\n\n    # component {comp_no} surf params\n')
                else:
                    master_string_list.append(f'\n\n    # component {comp_no} ({comp.name}) surf params\n')

                # surface coverage
                master_string_list.append(surf_cover_str)

                # make alpha_s_0 composition-dependent if desired for this component
                if comp.comp_dependent_adsorption:
                    alpha_s_0_str = f'\n    alpha_s_0_{comp_no} = '
                    # build up alpha_s_0 string
                    for k in range(len(mod_comps)):
                        cn = k+1
                        if cn != comp_no:
                            if mod_type.lower() == 'km-gap':
                                alpha_s_0_str += f'+ alpha_s_0_{comp_no}_{cn} * delta_{cn}**2 * (y[{cn-1}*Lorg+2*{cn-1}+1] / A[0]) '
                            else:
                                alpha_s_0_str += f'+ alpha_s_0_{comp_no}_{cn} * delta_{cn}**2 * y[{cn-1}*Lorg+{cn-1}] '
                            # update the required parameters
                            self.req_params.add(f'alpha_s_0_{comp_no}_{cn}')

                    # update the master string list
                    master_string_list.append(alpha_s_0_str)

                # alpha_s_X, Z if km-gap
                alpha_s_str = f'\n    alpha_s_{comp_no} = alpha_s_0_{comp_no} * (1-surf_cover)'
                master_string_list.append(alpha_s_str)

                # calculate [Z]ss,eq and [Z]g,eq - km-gap
                if mod_type.lower() == 'km-gap':
                    ka_str = f'\n    ka_{comp_no} = alpha_s_{comp_no} * w_{comp_no} / 4'
                    kd_str = f'\n    kd_{comp_no} = 1 / Td_{comp_no}'
                    Zss_eq_str = f'\n    Zss_eq_{comp_no} = (ka_{comp_no}/kd_{comp_no}) * Zgs_{comp_no}'

                    Zg_eq_str = f'\n    Zg_eq_{comp_no} = (p_{comp_no} * Na) / (R_cm3_units * T)'

                    master_string_list.append(ka_str)
                    master_string_list.append(kd_str)
                    master_string_list.append(Zss_eq_str)
                    master_string_list.append(Zg_eq_str)
                    self.req_params.add(f'p_{comp_no}')
                    self.req_params.add(f'Td_{comp_no}')

                # J_coll_X
                j_coll_str = f'\n    J_coll_X_{comp_no} = Xgs_{comp_no} * w_{comp_no}/4'
                if mod_type.lower() == 'km-gap':
                    j_coll_str = f'\n    J_coll_Z_{comp_no} = Zgs_{comp_no} * w_{comp_no}/4'
                master_string_list.append(j_coll_str)

                # J_ads_X
                j_ads_str = f'\n    J_ads_X_{comp_no} = alpha_s_{comp_no} * J_coll_X_{comp_no}'
                if mod_type.lower() == 'km-gap':
                    j_ads_str = f'\n    J_ads_Z_{comp_no} = alpha_s_{comp_no} * J_coll_Z_{comp_no}'
                master_string_list.append(j_ads_str)

                # J_des_X
                j_des_str = f'\n    J_des_X_{comp_no} = (1 / Td_{comp_no}) * y[{comp_no-1}*Lorg+{comp_no-1}]'
                if mod_type.lower() == 'km-gap':
                    j_des_str = f'\n    J_des_Z_{comp_no} = (1 / Td_{comp_no}) * (y[{comp_no-1}*Lorg+2*{comp_no-1}] / A[0])'
                master_string_list.append(j_des_str)

                if not comp.comp_dependent_adsorption:
                    self.req_params.add(f'alpha_s_0_{comp_no}')

                if mod_type.lower() == 'km-sub':
                    self.req_params.add(f'Xgs_{comp_no}')

                elif mod_type.lower() == 'km-gap':
                    self.req_params.add(f'Zgs_{comp_no}')

                self.req_params.add(f'w_{comp_no}')
                # self.req_params.add(f'kd_X_{comp_no}')
                if use_scaled_k_surf:
                    self.req_params.add('scale_bulk_to_surf')

                # add surface ads/des to the surface string of the gaseous component
                if mod_type.lower() == 'km-sub':
                    comp.surf_string += f'+ J_ads_X_{comp_no} - J_des_X_{comp_no}'
                elif mod_type.lower() == 'km-gap':
                    comp.surf_string += f'+ J_ads_Z_{comp_no} * A[0] - J_des_Z_{comp_no} * A[0]'

        # diffusion evolution

        master_string_list.append('\n\n    #--------------Bulk Diffusion evolution---------------\n')

        # calculate bulk fraction array for each component in each layer
        master_string_list.append('\n    # Db and fb arrays')

        for comp in mod_comps.values():
            comp_no = comp.component_number
            fb_string = f'\n    fb_{comp_no} = y[{comp_no-1}*Lorg+{comp_no}:{comp_no}*Lorg+{comp_no-1}+1] / ('
            if mod_type.lower() == 'km-gap':
                fb_string = f'\n    fb_{comp_no} = y[{comp_no-1}*Lorg+2*{comp_no-1}+2:{comp_no}*Lorg+{comp_no}+{comp_no-1}+1] / ('

            # loop over each other component and add to the fb_string
            for ind, c in enumerate(mod_comps.values()):

                cn = c.component_number
                if ind == 0:
                    if mod_type.lower() == 'km-sub':
                        fb_string += f'y[{cn-1}*Lorg+{cn}:{cn}*Lorg+{cn-1}+1] '
                    elif mod_type.lower() == 'km-gap':
                        fb_string += f'y[{cn-1}*Lorg+2*{cn-1}+2:{cn}*Lorg+{cn}+{cn-1}+1] '

                else:
                    if mod_type.lower() == 'km-sub':
                        fb_string += f'+ y[{cn-1}*Lorg+{cn}:{cn}*Lorg+{cn-1}+1]'
                    elif mod_type.lower() == 'km-gap':
                        fb_string += f'+ y[{cn-1}*Lorg+2*{cn-1}+2:{cn}*Lorg+{cn}+{cn-1}+1]'

            # close the bracket on the demoninator
            fb_string += ')'

            # add conditional checking if Db_{cn} is 0, fb will be 0 (monolayer)
            if_string = f'\n    if float(Db_{comp_no}) == 0.0:'
            ifclause_string = f'\n        fb_{comp_no} = np.zeros(Lorg)'
            else_string = '\n    else:'

            master_string_list.append(if_string)
            master_string_list.append(ifclause_string)
            master_string_list.append(else_string)
            master_string_list.append('\n     ' + fb_string[2:])

        if self.diffusion_regime.regime == 'obstruction':
            master_string_list.append('\n\n    # total product fraction (obstruction theory)')
            master_string_list.append('\n    ' + self.diffusion_regime.sum_product_fractions)
            master_string_list.append('\n    ' + self.diffusion_regime.sum_product_fractions_surf)

        master_string_list.append('\n')
        # Db arrays for each component
        accounted_Db_strings = []
        for comp in mod_comps.values():
            comp_no = comp.component_number
            if comp_no not in self.diffusion_regime._equivalent_diffusion_component_numbers:
                Db_arr_string = f'\n    Db_{comp_no}_arr = np.ones(Lorg) * Db_{comp_no}'

                accounted_Db_strings.append(f'Db_{comp_no}')
                master_string_list.append(Db_arr_string)

            # self.req_params.add(f'Db_{comp_no}')

        # make Db arrays for mixed compositions
        for param in self.req_params:
            if param.startswith('Db') and param not in accounted_Db_strings:
                Db_arr_string = '\n    ' + param + '_arr' + ' = np.ones(Lorg) * ' + param
                master_string_list.append(Db_arr_string)

        diff_regime = self.diffusion_regime

        master_string_list.append('\n\n    # bulk diffusion\n')
        master_string_list.append('\n')
        # Db
        for s in diff_regime.Db_strings:
            master_string_list.append(four_space+s+'\n')
        # kbb
        for s in diff_regime.kbb_strings:
            master_string_list.append(four_space+s+'\n')

        master_string_list.append('\n\n    # surface diffusion\n')
        # Ds
        for s in diff_regime.Ds_strings:
            master_string_list.append(four_space+s+'\n')

        # kbs
        for s in diff_regime.kbs_strings:
            master_string_list.append(four_space+s+'\n')

        # ksb
        for s in diff_regime.ksb_strings:
            master_string_list.append(four_space+s+'\n')

        # kbss
        for s in diff_regime.kbss_strings:
            master_string_list.append(four_space+s+'\n')

        # kssb
        for s in diff_regime.kssb_strings:
            master_string_list.append(four_space+s+'\n')

        if mod_type.lower() == 'km-gap':
            master_string_list.append('\n    # sorption-static surface layer\n')
            master_string_list.append('\n')
            # kss_s (km-gap)
            for s in diff_regime.kss_s_strings:
                master_string_list.append(four_space+s+'\n')

            # ks_ss (km-gap)
            for s in diff_regime.ks_ss_strings:
                master_string_list.append(four_space+s+'\n')

        master_string_list.append('\n\n    # mass fluxes\n')
        master_string_list.append('\n')
        # now add mass fluxes for each component
        for comp in mod_comps.values():
            # loop through mass transport strings (Jbb, Jssb etc.)
            for s in comp.mass_transport_rate_strings:
                master_string_list.append(four_space+s+'\n')

        # For each component, loop over the reaction scheme and identify which
        # rxns the component is involved in. Check if stoichiometry involved.
        # Add the reaction as necessary to the relevant string. Add diffusion

        # for each component in model components
        for comp in mod_comps.values():

            # for each reaction (reactants and products)
            for i in np.arange(len(self.reaction_scheme.reaction_reactants)):
                reactants = self.reaction_scheme.reaction_reactants[i]

                # if no products given in product list, default to the empty reaction products list
                try:
                    products = self.reaction_scheme.reaction_products[i]
                except IndexError:
                    products = self.reaction_scheme.reaction_products

                if self.reaction_scheme.reactant_stoich is None:
                    reactant_stoich = None
                else:
                    reactant_stoich = self.reaction_scheme.reactant_stoich[i]

                if self.reaction_scheme.product_stoich is None:
                    product_stoich = None
                else:
                    try:
                        product_stoich = self.reaction_scheme.product_stoich[i]
                    except IndexError:
                        product_stoich = None

                # if this component is lost as a reactant
                if int(comp.component_number) in reactants:

                    # build up surface, bulk and core strings for reactive loss
                    # for each reactant

                    # check if not first order decay (if len(reactants) != 1)
                    if len(reactants) != 1:
                        for ind, rn in enumerate(reactants):

                            # add reaction strings for each reactant that isn't the current component
                            # i.e. if the component number != reactant number
                            cn = comp.component_number
                            if rn != int(comp.component_number):

                                # if no stoich given, assume a coefficient of 1
                                if reactant_stoich is None:

                                    # INCLUDE STOICHIOMETRY (MULTIPLY BY STOICH COEFF)
                                    if mod_type.lower() == 'km-sub':
                                        comp.surf_string += f'- y[{cn-1}*Lorg+{cn-1}] * y[{rn-1}*Lorg+{rn-1}] * k'
                                        comp.firstbulk_string += f'- y[{cn-1}*Lorg+{cn}] * y[{rn-1}*Lorg+{rn}] * k'
                                        comp.bulk_string += f'- y[{cn-1}*Lorg+{cn}+i] * y[{rn-1}*Lorg+{rn}+i] * k'
                                        comp.core_string += f'- y[{cn}*Lorg+{cn-1}] * y[{rn}*Lorg+{rn-1}] * k'

                                    elif mod_type.lower() == 'km-gap':
                                        comp.surf_string += f'- A[0] * (y[{cn-1}*Lorg+2*{cn-1}]/A[0]) * (y[{rn-1}*Lorg+2*{rn-1}]/A[0]) * k'
                                        comp.static_surf_string += f'- A[0] * (y[{cn-1}*Lorg+2*{cn-1}+1]/A[0]) * (y[{rn-1}*Lorg+2*{rn-1}+1]/A[0]) * k'
                                        comp.firstbulk_string += f'- V[0] * (y[{cn-1}*Lorg+2*{cn-1}+2]/V[0]) * (y[{rn-1}*Lorg+2*{rn-1}+2]/V[0]) * k'
                                        comp.bulk_string += f'- V[i] * (y[{cn-1}*Lorg+{cn}+i+{cn}]/V[i]) * (y[{rn-1}*Lorg+{rn}+i+{rn}]/V[i]) * k'
                                        comp.core_string += f'- V[-1] *  (y[{cn}*Lorg+{cn}+{cn-1}]/V[-1]) * (y[{rn}*Lorg+{rn}+{rn-1}]/V[-1]) * k'

                                    # sorted array of cn and rn to define correct reaction constant (k)
                                    sorted_cn_rn = np.array([cn,rn])
                                    sorted_cn_rn = np.sort(sorted_cn_rn)
                                    # print(cn,rn)
                                    # print(sorted_cn_rn)
                                    k_string = 'k'
                                    for n in sorted_cn_rn:
                                        comp.surf_string += f'_{n}'
                                        comp.firstbulk_string += f'_{n}'
                                        if mod_type.lower() == 'km-gap':
                                            comp.static_surf_string += f'_{n}'
                                        comp.bulk_string += f'_{n}'
                                        comp.core_string += f'_{n}'
                                        k_string += f'_{n}'

                                    comp.surf_string += '_surf'
                                    if mod_type.lower() == 'km-gap':
                                        comp.static_surf_string += '_surf'
                                    self.req_params.add(k_string)

                                # otherwise, add in the stoichiometry
                                else:
                                    # extract the stoich coefficients from reactant_stoich tuple
                                    react_1_stoich, react_2_stoich = reactant_stoich

                                    if mod_type.lower() == 'km-sub':
                                        comp.surf_string += f'- {react_1_stoich} * y[{cn-1}*Lorg+{cn-1}] * {react_2_stoich} * y[{rn-1}*Lorg+{rn-1}] * k'
                                        comp.firstbulk_string += f'- {react_1_stoich} * y[{cn-1}*Lorg+{cn}] * {react_2_stoich} * y[{rn-1}*Lorg+{rn}] * k'
                                        comp.bulk_string += f'- {react_1_stoich} * y[{cn-1}*Lorg+{cn}+i] * {react_2_stoich} * y[{rn-1}*Lorg+{rn}+i] * k'
                                        comp.core_string += f'- {react_1_stoich} * y[{cn}*Lorg+{cn-1}] * {react_2_stoich} * y[{rn}*Lorg+{rn-1}] * k'

                                    elif mod_type.lower() == 'km-gap':
                                        comp.surf_string += f'- A[0] * {react_1_stoich} * (y[{cn-1}*Lorg+2*{cn-1}]/A[0]) * {react_2_stoich} * (y[{rn-1}*Lorg+2*{rn-1}]/A[0]) * k'
                                        comp.static_surf_string += f'- A[0] * {react_1_stoich} * (y[{cn-1}*Lorg+2*{cn-1}+1]/A[0]) * {react_2_stoich} * (y[{rn-1}*Lorg+2*{rn-1}+1]/A[0]) * k'
                                        comp.firstbulk_string += f'- V[0] * {react_1_stoich} * (y[{cn-1}*Lorg+2*{cn-1}+2]/V[0]) * {react_2_stoich} * (y[{rn-1}*Lorg+2*{rn-1}+2]/V[0]) * k'
                                        comp.bulk_string += f'- V[i] * {react_1_stoich} * (y[{cn-1}*Lorg+{cn}+i+{cn}]/V[i]) * {react_2_stoich} * (y[{rn-1}*Lorg+{rn}+i+{rn}]/V[i]) * k'
                                        comp.core_string += f'- V[-1] * {react_1_stoich} * (y[{cn}*Lorg+{cn}+{cn-1}]/V[-1]) * {react_2_stoich} * (y[{rn}*Lorg+{rn}+{rn-1}]/V[-1]) * k'

                                    # sorted array of cn and rn to define correct reaction constant (k)
                                    sorted_cn_rn = np.array([cn,rn]).sort()
                                    k_string = 'k'
                                    for n in sorted_cn_rn:
                                        comp.surf_string += f'_{n}'
                                        if mod_type.lower() == 'km-gap':
                                            comp.static_surf_string += f'_{n}'
                                        comp.firstbulk_string += f'_{n}'
                                        comp.bulk_string += f'_{n}'
                                        comp.core_string += f'_{n}'
                                        k_string += f'_{n}'

                                    comp.surf_string += '_surf'
                                    if mod_type.lower() == 'km-gap':
                                        comp.static_surf_string += '_surf'
                                    self.req_params.add(k_string)

                    # add in first order decay
                    elif mod_type.lower() == 'km-sub':
                        cn = comp.component_number
                        comp.surf_string += f'- y[{cn-1}*Lorg+{cn-1}] * k1_{cn}'
                        comp.firstbulk_string += f'- y[{cn-1}*Lorg+{cn}] * k1_{cn}'
                        comp.bulk_string += f'- y[{cn-1}*Lorg+{cn}+i] * k1_{cn}'
                        comp.core_string += f'- y[{cn}*Lorg+{cn-1}] * k1_{cn}'

                        self.req_params.add(f'k1_{cn}')

                    elif mod_type.lower() == 'km-gap':
                        cn = comp.component_number
                        comp.surf_string += f'- A[0] * (y[{cn-1}*Lorg+2*{cn-1}]/A[0]) * k1_{cn}'
                        comp.static_surf_string += f'- A[0] * (y[{cn-1}*Lorg+2*{cn-1}+1]/A[0]) * k1_{cn}'
                        comp.firstbulk_string += f'- V[0] * (y[{cn-1}*Lorg+2*{cn-1}+2]/V[0]) * k1_{cn}'
                        comp.bulk_string += f'- V[i] * (y[{cn-1}*Lorg+{cn}+i+{cn}]/V[i]) * k1_{cn}'
                        comp.core_string += f'- V[-1] * (y[{cn}*Lorg+{cn}+{cn-1}]/V[-1]) * k1_{cn}'

                        self.req_params.add(f'k1_{cn}')

                if int(comp.component_number) in products:

                    r1, r2 = reactants

                    # get stoich coefficient index
                    stoich_index = None
                    if product_stoich is not None:
                        for ind, prod_number in enumerate(products):
                            if prod_number == comp.component_number:
                                stoich_index = ind

                    if stoich_index is not None:
                        stoich_coefficient = product_stoich[stoich_index]

                    # if no stoich given, assume a coefficient of 1 for reactants
                    if reactant_stoich is None:

                        # INCLUDE STOICHIOMETRY (MULTIPLY BY STOICH COEFF)
                        if mod_type.lower() == 'km-sub':
                            if stoich_index is None:
                                comp.surf_string += f'+ y[{r1-1}*Lorg+{r1-1}] * y[{r2-1}*Lorg+{r2-1}] * k'
                                comp.firstbulk_string += f'+ y[{r1-1}*Lorg+{r1}] * y[{r2-1}*Lorg+{r2}] * k'
                                comp.bulk_string += f'+ y[{r1-1}*Lorg+{r1}+i] * y[{r2-1}*Lorg+{r2}+i] * k'
                                comp.core_string += f'+ y[{r1}*Lorg+{r1-1}] * y[{r2}*Lorg+{r2-1}] * k'

                            # add product stoichiometry coefficient if supplied
                            else:
                                comp.surf_string += f'+ {stoich_coefficient} * y[{r1-1}*Lorg+{r1-1}] * y[{r2-1}*Lorg+{r2-1}] * k'
                                comp.firstbulk_string += f'+ {stoich_coefficient} * y[{r1-1}*Lorg+{r1}] * y[{r2-1}*Lorg+{r2}] * k'
                                comp.bulk_string += f'+ {stoich_coefficient} * y[{r1-1}*Lorg+{r1}+i] * y[{r2-1}*Lorg+{r2}+i] * k'
                                comp.core_string += f'+ {stoich_coefficient} * y[{r1}*Lorg+{r1-1}] * y[{r2}*Lorg+{r2-1}] * k'

                        elif mod_type.lower() == 'km-gap':
                            if stoich_index is None:
                                comp.surf_string += f'+ A[0] * (y[{r1-1}*Lorg+2*{r1-1}]/A[0]) * (y[{r2-1}*Lorg+2*{r2-1}]/A[0]) * k'
                                comp.static_surf_string += f'+ A[0] * (y[{r1-1}*Lorg+2*{r1-1}+1]/A[0]) * (y[{r2-1}*Lorg+2*{r2-1}+1]/A[0]) * k'
                                comp.firstbulk_string += f'+ V[0] * (y[{r1-1}*Lorg+2*{r1-1}+2]/V[0]) * (y[{r2-1}*Lorg+2*{r2-1}+2]/V[0]) * k'
                                comp.bulk_string += f'+ V[i] * (y[{r1-1}*Lorg+{r1}+i+{r1}]/V[i]) * (y[{r2-1}*Lorg+{r2}+i+{r2}]/V[i]) * k'
                                comp.core_string += f'+ V[-1] *  (y[{r1}*Lorg+{r1}+{r1-1}]/V[-1]) * (y[{r2}*Lorg+{r2}+{r2-1}]/V[-1]) * k'

                            # add product stoichiometry coefficient if supplied
                            else:
                                comp.surf_string += f'+ {stoich_coefficient} * A[0] * (y[{r1-1}*Lorg+2*{r1-1}]/A[0]) * (y[{r2-1}*Lorg+2*{r2-1}]/A[0]) * k'
                                comp.static_surf_string += f'+ {stoich_coefficient} * A[0] * (y[{r1-1}*Lorg+2*{r1-1}+1]/A[0]) * (y[{r2-1}*Lorg+2*{r2-1}+1]/A[0]) * k'
                                comp.firstbulk_string += f'+ {stoich_coefficient} * V[0] * (y[{r1-1}*Lorg+2*{r1-1}+2]/V[0]) * (y[{r2-1}*Lorg+2*{r2-1}+2]/V[0]) * k'
                                comp.bulk_string += f'+ {stoich_coefficient} * V[i] * (y[{r1-1}*Lorg+{r1}+i+{r1}]/V[i]) * (y[{r2-1}*Lorg+{r2}+i+{r2}]/V[i]) * k'
                                comp.core_string += f'+ {stoich_coefficient} * V[-1] *  (y[{r1}*Lorg+{r1}+{r1-1}]/V[-1]) * (y[{r2}*Lorg+{r2}+{r2-1}]/V[-1]) * k'

                        # sorted array of cn and rn to define correct reaction constant (k)
                        sorted_r1_r2 = np.array([r1,r2])
                        sorted_r1_r2 = np.sort(sorted_r1_r2)
                        # print(cn,rn)
                        # print(sorted_cn_rn)
                        k_string = 'k'
                        for n in sorted_r1_r2:
                            comp.surf_string += f'_{n}'
                            if mod_type.lower() == 'km-gap':
                                comp.static_surf_string += f'_{n}'
                            comp.firstbulk_string += f'_{n}'
                            comp.bulk_string += f'_{n}'
                            comp.core_string += f'_{n}'
                            k_string += f'_{n}'
                        # print('sorted r1 r2: ',sorted_r1_r2)
                        # print(f'Comp: {comp.component_number}, reactants = {r1} + {r2}, k_string: {k_string}')

                        comp.surf_string += '_surf'
                        if mod_type.lower() == 'km-gap':
                            comp.static_surf_string += '_surf'
                        # self.req_params.add(k_string)

                    # otherwise, add in the stoichiometry
                    else:
                        # extract the stoich coefficients from reactant_stoich tuple
                        react_1_stoich, react_2_stoich = reactant_stoich

                        if mod_type.lower() == 'km-sub':
                            if stoich_index is None:
                                comp.surf_string += f'+ {react_1_stoich} * y[{r1-1}*Lorg+{r1-1}] * {react_2_stoich} * y[{r2-1}*Lorg+{r2-1}] * k'
                                comp.firstbulk_string += f'+ {react_1_stoich} * y[{r1-1}*Lorg+{r1}] * {react_2_stoich} * y[{r2-1}*Lorg+{r2}] * k'
                                comp.bulk_string += f'+ {react_1_stoich} * y[{r1-1}*Lorg+{r1}+i] * {react_2_stoich} * y[{r2-1}*Lorg+{r2}+i] * k'
                                comp.core_string += f'+ {react_1_stoich} * y[{r1}*Lorg+{r1-1}] * {react_2_stoich} * y[{r2}*Lorg+{r2-1}] * k'

                            # add product stoichiometry coefficient if supplied
                            else:
                                comp.surf_string += f'+ {stoich_coefficient} * {react_1_stoich} * y[{r1-1}*Lorg+{r1-1}] * {react_2_stoich} * y[{r2-1}*Lorg+{r2-1}] * k'
                                comp.firstbulk_string += f'+ {stoich_coefficient} * {react_1_stoich} * y[{r1-1}*Lorg+{r1}] * {react_2_stoich} * y[{r2-1}*Lorg+{r2}] * k'
                                comp.bulk_string += f'+ {stoich_coefficient} * {react_1_stoich} * y[{r1-1}*Lorg+{r1}+i] * {react_2_stoich} * y[{r2-1}*Lorg+{r2}+i] * k'
                                comp.core_string += f'+ {stoich_coefficient} * {react_1_stoich} * y[{r1}*Lorg+{r1-1}] * {react_2_stoich} * y[{r2}*Lorg+{r2-1}] * k'

                        elif mod_type.lower() == 'km-gap':
                            if stoich_index is None:
                                comp.surf_string += f'+ A[0] * {react_1_stoich} * (y[{r1-1}*Lorg+2*{r1-1}]/A[0]) * {react_2_stoich} * (y[{r2-1}*Lorg+2*{r2-1}]/A[0]) * k'
                                comp.static_surf_string += f'+ A[0] * {react_1_stoich} * (y[{r1-1}*Lorg+2*{r1-1}+1]/A[0]) * {react_2_stoich} * (y[{r2-1}*Lorg+2*{r2-1}+1]/A[0]) * k'
                                comp.firstbulk_string += f'+ V[0] * {react_1_stoich} * (y[{r1-1}*Lorg+2*{r1-1}+2]/V[0]) * {react_2_stoich} * (y[{r2-1}*Lorg+2*{r2-1}+2]/V[0]) * k'
                                comp.bulk_string += f'+ V[i] * {react_1_stoich} * (y[{r1-1}*Lorg+{r1}+i+{r1}]/V[i]) * {react_2_stoich} * (y[{r2-1}*Lorg+{r2}+i+{r2}]/V[i]) * k'
                                comp.core_string += f'+ V[-1] * {react_1_stoich} * (y[{r1}*Lorg+{r1}+{r1-1}]/V[-1]) * {react_2_stoich} * (y[{r2}*Lorg+{r2}+{r2-1}]/V[-1]) * k'

                            # add product stoichiometry coefficient if supplied
                            else:
                                comp.surf_string += f'+ {stoich_coefficient} * A[0] * {react_1_stoich} * (y[{r1-1}*Lorg+2*{r1-1}]/A[0]) * {react_2_stoich} * (y[{r2-1}*Lorg+2*{r2-1}]/A[0]) * k'
                                comp.static_surf_string += f'+ {stoich_coefficient} * A[0] * {react_1_stoich} * (y[{r1-1}*Lorg+2*{r1-1}+1]/A[0]) * {react_2_stoich} * (y[{r2-1}*Lorg+2*{r2-1}+1]/A[0]) * k'
                                comp.firstbulk_string += f'+ {stoich_coefficient} * V[0] * {react_1_stoich} * (y[{r1-1}*Lorg+2*{r1-1}+2]/V[0]) * {react_2_stoich} * (y[{r2-1}*Lorg+2*{r2-1}+2]/V[0]) * k'
                                comp.bulk_string += f'+ {stoich_coefficient} * V[i] * {react_1_stoich} * (y[{r1-1}*Lorg+{r1}+i+{r1}]/V[i]) * {react_2_stoich} * (y[{r2-1}*Lorg+{r2}+i+{r2}]/V[i]) * k'
                                comp.core_string += f'+ {stoich_coefficient} * V[-1] * {react_1_stoich} * (y[{r1}*Lorg+{r1}+{r1-1}]/V[-1]) * {react_2_stoich} * (y[{r2}*Lorg+{r2}+{r2-1}]/V[-1]) * k'

                        # sorted array of cn and rn to define correct reaction constant (k)
                        sorted_r1_r2 = np.array([r1, r2]).sort()
                        k_string = 'k'
                        for n in sorted_r1_r2:
                            comp.surf_string += f'_{n}'
                            if mod_type.lower() == 'km-gap':
                                comp.static_surf_string += f'_{n}'
                            comp.firstbulk_string += f'_{n}'
                            comp.bulk_string += f'_{n}'
                            comp.core_string += f'_{n}'
                            k_string += f'_{n}'

                        comp.surf_string += '_surf'
                        if mod_type.lower() == 'km-gap':
                            comp.static_surf_string += '_surf'
                        # self.req_params.add(k_string)

            # account for volatilisation from surface

            # append the completed strings for this component to the master string list
            master_string_list.append(f'\n    #========component number {comp.component_number}, {comp.name}========\n')
            if mod_type.lower() == 'km-sub':
                master_string_list.append('\n    # sorption layer (Xs) /static surface layer (Yss)\n')
            elif mod_type.lower() == 'km-gap':
                master_string_list.append('\n    # sorption layer (Zs)\n')
            master_string_list.append('\n'+four_space+comp.surf_string+'\n')
            if mod_type.lower() == 'km-gap':
                master_string_list.append('\n    # static surface layer (Zss)\n')
                master_string_list.append(four_space+comp.static_surf_string+'\n')

            master_string_list.append('\n    # first bulk layer (Xb1/Yb1/Zb1)\n')
            master_string_list.append(four_space+comp.firstbulk_string+'\n')
            master_string_list.append(four_space+'for i in np.arange(1,Lorg-1):'+'\n')
            # if mod_type.lower() == 'km-gap':

            for s in comp.mass_transport_rate_inloop_strings:
                master_string_list.append(four_space+four_space+s+'\n')

            master_string_list.append('\n        # bulk i layer (Xbi/Ybi/Zbi)\n')
            master_string_list.append(four_space+four_space+comp.bulk_string+'\n')
            master_string_list.append('\n    # core layer n (Xbn/Ybn/Zbn)\n')
            master_string_list.append(four_space+comp.core_string+'\n')

        # unpack all params from params dictionary
        # doing this here because self.req_params built up in the model building
        # process
        unpack_params_string_list = []

        unpack_params_string_list.append('\n    #--------------Unpack parameters (random order)---------------\n')

        # add k_surf scale factor to req_params if desired
        if use_scaled_k_surf:
            self.req_params.add('scale_bulk_to_surf')

        # need the scale_bulk_to_surf defined first, order of rest does not matter
        # if k_surf to be defined explicitly, do not use scale_bulk_to_surf
        # if use_scaled_k_surf:
        #     unpack_params_string_list.append('\n    scale_bulk_to_surf = param_dict["scale_bulk_to_surf"]')

        # need to iterate over a list representation of req_params
        # otherwise set changes size during iteration (k_surf defined and added
        # to req_params set in loop)

        list_req_params = list(self.req_params)
        # unpacking parameter values from dict of Parameter objects
        unpack_params_string_list.append('\n    try:')
        count = 0
        for param_str in list_req_params:
            # add scale_bulk_to_surf if desired
            if count == 0 and use_scaled_k_surf:
                unpack_params_string_list.append('\n        scale_bulk_to_surf = param_dict["scale_bulk_to_surf"].value')
                count += 1

            param_unpack_str = '\n        ' + param_str + ' = param_dict[' + '"' + param_str + '"].value'
            # add in if varying param == True

            if param_str != 'scale_bulk_to_surf':  # don't duplicate this param
                unpack_params_string_list.append(param_unpack_str)

            # convert to surface reaction rates from bulk reaction rates
            if 'k_' in param_str and '_surf' not in param_str:
                k_surf_string = '\n        ' + param_str + '_surf = ' + param_str + ' * scale_bulk_to_surf'

                if not use_scaled_k_surf:
                    k_surf_string = '\n        ' + param_str + '_surf' +' = param_dict[' + '"' + param_str+'_surf' + '"].value'

                    self.req_params.add(param_str+'_surf')
                unpack_params_string_list.append(k_surf_string)

        # unpacking parameter values from dict of non-Parameter objects (they would throw an attribute error)
        unpack_params_string_list.append('\n\n    except AttributeError:')
        count = 0
        for param_str in list_req_params:
            # add scale_bulk_to_surf if desired
            if count == 0 and use_scaled_k_surf:
                unpack_params_string_list.append('\n        scale_bulk_to_surf = param_dict["scale_bulk_to_surf"]')
                count += 1

            param_unpack_str = '\n        ' + param_str + ' = param_dict[' + '"' + param_str + '"]'
            # add in if varying param == True

            if param_str != 'scale_bulk_to_surf':  # don't duplicate this param
                unpack_params_string_list.append(param_unpack_str)

            # convert to surface reaction rates from bulk reaction rates
            if 'k_' in param_str and '_surf' not in param_str:
                k_surf_string = '\n        ' + param_str + '_surf = ' + param_str + ' * scale_bulk_to_surf'

                if not use_scaled_k_surf:
                    k_surf_string = '\n        ' + param_str + '_surf' +' = param_dict[' + '"' + param_str+'_surf' + '"]'

                    self.req_params.add(param_str+'_surf')
                unpack_params_string_list.append(k_surf_string)

        # define the gas constant
        unpack_params_string_list.append('\n\n    R = 82.0578')
        unpack_params_string_list.append('\n\n    R_cm3_units = 8.314 * 1e6')
        unpack_params_string_list.append('\n    Na = 6.022e23')

        # wrapping up the dydt function
        master_string_list.append('\n')
        master_string_list.append(four_space+'return dydt')

        # open and write to the model .py file
        if date_tag:
            filename = date + '_' + mod_type.lower() + '_' + rxn_name + extention + '.py'
        else:
            filename = mod_type.lower() + '_' + rxn_name + extention + '.py'

        self.filename = filename

        with open(filename, 'w') as f:
            f.writelines(heading_strings)
            f.writelines(func_def_strings)
            f.writelines(unpack_params_string_list)
            f.writelines(master_string_list)

        self.constructed = True


class Parameter():
    '''
    A class which defines a parameter object along with its value, name, bounds (optional) and stats.

    Parameters
    ----------
    value : float
        The numerical value of the parameter

    name : str
        The name of the parameter.

    bounds : tup or list
        A tuple or list defining the lower and upper bounds for the parameter.

    vary : bool
        Whether this parameter is to vary during model optimisation.

    stats : dict
        A dictionary which holds statistics derived from MCMC sampling.

    '''
    
    def __init__(self, value=np.inf, name='', bounds=None, vary=False):

        self.name = name
        self.bounds = bounds
        if bounds is not None:
            assert len(bounds) == 2, f"There are {len(bounds)} numbers supplied to bounds, there needs to be 2."
        self.value = value
        self.vary = vary
        self.stats = {}

        if bounds is not None:
            assert vary is True, "Bounds have been supplied but the parameter has not been set to vary."







