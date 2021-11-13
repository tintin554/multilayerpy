# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 13:55:44 2021

@author: Adam
"""

import unittest

from multilayerpy.build import ModelType, ReactionScheme, DiffusionRegime, ModelComponent, ModelBuilder, Parameter
from multilayerpy.simulate import Simulate, make_layers, initial_concentrations
import numpy as np
import matplotlib.pyplot as plt

class TestModelConstruction(unittest.TestCase):
    
    # model type
    def test_type(self):
        with self.assertRaises(AssertionError):
            mod = ModelType(1,'triangle')
            
    # reaction scheme
    def test_reactionscheme(self):
        
        mod_type = ModelType('km-sub',geometry='film')
        
        react_tup_list_wrong = [(1,1),
                  (4,5),
                  (4,6)]
        
        react_tup_list = [(1,2),
                  (4,5),
                  (4,6)]

        prod_tup_list = [(3,4,5),
                         (6,),
                         (7,)]
        prod_tup_wrong = [(3,3,5),
                         (6,),
                         (7,)]
        
        prod_stoich = [(0.454, 1.0, 1-0.454)]
        
        # test failure when repeated reactants in reaction
        with self.assertRaises(AssertionError):
            rs = ReactionScheme(mod_type, 
                              reactants = react_tup_list_wrong,
                              products = prod_tup_list,
                              product_stoich = prod_stoich)
        
        # test failure when repeated products
        with self.assertRaises(AssertionError):
            rs = ReactionScheme(mod_type, 
                              reactants = react_tup_list,
                              products = prod_tup_wrong,
                              product_stoich = prod_stoich)
            
    def test_modelcomponent(self):
        
        mod_type = ModelType('km-sub',geometry='film')
        
        react_tup_list = [(1,2),
                  (4,5),
                  (4,6)]
        
        prod_tup_list = [(3,4,5),
                         (6,),
                         (7,)]
        
        rs = ReactionScheme(mod_type, 
                              reactants = react_tup_list,
                              products = prod_tup_list,
                              )
        
        model_comp = ModelComponent(1,rs,name='component',gas=False)
        
        # make sure types not changed when instantiated 
        self.assertEqual(type(model_comp.component_number), int, 'Model component number did not return an int')
        self.assertEqual(type(model_comp.reaction_scheme.model_type), ModelType,'ModelComponent.ReactionScheme.model_type was not a ModelType object')
        self.assertEqual(type(model_comp.gas), bool, 'ModelComponent.gas was not a boolean')
        self.assertEqual(type(model_comp.comp_dependent_adsorption), bool, 'ModelComponent.comp_dependent_adsorption is not a boolean')
        
        # make sure all strings are actually strings
        for attrib in dir(model_comp):
            if 'string' in attrib:
                self.assertEqual(type(attrib), str)
         
    def test_modelsim(self):
        # make model and run the simulation
        
        mod_type = ModelType('km-sub',geometry='spherical')
        
        react_tup_list = [(1,2),
                  (4,5),
                  (4,6)]
        
        prod_tup_list = [(3,4,5),
                         (6,),
                         (7,)]
        
        rs = ReactionScheme(mod_type, 
                              reactants = react_tup_list,
                              products = prod_tup_list,
                              )
        
        OA = ModelComponent(1,rs,name='oleic acid')
        O3 = ModelComponent(2,rs,name='ozone',gas=True)
        prod = ModelComponent(3,rs,name='products')
        
        mod_comps_dict = {'1':OA,
                          '2':O3,
                          '3':prod}
        
        diff_dict = {'1':None,
                     '2':None,
                     '3':None}
        
        dr = DiffusionRegime(mod_type,mod_comps_dict,diff_dict=diff_dict)
        dr()
        
        model = ModelBuilder(rs,mod_comps_dict,dr)
        
        model.build(name_extention='unittesing')
        
        n_layers, rp = 10, 0.2e-4
        V, A, thick = make_layers(mod_type,n_layers,rp)
        
        bulk_conc_dict = {'1':1.21e21,'2':0,'3':0}
        surf_conc_dict = {'1':9.68e13,'2':0,'3':0}
        y0 = initial_concentrations(mod_type,bulk_conc_dict,surf_conc_dict,n_layers)
        
        param_dict = {'delta_3':Parameter(1e-7),
              'alpha_s_0_2':Parameter(4.2e-4),
              'delta_2':Parameter(0.4e-7),
              'Db_2':Parameter(1e-5),
              'delta_1':Parameter(0.8e-7),
              'Db_1':Parameter(1e-10),
              'Db_3':Parameter(1e-10),
              'k_1_2':Parameter(1.7e-15),
              'H_2':Parameter(4.8e-4),
              'Xgs_2': Parameter(7.0e13),
              'Td_2': Parameter(1e-2),
              'w_2':Parameter(3.6e4),
              'T':Parameter(298.0),
              'k_1_2_surf':Parameter(6.0e-12)}
        
        sim = Simulate(model,param_dict)
        output = sim.run(n_layers,rp,[0,40],100,V,A,thick,y0)
        
        data = sim.xy_data_total_number()
        test_data = np.genfromtxt('unittest_data.txt')
        #comp_bool_array = data == test_data
        
        plt.figure()
        plt.title('Testing KM-SUB')
        plt.plot(test_data[:,0],test_data[:,1],label='truth',lw=5,color=(1,0,0,0.4))
        plt.plot(data[:,0],data[:,1],label='unit test',ls='--',color='b')
        plt.xlabel('Time / s')
        plt.ylabel('Number of oleic acid molecules')
        plt.legend()
        plt.show()
        
        # check that output is within 0.01% of test data output
        precision = 0.0001
        violation = False
        for i in range(len(test_data)):
            test_datapoint = test_data[i,1]
            unittest_datapoint = data[i,1]
            
            one_percent_bound = test_datapoint * 0.0001
            
            ub, lb = test_datapoint + one_percent_bound, test_datapoint - one_percent_bound
        
            if unittest_datapoint < ub and unittest_datapoint > lb:
                continue
            else:
                violation = False
        
        #self.assertEqual(comp_bool_array.all(), True)
        self.assertEqual(violation, False,'Unit test model outut not equal to test data within the required precision of {} %'.format(precision * 100))

if __name__ == "__main__":
    unittest.main()
