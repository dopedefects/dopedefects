"""
File for tests of the descriptor_analysis.py file
"""

import unittest
import numpy as np
import os
import pandas as pd

import dopedefects.tests.context as context

#Variables for testing
testing_dir = os.path.join(os.path.dirname(__file__), 'test_data/')
test_df = pd.read_csv(testing_dir + '/test_data.csv')

rr_out_1 = np.array([-0.41841004, -0.41841004, -0.41841004,  0.55788006, -0.28571429,
       -0.28571429, -0.28571429,  0.38095238, -0.07407407, -0.07407407,
       -0.07407407,  0.09876543, -0.00881057, -0.00881057, -0.00881057,
        0.01174743])
rr_out_2 = np.array([-0.00881057, -0.00881057, -0.00881057,  0.01174743])

lasso_out_1 = np.array([[-0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
         1.25000000e+00],
       [-0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
         1.47500000e+00],
       [-0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
         1.49750000e+00],
       [-1.99944444e+00, -4.20188575e-16, -0.00000000e+00,
         1.66666667e-04]])
lasso_out_2 = np.array([[-1.99944444e+00, -4.20188575e-16, -0.00000000e+00,
         1.66666667e-04]])

rf_out = np.array([[0.25301205, 0.22891566, 0.25100402, 0.26706827]])


data_file = open('descriptor_analysis_example.dat', 'w')

#Unit tests
class do_feature_selection(unittest.TestCase):
    """
    Test suite for the descriptor_analysis.do_feature_selection function
    """
    def test_do_feature_selection(self):

        return

class rf_reg(unittest.TestCase):
    """
    Test suite for the descriptor_analysis.rfe_selection function
    """
    def test_rf_selection(self):
        X = test_df.iloc[:, 0:4].values
        Y = test_df.iloc[:,5].values
        data_file = open('descriptor_analysis_example.dat', 'w')

        Z = context.descriptor_analysis.rf_reg(X,Y,data_file,p=False)

        assert np.allclose(Z, rf_out)
        return

class ridge_reg(unittest.TestCase):
    """
    Test suite for the descriptor_analysis.ridge_reg function
    """
    def test_ridge_reg(self):
        X = test_df.iloc[:, 0:4].values
        Y = test_df.iloc[:,5].values
        data_file = open('descriptor_analysis_example.dat', 'w')

        Z1, Z2 = context.descriptor_analysis.ridge_reg(X,Y,data_file,p=False)

        assert np.allclose(Z1, rr_out_1)
        assert np.allclose(Z2, rr_out_2)

        return

class lasso_reg(unittest.TestCase):
    """
    Test suite for the descriptor_analysis.lasso_reg function
    """
    def test_lasso_reg(self):
        X = test_df.iloc[:, 0:4].values
        Y = test_df.iloc[:,5].values
        data_file = open('descriptor_analysis_example.dat', 'w')

        Z1, Z2 = context.descriptor_analysis.lasso_reg(X,Y,data_file,p=False)

        assert np.allclose(Z1, lasso_out_1)
        assert np.allclose(Z2, lasso_out_2)
        return

class get_data(unittest.TestCase):
    """
    Test suite for the descriptor_analysis.get_data function
    """
    def test_get_data(self):
        return
