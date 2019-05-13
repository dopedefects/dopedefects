"""
File for tests of the descriptor_analysis.py file
"""

import unittest
import numpy as np
import os

import dopedefects.tests.context as context

#Variables for testing
testing_dir = os.path.join(os.path.dirname(__file__), 'test_data/')

#Unit tests
class make_subplot(unittest.TestCase):
    """
    Test suite for the descriptor_analysis.make_subplot function
    """
    def test_make_subplot(self):
        return

class do_feature_selection(unittest.TestCase):
    """
    Test suite for the descriptor_analysis.do_feature_selection function
    """
    def test_do_feature_selection(self):
        return

class rfe_selection(unittest.TestCase):
    """
    Test suite for the descriptor_analysis.rfe_selection function
    """
    def test_rfe_selection(self):
        return

class ridge_reg(unittest.TestCase):
    """
    Test suite for the descriptor_analysis.ridge_reg function
    """
    def test_ridge_reg(self):
        return

class lasso_reg(unittest.TestCase):
    """
    Test suite for the descriptor_analysis.lasso_reg function
    """
    def test_lasso_reg(self):
        return

class get_data(unittest.TestCase):
    """
    Test suite for the descriptor_analysis.get_data function
    """
    def test_get_data(self):
        return
