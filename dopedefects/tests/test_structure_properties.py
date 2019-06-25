"""
File for tests of the structure_properties.py file
"""

import unittest
import numpy as np

import dopedefects.tests.context as context

#Testing variables
test_two_atoms =  [[1., [0., 0., 0.], 'H'], [2., [3., 0., 0.], 'H']]
test_two_defect = [0., [1., 0., 0.], 'H']
answer_two = np.asarray([[0.5, 1., 0.5], [0, 0.5, 1.], [0., 0., 0.5]])  

#Unit tests
class coulomb(unittest.TestCase):
    """
    Test suite for the colomb function
    """
    def test_three_atom_molec(self):
        """
        Tests to ensure that the coulomb matrix is able to return the
        properly calculated values for a 3 atom hydrogen chain.
        """
        col_test = context.structure_properties.coulomb(test_two_atoms, \
            test_two_defect)
        assert np.isclose(answer_two, col_test).all(), "Coulomb matrix for H \
doesn't match pretabulated answer"
        return
