"""
File for tests of the structure_extract.py file
"""

import unittest
import numpy as np
import os

import dopedefects.tests.context as context

#Variables for testing
testing_dir = os.path.join(os.path.dirname(__file__), 'test_data/')

#Unit tests
class find_files(unittest.TestCase):
    """
    Test suite for the structure_extract.find_files function
    """
    def test_nonexistant_directory(self):
        """
        Tests that a non-existant directory returns an error.  Should be
        handeled by built in python os package.
        """
        self.assertRaises(Exception, lambda:context.structure_extract.\
            find_files("NONEXISTANTDIRECTORY"))
        return

    def test_single_directory(self):
        """
        Tests that the function is able to return single POSCAR file in
        a directory.
        """
        assert len(context.structure_extract.find_files(testing_dir + \
            "/SINGLE_DIR/")) == 1, "More than one POSCAR found in the %s \
            directory" %(testing_dir + "/SINGLE_DIR/")
        return

    def test_multi_directory(self):
        """
        Tests that is able to return multiple POSCAR files in multiple
        sub directories.
        """
        num_found =  len(context.structure_extract.find_files(testing_dir + \
            "/MULT_DIR/"))
        assert num_found == 2, "%i POSCAR files found in %s where only two \
            should be present" %(num_found, testing_dir + "/MULT_DIR/")
        return

class id_crystal(unittest.TestCase):
    """
    Test suite for the structure_extract.id_crystal function
    """
    def test_nonexisitant_file(self):
        """
        Tests a non-existant file returns an error.  Should be handled
        by built in python os package.\
        """
        self.assertRaises(Exception, lambda:context.structure_extract.\
        id_crystal("NONEXISTANTFILE"))
        return

    def test_fail_unequal_count_atoms(self):
        """
        Tests that a POSCAR with an unequal number of defined atoms as
        defined atom numbers raises an exception.
        """
        self.assertRaises(Exception, lambda:context.structure_extract.\
            id_crystal(testing_dir + "/unequal_counts"))
        return
        
    def test_contains_Cd(self):
        """
        Tests that a POSCAR which does not contain Cd raises an
        exception.
        """
        self.assertRaises(Exception, lambda:context.structure_extract.\
            id_crystal(testing_dir + "/equal_counts_no_Cd"))
        return

    def test_cdte_crystal(self):
        """
        Tests a POSCAR file containing a Cd/Te crystal returns as cdte
        """
        assert context.structure_extract.id_crystal(testing_dir + \
            "/cdte_crystal") == 'cdte', "Unable to identify a Cd/Te crystal"
        return

    def test_cdtese_crystal(self):
        """
        Tests a POSCAR file containing a Cd/Te/Secrystal returns as cdtese
        """
        assert context.structure_extract.id_crystal(testing_dir + \
            "/cdtese_crystal") == 'cdtese', \
            "Unable to identify a Cd/(Te/Se) crystal"
        return

    def test_cdse_crystal(self):
        """
        Tests a POSCAR file containing a Cd/Se crystal returns as cdse
        """
        assert context.structure_extract.id_crystal(testing_dir + \
            "/cdse_crystal") == 'cdse', "Unable to identify a Cd/Se crystal"
        return

    def test_cdses_crystal(self):
        """
        Tests a POSCAR file containing a Cd/(50/50 Se/S) crystal returns
        as cdses
        """
        assert context.structure_extract.id_crystal(testing_dir + \
            "/cdses_crystal") == 'cdses', \
            "Unable to identify a Cd/(Se/S) crystal"
        return

    def test_cds_crystal(self):
        """
        Tests a POSCAR file containing a Cd/S crystal returns as cds
        """
        assert context.structure_extract.id_crystal(testing_dir + \
            "/cds_crystal") == 'cds', "Unable to identify a Cd/S crystal"
        return

    def test_unknown_crystal(self):
        """
        Tests a POSCAR file containing an unknown crystal returns
        throws an exception
        """
        self.assertRaises(Exception, lambda:context.structure_extract.\
            id_crystal(testing_dir + "unkonown_crystal"))
        return

class impurity_type(unittest.TestCase):
    """
    Test suite for the structure_extract.impurity_type function
    """
    def test_fail_unequal_count_atoms(self):
        """
        Tests that a POSCAR with an unequal number of defined atoms as
        defined atom numbers raises an exception.
        """
        self.assertRaises(Exception, lambda:context.structure_extract.\
            id_crystal(testing_dir + "/unequal_counts"))
        return

    def test_pure(self):
        """
        Tests to ensure that if a pure cell is passed in it will return
        that it is a pure crystal.
        """
        assert context.structure_extract.id_crystal(testing_dir + \
            "/cds_crystal") == 'cds', "Unable to identify a Cd/S crystal"

        return

    def test_impurity(self):
        """
        Tests to ensure that if a doped cell is passed in the type of
        defect will be properly returned.
        """
        assert context.structure_extract.impurity_type(testing_dir + \
            "/unknown_crystal") == 'Ge', \
            "Couldn't identify the dopant in structure"
        return

class unit_vector(unittest.TestCase):
    """
    Test suite for the structure_extract.unit_vector function
    """
    def test_unit_vector(self):
        vector = [1,4,7]
        length = np.sqrt(sum(context.structure_extract.unit_vector(vector) \
            ** 2))
        assert np.isclose(length, 1), "unit vector returned is not of length 1"
        return

class angle_between(unittest.TestCase):
    """
    Test suite for the structure_extract.angle_between function
    """
    #test parallel, perp, opposite
    def test_angle_xy_perp(self):
        """
        Test the xy perpendicular angle
        """
        assert np.isclose(context.structure_extract.angle_between([1,0,0], \
            [0,1,0]), 90), "Perp XY angle not 90 degrees"
        return
    
    def test_angle_yz_perp(self):
        """
        Test teh yz perpendicular angle
        """
        assert np.isclose(context.structure_extract.angle_between([0,1,0], \
            [0,0,1]), 90), "Perp XY angle not 90 degrees"
        return
    
    def test_angle_parallel(self):
        """
        Test a parallel angle
        """
        assert np.isclose(context.structure_extract.angle_between([1,1,1], \
            [2,2,2]), 0), "Parallel X angle not 0"
        return

    def test_angle_anti_parallel(self):
        """
        Test a 180 angle
        """
        assert np.isclose(context.structure_extract.angle_between([1,1,1], \
            [-1,-1,-1]), 180), "Opposite angle not 180"
        return

class direct_to_cart(unittest.TestCase):
    """
    Test suite for the structure_extract.direct_to_cart function
    """


class dist_between(unittest.TestCase):\
    """
    Test suite for the structure_extract.dist_between function
    """

class determine_closest_atoms(unittest.TestCase):
    """
    Test suite for the structure_extract.determine_closest_atoms
    function
    """

class atom_angles(unittest.TestCase):
    """
    Test suite for the structure_extract.atom_angles function
    """

class geometry_defect(unittest.TestCase):
    """
    Test suite for the structure_extract.geometry_defect function
    """


