"""
File for tests of the structure_extract.py file
"""

import unittest
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
        self.assertRaises(Exception, lambda:context.data_extract.\
        id_crystal("NONEXISTANTFILE"))
        return

    def test_fail_unequal_count_atoms(self):
        """
        Tests that a POSCAR with an unequal number of defined atoms as
        defined atom numbers raises an exception.
        """

        return
        
    def test_contains_Cd(self):
        """
        Tests that a POSCAR which does not contain Cd raises an
        exception.
        """

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

        return

    def test_pure(self):
        """
        Tests to ensure that if a pure cell is passed in it will return
        that it is a pure crystal.
        """

        return

    def test_impurity(self):
        """
        Tests to ensure that if a doped cell is passed in the type of
        defect will be properly returned.
        """

        return

class unit_vector(unittest.TestCase):
    """
    Test suite for the structure_extract.unit_vector function
    """

class angle_between(unittest.TestCase):
    """
    Test suite for the structure_extract.angle_between function
    """
    #test parallel, perp, opposite
    
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


