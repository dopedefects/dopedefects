"""
File for tests of the structure_extract.py file
"""

import unittest
import numpy as np
import os

import dopedefects.tests.context as context

#Variables for testing
testing_dir = os.path.join(os.path.dirname(__file__), 'test_data/')
vectors = np.asarray([[6.86700013030, 0.0, 0.0], [-3.4335000515, 5.949965370,\
    0.0], [0.0, 0.0, 19.8069992065]])
direct_coords = np.asarray([[0.0, 0.0, 0.33676992], [0.0, 0.0, 0.666322984], \
    [0.333332981, 0.666667004, 0.332989987]])
cart_coords = np.asarray([[0.0, 0.0, 6.609139919], [0.0, 0.0 ,13.197858810], \
    [-0.000003576, 3.964666367, 6.595532417]])
types = ['Ar', 'Cr', 'I']

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
            "/cdte_crystal") == 'CdTe', "Unable to identify a Cd/Te crystal"
        return

    def test_cdtese_crystal(self):
        """
        Tests a POSCAR file containing a Cd/Te/Secrystal returns as cdtese
        """
        assert context.structure_extract.id_crystal(testing_dir + \
            "/cdtese_crystal") == 'CdTe_0.5Se_0.5', \
            "Unable to identify a Cd/(Te/Se) crystal"
        return

    def test_cdse_crystal(self):
        """
        Tests a POSCAR file containing a Cd/Se crystal returns as cdse
        """
        assert context.structure_extract.id_crystal(testing_dir + \
            "/cdse_crystal") == 'CdSe', "Unable to identify a Cd/Se crystal"
        return

    def test_cdses_crystal(self):
        """
        Tests a POSCAR file containing a Cd/(50/50 Se/S) crystal returns
        as cdses
        """
        assert context.structure_extract.id_crystal(testing_dir + \
            "/cdses_crystal") == 'CdSe_0.5S_0.5', \
            "Unable to identify a Cd/(Se/S) crystal"
        return

    def test_cds_crystal(self):
        """
        TesAts a POSCAR file containing a Cd/S crystal returns as cds
        """
        assert context.structure_extract.id_crystal(testing_dir + \
            "/cds_crystal") == 'CdS', "Unable to identify a Cd/S crystal"
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
        crystal = context.structure_extract.id_crystal(testing_dir + \
            "/cds_crystal")
        assert crystal  == 'CdS', "Unable to identify a Cd/S crystal"
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
    def test_direct_to_cart(self):
        """
        """
        transformed = context.structure_extract.direct_to_cart(direct_coords,\
            vectors)
        assert np.isclose(transformed, cart_coords, rtol=1e-01).all(),\
            "Direct and Cart coords are not the same."
        return

class dist_between(unittest.TestCase):
    """
    Test suite for the structure_extract.dist_between function
    """
    def test_dist_between(self):
        """
        """
        distance = context.structure_extract.dist_between([1,2,3], [4,5,6])
        assert np.isclose(distance, 5.19615242271), "Unable to properly \
calculate distance between two points."

class determine_closest_atoms(unittest.TestCase):
    """
    Test suite for the structure_extract.determine_closest_atoms
    function
    """
    closest_atoms = context.structure_extract.determine_closest_atoms(2, [0., \
        0., 6.609139919], cart_coords, types)
    def test_closest_atoms_length(self):
        """
        """
        assert len(determine_closest_atoms.closest_atoms) == 2, \
            "more atoms returned than asked for"
        return

    def test_closest_atoms_no_defect(self):
        """
        """
        assert 'Ar' not in determine_closest_atoms.closest_atoms, \
            "Defect included in returned atoms"
        return

    def test_type_association(self):
        """
        """
        print("CLOSEST ATOMS = ", determine_closest_atoms.closest_atoms)
        assert determine_closest_atoms.closest_atoms[0][2] == 'I',\
            "Improper coord/type association"
        return

    def test_defect_return(self):
        """
        """
        defected_return = context.structure_extract.determine_closest_atoms(0,\
            [0., 0., 6.609139919], cart_coords, types)
        assert len(defected_return) == 3, "Improper length for defect return"
        return

    def test_defect_list_return(self):
        """
        """
        assert len(context.structure_extract.determine_closest_atoms(2, [0., \
            0., 6.609139919], [[0., 0., 6.609139919]], ['Ar'])) == 3, "Improper\
 length for defect return with shortened list"
        return

class atom_angles(unittest.TestCase):
    """
    Test suite for the structure_extract.atom_angles function
    """
    right_angle_atoms = context.structure_extract.atom_angles([0., 0., 0.,], \
        [[ 12., [1., 1., 0.], 'Ar'], [14.,  [0., 0., 1.], 'F']])
    def test_linear_angle(self):
        """
        """
        atom_angle = context.structure_extract.atom_angles([0., 0., 6.609139919], \
            [12., [-1.209587812, 5.848490715, 5.038246632], 'Cr'])
        assert atom_angle[0] == 180, "Linear molecule doesn't return 180"
        return
    
    def test_perp_angle_angle(self):
        """
        """
        assert np.isclose(atom_angles.right_angle_atoms[0], 90),\
            "right angle atoms not 90"
        return

class geometry_defect(unittest.TestCase):
    """
    Test suite for the structure_extract.geometry_defect function
    """
    def test_geometry_defects(self):
        """
        """
        return
