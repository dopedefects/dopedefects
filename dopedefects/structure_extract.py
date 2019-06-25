"""
Helper functions to extract information from given POSCAR files
"""
import os
import numpy as np


def find_files(data_dir):
    """
    Locates the POSCAR files within the 'data_dir'

    Inputs
    ------
        data_dir: String with the top level of the directory to recurse
              through

    Outputs
    -------
        poscar:   List containing paths to all found VASP POSCAR files
    """
    poscar = []
    for root, dirs, files in os.walk(data_dir):
        list_file_path = os.path.join(root, 'POSCAR')
        if os.path.isfile(list_file_path):
            count = sum(1 for line in open(list_file_path))
            if count > 0:
                poscar.append(list_file_path)
            else:
                print("File %s empty, skipping" %list_file_path)
    assert len(poscar) > 0, 'No POSCAR files found in %s.' %data_dir
    return poscar

def id_crystal(poscar):
    """
    With the given poscar file, will attempt to identify which crystal
    type it is.  Has the following options for return:
        Cd/Te
        Cd/Se
        Cd/S
        Cd Te/Se (50/50)
        Cd Se/S  (50/50)

    Inputs
    ------
    poscar:       string containing the path for the POSCAR file

    Outputs
    -------
    Crystal type as string
    """
    types = []
    count = []
    te_amount = 0
    se_amount = 0
    s_amount  = 0
    with open(poscar, 'r') as fileIn:
        for i, line in enumerate(fileIn):
            if i == 5:
                types = line.split()
            if i == 6:
                count = line.split()
                for j in range(len(count)):
                    count[j] = int(count[j])
            if i >= 6:
                break
    assert len(types) == len(count), \
        "Unequal number atom types and atom counts in %s" %poscar
    assert 'Cd' in types, 'Crystal defied by %s does not contain Cd' %poscar
    if 'Te' in types:
        te_amount = count[types.index('Te')]
    if 'Se' in types:
        se_amount = count[types.index('Se')]
    if 'S' in types:
        s_amount = count[types.index('S')]

    #comparisons not set to 0 given it could be an impurity in crystal:
    if te_amount > 0 and se_amount < 2 and s_amount < 2 and '/CdTe/' in poscar:
        return 'CdTe'
    if se_amount > 0:
        if te_amount / se_amount > 0.4 and '/CdTe_0.5Se_0.5/' in poscar:
            return 'CdTe_0.5Se_0.5'

    if se_amount > 0 and te_amount < 2 and s_amount < 2 and '/CdSe/' in poscar:
        return 'CdSe'
    if s_amount > 0:
        if se_amount / s_amount > 0.4 and '/CdSe_0.5S_0.5/' in poscar:
            return 'CdSe_0.5S_0.5'

    if s_amount > 0 and te_amount < 2 and se_amount < 2 and '/CdS/' in poscar:
        return 'CdS'

    raise Exception("Unknown crystal type given by %s" %poscar)

def impurity_type(poscar):
    """
    With the given VASP POSCAR file, determine defect type

    Inputs
    ------
    poscar:   string containing the path for the POSCAR file

    Outputs
    -------
    atom type of the defect
    """
    types = []
    count = []
    num_one = 0
    with open(poscar, 'r') as fileIn:
        for i, line in enumerate(fileIn):
            if i == 5:
                types = line.split()
            if i == 6:
                count = line.split()
                for j in range(len(count)):
                    count[j] = int(count[j])
                    if count[j] == 1:
                        num_one += 1
            if i >= 6:
                break
    assert len(types) == len(count), \
        "Unequal number atom types and atom counts in %s" %poscar
    #Assuming will only be 1 defect per unit cell
    if min(count) > 5:
        return "pure"
    #if there's more than one with only 1 entry
    if num_one > 1:
        #print("num_one > 1 for %s" %poscar)
        for x in range(len(types)):
            #print("TYPES TO CONSIDER: ", types[x], "COUNT: ", count[x])
            if '/' + types[x] + '/' in poscar and count[x] == 1:
                #print("RETURNING: ", types[x])
                return types[x]
    else:
        return types[count.index(min(count))]

def dopant_site(poscar):
    """
    With the given VASP POSCAR file, determine the impurity location
    given the folder structure

    Inputs
    ------
    poscar:  File path for the relevant POSCAR file

    Outputs
    -------
    impurity type of the defect as defined by collaborators
    """
    if 'M_Cd/' in poscar:
        return 'M_Cd'
    if 'M_S/' in poscar:
        return 'M_S'
    if 'M_Se/' in poscar:
        return 'M_Se'
    if 'M_Te/' in poscar:
        return 'M_Te'
    if 'M_i_Cd_site/' in poscar:
        return 'M_i_Cd_site'
    if 'M_i_S_site/' in poscar:
        return 'M_i_S_site'
    if 'M_i_Se_site/' in poscar:
        return 'M_i_Se_site'
    if 'M_i_Te_site/' in poscar:
        return 'M_i_Te_site'
    if 'M_i_other/' in poscar:
        return 'M_i_old'
    if 'M_i_neutral_site/' in poscar:
        return 'M_i_old'
    else:
        raise Exception("Unknown dopant site given by %s" %poscar)

def unit_vector(vector):
    """
    Returns unit vector of the vector.

    Inputs
    ------
    vector :  numpy vector which to return the unit vector of

    Outputs
    -------
    vector :  numpy unit vector for input vector

    """
    return vector / np.linalg.norm(vector)

def angle_between(a, b):
    """
    Determine the angle (in degrees) between two vectors
    inspired from: https://stackoverflow.com/questions/2827393/angles
    -between-two-n-dimensional-vectors-in-python/13849249#13849249

    Inputs
    ------
    a     : numpy array 1
    b     : numpy array 2

    Outputs
    -------
    angle : float of the angle (in degrees) between a and b
    """
    a_u = unit_vector(a)
    b_u = unit_vector(b)
    return np.rad2deg(np.arccos(np.clip(np.dot(a_u, b_u), -1.0, 1.0)))

def direct_to_cart(direct, vectors):
    """
    Given the direct coordinates, transform into cartesian

    Inputs
    ------
    direct    : numpy matrix containing the direct coordinates
    vectors   : numpy matrix containing the a, b, and c unit cell
                vectors

    Outputs
    -------
    xyz       : numpy matrix with the xyz coordinates
    """
    a = np.sqrt(np.sum(vectors[0, :] ** 2))
    b = np.sqrt(np.sum(vectors[1, :] ** 2))
    c = np.sqrt(np.sum(vectors[2, :] ** 2))
    alpha = np.deg2rad(angle_between(vectors[1, :], vectors[2, :]))
    beta  = np.deg2rad(angle_between(vectors[2, :], vectors[0, :]))
    gamma = np.deg2rad(angle_between(vectors[0, :], vectors[1, :]))

    omega = a * b * c * np.sqrt(1 - np.square(np.cos(alpha)) - \
        np.square(np.cos(beta)) - np.square(np.cos(gamma)) + 2 * np.cos(alpha)\
        * np.cos(beta) * np.cos(gamma))

    #transpose to do matrix multiplication through numpy
    direct = np.asmatrix(np.transpose(direct))

    mult = np.asmatrix(np.asarray([[a, b * np.cos(gamma), c * np.cos(beta)],\
                        [0, b * np.sin(gamma), (c * np.cos(alpha) -\
                          np.cos(beta) * np.cos(gamma)) / np.sin(gamma)],\
                        [0, 0, omega / (a * b * np.sin(gamma))]]))

    xyz = np.dot(mult, direct)

    #transpose so is easier to parse
    return np.transpose(xyz)

def dist_between(xyz1, xyz2):
    """
    Determine the distance between two points in cartesian space

    Inputs
    ------
    xyz1: list array of doubles containing the coordinates of the first point.
    xyz2: list of doubles containing the coordinates of the second point.

    Outputs
    -------
    double value of the distance between the two points.
    """
    xyz1 = np.asarray(xyz1)
    xyz2 = np.asarray(xyz2)
    return np.sqrt(np.sum((xyz1-xyz2) ** 2))

def determine_closest_atoms(number, defect, xyz, types):
    """
    Determine the atoms closest to the defect
        np.square(vectors[0,2]))

    Inputs
    ------
    number :    int declaring how many surrounding atoms to use
    defect :    list containing xyz coords for the defect atom
    xyz    :    list containing xyz coordinates for all atoms
    types  :    list containing the atom types (in order of the coords)

    Outputs
    -------
    array containing the distance, the cartesian coord., and
    the type of atom.
    """
    if number < 1 or len(xyz) < 2:
        #Presuming want the coordinate of the defect itself
        return [0, np.asmatrix(defect), types[0]]
    else:
        atm_list = [[0, [0., 0., 0.], 'A'] for x in range(len(xyz))]
        i = 0
        for coord in xyz:
            atm_list[i][0] = dist_between(defect, coord)
            atm_list[i][1] = coord
            atm_list[i][2] = types[i]
            i += 1
        #atm_list.sort(axis=0)
        atm_list = sorted(atm_list, key=lambda x: x[0])

        #Return first number, skipping the 1st value, presuming is defect
        return atm_list[1:number + 1]

def atom_angles(defect, bonds):
    """
    Determine angles around the defect coordinate.

    Inputs
    ------
    defect:   list containing the xyz coordinate for the defect
    xyz   :   2-D list containing the xyz coordinates for the
              atoms surrounding the defect.

    Outputs
    -------
    angles:   numpy array containing the angles for the atoms around the
              defect
    """
    angles = []
    defect = np.asarray(defect)
    if len(bonds) > 2 and isinstance(bonds[2], str):
        #Only should trip if there's only one entry in the bonds matrix
        #and as such the molecule being tested is linear
        angles = [180.]
    else:
        for i in range(len(bonds)):
            for j in range(i, len(bonds)):
                if i == j:
                    continue
                else:
                    center1 = np.asarray(bonds[i][1]) - np.asarray(defect)[0]
                    center2 = np.asarray(bonds[j][1]) - np.asarray(defect)[0]
                    angles.append(angle_between(center1, center2))
    return angles

def geometry_defect(number, defect, poscar, return_coord=False):
    """
    Determine the Atom type for the closest several atoms (ratio and
    type), determine the bond lengths to the first several atoms around
    the defect (return average), and determine the bond angle for the
    first several atoms around the defect (return average).

    Inputs
    ------
    defect:     output from impurity_type, is the defect type
    poscar:     VASP POSCAR file (cart or direct) with atom types and
                positions.

    Output
    ------
    bonds :     numpy array containing the bondlength, coordinates, and
                atom type of the atoms surrounding the defect atom.
    angles:     numpy array containing the angles for the  atoms
                surrounding the defect atom.
    """
    vectors     = []
    types       = []
    count       = []
    coord_type  = []
    coord       = []
    atoms       = []
    with open(poscar, 'r') as fileIn:
        i = -1
        selective = 0
        for line in fileIn:
            i += 1
            if i >= 2 and i < 5:
                #lattice vecotrs
                vectors.append([float(_) for _ in line.split()])
            if i == 5:
                #Atom Types
                types = line.split()
            if i == 6:
                #Atom counts
                count = [int(_) for _ in line.split()]
            if i == 7 or selective:
                #coordinate type
                if "Selective" in line:
                    selective = 1
                else:
                    coord_type = line
                    if selective:
                        i -= 1
                    selective = 0
            if i >= 8 and i < 8 + sum(count):
                #coordinates
                coord.append([float(_) for _ in line.split()[0:3]])

    assert len(types) == len(count), \
        "Unequal number atom types and atom counts in %s" %poscar

    assert sum(count) == len(coord), "Unequal number of declard atoms (%i) and\
        coordinates (%i) in %s" %(sum(count), len(coord), poscar)

    #Direct to Cartesian transformation if needed
    if 'Cart' in coord_type:
        pass
    else:
        coord = direct_to_cart(np.asarray(coord), np.asarray(vectors))

    #List of all the atom types in it
    for i in range(len(types)):
        for j in range(int(count[i])):
            atoms.append(types[i])

    #Determine which coord is the defect
    defect_coord = coord[atoms.index(defect)]

    #Returning for the bond differences:
    if return_coord:
        return (defect_coord, np.asarray(coord))

    #Determine bond lengths around the defect:
    bonds = determine_closest_atoms(number, defect_coord, np.asarray(coord), \
        atoms)

    #Determine bond angles around the defect:
    angles = atom_angles(defect_coord, bonds)

    #Return:
    #print("BONDS = ", bonds)
    #print("ANGLES = ", angles)
    return [bonds, angles]
