"""
Functions to calculate information given the defect structure.
"""
import copy
import numpy as np
try:
    import atomic_table
    import structure_extract
except:
    import dopedefects.atomic_table as atomic_table
    import dopedefects.structure_extract as structure_extract

def coulomb(bond_length, defect):
    """
    Calculate the atomic coulomb matrix:
                  /   0.5 * Z_i^{2.4}   i == j
        C_{i,j} = |      Z_i * Z_j
                  |      ---------      i =/= j
                  \     |R_i - R_j|

    Inputs
    ------
    bond_length : The bond length matrix as output from the
                  geometry_defect function.
    defect      : The coord and type of the defect

    Outputs
    -------
    coulomb     : Numpy array containing the upper-triangular coulomb
                  matrix
    """
    #prepend the defect coord (so as to keep in order of bond length)
    bond_length = [defect] + bond_length

    #initialize coloumb matrix
    coulomb = np.zeros(shape=(len(bond_length), len(bond_length)), dtype=float)

    #calculate coloumb matrix
    for i in range(len(bond_length)):
        for j in range(i, len(bond_length)):
            if j < i:
                continue
            elif i == j:
                coulomb[i, i] = 0.5 * np.power(atomic_table.atomic_weight(\
                    bond_length[i][2])[0], 2.4)
            else:
                coulomb[i, j] = (atomic_table.atomic_weight(bond_length[i][2])\
                    [0] * atomic_table.atomic_weight(bond_length[j][2])[0]) / (\
                    np.absolute(bond_length[i][0] - bond_length[j][0]))
    return coulomb

def bond_difference(pure, defected, defected_ang, defect_point):
    """
    Return the bond lengthdifference between the given cell and a pure 
    cell.

    Inputs
    ------
    pure          : The pure xyz coords
    defected      : The defect bond lengths
    defected_ang  : The defect angles
    defect_point  : The location of the defect
    """
    pure_coords = copy.copy(pure)
    defect_location = copy.copy(defect_point)
    defect_angles = np.asarray(copy.copy(defected_ang))
    defected_bonds = []
    pure_bonds = []
    number = len(defected)
    types = ['x' for x in range(len(pure_coords))]

    bonds = structure_extract.determine_closest_atoms(number, \
        defect_location, np.asarray(pure_coords), types)
    #Bond Diff
    for i, entry in enumerate(bonds):
        defected_bonds.append(defected[i][0])
        pure_bonds.append(bonds[i][0])
    bond_diff = np.subtract(np.asarray(pure_bonds), np.asarray(defected_bonds))
    
    #Ang Diff
    angles = np.asarray(structure_extract.atom_angles(defect_point, bonds))
    angle_diff = np.subtract(angles, defect_angles)
    return (abs(bond_diff), abs(angle_diff))
