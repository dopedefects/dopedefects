"""
Functions to calculate information given the defect structure.
"""
import numpy as np
try:
    import atomic_table
except:
    import dopedefects.atomic_table as atomic_table


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
                coulomb[i,i] = 0.5 * np.power(atomic_table.atomic_weight(\
                    bond_length[i][2])[0], 2.4)
            else:
                coulomb[i,j] = (atomic_table.atomic_weight(bond_length[i][2])[0]\
                    * atomic_table.atomic_weight(bond_length[j][2])[0]) / (\
                    np.absolute(bond_length[i][0] - bond_length[j][0]))
    return coulomb
                

