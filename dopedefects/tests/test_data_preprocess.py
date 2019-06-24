import pandas as pd
import unittest
import dopedefects.tests.context as context
import numpy as np



def test_resize_descriptors():
    """Test function for resizing descriptors"""

    # Small dataframe test case
    test_df = pd.DataFrame({'Bond_Difference': [[1, 2], [3, 4], [5, 6]],
                            'Angle_Difference': [[7, 8], [9, 10], [11, 12]],
            'Coulomb':[np.asarray([[1, 0, 10], [0, np.inf, 1], [0, 0, np.inf]]),
                     np.asarray([[2, 0, 20], [0, np.inf, 2], [0, 0, np.inf]]),
                     np.asarray([[3, 0, 30], [0, np.inf, 3], [0, 0, np.inf]])]})

    bond_mat, angle_mat, coulomb_mat = \
                    context.data_preprocess.resize_descriptors(test_df)

    assert bond_mat[0][0] == 1, \
        "Bond matrix is incorrectly constructed"
    assert angle_mat[1][1] == 10, \
        "Angle matrix is incorrectly constructed"
    assert coulomb_mat[0].size == 5, \
        "Coulomb matrix does not contain correct number of entries"
    assert coulomb_mat[2][1] == 30, \
        "Coulomn matrix is incorrectly constructed"

    return


def test_manage_inf():
    """Test function for managing infinity values"""

    x = np.asarray([np.inf, 5, 10, np.inf, 100])
    x = context.data_preprocess.manage_inf(x)

    assert x[0] == 100, "Inf value is not replaced by max non-inf value"

    return


def test_load_data():
    """Does not need to be tested since function loads data from file and
    selects descriptors to use"""

    return


def test_split_and_scale():
    """Test function for spliting data and scaling in [0, 1]"""

    X = np.asarray([[0 , 100], [0.02, 99], [0.5, 0.5], [0.3, 0.3],
                    [0.4, 0.4], [50, 50], [60, 60], [80, 80],
                    [55, 65], [100, 0]])
    y = np.asarray([-0.5, 0, 1, 0, 0.5, 0.1, 0.2, 0.3, 0.4, 1.2])

    X_train, y_train, X_test, y_test, scalarX, scalarY = \
        context.data_preprocess.split_and_scale(X, y)

    assert len(X_train) == 8, \
        "Split is of wrong length"
    assert len(X_test) == 2, \
        "Split is of wrong length"
    assert np.isclose(y_train.min(), 0), \
        "Min value for y_train not scaled to 0"
    assert np.isclose(y_train.max(), 1), \
        "Max value for y_train is not scaled to 1"

    return


def test_data_unscale():
    """Does nto require testing because using sklearn inbuilt functions here"""

    return



if __name__ == '__main__':
    unittest.main()
