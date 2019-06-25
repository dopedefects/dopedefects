'''Functions to preprocess data from hdf5 dataframe to load into ML model'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def resize_descriptors(df):
    '''Resizes bond, angle, coulomb matrix such that each value is a single
    descriptor'''

    bond_mat = np.zeros((len(df), len(df['Bond_Difference'][0])))
    for i in range(len(df)):
        bond_mat[i] = df['Bond_Difference'][i]


    angle_mat = np.zeros((len(df), len(df['Angle_Difference'][0])))
    for i in range(len(df)):
        angle_mat[i] = df['Angle_Difference'][i]

    x = df['Coulomb'][0]
    x = x.reshape(x.size,1)
    x = x[x > 0]
    coulomb_mat = np.zeros((len(df), len(x)))
    for i in range(len(df)):
        x = df['Coulomb'][i]
        x = x.reshape(x.size,1)
        x = x[x > 0]              # remove lower triangular zeros
        coulomb_mat[i] = manage_inf(x)

    return bond_mat, angle_mat, coulomb_mat


def manage_inf(x):
    '''Replaces values that are inf to the max non-inf value'''

    inf_ind = np.nonzero(np.isinf(x))[0]  # indices with inf value
    x_noinf = x
    max_val = np.sort(np.delete(x_noinf, inf_ind))[-1]   # max (non-inf) value
    np.put(x, inf_ind, np.ones(len(inf_ind))*max_val)

    return x


def load_data(filepath):
    '''Load training and testing data from df'''

    df = pd.read_hdf(filepath)
    df.reset_index(drop=True, inplace=True)
    bond_mat, angle_mat, coulomb_mat = resize_descriptors(df)


    ### decide X and y values
    X1 = df.iloc[:, 12:34].values   # base
    X2 = df.iloc[:, 40:].values     # additional elemental properties
    X3 = np.column_stack((bond_mat, angle_mat, coulomb_mat))
    Xall = np.column_stack((X1, X2, X3))

    Y = df.iloc[:, 3:12].values
    Y_labels = df.columns[3:12]


    ### select data
    X = X1
#     y = Y[:,0]

    return X, Y


def split_and_scale(X, y):
    """Preprocesses data such that all values are scaled in range [0,1],
    defined by training set only"""

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                        random_state=110)

    scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
    scalarX.fit(X_train)
    scalarY.fit(y_train.reshape(len(y_train),1))
    X_train = scalarX.transform(X_train)
    y_train = scalarY.transform(y_train.reshape(len(y_train),1))
    X_test = scalarX.transform(X_test)
    y_test = scalarY.transform(y_test.reshape(len(y_test),1))

    return X_train, y_train, X_test, y_test, scalarX, scalarY


def data_unscale(X, y, scalarX, scalarY):
    '''Unscale data to original range'''

    X = scalarX.inverse_transform(X)
    y = scalarY.inverse_transform(y)

    return X, y
