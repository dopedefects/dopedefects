import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE

def main():

    df, el, unit, dEl, coul, dCell, cell = get_data_h5('data.hdf5')

    print(df.dtypes)



def do_feature_selection(name, data_file):
    """
    Does lasso regression, ridge regression, recursive feature elimination,
    and random forrest modeling for each of the nine properties being modeled

    Inputs
    ------
    name :      String containing the name of the csv file containing the data
                to be analyzed (i.e. "CdTe" for CdTe.csv)

    data_file : String containing the name of the file the model statistics will
                be stored in, where the RMSE and R-Squared values for each model
                will be stored

    Outputs
    -------
    cd_list :   Contains four lists, one for each model type (Lasso, Ridge,
                Recursive Feature Elimination, and Random Forrest), each of which
                contains the coefficients for each descriptor used in the model
                to predict $\Delta$H(Cd-rich)

    mod_list :  Contains four lists, one for each model type (Lasso, Ridge,
                Recursive Feature Elimination, and Random Forrest), each of which
                contains the coefficients for each descriptor used in the model
                to predict $\Delta$H(Mod)

    x_list :    Contains four lists, one for each model type (Lasso, Ridge,
                Recursive Feature Elimination, and Random Forrest), each of which
                contains the coefficients for each descriptor used in the model
                to predict $\Delta$H(X-rich)

    plus_3_list : Contains four lists, one for each model type (Lasso, Ridge,
                Recursive Feature Elimination, and Random Forrest), each of which
                contains the coefficients for each descriptor used in the model
                to predict the (+3/+2) charge transfer state

    plus_2_list : Contains four lists, one for each model type (Lasso, Ridge,
                Recursive Feature Elimination, and Random Forrest), each of which
                contains the coefficients for each descriptor used in the model
                to predict the (+2/+1) charge transfer state

    plus_1_list : Contains four lists, one for each model type (Lasso, Ridge,
                Recursive Feature Elimination, and Random Forrest), each of which
                contains the coefficients for each descriptor used in the model
                to predict the (+1/0) charge transfer state

    minus_1_list : Contains four lists, one for each model type (Lasso, Ridge,
                Recursive Feature Elimination, and Random Forrest), each of which
                contains the coefficients for each descriptor used in the model
                to predict the (0/-1) charge transfer state

    minus_2_list : Contains four lists, one for each model type (Lasso, Ridge,
                Recursive Feature Elimination, and Random Forrest), each of which
                contains the coefficients for each descriptor used in the model
                to predict the (-1/-2) charge transfer state

    minus_3_list : Contains four lists, one for each model type (Lasso, Ridge,
                Recursive Feature Elimination, and Random Forrest), each of which
                contains the coefficients for each descriptor used in the model
                to predict the (-2/-3) charge transfer state
    """

    #assert len(types) == len(count)

    CdS_df = get_data_csv(name)

    X = CdS_df[['Period', 'Group', 'Site', 'Delta Ion. Rad.', 'Delta At. Wt.',
       'Delta Cov. Rad.', 'Delta Ion. En.', 'Delta At. Rad.', 'Delta EA',
       'Delta EN', 'Delta At. Num.', 'Delta Val.', '# Cd Neighbors',
       '# X Neighbors', 'Corrected VBM (eV)', 'Corrected CBM (eV)',
        '∆H_uc(Cd-rich)', '∆H_uc(Mod)', '∆H_uc(X-rich)']]

    data_file.write(name)
    data_file.write('\n\t# rows: ' + str(CdS_df.shape[0]))

    data_file.write("\n\n\t∆H(Cd-rich):")
    Y = CdS_df[['∆H(Cd-rich)']]
    cd_las_coefs, cd_las_list = lasso_reg(X,Y,data_file)
    cd_rr_coefs, cd_rr_list = ridge_reg(X,Y,data_file)
    #cd_rfe_list = rfe_selection(X,Y.values.ravel(),data_file)

    cd_rfr_list = rf_reg(X,Y.values.ravel(),data_file)

    data_file.write("\n\n\t∆H(Mod):")
    Y = CdS_df[['∆H(Mod)']]
    mod_las_coefs, mod_las_list = lasso_reg(X,Y,data_file)
    mod_rr_coefs, mod_rr_list = ridge_reg(X,Y,data_file)
    #mod_rfe_list = rfe_selection(X,Y.values.ravel(),data_file)

    mod_rfr_list = rf_reg(X,Y.values.ravel(),data_file)

    data_file.write("\n\n\t∆H(X-rich):")
    Y = CdS_df[['∆H(X-rich)']]
    x_las_coefs, x_las_list = lasso_reg(X,Y,data_file)
    x_rr_coefs, x_rr_list = ridge_reg(X,Y,data_file)
    #x_rfe_list = rfe_selection(X,Y.values.ravel(),data_file)

    x_rfr_list = rf_reg(X,Y.values.ravel(),data_file)

    data_file.write("\n\n\t(+3/+2):")
    Y = CdS_df[['(+3/+2)']]
    plus_3_las_coefs, plus_3_las_list = lasso_reg(X,Y,data_file)
    plus_3_rr_coefs, plus_3_rr_list = ridge_reg(X,Y,data_file)
    #plus_3_rfe_list = rfe_selection(X,Y.values.ravel(),data_file)

    plus_3_rfr_list = rf_reg(X,Y.values.ravel(),data_file)

    data_file.write("\n\n\t(+2/+1):")
    Y = CdS_df[['(+2/+1)']]
    plus_2_las_coefs, plus_2_las_list = lasso_reg(X,Y,data_file)
    plus_2_rr_coefs, plus_2_rr_list = ridge_reg(X,Y,data_file)
    #plus_2_rfe_list = rfe_selection(X,Y.values.ravel(),data_file)

    plus_2_rfr_list = rf_reg(X,Y.values.ravel(),data_file)

    data_file.write("\n\n\t(+1/0):")
    Y = CdS_df[['(+1/0)']]
    plus_1_las_coefs, plus_1_las_list = lasso_reg(X,Y,data_file)
    plus_1_rr_coefs, plus_1_rr_list = ridge_reg(X,Y,data_file)
    #plus_1_rfe_list = rfe_selection(X,Y.values.ravel(),data_file)

    plus_1_rfr_list = rf_reg(X,Y.values.ravel(),data_file)

    data_file.write("\n\n\t(0/-1):")
    Y = CdS_df[['(0/-1)']]
    minus_1_las_coefs, minus_1_las_list = lasso_reg(X,Y,data_file)
    minus_1_rr_coefs, minus_1_rr_list = ridge_reg(X,Y,data_file)
    #minus_1_rfe_list = rfe_selection(X,Y.values.ravel(),data_file)

    minus_1_rfr_list = rf_reg(X,Y.values.ravel(),data_file)

    data_file.write("\n\n\t(-1/-2):")
    Y = CdS_df[['(-1/-2)']]
    minus_2_las_coefs, minus_2_las_list = lasso_reg(X,Y,data_file)
    minus_2_rr_coefs, minus_2_rr_list = ridge_reg(X,Y,data_file)
    #minus_2_rfe_list = rfe_selection(X,Y.values.ravel(),data_file)

    minus_2_rfr_list = rf_reg(X,Y.values.ravel(),data_file)

    data_file.write("\n\n\t(-2/-3):")
    Y = CdS_df[['(-2/-3)']]
    minus_3_las_coefs, minus_3_las_list = lasso_reg(X,Y,data_file)
    minus_3_rr_coefs, minus_3_rr_list = ridge_reg(X,Y,data_file)
    #minus_3_rfe_list = rfe_selection(X,Y.values.ravel(),data_file)

    minus_3_rfr_list = rf_reg(X,Y.values.ravel(),data_file)


    data_file.write("\n\n")

    cd_list = np.concatenate((cd_las_list, cd_rr_list, cd_rfr_list), axis=0)
    mod_list = np.concatenate((mod_las_list, mod_rr_list, mod_rfr_list), axis=0)
    x_list = np.concatenate((x_las_list, x_rr_list, x_rfr_list), axis=0)

    plus_3_list = np.concatenate((plus_3_las_list, plus_3_rr_list, plus_3_rfr_list), axis=0)
    plus_2_list = np.concatenate((plus_2_las_list, plus_2_rr_list, plus_2_rfr_list), axis=0)
    plus_1_list = np.concatenate((plus_1_las_list, plus_1_rr_list, plus_1_rfr_list), axis=0)

    minus_1_list = np.concatenate((minus_1_las_list, minus_1_rr_list, minus_1_rfr_list), axis=0)
    minus_2_list = np.concatenate((minus_2_las_list, minus_2_rr_list, minus_2_rfr_list), axis=0)
    minus_3_list = np.concatenate((minus_3_las_list, minus_3_rr_list, minus_3_rfr_list), axis=0)

    return cd_list, mod_list, x_list, \
    plus_3_list, plus_2_list, plus_1_list, \
    minus_1_list, minus_2_list, minus_3_list

def rfe_selection(X,Y,data_file,p=False):
    """
    Does recursive feature elimination on the data provided

    Inputs
    ------
    X :         Coulumns of the pandas dataframe that contains the data for each
                of the descriptors to be used

    Y :         Column of the pandas dataframe that contains the values to be
                predicted

    data_file : String containing the name of the file the model statistics will
                be stored in, where the RMSE and R-Squared values for each model
                will be stored

    Outputs
    -------
    coefs :     Contains a list of the coefficient for each descriptor used
    """

    n=len(X.columns)
    high_score=0
    nof=0
    score_list =[]
    temp = []


    for n in range(len(nof_list)):
        X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)
        model = LinearRegression()
        rfe = RFE(model,nof_list[n])
        X_train_rfe = rfe.fit_transform(X_train,y_train)
        X_test_rfe = rfe.transform(X_test)
        model.fit(X_train_rfe,y_train)
        score = model.score(X_test_rfe,y_test)
        score_list.append(score)

        if(n==0 or score>high_score):
            high_score = score
            nof = nof_list[n]
            coefs_ = model.coef_
            temp = rfe.support_


    count = 0
    coefs = np.zeros(19)

    for i in range(len(temp)):
        if (temp[i] == True):
            coefs[i] = coefs_[count]
            count += 1

    data_file.write('\n\t\tRFE Score with %d features: \t\t\t%f' % (nof, high_score))

    if(p==True): print('\n\t\tRFE Score with %d features: \t\t\t%f' % (nof, high_score)) #Fix spacing

    return np.array([coefs])

def ridge_reg(X,Y,data_file,p=False):
    """
    Does ridge regression on the data provided

    Inputs
    ------
    X :         Coulumns of the pandas dataframe that contains the data for each
                of the descriptors to be used

    Y :         Column of the pandas dataframe that contains the values to be
                predicted

    data_file : String containing the name of the file the model statistics will
                be stored in, where the RMSE and R-Squared values for each model
                will be stored

    Outputs
    -------
    coefs :     Contains a list of the coefficient for each descriptor used
    """

    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=3)

    high_score = 0
    alpha_ = 0
    #coefs = np.zeros(19)

    rr0001 = Ridge(alpha=0.001)
    rr0001.fit(X_train, y_train)
    Ridge_train_score0001 = rr0001.score(X_train,y_train)
    Ridge_test_score0001 = rr0001.score(X_test, y_test)
    high_score = Ridge_test_score0001
    alpha_ = 0.001
    coefs = rr0001.coef_
    pred = rr0001.predict(X_test)
    rmse = np.sqrt(MSE(y_test, pred))

    rr001 = Ridge(alpha=0.01)
    rr001.fit(X_train, y_train)
    Ridge_train_score001 = rr001.score(X_train,y_train)
    Ridge_test_score001 = rr001.score(X_test, y_test)
    if(Ridge_test_score001 > high_score):
        high_score = Ridge_test_score001
        alpha_ = 0.01
        coefs = rr001.coef_
        pred = rr001.predict(X_test)
        rmse = np.sqrt(MSE(y_test, pred))

    rr01 = Ridge(alpha=0.1)
    rr01.fit(X_train, y_train)
    Ridge_train_score01 = rr01.score(X_train,y_train)
    Ridge_test_score01 = rr01.score(X_test, y_test)
    if(Ridge_test_score01 > high_score):
        high_score = Ridge_test_score01
        alpha_ = 0.1
        coefs = rr01.coef_
        pred = rr01.predict(X_test)
        rmse = np.sqrt(MSE(y_test, pred))

    rr10 = Ridge(alpha=10)
    rr10.fit(X_train, y_train)
    Ridge_train_score10 = rr10.score(X_train,y_train)
    Ridge_test_score10 = rr10.score(X_test, y_test)
    if(Ridge_test_score10 > high_score):
        high_score = Ridge_test_score10
        alpha_ = 10
        coefs = rr10.coef_
        pred = rr10.predict(X_test)
        rmse = np.sqrt(MSE(y_test, pred))

    rr100 = Ridge(alpha=100)
    rr100.fit(X_train, y_train)
    Ridge_train_score100 = rr100.score(X_train,y_train)
    Ridge_test_score100 = rr100.score(X_test, y_test)
    if(Ridge_test_score100 > high_score):
        high_score = Ridge_test_score100
        alpha_ = 100
        coefs = rr100.coef_
        pred = rr100.predict(X_test)
        rmse = np.sqrt(MSE(y_test, pred))

    rr1000 = Ridge(alpha=1000)
    rr1000.fit(X_train, y_train)
    Ridge_train_score1000 = rr1000.score(X_train,y_train)
    Ridge_test_score1000 = rr1000.score(X_test, y_test)
    if(Ridge_test_score1000 > high_score):
        high_score = Ridge_test_score1000
        alpha_ = 1000
        coefs = rr1000.coef_
        pred = rr1000.predict(X_test)
        rmse = np.sqrt(MSE(y_test, pred))

    data_file.write('\n\t\tRidge Regression Score with alpha=%f: \t%f' % (alpha_, high_score))
    data_file.write('\n\t\t\tRMSE: \t\t%f' % (rmse))

    if(p==True):
        print('\n\t\tRidge Regression Score with alpha=%f: \t%f' % (alpha_, high_score))
        print('\n\t\tRMSE: \t\t%f' % (rmse))

    return np.concatenate((rr001.coef_, rr10.coef_, rr100.coef_, rr1000.coef_), axis=0), np.array(coefs)

def lasso_reg(X,Y,data_file,p=False):
    """
    Does lasso regression on the data provided

    Inputs
    ------
    X :         Coulumns of the pandas dataframe that contains the data for each
                of the descriptors to be used

    Y :         Column of the pandas dataframe that contains the values to be
                predicted

    data_file : String containing the name of the file the model statistics will
                be stored in, where the RMSE and R-Squared values for each model
                will be stored

    Outputs
    -------
    coefs :     Contains a list of the coefficient for each descriptor used
    """

    X_train,X_test,y_train,y_test=train_test_split(X,Y, test_size=0.3, random_state=31)

    high_score = 0
    alpha_ = 0
    #coefs = np.zeros(19)

    lasso = Lasso()
    lasso.fit(X_train,y_train)
    train_score=lasso.score(X_train,y_train)
    test_score=lasso.score(X_test,y_test)
    coeff_used = np.sum(lasso.coef_!=0)
    high_score = test_score
    alpha_ = 1
    coefs = lasso.coef_
    pred = lasso.predict(X_test)
    rmse = np.sqrt(MSE(y_test, pred))

    lasso01 = Lasso(alpha=0.1, max_iter=10e5)
    lasso01.fit(X_train,y_train)
    train_score01=lasso01.score(X_train,y_train)
    test_score01=lasso01.score(X_test,y_test)
    coeff_used01 = np.sum(lasso01.coef_!=0)
    if(test_score01 > high_score):
        high_score = test_score01
        alpha_ = 0.1
        coefs = lasso01.coef_
        pred = lasso01.predict(X_test)
        rmse = np.sqrt(MSE(y_test, pred))

    lasso001 = Lasso(alpha=0.01, max_iter=10e5)
    lasso001.fit(X_train,y_train)
    train_score001=lasso001.score(X_train,y_train)
    test_score001=lasso001.score(X_test,y_test)
    coeff_used001 = np.sum(lasso001.coef_!=0)
    if(test_score001 > high_score):
        high_score = test_score001
        alpha_ = 0.01
        coefs = lasso001.coef_
        pred = lasso001.predict(X_test)
        rmse = np.sqrt(MSE(y_test, pred))

    lasso0001 = Lasso(alpha=0.001, max_iter=10e5)
    lasso0001.fit(X_train,y_train)
    train_score0001=lasso0001.score(X_train,y_train)
    test_score0001=lasso0001.score(X_test,y_test)
    coeff_used0001 = np.sum(lasso0001.coef_!=0)
    if(test_score0001 > high_score):
        high_score = test_score0001
        alpha_ = 0.001
        coefs = lasso0001.coef_
        pred = lasso0001.predict(X_test)
        rmse = np.sqrt(MSE(y_test, pred))

    data_file.write('\n\t\tLasso Regression Score with alpha=%f: \t%f' % (alpha_, high_score))
    data_file.write('\n\t\t\tRMSE: \t\t%f' % (rmse))

    if(p==True):
        print('\n\t\tLasso Regression Score with alpha=%f: \t%f' % (alpha_, high_score))
        print('\n\t\tRMSE: \t\t%f' % (rmse))

    return np.concatenate((np.array([lasso.coef_]), np.array([lasso01.coef_]),
                           np.array([lasso001.coef_]), np.array([lasso0001.coef_])), axis=0), np.array([coefs])

def rf_reg(X,Y,data_file,p=False):
    """
    Does random forrest regression on the data provided

    Inputs
    ------
    X :         Coulumns of the pandas dataframe that contains the data for each
                of the descriptors to be used

    Y :         Column of the pandas dataframe that contains the values to be
                predicted

    data_file : String containing the name of the file the model statistics will
                be stored in, where the RMSE and R-Squared values for each model
                will be stored

    Outputs
    -------
    coefs :     Contains a list of the coefficient for each descriptor used
    """

    X_train,X_test,y_train,y_test=train_test_split(X,Y, test_size=0.3, random_state=31)

    high_score = 0
    n_est = 0
    #coefs = np.zeros(19)

    rf = RandomForestRegressor(n_estimators = 10, random_state = 31)
    rf.fit(X_train, y_train)
    train_score=rf.score(X_train,y_train)
    test_score=rf.score(X_test,y_test)
    pred = rf.predict(X_test)
    rmse = np.sqrt(MSE(y_test, pred))
    high_score = test_score
    n_est = 10
    coefs = rf.feature_importances_


    rf = RandomForestRegressor(n_estimators = 100, random_state = 31)
    rf.fit(X_train, y_train)
    train_score=rf.score(X_train,y_train)
    test_score=rf.score(X_test,y_test)
    if(test_score > high_score):
        high_score = test_score
        n_est = 100
        coefs = rf.feature_importances_
        pred = rf.predict(X_test)
        rmse = np.sqrt(MSE(y_test, pred))

    rf = RandomForestRegressor(n_estimators = 1000, random_state = 31)
    rf.fit(X_train, y_train)
    train_score=rf.score(X_train,y_train)
    test_score=rf.score(X_test,y_test)
    if(test_score > high_score):
        high_score = test_score
        n_est = 1000
        coefs = rf.feature_importances_
        pred = rf.predict(X_test)
        rmse = np.sqrt(MSE(y_test, pred))

    data_file.write('\n\t\tRandom Forest Regression n=%d: \t\t%f' % (n_est, test_score))
    data_file.write('\n\t\t\tRMSE: \t\t%f' % (rmse))

    if(p==True):
        print('\n\t\tRandom Forest Regression n=%d: \t\t%f' % (n_est, test_score))
        print('\n\t\tRMSE: \t\t%f' % (rmse))

    return np.array([coefs])

def get_data_csv(name):
    """
    Retrieves data from a csv file and puts it into a pandas dataframe

    Inputs
    ------
    name :      String containing the name of the csv file containing the data
                to be analyzed

    Outputs
    -------
    df :        Pandas dataframe containing all of the data extracted from the
                csv file
    """

    df = pd.read_csv(name)

    df = df[pd.notnull(df['(+3/+2)'])]

    df.drop('Unnamed: 12', axis=1, inplace=True)
    df.drop('Descriptors -->', axis=1, inplace=True)
    df.drop('M', axis=1, inplace=True)
    df.drop('CdX', axis=1, inplace=True)
    df.drop('Doping Site', axis=1, inplace=True)

    if '∆H(Te-rich)' in df:
        df.rename(columns={'∆H(Te-rich)':'∆H(X-rich)',
                           '∆H(Te-rich).1':'∆H(X-rich).1',
                           '# Te Neighbors':'# X Neighbors'}, inplace=True)
    elif '∆H(Se-rich)' in df:
        df.rename(columns={'∆H(Se-rich)':'∆H(X-rich)',
                           '∆H(Se-rich).1':'∆H(X-rich).1',
                           '# Se Neighbors':'# X Neighbors'}, inplace=True)
    elif '∆H(S-rich)' in df:
        df.rename(columns={'∆H(S-rich)':'∆H(X-rich)',
                           '∆H(S-rich).1':'∆H(X-rich).1',
                           '# S Neighbors':'# X Neighbors'}, inplace=True)
    elif '∆H(X-rich)' in df:
        df.rename(columns={'∆H(S-rich)':'∆H(X-rich)',
                           '∆H(S-rich).1':'∆H(X-rich).1',
                           '# Se/Te Neighbors':'# X Neighbors'}, inplace=True)

    df.rename(columns={'∆H(Cd-rich).1':'∆H_uc(Cd-rich)',
                       '∆H(Mod).1':'∆H_uc(Mod)',
                       '∆H(X-rich).1':'∆H_uc(X-rich)'}, inplace=True)

    return df

def get_data_h5(name,key=None):
    """
    Retrieves data from hdf5 file and puts it into a pandas dataframe, then puts
    the descriptors into groups

    Inputs
    ------
    name :      String containing the name of the hdf5 file containing the data
                to be analyzed (i.e. data.hdf5)

    Outputs
    -------
    df :        Pandas dataframe containing all of the data extracted from the
                hdf5 file

    el :        contains the names of the columns of the elemental descriptors

    unit :      contains the names of the columns of the unit cell descriptors

    dEl :       contains the names of the columns of the $\Delta$ elemental descriptors

    coul :      contains the names of the columns of the coulomb matrix descriptors

    dCell :     contains the names of the columns of the $\Delta$ cell descriptors

    cell :      contains the names of the columns of the cell descriptors
    """
    df = pd.read_hdf(name, key)

    #Separate Bonds
    for i in range(7):
        name = 'Bonds_'+str(i)
        df[name] = df['Bonds'].apply(lambda x: x[i][0])

    #Separate Angles
    for i in range(21):
        name = 'Angles_'+str(i)
        df[name] = df['Angles'].apply(lambda x: x[i])

    #Separate Bond Difference
    for i in range(7):
        name = 'Bond_Difference_'+str(i)
        df[name] = df['Bond_Difference'].apply(lambda x: x[i])

    #Separate Angle Difference
    for i in range(21):
        name = 'Angle_Difference_'+str(i)
        df[name] = df['Angle_Difference'].apply(lambda x: x[i])

    #Separate Coulomb Matrix
    for i in range(8):
        for j in range(8):
            name = 'Coulomb_'+str(i*8+j)
            df[name] = df['Coulomb'].apply(lambda x: x[i][j])
            df[name].replace(np.inf, 10000000, inplace=True)

    #print(df.dtypes)

    #get list column names for each type of descriptor
    el    = df.columns[40:63]   #'Ionic_radius', 'Boiling_point', 'Melting_point', 'Density', 'Atomic_weight', 'ICSD_volume', 'Cov_radius', 'Atomic_radius', 'Electron_affinity', 'Atomic_vol', 'Mendeleev_number', 'Ionization_pot_1', 'Ionization_pot_2', 'Ionization_pot_3', 'Therm_expn_coeff', 'Sp_heat_cap', 'Therm_cond', 'Heat_of_fusion', 'Heat_of_vap', 'Electronegativity', 'At_num', 'Valence', 'Ox_state'
    unit  = df.columns[31:34]   #'dH(Cd-rich) UC', 'dH(Mod) UC', 'dH(X-rich) UC'
    dEl   = df.columns[12:31]   #'Period', 'Group', 'Site', 'Delta Ion. Rad.', 'Delta At. Wt.', 'Delta Cov. Rad.', 'Delta Ion. En.', 'Delta At. Rad.', 'Delta EA', 'Delta EN', 'Delta At. Num.', 'Delta Val.', '# Cd Neighbors', '# S Neighbors', '# Se Neighbors', '# Te Neighbors', '# Se/Te Neighbors', 'Corrected VBM (eV)', 'Corrected CBM (eV)'
    coul  = df.columns[119:183] #'Coulomb_0' - 'Coulomb_63'
    dCell = df.columns[63:91]   #'Bonds_0' - 'Bonds_6', 'Angles_0' - 'Angles_20'
    cell  = df.columns[91:119]  #'Bond_Difference_0' - 'Bond_Difference_6', 'Angle_Difference_0' - 'Angle_Difference_20'

    return df, el, unit, dEl, coul, dCell, cell


if __name__ == '__main__':
    main()
