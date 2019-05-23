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
from sklearn.svm import SVR

def main():
    # #Create Data File
    # data_file = open('featureSelectionTestScores.dat', 'w')
    #
    # #Get Coefficients
    # CdTeCoefs = do_feature_selection('CdTe',data_file)
    # CdSeCoefs = do_feature_selection('CdSe',data_file)
    # CdSCoefs = do_feature_selection('CdS',data_file)
    # CdSeSCoefs = do_feature_selection('CdSe_0.5S_0.5',data_file)
    # CdTeSCoefs = do_feature_selection('CdTe_0.5S_0.5',data_file)
    #
    #
    # #Close Data File
    # data_file.close()
    #
    # #Set Up Figure
    # fig, ax = plt.subplots(9,5,figsize=(20,10))
    # plt.xlim((0,19))
    #
    # #Make Subplots
    # im = make_subplot('CdTe', CdTeCoefs, 0, ax)
    # make_subplot('CdSe', CdSeCoefs, 1, ax)
    # make_subplot('CdS', CdSCoefs, 2, ax)
    # make_subplot('CdSe_0.5S_0.5', CdSeSCoefs, 3, ax)
    # make_subplot('CdTe_0.5S_0.5', CdTeSCoefs, 4, ax)
    #
    # #Format Figure, Add Colorbar
    # plt.tight_layout()
    # fig.subplots_adjust(right=0.95) #6
    # cbar_ax = fig.add_axes([0.956, 0.185, 0.01, 0.796])
    # bar = fig.colorbar(im, cax=cbar_ax)
    #
    # #Save and Display Figure
    # plt.savefig('featureSelection.png')
    # plt.show()


    #Create Data File
    data_file = open('featureSelectionTestScores_all.dat', 'w')

    df = pd.read_hdf('data.hdf5')

    print(df.head())

    print(df.dtypes)

    X = df[['Period', 'Group', 'Site', 'Delta Ion. Rad.', 'Delta At. Wt.',
       'Delta Cov. Rad.', 'Delta Ion. En.', 'Delta At. Rad.', 'Delta EA',
       'Delta EN', 'Delta At. Num.', 'Delta Val.', '# Cd Neighbors',
       '# S Neighbors', '# Se Neighbors', '# Te Neighbors',
       '# Se/Te Neighbors', 'Corrected VBM (eV)', 'Corrected CBM (eV)',
       'dH(Cd-rich) UC', 'dH(Mod) UC', 'dH(X-rich) UC',
       'Ionic_radius', 'Boiling_point', 'Melting_point',
       'Density', 'Atomic_weight', 'ICSD_volume', 'Cov_radius',
       'Atomic_radius', 'Electron_affinity', 'Atomic_vol', 'Mendeleev_number',
       'Ionization_pot_1', 'Ionization_pot_2', 'Ionization_pot_3',
       'Therm_expn_coeff', 'Sp_heat_cap', 'Therm_cond', 'Heat_of_fusion',
       'Heat_of_vap', 'Electronegativity', 'At_num', 'Valence', 'Ox_state']]

    Y = ['dH(Cd-rich)', 'dH(Mod)', 'dH(X-rich)',
       '(+3/+2)', '(+2/+1)', '(+1/0)', '(0/-1)', '(-1/-2)', '(-2/-3)']

    data_file.write("\n\n\t∆H(Cd-rich):")

    for i in range(0,9):
        data_file.write("\n\n\t"+str(Y[i])+":")

        cd_las_coefs, cd_las_list = lasso_reg(X,df[Y[i]],data_file)
        cd_rr_coefs, cd_rr_list = ridge_reg(X,df[Y[i]],data_file)

        cd_rfr_list = rf_reg(X,df[Y[i]].values.ravel(),data_file)

        print("Min: " + str(np.amin(cd_las_list)) + "\tMax: " + str(np.amax(cd_las_list)))
        print("Min: " + str(np.amin(cd_rr_list)) + "\tMax: " + str(np.amax(cd_rr_list)))
        print("Min: " + str(np.amin(cd_rfr_list)) + "\tMax: " + str(np.amax(cd_rfr_list)))

        cd_list = np.concatenate((cd_las_list, np.array([cd_rr_list]), cd_rfr_list), axis=0)

        plt.figure(figsize=(10,3))
        plt.pcolor(cd_list, cmap=plt.cm.PRGn, vmin=-1, vmax=1)
        plt.colorbar()
        plt.yticks(np.arange(0.5,3),['Lasso', 'Ridge', 'RFR'])
        plt.xticks(np.arange(0.5,45),['Period', 'Group', 'Site', 'Delta Ion. Rad.', 'Delta At. Wt.',
           'Delta Cov. Rad.', 'Delta Ion. En.', 'Delta At. Rad.', 'Delta EA',
           'Delta EN', 'Delta At. Num.', 'Delta Val.', '# Cd Neighbors',
           '# S Neighbors', '# Se Neighbors', '# Te Neighbors',
           '# Se/Te Neighbors', 'Corrected VBM (eV)', 'Corrected CBM (eV)',
           'dH(Cd-rich) UC', 'dH(Mod) UC', 'dH(X-rich) UC',
           'Ionic_radius', 'Boiling_point', 'Melting_point',
           'Density', 'Atomic_weight', 'ICSD_volume', 'Cov_radius',
           'Atomic_radius', 'Electron_affinity', 'Atomic_vol', 'Mendeleev_number',
           'Ionization_pot_1', 'Ionization_pot_2', 'Ionization_pot_3',
           'Therm_expn_coeff', 'Sp_heat_cap', 'Therm_cond', 'Heat_of_fusion',
           'Heat_of_vap', 'Electronegativity', 'At_num', 'Valence', 'Ox_state'], rotation=90)

        plt.ylabel(str(Y[i]))

        plt.tight_layout()

        plt.savefig('featureSelection_all_'+ str(i) +'.pdf')
        plt.show()

        Las_score = []
        Ridge_score = []
        Rfr_score = []


    exit()

    #Get Coefficients
    CdTeCoefs = do_feature_selection('CdTe',data_file)
    CdSeCoefs = do_feature_selection('CdSe',data_file)
    CdSCoefs = do_feature_selection('CdS',data_file)
    CdSeSCoefs = do_feature_selection('CdSe_0.5S_0.5',data_file)
    CdTeSCoefs = do_feature_selection('CdTe_0.5S_0.5',data_file)


    #Close Data File
    data_file.close()

    #Set Up Figure
    fig, ax = plt.subplots(9,5,figsize=(20,10))
    plt.xlim((0,19))

    #Make Subplots
    im = make_subplot('CdTe', CdTeCoefs, 0, ax)
    make_subplot('CdSe', CdSeCoefs, 1, ax)
    make_subplot('CdS', CdSCoefs, 2, ax)
    make_subplot('CdSe_0.5S_0.5', CdSeSCoefs, 3, ax)
    make_subplot('CdTe_0.5S_0.5', CdTeSCoefs, 4, ax)

    #Format Figure, Add Colorbar
    plt.tight_layout()
    fig.subplots_adjust(right=0.95) #6
    cbar_ax = fig.add_axes([0.956, 0.185, 0.01, 0.796])
    bar = fig.colorbar(im, cax=cbar_ax)

    #Save and Display Figure
    plt.savefig('featureSelection_all.pdf')
    plt.show()



def make_subplot(name, coefs, x, ax):
    #Get Coefficient Lists
    cd_list, mod_list, x_list, \
    plus_3_list, plus_2_list, plus_1_list, \
    minus_1_list, minus_2_list, minus_3_list = coefs

    #Plot Data
    im = ax[6,x].pcolor(cd_list, cmap=plt.cm.PRGn, vmin=-1, vmax=1)
    ax[7,x].pcolor(mod_list, cmap=plt.cm.PRGn, vmin=-1, vmax=1)
    ax[8,x].pcolor(x_list, cmap=plt.cm.PRGn, vmin=-1, vmax=1)

    ax[0,x].pcolor(plus_3_list, cmap=plt.cm.PRGn, vmin=-1, vmax=1)
    ax[1,x].pcolor(plus_2_list, cmap=plt.cm.PRGn, vmin=-1, vmax=1)
    ax[2,x].pcolor(plus_1_list, cmap=plt.cm.PRGn, vmin=-1, vmax=1)
    ax[3,x].pcolor(minus_1_list, cmap=plt.cm.PRGn, vmin=-1, vmax=1)
    ax[4,x].pcolor(minus_2_list, cmap=plt.cm.PRGn, vmin=-1, vmax=1)
    ax[5,x].pcolor(minus_3_list, cmap=plt.cm.PRGn, vmin=-1, vmax=1)

    #Format Subplot Labels
    plt.sca(ax[0,x])
    plt.xticks([])
    if(x==0):
        plt.ylabel('(+3/+2)')
        plt.yticks(np.arange(0.5,3),['Lasso', 'Ridge', 'RFE', 'RFR'])
    else: plt.yticks([])

    plt.sca(ax[1,x])
    plt.xticks([])
    if(x==0):
        plt.ylabel('(+2/+1)')
        plt.yticks(np.arange(0.5,3),['Lasso', 'Ridge', 'RFE', 'RFR'])
    else: plt.yticks([])

    plt.sca(ax[2,x])
    plt.xticks([])
    if(x==0):
        plt.ylabel('(+1/0)')
        plt.yticks(np.arange(0.5,3),['Lasso', 'Ridge', 'RFE', 'RFR'])
    else: plt.yticks([])

    plt.sca(ax[3,x])
    plt.xticks([])
    if(x==0):
        plt.ylabel('(0/-1)')
        plt.yticks(np.arange(0.5,3),['Lasso', 'Ridge', 'RFE', 'RFR'])
    else: plt.yticks([])

    plt.sca(ax[4,x])
    plt.xticks([])
    if(x==0):
        plt.ylabel('(-1/-2)')
        plt.yticks(np.arange(0.5,3),['Lasso', 'Ridge', 'RFE', 'RFR'])
    else: plt.yticks([])

    plt.sca(ax[5,x])
    plt.xticks([])
    if(x==0):
        plt.ylabel('(-2/-3)')
        plt.yticks(np.arange(0.5,3),['Lasso', 'Ridge', 'RFE', 'RFR'])
    else: plt.yticks([])

    plt.sca(ax[6,x])
    plt.xticks([])
    if(x==0):
        plt.ylabel('∆H(Cd-rich)')
        plt.yticks(np.arange(0.5,3),['Lasso', 'Ridge', 'RFE', 'RFR'])
    else: plt.yticks([])

    plt.sca(ax[7,x])
    plt.xticks([])
    if(x==0):
        plt.ylabel('∆H(Mod)')
        plt.yticks(np.arange(0.5,3),['Lasso', 'Ridge', 'RFE', 'RFR'])
    else: plt.yticks([])

    plt.sca(ax[8,x])
    if(x==0):
        plt.ylabel('∆H(X-rich)')
        plt.yticks(np.arange(0.5,3),['Lasso', 'Ridge', 'RFE', 'RFR'])
    else: plt.yticks([])
    if (name == 'CdTe' or name == 'CdSe' or name == 'CdS'):
        plt.xticks(np.arange(0.5,19),['Period', 'Group', 'Site', 'Delta Ion. Rad.', 'Delta At. Wt.',
            'Delta Cov. Rad.', 'Delta Ion. En.', 'Delta At. Rad.', 'Delta EA',
            'Delta EN', 'Delta At. Num.', 'Delta Val.', '# Cd Neighbors',
            '# Te Neighbors', 'Corrected VBM (eV)', 'Corrected CBM (eV)',
            '∆H(Cd-rich)', '∆H(Mod)', '∆H('+name[2:]+'-rich)'], rotation=90)
        plt.xlabel(name)
    elif (name == 'CdSe_0.5S_0.5' or name == 'CdTe_0.5S_0.5'):
        plt.xticks(np.arange(0.5,19),['Period', 'Group', 'Site', 'Delta Ion. Rad.', 'Delta At. Wt.',
            'Delta Cov. Rad.', 'Delta Ion. En.', 'Delta At. Rad.', 'Delta EA',
            'Delta EN', 'Delta At. Num.', 'Delta Val.', '# Cd Neighbors',
            '# '+name[2:4]+' Neighbors', 'Corrected VBM (eV)', 'Corrected CBM (eV)',
            '∆H(Cd-rich)', '∆H(Mod)', '∆H(X-rich)'], rotation=90)
        plt.xlabel(name[:4]+'$_{0.5}$S$_{0.5}$')

    #Return Axes Instance
    return im


def do_feature_selection(name, data_file):
    CdS_df = get_data(name)

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
    cd_rfe_list = rfe_selection(X,Y.values.ravel(),data_file)

    cd_rfr_list = rf_reg(X,Y.values.ravel(),data_file)

    data_file.write("\n\n\t∆H(Mod):")
    Y = CdS_df[['∆H(Mod)']]
    mod_las_coefs, mod_las_list = lasso_reg(X,Y,data_file)
    mod_rr_coefs, mod_rr_list = ridge_reg(X,Y,data_file)
    mod_rfe_list = rfe_selection(X,Y.values.ravel(),data_file)

    mod_rfr_list = rf_reg(X,Y.values.ravel(),data_file)

    data_file.write("\n\n\t∆H(X-rich):")
    Y = CdS_df[['∆H(X-rich)']]
    x_las_coefs, x_las_list = lasso_reg(X,Y,data_file)
    x_rr_coefs, x_rr_list = ridge_reg(X,Y,data_file)
    x_rfe_list = rfe_selection(X,Y.values.ravel(),data_file)

    x_rfr_list = rf_reg(X,Y.values.ravel(),data_file)

    data_file.write("\n\n\t(+3/+2):")
    Y = CdS_df[['(+3/+2)']]
    plus_3_las_coefs, plus_3_las_list = lasso_reg(X,Y,data_file)
    plus_3_rr_coefs, plus_3_rr_list = ridge_reg(X,Y,data_file)
    plus_3_rfe_list = rfe_selection(X,Y.values.ravel(),data_file)

    plus_3_rfr_list = rf_reg(X,Y.values.ravel(),data_file)

    data_file.write("\n\n\t(+2/+1):")
    Y = CdS_df[['(+2/+1)']]
    plus_2_las_coefs, plus_2_las_list = lasso_reg(X,Y,data_file)
    plus_2_rr_coefs, plus_2_rr_list = ridge_reg(X,Y,data_file)
    plus_2_rfe_list = rfe_selection(X,Y.values.ravel(),data_file)

    plus_2_rfr_list = rf_reg(X,Y.values.ravel(),data_file)

    data_file.write("\n\n\t(+1/0):")
    Y = CdS_df[['(+1/0)']]
    plus_1_las_coefs, plus_1_las_list = lasso_reg(X,Y,data_file)
    plus_1_rr_coefs, plus_1_rr_list = ridge_reg(X,Y,data_file)
    plus_1_rfe_list = rfe_selection(X,Y.values.ravel(),data_file)

    plus_1_rfr_list = rf_reg(X,Y.values.ravel(),data_file)

    data_file.write("\n\n\t(0/-1):")
    Y = CdS_df[['(0/-1)']]
    minus_1_las_coefs, minus_1_las_list = lasso_reg(X,Y,data_file)
    minus_1_rr_coefs, minus_1_rr_list = ridge_reg(X,Y,data_file)
    minus_1_rfe_list = rfe_selection(X,Y.values.ravel(),data_file)

    minus_1_rfr_list = rf_reg(X,Y.values.ravel(),data_file)

    data_file.write("\n\n\t(-1/-2):")
    Y = CdS_df[['(-1/-2)']]
    minus_2_las_coefs, minus_2_las_list = lasso_reg(X,Y,data_file)
    minus_2_rr_coefs, minus_2_rr_list = ridge_reg(X,Y,data_file)
    minus_2_rfe_list = rfe_selection(X,Y.values.ravel(),data_file)

    minus_2_rfr_list = rf_reg(X,Y.values.ravel(),data_file)

    data_file.write("\n\n\t(-2/-3):")
    Y = CdS_df[['(-2/-3)']]
    minus_3_las_coefs, minus_3_las_list = lasso_reg(X,Y,data_file)
    minus_3_rr_coefs, minus_3_rr_list = ridge_reg(X,Y,data_file)
    minus_3_rfe_list = rfe_selection(X,Y.values.ravel(),data_file)

    minus_3_rfr_list = rf_reg(X,Y.values.ravel(),data_file)


    data_file.write("\n\n")

    cd_list = np.concatenate((cd_las_list, cd_rr_list, cd_rfe_list, cd_rfr_list), axis=0)
    mod_list = np.concatenate((mod_las_list, mod_rr_list, mod_rfe_list, mod_rfr_list), axis=0)
    x_list = np.concatenate((x_las_list, x_rr_list, x_rfe_list, x_rfr_list), axis=0)

    plus_3_list = np.concatenate((plus_3_las_list, plus_3_rr_list, plus_3_rfe_list, plus_3_rfr_list), axis=0)
    plus_2_list = np.concatenate((plus_2_las_list, plus_2_rr_list, plus_2_rfe_list, plus_2_rfr_list), axis=0)
    plus_1_list = np.concatenate((plus_1_las_list, plus_1_rr_list, plus_1_rfe_list, plus_1_rfr_list), axis=0)

    minus_1_list = np.concatenate((minus_1_las_list, minus_1_rr_list, minus_1_rfe_list, minus_1_rfr_list), axis=0)
    minus_2_list = np.concatenate((minus_2_las_list, minus_2_rr_list, minus_2_rfe_list, minus_2_rfr_list), axis=0)
    minus_3_list = np.concatenate((minus_3_las_list, minus_3_rr_list, minus_3_rfe_list, minus_3_rfr_list), axis=0)

    return cd_list, mod_list, x_list, \
    plus_3_list, plus_2_list, plus_1_list, \
    minus_1_list, minus_2_list, minus_3_list

def rfe_selection(X,Y,data_file):
    nof_list=np.arange(1,19)
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

    return np.array([coefs])


def ridge_reg(X,Y,data_file):
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

    rr001 = Ridge(alpha=0.01)
    rr001.fit(X_train, y_train)
    Ridge_train_score001 = rr001.score(X_train,y_train)
    Ridge_test_score001 = rr001.score(X_test, y_test)
    if(Ridge_test_score001 > high_score):
        high_score = Ridge_test_score001
        alpha_ = 0.01
        coefs = rr001.coef_

    rr01 = Ridge(alpha=0.1)
    rr01.fit(X_train, y_train)
    Ridge_train_score01 = rr01.score(X_train,y_train)
    Ridge_test_score01 = rr01.score(X_test, y_test)
    if(Ridge_test_score01 > high_score):
        high_score = Ridge_test_score01
        alpha_ = 0.1
        coefs = rr01.coef_

    rr10 = Ridge(alpha=10)
    rr10.fit(X_train, y_train)
    Ridge_train_score10 = rr10.score(X_train,y_train)
    Ridge_test_score10 = rr10.score(X_test, y_test)
    if(Ridge_test_score10 > high_score):
        high_score = Ridge_test_score10
        alpha_ = 10
        coefs = rr10.coef_

    rr100 = Ridge(alpha=100)
    rr100.fit(X_train, y_train)
    Ridge_train_score100 = rr100.score(X_train,y_train)
    Ridge_test_score100 = rr100.score(X_test, y_test)
    if(Ridge_test_score100 > high_score):
        high_score = Ridge_test_score100
        alpha_ = 100
        coefs = rr100.coef_

    rr1000 = Ridge(alpha=1000)
    rr1000.fit(X_train, y_train)
    Ridge_train_score1000 = rr1000.score(X_train,y_train)
    Ridge_test_score1000 = rr1000.score(X_test, y_test)
    if(Ridge_test_score1000 > high_score):
        high_score = Ridge_test_score1000
        alpha_ = 1000
        coefs = rr1000.coef_

    data_file.write('\n\t\tRidge Regression Score with alpha=%f: \t%f' % (alpha_, high_score))

    return np.concatenate((rr001.coef_, rr10.coef_, rr100.coef_, rr1000.coef_), axis=0), np.array(coefs)

def lasso_reg(X,Y,data_file):
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

    lasso01 = Lasso(alpha=0.1, max_iter=10e5)
    lasso01.fit(X_train,y_train)
    train_score01=lasso01.score(X_train,y_train)
    test_score01=lasso01.score(X_test,y_test)
    coeff_used01 = np.sum(lasso01.coef_!=0)
    if(test_score01 > high_score):
        high_score = test_score01
        alpha_ = 0.1
        coefs = lasso01.coef_

    lasso001 = Lasso(alpha=0.01, max_iter=10e5)
    lasso001.fit(X_train,y_train)
    train_score001=lasso001.score(X_train,y_train)
    test_score001=lasso001.score(X_test,y_test)
    coeff_used001 = np.sum(lasso001.coef_!=0)
    if(test_score001 > high_score):
        high_score = test_score001
        alpha_ = 0.01
        coefs = lasso001.coef_

    lasso0001 = Lasso(alpha=0.001, max_iter=10e5)
    lasso0001.fit(X_train,y_train)
    train_score0001=lasso0001.score(X_train,y_train)
    test_score0001=lasso0001.score(X_test,y_test)
    coeff_used0001 = np.sum(lasso0001.coef_!=0)
    if(test_score0001 > high_score):
        high_score = test_score0001
        alpha_ = 0.001
        coefs = lasso0001.coef_

    data_file.write('\n\t\tLasso Regression Score with alpha=%f: \t%f' % (alpha_, high_score))

    return np.concatenate((np.array([lasso.coef_]), np.array([lasso01.coef_]),
                           np.array([lasso001.coef_]), np.array([lasso0001.coef_])), axis=0), np.array([coefs])


def rf_reg(X,Y,data_file):
    X_train,X_test,y_train,y_test=train_test_split(X,Y, test_size=0.3, random_state=31)

    high_score = 0
    n_est = 0
    #coefs = np.zeros(19)

    rf = RandomForestRegressor(n_estimators = 10, random_state = 31)
    rf.fit(X_train, y_train)
    train_score=rf.score(X_train,y_train)
    test_score=rf.score(X_test,y_test)
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

    rf = RandomForestRegressor(n_estimators = 1000, random_state = 31)
    rf.fit(X_train, y_train)
    train_score=rf.score(X_train,y_train)
    test_score=rf.score(X_test,y_test)
    if(test_score > high_score):
        high_score = test_score
        n_est = 1000
        coefs = rf.feature_importances_

    data_file.write('\n\t\tRandom Forest Regression n=%d: \t\t%f' % (n_est, test_score))

    return np.array([coefs])

def get_data(name):
    df = pd.read_csv('./data/' + name + '.csv')

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

if __name__ == '__main__':
    main()
