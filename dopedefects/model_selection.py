import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import data_preprocess


def rmse_train_test(model, X_train, X_test, y_train, y_test, scalarX, scalarY):
    """Reports RMSE of training and testing data given trained model"""

    y_pred1 = model.predict(X_train)
    y_pred2 = model.predict(X_test)

    X_train, y_train = data_preprocess.data_unscale(X_train, y_train, scalarX, scalarY)
    X_test, y_test = data_preprocess.data_unscale(X_test, y_test, scalarX, scalarY)
    y_pred1 = scalarY.inverse_transform(y_pred1.reshape(len(y_pred1),1))
    y_pred2 = scalarY.inverse_transform(y_pred2.reshape(len(y_pred2),1))

    rmse = [np.sqrt(mean_squared_error(y_train, y_pred1)),
            np.sqrt(mean_squared_error(y_test, y_pred2))]

    return rmse


def model_select(X, Y):
    """Returns RMSE of test and training data for different regressors"""

    rmse_mat = np.zeros((9,3,2))
    for i in range(9):
        print(i)
        X_train, y_train, X_test, y_test, scalarX, scalarY = \
                    data_preprocess.split_and_scale(X,Y[:,i])


        reg = RandomForestRegressor(max_depth=15, random_state=0,
                                    n_estimators=100)
        reg.fit(X_train, y_train.reshape(len(y_train)))
        rmse = rmse_train_test(reg, X_train, X_test, y_train, y_test,
                               scalarX, scalarY)
        rmse_mat[i,0] = rmse
        print("Random Forest", rmse)


        reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                                  n_estimators=1000, random_state=100)
        reg.fit(X_train, y_train.reshape(len(y_train)))
        rmse = rmse_train_test(reg, X_train, X_test, y_train, y_test,
                               scalarX, scalarY)
        rmse_mat[i,1] = rmse
        print("AdaBoost", rmse)

#         nn = nn_model(X_train, y_train)
#         rmse = rmse_train_test(nn, X_train, X_test, y_train, y_test,
#                               scalarX, scalarY)
#         rmse_mat[i,2] = rmse
#         print("Neural Net", rmse)

    return rmse_mat
