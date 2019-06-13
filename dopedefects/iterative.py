from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np


def iter_model(X, y, test_split, choose_opt):
    """Iteratively trains model by adding points based on choose_opt specified
    choose_opt can be:
    'GPR_std' - Gaussian process regression, maximize uncertainty
    'GPR_rand' - Gaussian process regression, next point chosen at random
    'RF_rand' - Random forest model, next point chosen at random"""

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=test_split,
                                                    random_state=110)

    y_r2 = []
    y_rmse = []

    for n in range(len(y_test)-2):

        if choose_opt == "GPR_std":
            kernel = RBF()
            gpr = GaussianProcessRegressor(kernel=kernel,
                                    random_state=100).fit(X_train, y_train)
            y_mean, y_std = gpr.predict(X_test, return_std=True)
            opt = y_std                # optimizing function, the standard dev
            ind = np.argsort(opt)[-1]  # select optimum test point
            model = gpr

        elif choose_opt == "GPR_rand":
            kernel = RBF()
            gpr = GaussianProcessRegressor(kernel=kernel,
                                    random_state=100).fit(X_train, y_train)
            y_mean, y_std = gpr.predict(X_test, return_std=True)
            np.random.seed(10)
            ind = np.random.randint(0, len(y_test))
            model = gpr

        elif choose_opt == "RF_rand":
            reg = RandomForestRegressor(max_depth=15,
                                    random_state=0, n_estimators=100)
            # reg = LinearRegression().fit(X_train, y_train)
            reg.fit(X_train, y_train.reshape(len(y_train)))
            np.random.seed(10)
            ind = np.random.randint(0, len(y_test))
            model = reg

        else:
            assert "Not a valid model choice"

        # add test point to training set, delete from testing set
        y_train = np.append(y_train, y_test[ind])
        y_test = np.delete(y_test, (ind))
        X_train = np.append(X_train, [X_test[ind,:]], axis=0)
        X_test = np.delete(X_test, (ind), axis=0)

        y_r2 = np.append(y_r2, model.score(X_test, y_test))

        y_pred = model.predict(X_test)
        y_rmse = np.append(y_rmse, np.sqrt(mean_squared_error(y_test, y_pred)))

    return y_r2, y_rmse
