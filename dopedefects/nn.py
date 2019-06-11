from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

import numpy as np

def nn_model(X_train, y_train, num_nodes1, num_nodes2, do_rate, lr):
    """Build NN with 1 hidden layer"""

    seed = 7
    np.random.seed(seed)

    model = Sequential()

    # Hidden layers
    model.add(Dense(num_nodes1, input_dim=X_train.shape[1],
                    activation='relu'))
    model.add(Dropout(do_rate, seed=seed))
    model.add(Dense(num_nodes2, activation='relu'))
    model.add(Dropout(do_rate, seed=seed))

    # Output Layer
    model.add(Dense(1))

    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=['mae'])

    return model


def tune_parameters(X, y):
    """Tune hyperparameters to optimize neural net using gridsearch.
    Output runs is a list of trained neural networks with paramters:
    epoch, batch, num_nodes1, num_nodes2, do_rate, lr, rmse, where rmse
    is a list of training error and validation error"""

    X_train, y_train, X_test, y_test, scalarX, scalarY = data_preprocess(X, y)

    # RF to compare to
    reg = RandomForestRegressor(max_depth=15, random_state=0, n_estimators=100)
    reg.fit(X_train, y_train.reshape(len(y_train)))
    rmse = rmse_train_test(reg, X_train, X_test, y_train, y_test,
                           scalarX, scalarY)
    print("RF", rmse)


    # validation set for hyperparamter tuning
    X_train_nn, X_val, y_train_nn, y_val = train_test_split(X_train, y_train,
                                                        test_size=0.2,
                                                        random_state=110)

    # hyperparameters to tune
    epoch_arr = [250, 500, 750]
    batch_size_arr = [50, 100, 400]
    num_nodes1_arr = [8, 16, 32]
    num_nodes2_arr = [8, 16, 32]
    do_rate_arr = [0.05, 0.1, 0.25]
    lr_arr = [0.001, 0.01, 0.1]

    runs = []
    count = 0
    for epoch in epoch_arr:
        for batch in batch_size_arr:
            for num_nodes1 in num_nodes1_arr:
                for num_nodes2 in num_nodes2_arr:
                    for do_rate in do_rate_arr:
                        for lr in lr_arr:
                            model = nn_model(X_train_nn, y_train_nn, num_nodes1,
                                            num_nodes2, do_rate, lr)
                            model.fit(X_train_nn, y_train_nn, shuffle=True,
                                    epochs=epoch, batch_size=batch, verbose=0)
                            rmse = rmse_train_test(model, X_train_nn, X_val,
                                            y_train_nn, y_val, scalarX, scalarY)
                            this_run = (epoch, batch, num_nodes1, num_nodes2,
                                        do_rate, lr, rmse)
                            runs.append(this_run)
                            print(this_run)
                            count = count + 1

    return runs
