from sklearn.neural_network import MLPRegressor

def apply_MLPRegressor(train_x, train_y, test_x):
    # apply Linear Regression:
    mlp = MLPRegressor(
    hidden_layer_sizes=(100, ), 
    activation='relu', 
    solver='lbfgs', 
    alpha=0.0001, 
    batch_size='auto', 
    learning_rate='constant', 
    learning_rate_init=0.001, 
    power_t=0.5, 
    max_iter=1000, 
    shuffle=True, 
    random_state=None, 
    tol=0.0001, 
    verbose=False, 
    warm_start=False, 
    momentum=0.9, 
    nesterovs_momentum=True, 
    early_stopping=False, 
    validation_fraction=0.1, 
    beta_1=0.9, 
    beta_2=0.999, 
    epsilon=1e-08, 
    n_iter_no_change=10)
    mlp.fit(train_x, train_y)

    # predict the results:
    y_prediction = mlp.predict(test_x)

    # return predictions:
    return y_prediction
