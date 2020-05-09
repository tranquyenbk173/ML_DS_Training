from utils import load_data, normalize_and_add_bias, train_test_split
import ridge_regression


if __name__ == '__main__':
    #load data and normalize
    X, y = load_data('death_rate_data.txt')
    normalize_and_add_bias(X)

    #split_train, test sets
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    #Build the model:
    model = ridge_regression.RidgeReg()

    best_lambda = model.get_best_lambdaa(X_train, y_train) #find hyper-param
    print("Best lambda: ", best_lambda)

    w = model.fit(X_train, y_train, best_lambda) #find model params

    #Predict:
    y_pred = model.predict(X_test, w)
    loss = model.compute_RSS(y_test, y_pred)
    print("RSS = ", loss)

