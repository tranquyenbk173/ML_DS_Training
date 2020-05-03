import numpy as np

class RidgeReg:
    """
    this class is used for building Ridge regression model
    """
    def __init__(self):
        return

    def fit(self, X_train, y_train, lambdaa):
        """
        matrix multiplication - when data "small"
        :param X_train: features matrix
        :param y_train: values vector
        :param lambdaa: hyper-parameter
        :return: w
            w is a vector - model parameter
        """
        I = np.eye(X_train.shape[1])
        #print(type(I))
        w = np.linalg.inv(X_train.transpose().dot(X_train) + lambdaa * np.eye(X_train.shape[1])).dot(X_train.transpose()).dot(y_train)
        return w

    def fitSGD(self, X_train, y_train, lambdaa, learning_rate = 0.01, \
               epochs = 20, batch_size = 30):
        """
        find model param with GD
        :param X_train: features matrix
        :param y_train: values vectors
        :param lambdaa: hyper-parameter
        :param learning_rate: ##
        :param epochs: num of epochs when training
        :param batch_size: size of batchs in a training time
        :return: w
            w is model-parameter
        """
        w = np.random.rand(X_train.shape[1])
        last_loss = 10e8

        for i in range(epochs):

            #shuffle data when start an epoch:
            ids = range(0, X_train.shape[0])
            ids = np.random.shuffle(ids)
            X_train = X_train[ids]
            y_train = y_train[ids]

            #divide into batchs and train
            num_of_minibatchs = X_train.shape[0]//batch_size
            for ib in range(num_of_minibatchs):
                index = i*batch_size
                X_train_sub = X_train[index:index + batch_size]
                y_train_sub = y_train[index:index + batch_size]
                gradLoss = X_train_sub.T.dot(X_train_sub.dot(w) - y_train_sub) +\
                    lambdaa*w

                #update w:
                w = w - learning_rate*gradLoss

            #condition for stopping: w change -- not more than 10^-5
            new_loss = self.compute_RSS(self.predict(X_train, w), y_train)
            if np.abs(new_loss - last_loss) <= 1e-5:
                break
            last_loss = new_loss
        return w

    def predict(self, X_test, w):
        """
        predict value from features given
        :param X_test:features
        :param w: model parameter
        :return: predicted value/values
        """
        y_pred = X_test.dot(w)
        return y_pred

    def compute_RSS(self, y_test, y_pred):
        """
        compute loss function
        :param y_test: real_values
        :param y_pred: predicted_values
        :return: value of loss function RSS
        """
        loss = (1.0/y_test.shape[0])*np.linalg.norm(y_test - y_pred)**2
        return loss

    def get_best_lambdaa(self, X_train, y_train):
        """
        (use cross-validation in order to find best lambda)
        + Find best lambda in lambda_values
        + Find value around lambda that is the best lambda
        :param X_train: data_train X (features)
        :param y_train: data_train y (values)
        :return: best_lambda
        """
        def cross_validation(num_folds, initial_lambda):
            """
            Implement k-fold algorithm with train_data
            :param num_folds: k_fold
            :param initial_lambda:
            :return: average of RSS
            """

            lambdaa = initial_lambda
            print("L = ", lambdaa)

            #divide training set into num_folds parts follow indexes:
            row_ids = np.array(range(X_train.shape[0]))
            valid_ids = np.split(row_ids[:len(row_ids) - len(row_ids)%num_folds], num_folds)
            valid_ids[-1] = np.append(valid_ids[-1],
                                      row_ids[len(row_ids) - len(row_ids) % num_folds:])

            train_ids = [[k for k in row_ids if k not in valid_ids[i]] for i in range(num_folds)]
            avg_RSS = 0

            for i in range(num_folds):
                #with each i, we have corresponding train-val sets:
                #(k-1) parts for train and the rest for val
                valid_X_part, valid_y_part = X_train[valid_ids[i]], y_train[valid_ids[i]]
                train_X_part, train_y_part = X_train[train_ids[i]], y_train[train_ids[i]]

                #fit and compute corresponding RSS:
                w = self.fit(train_X_part, train_y_part, lambdaa)
                y_pred = self.predict(valid_X_part, w)
                avg_RSS +=self.compute_RSS(valid_y_part, y_pred)
                print('ok')

            return avg_RSS/num_folds


        def range_scan(best_lambda, minRSS, lambda_values):
            """
            Use lambda_values given, find the best lambda from lambda_values
            :param best_lambda:
            :param minRSS:
            :param lambda_values:
            :return: best_lambda
            """
            print("Type", type(lambda_values))
            for l in lambda_values:
                lam = l
                print(type(l), l)
                loss = cross_validation(num_folds=5, initial_lambda=lam)
                if loss < minRSS:
                    minRSS = loss
                    best_lambda = lam

            return best_lambda


        #print("heelo!")
        #fisrt time - Find best lambda in lambda_values
        best_lambda, minRSS = range_scan(best_lambda = 0, minRSS = 1000 ** 2, lambda_values= range(50))

        #second time - Find value around recent-lambda that is the best lambda
        temp_range = range(max(0, best_lambda-1)*1000, (best_lambda+1)*1000, 1)
        lambda_values = [i*1./1000 for i in temp_range]
        best_lambda, minRSS = range_scan(best_lambda, minRSS, lambda_values)

        return best_lambda
