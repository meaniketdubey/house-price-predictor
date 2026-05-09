import numpy as np 


class LinearRegression():
    def __init__(self, learning_rate = 0.01, iterations = 100):
        
        self.learning_rate = learning_rate
        self.iterations = iterations

        self.weights = None
        self.bias = None

        self.mean = None
        self.std = None

        self.cost_history = []

    def normalize_features(self, X):

        if self.mean is None:
            self.mean = np.mean(X, axis=0)

        if self.std is None:
            self.std = np.std(X, axis=0)

        X_normalized = (X - self.mean) / self.std

        return X_normalized
    
    def initialize_parameters(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0

    def predict(self, x):


        x = (x - self.mean) / self.std
        predictions = np.dot(x ,self.weights) + self.bias
        

        return predictions
    
    def compute_cost(self, y_true, y_pred):
        n_sample = len(y_true)

        cost = (1/n_sample) * np.sum((y_pred - y_true) ** 2)

        return cost
    
    def fit(self, X,y):
        n_samples, n_features = X.shape

        X = self.normalize_features(X)

        self.initialize_parameters(n_features)

        for i in range(self.iterations):

            y_pred = self.predict(X)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

            cost = self.compute_cost(y, y_pred)

            self.cost_history.append(cost)

            if i % 1000 == 0:
                print(f"Iteration {i}, Cost: {cost}")
