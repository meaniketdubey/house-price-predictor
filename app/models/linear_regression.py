import numpy as np 


class LinearRegression():
    def __init__(self, learning_rate = 0.01, iterations = 100):
        
        self.learning_rate = learning_rate
        self.iterations = iterations

        self.weights = None
        self.bias = None

        self.cost_history = []
    
    def initialize_parameters(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0

    def predict(self, x):

        predictions = np.dot(x ,self.weights) + self.bias

        return predictions
    
    def compute_cost(self, y_true, y_pred):
        n_sample = len(y_true)

        cost = (1/n_sample) * np.sum((y_pred - y_true) ** 2)
    
    def fit(self, X,y):
        n_samples, n_features = X.shape

        self.initialize_parameters(n_features)

        for i in range(self.iterations):

            y_pred = self.predict(X)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

            cost = self.compute_cost(y, y_pred)

            self.cost_history.append(cost)

            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost}")
