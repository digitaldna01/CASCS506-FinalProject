import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param  # 규제 parameters
        self.n_iters = n_iters
        self.w = None  # weights Initiate Weights to zero
        self.b = 0     # bias Initiate bias to 0

    def fit(self, X, y):
        n_samples, n_features = X.shape # Get the shape of Train
        self.w = np.zeros(n_features) # Initiate the weights to 0

        # transfork: check y is {1, -1} shape
        y_ = np.where(y <= 0, -1, 1) # Based on Y value, if it is less than 0 change it to -1, so there are only two y values, -1 and 1 values.

        # Gradient Descent optimization
        for _ in range(self.n_iters): # Run each iterations. 
            for idx, x_i in enumerate(X): # Iterate each rows of Train Data
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    # Loss function Gradient (힌지 손실 X)
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # 힌지 손실 포함
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)  # class {1, -1} return

# Data Excample
if __name__ == "__main__":
    # generate data (XOR 문제)
    X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
    y = np.array([1, 1, 1, -1, -1])  # 클래스 라벨

    # Train SVM model
    model = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    model.fit(X, y)

    # Prediction
    predictions = model.predict(X)
    print("Predictions:", predictions)
