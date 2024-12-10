import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, C=1.0, n_iters=1000):
        self.lr = learning_rate
        self.C = C  # C parameters
        self.n_iters = n_iters
        self.w = None  # weights Initiate Weights to zero
        self.b = 0     # bias Initiate bias to 0

    def fit(self, X, y):
        n_samples, n_features = X.shape # Get the shape of Train
        self.w = np.zeros(n_features) # Initiate the weights to 0

        # transfork: check y is {1, -1} shape
        y_ = np.where(y <= 0, -1, 1) # Based on Y value, if it is less than 0 change it to -1, so there are only two y values, -1 and 1 values.
        
        # Gradient Descent optimization
        # prev_loss = float('inf')
        for _ in range(self.n_iters): # Run each iterations. 
            for idx, x_i in enumerate(X): # Iterate each rows of Train Data
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    # Loss function Gradient (힌지 손실 X)
                    self.w -= self.lr * (2 * self.w)
                else:
                    # 힌지 손실 포함
                    self.w -= self.lr * (2 * self.w - self.C *  np.dot(x_i, y_[idx]))
                    self.b -= self.lr * self.C * y_[idx]
        
        

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)  # class {1, -1} return
    
    def _compute_loss(self, X, y_):
        hinge_loss = np.maximum(0, 1 - y_ * (np.dot(X, self.w) + self.b))
        return 0.5 * np.dot(self.w, self.w) + self.C * np.sum(hinge_loss)

# Data Excample
if __name__ == "__main__":
    # generate data (XOR 문제)
    X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
    y = np.array([1, 1, 1, -1, -1])  # Class Label

    # Train SVM model
    model = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    model.fit(X, y)

    # Prediction
    predictions = model.predict(X)
    print("Predictions:", predictions)
