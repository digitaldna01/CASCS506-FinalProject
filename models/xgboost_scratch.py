import numpy as np

class XGBoostFromScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, lambda_=1, min_child_weight=1, gamma=0):
        self.n_estimators = n_estimators  # Number of trees
        self.learning_rate = learning_rate  # Learning rate
        self.max_depth = max_depth  # Maximum depth of each tree
        self.lambda_ = lambda_  # Regularization term
        self.min_child_weight = min_child_weight  # Minimum child weight
        self.gamma = gamma  # Minimum gain required to split
        self.trees = []  # Store all trees

    def fit(self, X, y):
        """
        Train the XGBoost model.
        X: Feature matrix
        y: Target vector
        """
        # Initialize predictions with zero
        y_pred = np.zeros_like(y, dtype=float)

        for i in range(self.n_estimators):
            # print("fitting " + i +"th iteration")
            # Compute gradients and hessians
            gradient = self._compute_gradient(y, y_pred)
            hessian = self._compute_hessian(y, y_pred)
            
            # Build a single tree
            tree = self._build_tree(X, gradient, hessian, depth=0)
            self.trees.append(tree)
            
            # Update predictions
            y_pred += self.learning_rate * self._predict_tree(tree, X)
            
    def predict(self, X, classification_threshold=0.5):
        """
        Predict using the trained XGBoost model.
        X: Feature matrix
        classification_threshold: Threshold for binary classification (default=0.5)
        """
        y_pred = np.zeros((X.shape[0],), dtype=float)
        for tree in self.trees:
            y_pred += self.learning_rate * self._predict_tree(tree, X)

        # 분류 문제의 경우 예측값을 이진값으로 변환
        if classification_threshold is not None:
            return (y_pred > classification_threshold).astype(int)
        return y_pred


    def _compute_gradient(self, y, y_pred):
        # Example: Gradient for MSE loss
        return y_pred - y

    def _compute_hessian(self, y, y_pred):
        # Example: Hessian for MSE loss
        return np.ones_like(y)
    
    def _build_tree(self, X, gradient, hessian, depth):
        """
        Recursively build a tree.
        """
        if depth == self.max_depth:
            return self._compute_leaf_value(gradient, hessian)
        
        # Find the best split
        split_feature, split_value = self._find_best_split(X, gradient, hessian)
        
        if split_feature is None:
            return self._compute_leaf_value(gradient, hessian)
        
        # Split the data
        left_idx = X[:, split_feature] <= split_value
        right_idx = X[:, split_feature] > split_value

        return {
            'split_feature': split_feature,
            'split_value': split_value,
            'left': self._build_tree(X[left_idx], gradient[left_idx], hessian[left_idx], depth + 1),
            'right': self._build_tree(X[right_idx], gradient[right_idx], hessian[right_idx], depth + 1),
        }

    def _find_best_split(self, X, gradient, hessian):
        """
        Find the best feature and threshold to split on.
        """
        best_gain = -float('inf')
        best_split = None
        
        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_idx = X[:, feature_idx] <= threshold
                right_idx = X[:, feature_idx] > threshold
                
                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue
                
                # Calculate gain
                gain = self._calculate_gain(gradient, hessian, left_idx, right_idx)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_idx, threshold)
        
        return best_split if best_split else (None, None)

    def _calculate_gain(self, gradient, hessian, left_idx, right_idx):
        """
        Calculate the gain of a split, incorporating min_child_weight and gamma.
        """
        # Sum gradients and hessians
        G_left = gradient[left_idx].sum()
        G_right = gradient[right_idx].sum()
        H_left = hessian[left_idx].sum()
        H_right = hessian[right_idx].sum()

        # Check if child nodes satisfy min_child_weight
        if H_left < self.min_child_weight or H_right < self.min_child_weight:
            return -float('inf')  # Invalid split

        # Calculate gain
        gain = 0.5 * (
            (G_left**2 / (H_left + self.lambda_)) +
            (G_right**2 / (H_right + self.lambda_))
        ) - (gradient.sum()**2 / (hessian.sum() + self.lambda_))

        # Apply gamma constraint
        return gain if gain > self.gamma else -float('inf')
    
    def _compute_leaf_value(self, gradient, hessian):
        """
        Compute the value of a leaf.
        """
        return -gradient.sum() / (hessian.sum() + self.lambda_)

    def _predict_tree(self, tree, X):
        """
        Predict using a single tree.
        """
        if isinstance(tree, dict):
            split_feature = tree['split_feature']
            split_value = tree['split_value']
            left_idx = X[:, split_feature] <= split_value
            right_idx = X[:, split_feature] > split_value
            
            y_pred = np.zeros(X.shape[0])
            y_pred[left_idx] = self._predict_tree(tree['left'], X[left_idx])
            y_pred[right_idx] = self._predict_tree(tree['right'], X[right_idx])
            return y_pred
        else:
            return np.full(X.shape[0], tree)


# Example 
if __name__ == "__main__":
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1.2, 2.3, 3.1, 3.9, 5.1])

    # Model reset and train
    model = XGBoostFromScratch(n_estimators=10, learning_rate=0.1, max_depth=2)
    model.fit(X, y)

    # Prediction
    predictions = model.predict(X)
    print("Predictions:", predictions)