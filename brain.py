import numpy as np

class Brain:
    def __init__(self, feature_size, action_size):
        self.weights = np.random.zeros(feature_size, action_size)
        self.bias = np.random.zeros(action_size)

    def predict(self, features):
        return np.dot(features, self.weights) + self.bias
    
    def loss(self, target, features):
        target = np.array(target)
        predictions = self.predict(features)
        return np.mean((predictions - target) ** 2)
    
    def gradient(self, target, features):
        N = features.shape[0]
        target = np.array(target).reshape(-1, self.bias.size)

        predictions = self.predict(features)
        error = predictions - target
        gradient = (2 / N ) * np.dot(features.T, error)
        bias_gradient = (2 / N) * np.sum(error, axis=0)

        return gradient, bias_gradient
    
    def update(self, gradient, bias_gradient, learning_rate):
        self.weights -= learning_rate * gradient
        self.bias -= learning_rate * bias_gradient

    def train(self, features, target, learning_rate, epochs):
        for epoch in range(epochs):
            grad, bias_grad = self.gradient(target, features)
            self.update(grad, bias_grad, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {self.loss(target, features)}")