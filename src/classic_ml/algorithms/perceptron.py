# src/classic_ml/algorithms/perceptron.py

import random

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = []
        self.bias = 0.0
        self.converged_epoch = None

    def predict(self, x):
        activation = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return activation, (1 if activation >= 0 else -1)

    def fit(self, X, y_bin, init_w=None, init_b=None):
        """
        y_bin should be a list of 1 and -1.
        init_w: optional initial weights list.
        init_b: optional initial bias.
        """
        if not X or not y_bin:
            return
        
        n_features = len(X[0])
        self.weights = init_w[:] if init_w else [0.0] * n_features
        self.bias = init_b if init_b is not None else 0.0
        self.converged_epoch = None
        
        for epoch in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, y_bin):
                _, pred = self.predict(xi)
                if pred != yi:
                    update = self.lr * yi
                    self.weights = [w + update * xij for w, xij in zip(self.weights, xi)]
                    self.bias += update
                    errors += 1
            
            if errors == 0:
                self.converged_epoch = epoch + 1
                break
                
        return self.weights, self.bias, self.converged_epoch

    def evaluate(self, X, y_bin):
        if not X:
            return 0.0
        hits = sum(self.predict(xi)[1] == yi for xi, yi in zip(X, y_bin))
        return 100.0 * hits / len(X)

    @staticmethod
    def train_ovr(X, y_all, lr=0.1, epochs=100):
        """
        One-vs-Rest multiclass training.
        Returns a dictionary of class: (model_weights, model_bias)
        """
        classes = sorted(list(set(y_all)))
        classifiers = {}
        for cls in classes:
            # Create a binary target for this class
            y_bin = [1 if y == cls else -1 for y in y_all]
            model = Perceptron(learning_rate=lr, epochs=epochs)
            w, b, _ = model.fit(X, y_bin)
            classifiers[cls] = (w, b)
        return classifiers

    @staticmethod
    def predict_ovr(classifiers, x):
        """
        Predict using a set of OvR classifiers.
        Returns the class with the highest activation.
        """
        activations = {}
        for cls, (w, b) in classifiers.items():
            activation = sum(wi * xi for wi, xi in zip(w, x)) + b
            activations[cls] = activation
        return max(activations, key=activations.get)

    @staticmethod
    def grid_search(X, y_bin, bias, w0_range, w1_range):
        """
        Search for weights in 2D that correctly classify the data.
        w0_range/w1_range: (start, stop, step)
        """
        def frange(start, stop, step):
            curr = start
            while curr <= stop + 1e-9:
                yield curr
                curr += step

        solutions = []
        for w0 in frange(*w0_range):
            for w1 in frange(*w1_range):
                # We assume 2D features
                weights = [w0, w1]
                hits = sum((sum(w * xi for w, xi in zip(weights, x)) + bias >= 0) == (y == 1) 
                          for x, y in zip(X, y_bin))
                if hits == len(X):
                    solutions.append((w0, w1))
        return solutions
