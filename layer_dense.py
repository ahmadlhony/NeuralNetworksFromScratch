import numpy as np


class Layer_Dense:
    # We’ll initialize the biases with the shape of (1, n_neurons), as a row vector,
    # which will let us easily add it to the result of the dot product later,
    # without additional operations like transposition.

    def __init__(self, n_inputs, n_neurons):
        self.dinputs = None
        self.dbiases = None
        self.dweights = None
        self.inputs = None
        self.output = None
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    # Backward pass
    def backward(self, dvalues):

        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
