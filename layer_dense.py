import numpy as np


class Layer_Dense:
    # We’ll initialize the biases with the shape of (1, n_neurons), as a row vector,
    # which will let us easily add it to the result of the dot product later,
    # without additional operations like transposition.

    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self._weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self._biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self._weights) + self._biases
