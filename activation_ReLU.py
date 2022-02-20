import numpy as np
 
class Activation_ReLU:
    # Forward pass
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        # Calculate output values from input
        self.output = np.maximum(0, inputs)
