
# Input "layer"
class Layer_Input:
    # Forward pass
    def __init__(self):
        self.output = None

    def forward(self, inputs, training):
        self.output = inputs
