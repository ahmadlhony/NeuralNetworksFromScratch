import numpy as np
import layer_dense as ld
import nnfs
from nnfs.datasets import spiral_data

from activation_ReLU import Activation_ReLU
from activation_softmax import Activation_Softmax

nnfs.init()


def firstNeuron():
    inputs = [1.2, 5.1, 2.1]
    weights = [3.1, 2.1, 8.7]
    bias = 3

    output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + bias
    print(output)


def firstNeuronLayer():
    inputs = [1, 2, 3, 2.5]
    weights = [[0.2, 0.8, -0.5, 1],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]]
    biases = [2, 3, 0.5]
    # Output of current layer
    layer_outputs = []
    # For each neuron
    for neuron_weights, neuron_bias in zip(weights, biases):
        # Zeroed output of given neuron
        neuron_output = 0
        # For each input and weight to the neuron
        for n_input, weight in zip(inputs, neuron_weights):
            # Multiply this input by associated weight
            # and add to the neuron’s output variable
            neuron_output += n_input * weight
        # Add bias
        neuron_output += neuron_bias
        # Put neuron’s result to the layer’s output list
        layer_outputs.append(neuron_output)
    print(layer_outputs)


def dotProductFirstExample():
    a = [1, 2, 3]
    b = [2, 3, 4]
    dot_product = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    print(dot_product)


def singleNeuronUsingNumPy():
    inputs = [1.0, 2.0, 3.0, 2.5]
    weights = [0.2, 0.8, -0.5, 1.0]
    bias = 2.0
    outputs = np.dot(weights, inputs) + bias
    print(outputs)


def layerOfNeuronUsingNumPy():
    inputs = [1.0, 2.0, 3.0, 2.5]
    weights = [[0.2, 0.8, -0.5, 1],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]]
    biases = [2.0, 3.0, 0.5]
    layer_outputs = np.dot(weights, inputs) + biases
    print(layer_outputs)


def layerOfNeuronWithBatchOfDataUsingNumPy():
    import numpy as np
    inputs = [[1.0, 2.0, 3.0, 2.5],
              [2.0, 5.0, -1.0, 2.0],
              [-1.5, 2.7, 3.3, -0.8]]
    weights = [[0.2, 0.8, -0.5, 1.0],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]]
    biases = [2.0, 3.0, 0.5]
    layer_outputs = np.dot(inputs, np.array(weights).T) + biases
    print(layer_outputs)


def multiLayerOfNeuronUsingNumPy():
    inputs = [[1, 2, 3, 2.5], [2., 5., -1., 2], [-1.5, 2.7, 3.3, -0.8]]
    weights = [[0.2, 0.8, -0.5, 1],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]]
    biases = [2, 3, 0.5]
    weights2 = [[0.1, -0.14, 0.5],
                [-0.5, 0.12, -0.33],
                [-0.44, 0.73, -0.13]]
    biases2 = [-1, 2, -0.5]
    layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
    layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
    print(layer2_outputs)


def firstDenseLayer():
    X = [
        [1,2,3,2.5],
        [2.0,5.0,-1.0,2.9],
        [-1.5,2.7,3.3,-0.8]
    ]


    # We’ll initialize the biases with the shape of (1, n_neurons), as a row vector,
    # which will let us easily add it to the result of the dot product later,
    # without additional operations like transposition.
    layer1 = ld.Layer_Dense(4, 5)
    layer2 = ld.Layer_Dense(5, 2)

    layer1.forward(X)
    print(layer1.output)
    layer2.forward(layer1.output)
    print(layer2.output)


def ReLUActivationFunction():
    X, y = spiral_data(samples=100, classes=3)
    dense1 = ld.Layer_Dense(2, 3)
    activation1 = Activation_ReLU()
    dense1.forward(X)
    activation1.forward(dense1.output)
    print(activation1.output[:5])

def softmaxActivationFunction():
    # Create dataset
    X, y = spiral_data(samples=100, classes=3)
    # Create Dense layer with 2 input features and 3 output values
    dense1 = ld.Layer_Dense(2, 3)
    # Create ReLU activation (to be used with Dense layer):
    activation1 = Activation_ReLU()
    # Create second Dense layer with 3 input features (as we take output
    # of previous layer here) and 3 output values (output values)
    dense2 = ld.Layer_Dense(3, 3)
    # Create Softmax activation (to be used with Dense layer):
    activation2 = Activation_Softmax()
    # Make a forward pass of our training data through this layer
    dense1.forward(X)
    # Make a forward pass through activation function
    # it takes the output of first dense layer here
    activation1.forward(dense1.output)
    # Make a forward pass through second Dense layer
    # it takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)
    # Make a forward pass through activation function
    # it takes the output of second dense layer here
    activation2.forward(dense2.output)
    # Let's see output of the first few samples:
    print(activation2.output[:5])

softmaxActivationFunction()