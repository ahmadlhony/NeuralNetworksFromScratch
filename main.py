import numpy as np
import layer_dense as ld
import nnfs
import loss_categorical_crossentropy
from nnfs.datasets import spiral_data

from activation_ReLU import Activation_ReLU
from activation_softmax import Activation_Softmax
from activation_softmax_loss_categorical_crossentropy import Activation_Softmax_Loss_CategoricalCrossentropy
from optimizer_RMSprop import Optimizer_RMSprop
from optimizer_adam import Optimizer_Adam
from optimizer_sgd import Optimizer_SGD

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

def calculationgLossWithCategoricalCrossEntropy():
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

    loss_function = loss_categorical_crossentropy.Loss_CategoricalCrossentropy()

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

    # Perform a forward pass through activation function
    # it takes the output of second dense layer here and returns loss
    loss = loss_function.calculate(activation2.output, y)
    # Print loss value
    print('loss:', loss)

def accuracyCalculation():
    # Probabilities of 3 samples
    softmax_outputs = np.array([[0.7, 0.2, 0.1],
                                [0.5, 0.1, 0.4],
                                [0.02, 0.9, 0.08]])
    # Target (ground-truth) labels for 3 samples
    class_targets = np.array([0, 1, 1])
    # Calculate values along second axis (axis of index 1)
    predictions = np.argmax(softmax_outputs, axis=1)
    # If targets are one-hot encoded - convert them
    if len(class_targets.shape) == 2:
        class_targets = np.argmax(class_targets, axis=1)
    # True evaluates to 1; False to 0
    accuracy = np.mean(predictions == class_targets)
    print('acc:', accuracy)


def applyBackpropagation():
    # Create dataset
    X, y = spiral_data(samples=100, classes=3)
    # Create Dense layer with 2 input features and 3 output values
    dense1 = ld.Layer_Dense(2, 3)
    # Create ReLU activation (to be used with Dense layer):
    activation1 = Activation_ReLU()
    # Create second Dense layer with 3 input features (as we take output
    # of previous layer here) and 3 output values (output values)
    dense2 = ld.Layer_Dense(3, 3)
    # Create Softmax classifier's combined loss and activation
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    # Perform a forward pass of our training data through this layer
    dense1.forward(X)
    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)
    # Perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)
    # Perform a forward pass through the activation/loss function
    # takes the output of second dense layer here and returns loss
    loss = loss_activation.forward(dense2.output, y)
    # Let's see output of the first few samples:
    print(loss_activation.output[:5])
    # Print loss value
    print('loss:', loss)
    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    # Print accuracy
    print('acc:', accuracy)
    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    # Print gradients
    print(dense1.dweights)
    print(dense1.dbiases)
    print(dense2.dweights)
    print(dense2.dbiases)

def applyOptimizerSGD():
    # Create dataset
    X, y = spiral_data(samples=100, classes=3)
    # Create Dense layer with 2 input features and 64 output values
    dense1 = ld.Layer_Dense(2, 64)
    # Create ReLU activation (to be used with Dense layer):
    activation1 = Activation_ReLU()
    # Create second Dense layer with 64 input features (as we take output
    # of previous layer here) and 3 output values (output values)
    dense2 = ld.Layer_Dense(64, 3)
    # Create Softmax classifier's combined loss and activation
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    # Create optimizer
    # optimizer = Optimizer_SGD(decay=1e-3,  momentum=0.9)
    # optimizer = Optimizer_SGD(decay=1e-4)
    # optimizer = Optimizer_RMSprop(learning_rate=0.02, decay=1e-5, rho=0.999)
    optimizer = Optimizer_Adam(learning_rate=0.05, decay=1e-7)
    for epoch in range(10001):
        # Perform a forward pass of our training data through this layer
        dense1.forward(X)
        # Perform a forward pass through activation function
        # takes the output of first dense layer here
        activation1.forward(dense1.output)
        # Perform a forward pass through second Dense layer
        # takes outputs of activation function of first layer as inputs
        dense2.forward(activation1.output)
        # Perform a forward pass through the activation/loss function
        # takes the output of second dense layer here and returns loss
        loss = loss_activation.forward(dense2.output, y)
        # Let's print loss value
        # Calculate accuracy from output of activation2 and targets
        # calculate values along first axis
        predictions = np.argmax(loss_activation.output, axis=1)

        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)

        if not epoch % 100:
            print(f'epoch: {epoch}, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}, ' +
                  f'lr: {optimizer.current_learning_rate}')

        # Backward pass
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update weights and biases
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()


accuracyCalculation()
