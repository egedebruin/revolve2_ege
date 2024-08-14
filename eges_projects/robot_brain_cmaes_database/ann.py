import numpy as np


class NeuralNetwork:
    def __init__(self):
        # Set up architecture
        self.input_size = 2
        self.hidden_size = 10
        self.output_size = 1

        # Initialize weights
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)

        # Initialize biases
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def set_weights(self, weights):
        self.weights_input_hidden = np.array_split(weights[:20], self.input_size)
        self.weights_hidden_output = np.array_split(weights[20:], self.hidden_size)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    def forward(self, inputs):
        # Forward pass
        hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.tanh(hidden_input)

        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        output_output = self.tanh(output_input)

        return output_output
