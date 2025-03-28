#   Shubh Khandelwal

import numpy as np
from skn.layers import Linear, Tanh
from skn.nn import NeuralNetwork
from skn.train import train

input = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

target = np.array([
    [0, 1],
    [1, 0],
    [1, 0],
    [0, 1]
])

nn = NeuralNetwork([
    Linear(2, 4),
    Tanh(),
    Linear(4, 2)
])

train(nn = nn, input = input, target = target, num_epochs = 5000)

for x, y in zip(input, target):
    predicted = nn.forward(x)
    print(f"Input: {x} | Predicted: {predicted} | True: {y}")