#   Shubh Khandelwal

from skn.layers import Layer
from skn.tensor import Tensor
from typing import Iterator, Sequence, Tuple

class NeuralNetwork:

    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def params_and_gradients(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                gradient = layer.gradients[name]
                yield param, gradient
    
    def forward(self, input: Tensor) -> Tensor:
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def backward(self, gradient: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient