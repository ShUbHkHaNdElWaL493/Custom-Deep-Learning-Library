#   Shubh Khandelwal

import numpy as np
from skn.tensor import Tensor
from typing import Callable, Dict

class Layer:

    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.gradients: Dict[str, Tensor] = {}

    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError
    
    def backward(self, gradient: Tensor) -> Tensor:
        raise NotImplementedError
    
class Linear(Layer):

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        return input @ self.params["w"] + self.params["b"]
    
    def backward(self, gradient: Tensor) -> Tensor:
        self.gradients["b"] = np.sum(gradient, axis = 0)
        self.gradients["w"] = self.input.T @ gradient
        return gradient @ self.params["w"].T
    
class Activation(Layer):

    def __init__(self, function: Callable[[Tensor], Tensor], function_1: Callable[[Tensor], Tensor]):
        super().__init__()
        self.function = function
        self.function_1 = function_1

    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        return self.function(input)
    
    def backward(self, gradient: Tensor) -> Tensor:
        return self.function_1(self.input) * gradient

def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_1(x: Tensor) -> Tensor:
    return (1 - (tanh(x) ** 2))

class Tanh(Activation):

    def __init__(self):
        super().__init__(tanh, tanh_1)