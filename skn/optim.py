#   Shubh Khandelwal

from skn.nn import NeuralNetwork

class Optimizer:

    def step(self, nn: NeuralNetwork) -> None:
        raise NotImplementedError
    
class SGD(Optimizer):

    def __init__(self, learn_rate: float = 0.01) -> None:
        self.learn_rate = learn_rate
    
    def step(self, nn: NeuralNetwork) -> None:
        for param, gradient in nn.params_and_gradients():
            param -= self.learn_rate * gradient