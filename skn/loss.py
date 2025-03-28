#   Shubh Khandelwal

import numpy as np
from skn.tensor import Tensor

class Loss:

    def loss(self, predicted: Tensor, target: Tensor) -> float:
        raise NotImplementedError
    
    def gradient(self, predicted: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError
    
class MSE(Loss):

    def loss(self, predicted: Tensor, target: Tensor) -> float:
        return np.sum((predicted - target) ** 2)
    
    def gradient(self, predicted: Tensor, target: Tensor) -> Tensor:
        return 2 * (predicted - target)