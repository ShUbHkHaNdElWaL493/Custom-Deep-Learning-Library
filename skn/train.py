#   Shubh Khandelwal

from skn.data import DataIterator, BatchIterator
from skn.loss import Loss, MSE
from skn.nn import NeuralNetwork
from skn.optim import Optimizer, SGD
from skn.tensor import Tensor

def train(
        nn: NeuralNetwork,
        input: Tensor,
        target: Tensor,
        num_epochs: int = 1,
        iterator: DataIterator = BatchIterator(),
        loss: Loss = MSE(),
        optimizer: Optimizer = SGD()
) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(input, target):
            predicted = nn.forward(batch.input)
            epoch_loss += loss.loss(predicted = predicted, target = batch.target)
            gradient = loss.gradient(predicted = predicted, target = batch.target)
            nn.backward(gradient)
            optimizer.step(nn)
        print(f"Epoch: [{epoch + 1}/{num_epochs}] | Loss: {epoch_loss}")