#   Shubh Khandelwal

import numpy as np
from sknpy.tensor import Tensor
from typing import Iterator, NamedTuple

Batch = NamedTuple("Batch", [("input", Tensor), ("target", Tensor)])

class DataIterator:

    def __call__(self, input: Tensor, target: Tensor) -> Iterator[Batch]:
        raise NotImplementedError
    
class BatchIterator(DataIterator):

    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, input: Tensor, target: Tensor) -> Iterator[Batch]:
        starts = np.arange(0, len(input), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)
        for start in starts:
            end = start + self.batch_size
            batch_input = input[start : end]
            batch_target = target[start : end]
            yield Batch(batch_input, batch_target)