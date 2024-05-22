import torch
import numpy as np


class FCLayer:
    def __init__(self, shape_in: int, shape_out: int, activation: str):
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.weights = torch.rand(shape_out, shape_in)
        self.weights.requires_grad(True)
        self.bias = torch.rand(shape_in)
        self.bias.requires_grad(True)
        self.activation = activation
        match activation:
            case "relu":
                # own implementation would just be torch.maximum(0,x)
                self.activation = torch.relu
            case "sigmoid":
                self.activation = torch.sigmoid
            case "tanh":
                self.activation = torch.tanh



    def propagate_forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape != self.weights.shape[0]:
            raise ValueError("x shape does not match weight shape")
        # output z
        z = self.weights @ x + self.bias
        return self.activation(z)
