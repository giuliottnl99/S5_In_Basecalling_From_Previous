import torch
import torch.nn as nn
from src.architecture.original.network_system.ann.backend_module.lmu.LMU import LMU


class LMUNet(nn.Module):
    def __init__(self, input_size, output_size, params):
        super().__init__()
        self.lmu = LMU(input_size=input_size, output=True, output_size=output_size, params=params)

    def forward(self, input):
        hidden, memory = None, None

        # Record the output
        output_list = []

        for step in range(input.size(0)):
            output, hidden, memory = self.lmu(input[step].flatten(1), hidden=hidden,
                                              memory=memory)
            output_list.append(output)

        return torch.stack(output_list, dim=0)
