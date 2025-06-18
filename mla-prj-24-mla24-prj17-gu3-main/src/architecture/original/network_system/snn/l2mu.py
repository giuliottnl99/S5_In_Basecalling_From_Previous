import torch
import torch.nn as nn
from src.architecture.original.network_system.snn.backend_module.l2mu import L2MULeaky, L2MUSynaptic


class L2MUNet(nn.Module):
    def __init__(self, input_size, output_size, params, neuron_type='leaky', encoder='plain'):
        super().__init__()
        self.slmu = None
        self.select_neuron_type(input_size, output_size, neuron_type, encoder, params)

    def select_neuron_type(self, input_size, output_size, neuron_type, encoder, params):
        if neuron_type == 'leaky':
            self.slmu = L2MULeaky(
                input_size=input_size,
                output=True,
                output_size=output_size,
                encoder=encoder,
                params=params
            )
        elif neuron_type == 'synaptic':
            self.slmu = L2MUSynaptic(
                input_size=input_size,
                output=True,
                output_size=output_size,
                encoder=encoder,
                params=params
            )
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")

    def forward(self, input):
        spk_hidden, spk_memory = self.slmu.init_l2mu()

        # Record the output
        spk_output_list = []

        for step in range(input.size(0)):
            spk_output, spk_hidden, spk_memory = self.slmu(input[step].flatten(1), spk_hidden=spk_hidden,
                                                           spk_memory=spk_memory)
            spk_output_list.append(spk_output)

        return torch.stack(spk_output_list, dim=0)
