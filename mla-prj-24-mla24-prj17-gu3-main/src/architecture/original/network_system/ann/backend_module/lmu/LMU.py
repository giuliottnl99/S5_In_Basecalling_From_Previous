import torch
from torch import nn
import torch.nn.functional as F
from src.architecture.original.network_system.bare_lmu_network.bare_LMU import BareLMU


class LMU(BareLMU):

    def __init__(
            self,
            input_size,
            params,
            bias=False,
            trainable_theta=False,
            output_size=None,
            output=False,
    ):
        super().__init__(input_size=input_size, hidden_size=int(params['hidden_size']),
                         memory_size=int(params['memory_size']), order=int(params['order']), theta=params['theta'],
                         output=output, output_size=output_size, bias=bias, trainable_theta=trainable_theta)

        self.f = nn.Tanh()
        self._gen_AB()
        self.init_parameters()

    def forward(self, input_, hidden, memory):

        if hidden is None and memory is None:
            hidden = torch.zeros(size=(input_.size(0), self.hidden_size), device=input_.device, requires_grad=True)
            memory = torch.zeros(size=(input_.size(0), self.memory_size * self.order), device=input_.device,
                                 requires_grad=True)

        # Equation (7) of the paper
        u = F.linear(input_, self.e_x, self.bias_x) + F.linear(hidden, self.e_h, self.bias_h) + \
            F.linear(memory, self.e_m, self.bias_m)  # [batch_size, memory_size]

        # Equation (4) of the paper

        # separate memory/order dimensions
        u = torch.unsqueeze(u, -1)
        memory = torch.reshape(memory, (-1, self.memory_size, self.order))

        if self.discretizer == 'zoh' and self.trainable_theta:
            A, B = self._cont2discrete_zoh(self._base_A * self.theta_inv, self._base_B * self.theta_inv)
        else:
            A, B = self.A, self.B

        memory = F.linear(memory, A) + F.linear(u, B)  # [batch_size, memory_size]

        if self.discretizer == 'euler' and self.trainable_theta:
            memory += memory * self.theta_inv

        # re-combine memory/order dimensions
        memory = torch.reshape(memory, (-1, self.memory_size * self.order))

        # Equation (6) of the paper
        hidden = self.f(F.linear(input_, self.W_x) + F.linear(hidden, self.W_h) + F.linear(memory, self.W_m))

        # Output
        if self.output:
            output = self.output_transformation(hidden)
            return output, hidden, memory

        return None, hidden, memory
