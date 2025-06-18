from snntorch._neurons.neurons import _SpikeTensor
import torch
import torch.nn.functional as F
from snntorch import Leaky
from src.architecture.original.signal_encoding.single_pop_encoder import SinglePopEncoderLeaky
from src.architecture.original.signal_encoding.stacked_pop_encoder import StackedPopEncoderLeaky
from src.architecture.utils import initialize_states
from src.architecture.original.network_system.bare_lmu_network.bare_LMU import BareLMU


class L2MU(BareLMU):

    def __init__(
            self,
            input_size,
            params,
            bias=False,
            trainable_theta=False,
            output_size=None,
            output=False,
            encoder='plain'
    ):

        super().__init__(input_size=input_size, hidden_size=int(params['hidden_size']),
                         memory_size=int(params['memory_size']), order=int(params['order']), theta=params['theta'],
                         output=output, output_size=output_size, bias=bias, trainable_theta=trainable_theta)

        self.input_encoder = None
        self.select_input_encoder(input_size, encoder, params)

        self._gen_AB()
        self.init_parameters()

        self.spk_u = Leaky(beta=params['beta_spk_u'], threshold=params['threshold_spk_u'])
        self.spk_h = Leaky(beta=params['beta_spk_h'], threshold=params['threshold_spk_h'])
        self.spk_m = Leaky(beta=params['beta_spk_m'], threshold=params['threshold_spk_m'])

        self.mem_m = None
        self.mem_h = None
        self.mem_u = None

        if self.output:
            self.spk_output = Leaky(beta=params['beta_spk_output'], threshold=params['threshold_spk_output'])
            self.mem_output = None

    def select_input_encoder(self, input_size, encoder, params):
        if encoder == 'plain':
            pass  # No action needed for 'plain'
        elif encoder == 'single':
            self.input_encoder = SinglePopEncoderLeaky(input_size=input_size, params=params)
            self.input_size = self.input_encoder.output_size
        elif encoder == 'stacked':
            self.input_encoder = StackedPopEncoderLeaky(input_size=input_size, params=params)
        else:
            raise ValueError(f"Unknown encoder type: {encoder}")

    def init_l2mu(self):
        self.mem_u = self.spk_u.init_leaky()
        self.mem_h = self.spk_h.init_leaky()
        self.mem_m = self.spk_m.init_leaky()
        if self.input_encoder:
            self.input_encoder.init_pop_encoder()

        if self.output:
            self.mem_output = self.spk_output.init_leaky()

        spike_h = _SpikeTensor(init_flag=False)
        spike_m = _SpikeTensor(init_flag=False)

        return spike_h, spike_m

    def forward(self, input_, spk_hidden, spk_memory):

        if hasattr(spk_hidden, 'init_flag') or hasattr(spk_memory, 'init_flag'):
            spk_hidden = initialize_states(spk_hidden, (input_.size(0), self.hidden_size), input_.device)
            spk_memory = initialize_states(spk_memory, (input_.size(0), self.memory_size * self.order), input_.device)

        spk_input = self.input_encoder(input_) if self.input_encoder else input_

        # Equation (7) of the paper
        curr_u = F.linear(spk_input, self.e_x, self.bias_x) + F.linear(spk_hidden, self.e_h, self.bias_h) + \
                 F.linear(spk_memory, self.e_m, self.bias_m)  # [batch_size, memory_size]

        # Equation (4) of the paper
        spk_u, self.mem_u = self.spk_u(curr_u, self.mem_u)

        # separate memory/order dimensions
        spk_u = torch.unsqueeze(spk_u, -1)
        spk_memory = torch.reshape(spk_memory, (-1, self.memory_size, self.order))

        if self.discretizer == 'zoh' and self.trainable_theta:
            A, B = L2MU._cont2discrete_zoh(self._base_A * self.theta_inv, self._base_B * self.theta_inv)
        else:
            A, B = self.A, self.B

        curr_m = F.linear(spk_memory, A) + F.linear(spk_u, B)  # [batch_size, memory_size]

        if self.discretizer == 'euler' and self.trainable_theta:
            curr_m += curr_m * self.theta_inv

        spk_memory, self.mem_m = self.spk_m(curr_m, self.mem_m)

        # re-combine memory/order dimensions
        spk_memory = torch.reshape(spk_memory, (-1, self.memory_size * self.order))

        # Equation (6) of the paper
        curr_h = F.linear(spk_input, self.W_x) + F.linear(spk_hidden, self.W_h) + F.linear(spk_memory, self.W_m)

        spk_hidden, self.mem_h = self.spk_h(curr_h, self.mem_h)  # [batch_size, hidden_size]

        # Output
        if self.output:
            curr_output = self.output_transformation(spk_hidden)
            spk_output, self.mem_output = self.spk_output(curr_output, self.mem_output)
            return spk_output, spk_hidden, spk_memory

        return None, spk_hidden, spk_memory
