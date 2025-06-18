import torch
from torch import nn
from torch.nn import init
import numpy as np
from abc import abstractmethod
from src.architecture.utils import lecun_uniform


class BareLMU(nn.Module):

    def __init__(
            self,
            input_size,
            hidden_size,
            memory_size,
            order,
            theta,
            output_size=None,
            output=False,
            bias=False,
            trainable_theta=False,
            discretizer='zoh'

    ):
        super().__init__()

        # Parameters passed
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.order = order
        self._init_theta = theta
        self.output_size = output_size
        self.bias = bias
        self.output = output
        self.trainable_theta = trainable_theta
        self.discretizer = discretizer

        # Parameters to be learned
        self.W_h = None
        self.W_m = None
        self.W_x = None
        self.bias_m = None
        self.bias_h = None
        self.bias_x = None
        self.e_m = None
        self.e_h = None
        self.e_x = None
        self.theta_inv = None
        self.output_transformation = None

    def init_parameters(self):
        if self.trainable_theta:
            self.theta_inv = nn.Parameter(torch.empty(()))
        else:
            self.theta_inv = 1 / self._init_theta

        self.e_x = nn.Parameter(torch.empty(self.memory_size, self.input_size))
        self.e_h = nn.Parameter(torch.empty(self.memory_size, self.hidden_size))
        self.e_m = nn.Parameter(torch.empty(self.memory_size, self.memory_size * self.order))

        self.bias_x, self.bias_h, self.bias_m = None, None, None

        if self.bias:
            self.bias_x = nn.Parameter(torch.empty(self.memory_size, ))
            self.bias_h = nn.Parameter(torch.empty(self.memory_size, ))
            self.bias_m = nn.Parameter(torch.empty(self.memory_size, ))

        # Kernels
        self.W_x = nn.Parameter(torch.empty(self.hidden_size, self.input_size))
        self.W_h = nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.W_m = nn.Parameter(torch.empty(self.hidden_size, self.memory_size * self.order))

        self.reset_parameters()

        if self.output:
            self.output_transformation = nn.Linear(self.hidden_size, self.output_size)

    @property
    def theta(self):
        if self.trainable_thera:
            return 1 / self.theta_inv
        return self._init_theta

    def reset_parameters(self):
        lecun_uniform(self.e_x)
        lecun_uniform(self.e_h)
        init.constant_(self.e_m, 0)

        if self.trainable_theta:
            init.constant_(self.theta_inv, 1 / self._init_theta)

        if self.bias:
            fan_in_x = init._calculate_correct_fan(self.e_x, mode='fan_in')
            bound = np.sqrt(3. / fan_in_x)
            init.uniform_(self.bias_x, -bound, bound)
            fan_in_h = init._calculate_correct_fan(self.e_h, mode='fan_in')
            bound = np.sqrt(3. / fan_in_h)
            init.uniform_(self.bias_h, -bound, bound)
            fan_in_m = init._calculate_correct_fan(self.e_m, mode='fan_in')
            bound = np.sqrt(3. / fan_in_m)
            init.uniform_(self.bias_m, -bound, bound)

        # Initialize kernels
        init.xavier_normal_(self.W_x)
        init.xavier_normal_(self.W_h)
        init.xavier_normal_(self.W_m)

    def _gen_AB(self):
        """Generates A and B matrices."""

        # compute analog A/B matrices
        Q = np.arange(self.order, dtype=np.float64)
        R = (2 * Q + 1)[:, None]
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
        B = (-1.0) ** Q[:, None] * R

        # discretize matrices
        if self.discretizer == "zoh":
            # save the un-discretized matrices for use in .call
            _base_A = torch.FloatTensor(A.T)
            _base_B = torch.FloatTensor(B.T)
            if self.trainable_theta:
                self.register_buffer('_base_A', _base_A)
                self.register_buffer('_base_B', _base_B)

            A, B = self._cont2discrete_zoh(
                _base_A / self._init_theta, _base_B / self._init_theta
            )
            self.register_buffer('A', A)
            self.register_buffer('B', B)

        else:
            if not self.trainable_theta:
                A = A / self._init_theta + np.eye(self.order)
                B = B / self._init_theta

            self.A = torch.FloatTensor(A.T)
            self.B = torch.FloatTensor(B.T)

    @staticmethod
    def _cont2discrete_zoh(A, B):
        """
        Function to discretize A and B matrices using Zero Order Hold method.

        Functionally equivalent to
        ``scipy.signal.cont2discrete((A.T, B.T, _, _), method="zoh", dt=1.0)``
        (but implemented in Pytorch so that it is differentiable).

        Note that this accepts and returns matrices that are transposed from the
        standard linear system implementation (as that makes it easier to use in
        `.call`).
        """

        # combine A/B and pad to make square matrix
        em_upper = torch.cat([A, B], dim=0)  # pylint: disable=no-value-for-parameter
        padding = (0, B.shape[0], 0, 0)
        em = torch.nn.functional.pad(em_upper, padding)

        # compute matrix exponential
        ms = torch.matrix_exp(em)

        # slice A/B back out of combined matrix
        discreet_A = ms[: A.shape[0], : A.shape[1]]
        discreet_B = ms[A.shape[0]:, : A.shape[1]]
        discreet_B = discreet_B.reshape(discreet_B.shape[1], discreet_B.shape[0])

        return discreet_A, discreet_B

    @classmethod
    @abstractmethod
    def forward(self, input_, _h, _m):
        pass
