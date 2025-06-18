import torch
from snntorch import Leaky
import torch.nn as nn


class SinglePopEncoder(nn.Module):
    """
    Input WISDM dataset:
        0: type: accelerometer, axes: x -> Leaky 1
        1: type: gyro, axes: x  -> Leaky 2
        2: type: accelerometer, axes: y -> Leaky 3
        3: type: gyro, axes: y  -> Leaky 4
        4: type: accelerometer, axes: z -> Leaky 5
        5: type: gyro, axes: z -> Leaky 6
    """

    def __init__(self, input_size, params, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mem_pop_encoder = []
        self.input_size = input_size
        self.pop_size = int(params['pop_size'])

        self.neurons_pop_encored_conf = [
            (params['beta_spk_x_acc'], params['threshold_spk_x_acc']),
            (params['beta_spk_x_gyro'], params['threshold_spk_x_gyro']),
            (params['beta_spk_y_acc'], params['threshold_spk_y_acc']),
            (params['beta_spk_y_gyro'], params['threshold_spk_y_gyro']),
            (params['beta_spk_z_acc'], params['threshold_spk_z_acc']),
            (params['beta_spk_z_gyro'], params['threshold_spk_z_gyro'])
        ]

        self.mem_pop_encoder = [None] * self.input_size

        self.pop_encoder_list = []
        self.spk_pop_encoder_list = []

        for i in range(self.input_size):
            self.pop_encoder_list.append(nn.Linear(1, self.pop_size))
            self.spk_pop_encoder_list.append(
                Leaky(beta=self.neurons_pop_encored_conf[i][0], threshold=self.neurons_pop_encored_conf[i][1]))

        self.pop_encoder_list = nn.ModuleList(self.pop_encoder_list)

    def forward(self, input_):

        spk_pop_encoded_signal = []
        for i in range(self.input_size):
            pop_encoded_feature_i = self.pop_encoder_list[i](
                torch.reshape(input_[:, i], (input_[:, 1].shape[0], 1)))
            spk_pop_encoded_feature_i, self.mem_pop_encoder[i] = self.spk_pop_encoder_list[i](
                pop_encoded_feature_i, self.mem_pop_encoder[i])
            spk_pop_encoded_signal.append(spk_pop_encoded_feature_i)

        spk_pop_encoded_signal_rec = torch.cat(spk_pop_encoded_signal, 1)

        return spk_pop_encoded_signal_rec  # output_dim : pop_size * input_size

    def init_pop_encoder(self):
        for i in range(self.input_size):
            self.mem_pop_encoder[i] = self.spk_pop_encoder_list[i].init_leaky()

    @property
    def output_size(self):
        return self.input_size * self.pop_size
