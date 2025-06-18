import torch
from snntorch import Synaptic
import torch.nn as nn


class StackedPopEncoder(nn.Module):
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

        self.syn_decoder = None
        self.syn_encoder = None
        self.mem_decoder = None
        self.mem_encoder = None
        self.mem_hidden_encoder = []
        self.syn_hidden_encoder = []
        self.input_size = input_size
        self.encoding_size = int(params['encoding_size'])
        self.pop_size = int(params['pop_size'])
        self.output_size = int(params['output_transformer_size'])

        self.neurons_hidden_encored_conf = [
            (params['beta_spk_x_acc'], params['threshold_spk_x_acc'], params['alpha_spk_x_acc']),
            (params['beta_spk_x_gyro'], params['threshold_spk_x_gyro'], params['alpha_spk_x_gyro']),
            (params['beta_spk_y_acc'], params['threshold_spk_y_acc'], params['alpha_spk_y_acc']),
            (params['beta_spk_y_gyro'], params['threshold_spk_y_gyro'], params['alpha_spk_y_gyro']),
            (params['beta_spk_z_acc'], params['threshold_spk_z_acc'], params['alpha_spk_z_acc']),
            (params['beta_spk_z_gyro'], params['threshold_spk_z_gyro'], params['alpha_spk_z_gyro'])
        ]

        for i in range(self.input_size):
            self.mem_hidden_encoder.append(None)
            self.syn_hidden_encoder.append(None)

        self.hidden_encoder_list = []
        self.spk_hidden_encoder_list = []
        for i in range(self.input_size):
            self.hidden_encoder_list.append(nn.Linear(1, self.pop_size))
            self.spk_hidden_encoder_list.append(
                Synaptic(beta=self.neurons_hidden_encored_conf[i][0], threshold=self.neurons_hidden_encored_conf[i][1], alpha=self.neurons_hidden_encored_conf[i][2] ))

        self.hidden_encoder_list = nn.ModuleList(self.hidden_encoder_list)
        self.spk_hidden_encoder_list = nn.ModuleList(self.spk_hidden_encoder_list)

        self.encoder = nn.Linear(self.input_size * self.pop_size, self.encoding_size)
        self.spk_encoder = Synaptic(beta=params['beta_spk_encoder'], threshold=params['threshold_spk_encoder'], alpha=params['alpha_spk_encoder'])

        self.decoder = nn.Linear(self.encoding_size, self.output_size)
        self.spk_decoder = Synaptic(beta=params['beta_spk_decoder'], threshold=params['threshold_spk_decoder'], alpha=params['alpha_spk_decoder'])

    def forward(self, input_):

        spk_hidden_encoded_signal = []
        for i in range(self.input_size):
            hidden_encoded_feature_i = self.hidden_encoder_list[i](
                torch.reshape(input_[:, i], (input_[:, 1].shape[0], 1)))
            spk_hidden_encoded_feature_i, self.syn_hidden_encoder[i], self.mem_hidden_encoder[i] = \
                self.spk_hidden_encoder_list[i](hidden_encoded_feature_i, self.syn_hidden_encoder[i],
                                                self.mem_hidden_encoder[i])
            spk_hidden_encoded_signal.append(spk_hidden_encoded_feature_i)

        spk_hidden_encoded_signal_rec = torch.cat(spk_hidden_encoded_signal, 1)

        encoder_signal = self.encoder(spk_hidden_encoded_signal_rec)
        spk_encoder_signal, self.syn_encoder, self.mem_encoder = self.spk_encoder(encoder_signal, self.syn_encoder,
                                                                                  self.mem_encoder)
        decoder_signal = self.decoder(spk_encoder_signal)

        spk_decoder_signal, self.syn_decoder, self.mem_decoder = self.spk_decoder(decoder_signal, self.syn_decoder,
                                                                                  self.mem_decoder)

        return spk_decoder_signal

    def init_pop_encoder(self):
        for i in range(self.input_size):
            self.syn_hidden_encoder[i], self.mem_hidden_encoder[i] = self.spk_hidden_encoder_list[i].init_synaptic()
        self.syn_encoder, self.mem_encoder = self.spk_encoder.init_synaptic()
        self.syn_decoder, self.mem_decoder = self.spk_decoder.init_synaptic()
