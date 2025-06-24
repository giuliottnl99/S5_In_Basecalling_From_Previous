"""Implementation of the Bonito-CTC model

Based on: 
https://github.com/nanoporetech/bonito
"""

import os
import sys
from torch import nn

from bonitosnn.classes import BaseModelImpl
from bonitosnn.layers import BonitoLSTM
from s5 import S5Block


class CommonBonitoS5Model(BaseModelImpl):
    """Bonito Model
    """
    def __init__(self, convolution = None, encoder = None, decoder = None, reverse = True, load_default = False, *args, **kwargs):
        super(CommonBonitoS5Model, self).__init__(*args, **kwargs)
        """
        Args:
            convolution (nn.Module): module with: in [batch, channel, len]; out [batch, channel, len]
            encoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
            decoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
            reverse (bool): if the first rnn layer starts with reverse 
        """
    
        self.convolution = convolution
        self.encoder = encoder
        self.decoder = decoder
        self.reverse = reverse
        # self.networkAndDecoderConfig => Must be defined in the subclass
        
        if load_default:
            self.load_default_configuration()

    def forward(self, x):
        """Forward pass of a batch
        
        Args:
            x (tensor) : [batch, channels (1), len]
        """
        
        x = self.convolution(x)
        x = x.permute(2, 0, 1) # [len, batch, channels]
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def build_cnn(self):

        cnn = nn.Sequential(
            nn.Conv1d(
                in_channels = 1, 
                out_channels = self.networkAndDecoderConfig['1_layer_out_channels'],  #4 for Bonito, 64 for S5
                kernel_size = 5, 
                stride= 1, 
                padding=5//2, 
                bias=True),
            nn.SiLU(),
            nn.Conv1d(
                in_channels = self.networkAndDecoderConfig['1_layer_out_channels'], 
                out_channels = self.networkAndDecoderConfig['2_layer_out_channels'],  #4 for Bonito, 64 for S5
                kernel_size = 5, 
                stride= 1, 
                padding=5//2, 
                bias=True),
            nn.SiLU(),
            nn.Conv1d(
                in_channels = self.networkAndDecoderConfig['2_layer_out_channels'],  #16 for Bonito, 64 for S5 
                out_channels = self.networkAndDecoderConfig['decoder_dimension'], #384 for Bonito, 256 for S5 
                kernel_size = 19, 
                stride= 5, 
                padding=19//2, 
                bias=True),
            nn.SiLU()
        )
        return cnn
        
    def load_default_configuration(self):
        """Sets the default configuration for one or more
        modules of the network
        """

        self.convolution = self.build_cnn()
        self.cnn_stride = self.get_defaults()['cnn_stride']
        self.encoder = self.build_encoder(input_size = self.networkAndDecoderConfig['decoder_dimension'], reverse = True)
        self.decoder = self.build_decoder(encoder_output_size = self.networkAndDecoderConfig['decoder_dimension'], decoder_type = 'crf')
        self.decoder_type = 'crf'


class BonitoModel(CommonBonitoS5Model):
    """Bonito Model
    """
    def __init__(self, convolution = None, encoder = None, decoder = None, reverse = True, load_default = False, networkAndDecoderConfig=None, *args, **kwargs):
        
        self.networkAndDecoderConfig = {'1_layer_out_channels': 4, '2_layer_out_channels': 16, 'decoder_dimension': 384}
        super(BonitoModel, self).__init__(load_default=load_default, *args, **kwargs)
        
    def build_encoder(self, input_size, reverse):

        if reverse:
            encoder = nn.Sequential(BonitoLSTM(input_size, 384, reverse = True),
                                    BonitoLSTM(384, 384, reverse = False),
                                    BonitoLSTM(384, 384, reverse = True),
                                    BonitoLSTM(384, 384, reverse = False),
                                    BonitoLSTM(384, 384, reverse = True))
        else:
            encoder = nn.Sequential(BonitoLSTM(input_size, 384, reverse = False),
                                    BonitoLSTM(384, 384, reverse = True),
                                    BonitoLSTM(384, 384, reverse = False),
                                    BonitoLSTM(384, 384, reverse = True),
                                    BonitoLSTM(384, 384, reverse = False))
        return encoder    

    def get_defaults(self):
        defaults = {
            'cnn_output_size': 384, 
            'cnn_output_activation': 'silu',
            'encoder_input_size': 384,
            'encoder_output_size': 384,
            'cnn_stride': 5,
        }
        return defaults
        

class S5Model(CommonBonitoS5Model):
    """Bonito Model
    """
    def __init__(self, convolution = None, encoder = None, decoder = None, reverse = True, load_default = False, networkAndDecoderConfig=None, *args, **kwargs):
        
        self.networkAndDecoderConfig = {'1_layer_out_channels': 64, '2_layer_out_channels': 64, 'decoder_dimension': 256}
        super(S5Model, self).__init__(load_default=load_default, *args, **kwargs)
        
    def build_encoder(self, input_size, reverse):

        """Build S5 SSM encoder
        
        Args:
            input_size (int): input feature dimension
        """
        
        # S5 SSM layers - wrapping to handle len-first format
        encoder = S5Wrapper(
            dim=input_size,
            state_dim=self.config.state_dim,
            bidir=True,
            block_count=self.config.s5_stacks,
            ff_dropout=0.0,
        )
        return encoder


class S5Wrapper(nn.Module):
    """Wrapper for S5Block to handle len-first tensor format"""
    
    def __init__(self, dim, state_dim, bidir, block_count, ff_dropout):
        super().__init__()
        self.s5_block = S5Block(
            dim=dim,
            state_dim=state_dim,
            bidir=bidir,
            block_count=block_count,
            ff_dropout=ff_dropout,
        )
    
    def forward(self, x):
        # x shape: [len, batch, features]
        # S5Block expects: [batch, len, features]
        x = x.permute(1, 0, 2)  # [batch, len, features]
        x = self.s5_block(x)    # [batch, len, features]
        x = x.permute(1, 0, 2)  # [len, batch, features]
        return x
