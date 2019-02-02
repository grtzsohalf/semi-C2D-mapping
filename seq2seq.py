import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import random

from attention import Attention
from utils import weight_init


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################
# The modules for a seq2seq model
# ===============================
#


######################################################################
# The BaseRNN module
# ------------------
#
# Implement a multi-layer RNN.
# It is a base class. Please use one of the sub-classes.
#

class BaseRNN(nn.Module):
    SYM_MASK = "MASK"

    def __init__(self, io_size, max_len, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell):
        super(BaseRNN, self).__init__()
        self.io_size = io_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.dropout_p = dropout_p

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


######################################################################
# The encoder module
# ------------------
#

class EncoderRNN(BaseRNN):
    """
    Inputs: inputs, input_lengths
        - **inputs**: (batch_size, seq_len, feat_size): list of sequences.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch_size, it must be provided when using variable length RNN (default: `None`)
    Outputs: output, hidden
        - **output** (batch_size, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the hidden state `h`
    """

    def __init__(self, input_size, max_len, hidden_size, 
                 input_dropout_p=0, dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru', 
                 variable_lengths=False):
        super(EncoderRNN, self).__init__(input_size, max_len, hidden_size,
                input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.variable_lengths = variable_lengths
        self.rnn = self.rnn_cell(input_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
        self.rnn.apply(weight_init)

    def forward(self, input_var, input_lengths=None):
        input_var = self.input_dropout(input_var)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(input_var, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden


######################################################################
# The decoder module
# ------------------
#

class DecoderRNN(BaseRNN):
    """
    Attributes:
        KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
    Inputs: inputs, encoder_hidden, encoder_outputs, teacher_forcing_ratio
        - **inputs** (batch_size, seq_len, input_size): list of sequences. It is used for teacher forcing when provided. (default `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch_size, seq_len, hidden_size): tensor which contains the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).
    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (seq_len, batch_size, output_size): list of tensors with size (batch_size, output_size) containing
          the outputs.
        - **decoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_ATTN_SCORE* : list of
          sequences, where each list is of attention weights }.
    """

    KEY_ATTN_SCORE = 'attention_score'

    def __init__(self, output_size, max_len, hidden_size,
            n_layers=1, rnn_cell='gru', bidirectional=False,
            input_dropout_p=0, dropout_p=0, use_attention=False):
        super(DecoderRNN, self).__init__(output_size, max_len, hidden_size,
                input_dropout_p, dropout_p,
                n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        self.rnn = self.rnn_cell(hidden_size, output_size, n_layers, batch_first=True, dropout=dropout_p)
        self.rnn.apply(weight_init)

        self.max_length = max_len
        self.use_attention = use_attention

        self.init_input = None

        if use_attention:
            self.attention = Attention(self.hidden_size)

    def forward_step(self, input_var, hidden, encoder_outputs):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)

        input_var = self.input_dropout(input_var)

        output, hidden = self.rnn(input_var, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        return output, hidden, attn

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None, teacher_forcing_ratio=0):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs, teacher_forcing_ratio)
        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
        if use_teacher_forcing:
            # decoder_input = inputs[:, :-1]
            decoder_output, decoder_hidden, attn = self.forward_step(inputs, decoder_hidden, encoder_outputs)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decode(di, step_output, step_attn)
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
                step_output = decoder_output.squeeze(1)
                decode(di, step_output, step_attn)
                decoder_input = step_output

        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        # if inputs is None:
            # if teacher_forcing_ratio > 0:
                # raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            # inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            # if torch.cuda.is_available():
                # inputs = inputs.cuda()
            # max_length = self.max_length
        # else:
            # max_length = inputs.size(1) - 1 # minus the start of sequence symbol

        # return inputs, batch_size, max_length
        return inputs, batch_size, inputs.size(1)


######################################################################
# The seq2seq module
# ------------------
#

class Seq2seq(nn.Module):
    """ 
    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN
    Inputs: input_variable, input_lengths, target_variable, teacher_forcing_ratio
        - **input_variable** (batch_size, seq_len, feat_size): list of sequences. This information is forwarded to the encoder.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
        - **target_variable** (list, optional): list of sequences. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (int, optional): probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0)
    Outputs: encoder_outputs, encoder_hidden, decoder_outputs, decoder_hidden, ret_dict
        - **encoder_outputs** (batch_size, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the hidden state `h`
        - **decoder_outputs** (seq_len, batch_size, output_size): batch-length list of tensors with size (max_length, hidden_size) 
          containing the outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_ATTN_SCORE* : list of
          sequences, where each list is of attention weights }.
    """

    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, 
                input_lengths=None, target_variable=None, teacher_forcing_ratio=0):
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)
        decoder_outputs, decoder_hidden, ret_dict = \
            self.decoder(inputs=target_variable,
                         encoder_hidden=encoder_hidden,
                         encoder_outputs=encoder_outputs,
                         teacher_forcing_ratio=teacher_forcing_ratio)
        return encoder_outputs, encoder_hidden, decoder_outputs, decoder_hidden, ret_dict
