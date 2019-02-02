from __future__ import unicode_literals, print_function, division
import random
import os
import time
from datetime import datetime 
from tensorboardX import SummaryWriter

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch import autograd

from A2VwD_RNN import A2VwD
from ../seq2seq import EncoderRNN, DecoderRNN
from ../WGAN-GP import Discriminator
from ../utils import weight_init


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################
# Solver
# ======
#
# Set all parameters and construct a graph for the model.
#

class Solver:
    def __init__(self, init_lr, batch_size, seq_len, feat_dim, p_hidden_dim, s_hidden_dim, 
                 dropout_rate, iter_d, phn_num_layers, spk_num_layers, 
                 dec_num_layers, D_num_layers, log_dir, model_dir):

        self.init_lr = init_lr
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.p_hidden_dim = p_hidden_dim
        self.s_hidden_dim = s_hidden_dim
        self.dropout_rate = dropout_rate
        self.iter_d = iter_d
        self.phn_num_layers=phn_num_layers
        self.spk_num_layers=spk_num_layers
        self.dec_num_layers=dec_num_layers
        self.D_num_layers=D_num_layers

        self.log_dir = log_dir
        self.model_dir = model_dir
        self.model = None


    ######################################################################
    # Train the Model
    # ------------------
    #

    def buildModel(model=None):
        if model:
            self.model = model
        else:
            phn_encoder = EncoderRNN(self.feat_dim, self.seq_len, self.p_hidden_dim, 
                                     input_dropout_p=self.dropout_rate, dropout_p=self.dropout_rate,
                                     n_layers=self.phn_num_layers, bidirectional=True, rnn_cell='gru', variable_lengths=True)
            spk_encoder = EncoderRNN(self.feat_dim, self.seq_len, self.s_hidden_dim, 
                                     input_dropout_p=self.dropout_rate, dropout_p=self.dropout_rate,
                                     n_layers=self.spk_num_layers, bidirectional=True, rnn_cell='gru', variable_lengths=True)
            decoder = DecoderRNN(self.feat_dim, self.seq_len, self.p_hidden_dim+self.s_hidden_dim, n_layers=self.dec_num_layers, 
                                 rnn_cell='gru', bidirectional=True, input_dropout_p=self.dropout_rate, dropout_p=self.dropout_rate)
            # size of discriminator input = phn_num_layers * num_directions * p_hidden_dim * 2
            discriminator = Discriminator(self.phn_num_layers*2*self.p_hidden_dim*2, self.p_hidden_dim*2, self.D_num_layers)
            self.model = self.A2VwD(phn_encoder, spk_encoder, decoder, discriminator)
        model.to(device)

    ######################################################################
    # The whole training process:
    #
    # -  Start a timer
    # -  Initialize optimizers and criterion
    # -  Create sets of training pairs
    # -  Start empty losses array
    #
    # Then we call ``train`` many times and occasionally print the progress (%
    # of examples, time so far, estimated time) and average loss.
    #

    def trainIters(train_data, n_epochs, print_every=200):
        writer = SummaryWriter()
        start = time.time()
        print_loss_total = 0  # Reset every print_every

        optimizer_D = optim.Adam(self.model.discriminator.parameters(), lr=self.init_lr, betas=(0.5, 0.9))
        optimizer_G = optim.Adam([self.model.phn_encoder.parameters(), self.model.spk_encoder.parameters(),
                                  self.model.decoder.parameters(), lr=self.init_lr, betas=(0.5, 0.9))

        num_examples = 
        batch_target, batch_pos, batch_neg = train_data(indices)
                          for i in range(n_iters)]

        for epoch in range(1, n_epochs + 1):
            ################
            # (1) Update D #
            ################
            for p in self.model.discriminator.parameters(): # reset requires_grad
                p.requires_grad = True # set to False below in G update

            for iter in self.iter_d:
                self.model.discriminator.zero_grad()

                losses (d_loss, GP_loss) = 
                losses.backward()
                optimizer_D.step()

            ################
            # (2) Update G #
            ################
            for p in self.model.discriminator.parameters():
                p.requires_grad = False # to avoid computation
            
            self.model.phn_encoder.zero_grad()
            self.model.spk_encoder.zero_grad()
            self.model.decoder.zero_grad()

            losses (g_loss, r_loss, pos_spk_loss, neg_spk_loss) = 
            losses.backward()
            optimizer_G.step()

            # Write logs, save model and print losses
            print_losses_total += losses
            writer.add_scalar('losses/...', loss, epoch)
            ckpt = Checkpoint(model, optimizers, epoch, 0)
            ckpt.save(path)

            if iter % print_every == 0:
                print_losses_avg = print_losses_total / print_every
                print_losses_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                             epoch, epoch / n_epochs * 100, print_losses_avg))

        writer.export_scalars_to_json(".../....json")
        writer.close()

