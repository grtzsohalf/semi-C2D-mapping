from __future__ import unicode_literals, print_function, division
import random
import os
import time
from datetime import datetime 
from tqdm import tqdm
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch import autograd
import torch.nn.functional as F

from WGAN_GP import FCDiscriminator
from model import A2VwD, A2V, Model
from seq2seq import EncoderRNN, DecoderRNN
from utils import weight_init, write_pkl, Logger, getNN, beam_search
from torch.optim.lr_scheduler import StepLR


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


WIDTH = 50
WEIGHT_LM = 0.01


######################################################################
# Solver
# ======
#
# Set all parameters and construct a graph for the model.
#

class Solver:
    def __init__(self, init_lr, batch_size, seq_len, feat_dim, hidden_dim, 
                 enc_num_layers, dec_num_layers, D_num_layers, dropout_rate, 
                 iter_d, log_dir, mode):

        self.init_lr = init_lr
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.enc_num_layers = enc_num_layers
        self.dec_num_layers = dec_num_layers
        self.D_num_layers = D_num_layers
        self.dropout_rate = dropout_rate
        self.iter_d = iter_d

        self.mode = mode
        if self.mode == 'train':
            self.logger = Logger(os.path.join(log_dir))

        self.model = None

        # work around for log
        # self.log_train_file = log_dir+'/loss_train_'+loss_type
        # self.log_test_file = log_dir+'/loss_test_'+loss_type
        # if not os.path.isfile(self.log_train_file):
        #     with open(self.log_train_file, 'w') as fo:
        #         fo.write("0 0 0 0\n")
        # if not os.path.isfile(self.log_test_file):
        #     with open(self.log_test_file, 'w') as fo:
        #         fo.write("0 0 0 0\n")

    ######################################################################
    # Build the model
    # ------------------
    #

    def build_model(self):

        class MLP(nn.Module):

            def __init__(self, dims):
                super(MLP, self).__init__()
                self.hidden = nn.ModuleList()
                for k in range(len(dims)-1):
                    self.hidden.append(nn.Linear(dims[k], dims[k+1]))

            def forward(self, x):
                for layer in self.hidden[:-1]:
                    x = F.relu(layer(x))
                output = self.hidden[-1](x.float())
                return output

        # A2V
        input_MLP = MLP([self.feat_dim, self.hidden_dim, self.hidden_dim])
        phn_encoder = EncoderRNN(self.hidden_dim, self.seq_len, self.hidden_dim, 
                                 input_dropout_p=self.dropout_rate, dropout_p=self.dropout_rate,
                                 n_layers=self.enc_num_layers, bidirectional=True, rnn_cell='gru', variable_lengths=True)
        spk_encoder = EncoderRNN(self.hidden_dim, self.seq_len, self.hidden_dim, 
                                 input_dropout_p=self.dropout_rate, dropout_p=self.dropout_rate,
                                 n_layers=self.enc_num_layers, bidirectional=True, rnn_cell='gru', variable_lengths=True)
        decoder = DecoderRNN(self.hidden_dim*4, self.seq_len, self.hidden_dim*4, n_layers=self.dec_num_layers, 
                             rnn_cell='gru', bidirectional=True, input_dropout_p=self.dropout_rate, dropout_p=self.dropout_rate)
        output_MLP = MLP([self.hidden_dim*4, self.hidden_dim, self.feat_dim])

        a2v = A2VwD(input_MLP, phn_encoder, spk_encoder, decoder, output_MLP, self.dec_num_layers)

        # T2V
        txt_input_MLP = MLP([60, self.hidden_dim])
        txt_encoder = EncoderRNN(self.hidden_dim, self.seq_len, self.hidden_dim, 
                                 n_layers=1, bidirectional=True, rnn_cell='gru', variable_lengths=True)
        txt_decoder = DecoderRNN(self.hidden_dim*2, self.seq_len, self.hidden_dim*2, n_layers=1, 
                             rnn_cell='gru', bidirectional=True)
        txt_output_MLP = MLP([self.hidden_dim*2, 60])

        t2v = A2V(txt_input_MLP, txt_encoder, txt_decoder, txt_output_MLP, 1)

        # size of discriminator input = num_directions * p_hidden_dim * 2
        discriminator = FCDiscriminator(2*self.hidden_dim*2, self.hidden_dim, self.D_num_layers)

        # the whole model
        self.model = Model(a2v, t2v, discriminator) 

        self.model.to(device)
        # print (next(self.model.parameters()).is_cuda)

    
    ######################################################################
    # The core compute function:
    #
    
    def compute(self, data, mode, optimizer_G=None, optimizer_D=None, result_file=None):
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        total_G_losses = 0
        total_D_losses = 0
        total_r_loss = 0
        total_txt_r_loss = 0
        total_g_loss = 0
        total_pos_spk_loss = 0
        total_neg_spk_loss = 0
        total_paired_loss = 0
        total_d_loss = 0
        total_gp_loss = 0

        total_indices = np.arange(data.n_total_wrds)
        if mode == 'train':
            np.random.shuffle(total_indices)
            if data.num_paired == -1:
                total_indices_paired = np.arange(data.n_total_wrds)
            else:
                total_indices_paired = np.arange(data.num_paired)

        # for i in tqdm(range(data.n_batches-1, data.n_batches)):
        for i in tqdm(range(data.n_batches)):
            indices = total_indices[i*self.batch_size:(i+1)*self.batch_size]
            batch_size = len(indices)
            if mode == 'train':
                indices_paired = np.random.choice(total_indices_paired, batch_size, False)
                indices = np.concatenate((indices, indices_paired), axis=0)

            batch_data, batch_length, batch_order, \
                batch_txt, batch_txt_length, batch_txt_order \
                = data.get_batch_data(indices, mode)

            if mode == 'train':
                ################
                # (1) Update D #
                ################
                for p in self.model.discriminator.parameters(): # reset requires_grad
                    p.requires_grad = True # set to False below in G update

                for _ in range(self.iter_d):
                    self.model.discriminator.zero_grad()

                    phn_hiddens, spk_hiddens, txt_hiddens, r_loss, txt_r_loss, g_loss, d_loss, \
                        gp_loss, pos_spk_loss, neg_spk_loss, paired_loss \
                        = self.model(batch_size, batch_data, batch_length, batch_order, 
                                     batch_txt, batch_txt_length, batch_txt_order, 'train')
                    D_losses = d_loss + 10 * gp_loss
                    D_losses.backward()
                    optimizer_D.step()

                ################
                # (2) Update G #
                ################
                for p in self.model.discriminator.parameters():
                    p.requires_grad = False # to avoid computation
                
                self.model.a2v.zero_grad()
                self.model.t2v.zero_grad()

                phn_hiddens, spk_hiddens, txt_hiddens, r_loss, txt_r_loss, g_loss, d_loss, \
                        gp_loss, pos_spk_loss, neg_spk_loss, paired_loss \
                        = self.model(batch_size, batch_data, batch_length, batch_order, 
                                     batch_txt, batch_txt_length, batch_txt_order, 'train') 
                G_losses = r_loss + txt_r_loss + g_loss + pos_spk_loss + neg_spk_loss + paired_loss
                G_losses.backward()
                optimizer_G.step()
            else:
                phn_hiddens, spk_hiddens, txt_hiddens, r_loss, txt_r_loss, g_loss, d_loss, \
                        gp_loss, pos_spk_loss, neg_spk_loss, paired_loss \
                        = self.model(batch_size, batch_data, batch_length, batch_order, 
                                     batch_txt, batch_txt_length, batch_txt_order, 'test') 
                D_losses = d_loss + 10 * gp_loss
                G_losses = r_loss + txt_r_loss + g_loss + pos_spk_loss + neg_spk_loss + paired_loss

            if mode == 'test' and result_file:
                write_pkl([phn_hiddens.cpu().detach().numpy(), 
                           spk_hiddens.cpu().detach().numpy(), 
                           txt_hiddens.cpu().detach().numpy()], 
                          result_file)

            total_G_losses += G_losses.item()
            total_D_losses += D_losses.item()
            total_r_loss += r_loss.item()
            total_txt_r_loss += txt_r_loss.item()
            total_g_loss += g_loss.item()
            total_pos_spk_loss += pos_spk_loss.item()
            total_neg_spk_loss += neg_spk_loss.item()
            total_paired_loss += paired_loss.item()
            total_d_loss += d_loss.item()
            total_gp_loss += gp_loss.item()

            # del loss, target_output

        return total_G_losses/data.n_batches, total_D_losses/data.n_batches, total_r_loss/data.n_batches, \
            total_txt_r_loss/data.n_batches, total_g_loss/data.n_batches, total_pos_spk_loss/data.n_batches, \
            total_neg_spk_loss/data.n_batches, total_paired_loss/data.n_batches, total_d_loss/data.n_batches, \
            total_gp_loss/data.n_batches


    ######################################################################
    # The scoring function:
    #

    def score(self, data, result_file, trans_file, acc_file=None, use_train=False, train_txt_hiddens=None, train_wrds=None):
        phn_hiddens = np.array([])
        if not use_train:
            txt_hiddens = np.array([])
        with open(result_file, 'rb') as f:
            while True:
                try:
                    hiddens = pickle.load(f)
                    phn = hiddens[0]
                    if not use_train:
                        txt = hiddens[2]
                    if phn_hiddens.size:
                        phn_hiddens = np.concatenate((phn_hiddens, phn), axis=0)
                        if not use_train:
                            txt_hiddens = np.concatenate((txt_hiddens, txt), axis=0)
                    else:
                        phn_hiddens = phn
                        if not use_train:
                            txt_hiddens = txt
                except EOFError:
                    break

        # 
        if not use_train:
            unique_txt_hiddens = []
            unique_phn_wrds = []
            for phn_wrd, hidden in zip(data.phn_wrds, txt_hiddens):
                if not phn_wrd in unique_phn_wrds:
                    unique_phn_wrds.append(phn_wrd)
                    unique_txt_hiddens.append(hidden)
            unique_txt_hiddens = np.array(unique_txt_hiddens)
            unique_wrds = np.array([w[0] for w in unique_phn_wrds])
        else:
            unique_txt_hiddens = train_txt_hiddens
            unique_wrds = train_wrds

        print (phn_hiddens.shape)
        print (unique_txt_hiddens.shape)

        sim_values, sim_wrds = getNN(200, phn_hiddens, unique_txt_hiddens, unique_wrds)
        utt_lens = [len(u) for u in data.wrd_meta]
        print (sum(utt_lens), len(sim_values), len(sim_wrds))
        start = 0
        sim_value_utts = []
        sim_wrd_utts = []
        for l in utt_lens:
            sim_value_utts.append(sim_values[start:start+l])
            sim_wrd_utts.append(sim_wrds[start:start+l])
            start += l
        # sim_value_utts = [sim_values[:utt_lens[0]]]
        # sim_word_utts = [sim_words[:utt_lens[0]]]

        trans = beam_search(sim_value_utts, sim_wrd_utts, data.LM, WIDTH, WEIGHT_LM, trans_file)

        acc = 0
        for w1, w2 in zip(trans, data.wrds):
            if w1 == w2:
                acc += 1
        print (acc / len(trans), acc, len(trans))
        if acc_file:
            with open(acc_file,'a') as f:
                f.write(str(acc / len(trans))+'\n')

        return unique_txt_hiddens, unique_wrds


    ######################################################################
    # The whole training process
    # --------------------------
    #

    def train_iters(self, train_data, test_data, saver, n_epochs, global_step, result_dir, print_every=1):
        # optimizer = optim.Adam(self.model.parameters(), lr=self.init_lr, betas=(0.5, 0.9))
        optimizer_D = optim.Adam(self.model.discriminator.parameters(), lr=self.init_lr, betas=(0.5, 0.9))
        optimizer_G = optim.Adam([{'params': self.model.a2v.parameters()}, {'params': self.model.t2v.parameters()}],
                                  lr=self.init_lr, betas=(0.5, 0.9))

        for epoch in range(global_step+1, global_step+1+n_epochs):
            print ('\nEpoch: ', epoch)
            # Train
            train_G_losses, train_D_losses, train_r_loss, train_txt_r_loss, train_g_loss, train_pos_spk_loss, \
                train_neg_spk_loss, train_paired_loss, train_d_loss, train_gp_loss \
                = self.compute(train_data, 'train', optimizer_D=optimizer_D, optimizer_G=optimizer_G)
        
            self.logger.scalar_summary('train_losses/G_losses', train_G_losses, epoch)
            self.logger.scalar_summary('train_losses/D_losses', train_D_losses, epoch)
            self.logger.scalar_summary('train_losses/r_loss', train_r_loss, epoch)
            self.logger.scalar_summary('train_losses/txt_r_loss', train_txt_r_loss, epoch)
            self.logger.scalar_summary('train_losses/g_loss', train_g_loss, epoch)
            self.logger.scalar_summary('train_losses/pos_spk_loss', train_pos_spk_loss, epoch)
            self.logger.scalar_summary('train_losses/neg_spk_loss', train_neg_spk_loss, epoch)
            self.logger.scalar_summary('train_losses/paired_loss', train_paired_loss, epoch)
            self.logger.scalar_summary('train_losses/d_loss', train_d_loss, epoch)
            self.logger.scalar_summary('train_losses/gp_loss', train_gp_loss, epoch)

            print ('Train -----> G_losses: ',train_G_losses, ' D_losses: ', train_D_losses, 
                   '\nr_loss:       ', train_r_loss, '\ntxt_r_loss:   ', train_txt_r_loss, '\ng_loss:       ',train_g_loss, 
                   '\npos_spk_loss: ', train_pos_spk_loss, '\nneg_spk_loss: ', train_neg_spk_loss, 
                   '\npaired_loss:  ', train_paired_loss, '\nd_loss:       ', train_d_loss, '\ngp_loss:      ', train_gp_loss)

            # Evaluate for train data
            train_G_losses, train_D_losses, train_r_loss, train_txt_r_loss, train_g_loss, train_pos_spk_loss, \
                train_neg_spk_loss, train_paired_loss, train_d_loss, train_gp_loss \
                = self.compute(train_data, 'test', result_file=os.path.join(result_dir, f'result_train_{epoch}.pkl'))
        
            train_txt_hiddens, train_wrds = self.score(train_data,
                                                       os.path.join(result_dir, f'result_train_{epoch}.pkl'), 
                                                       os.path.join(result_dir, f'trans_train_{epoch}_{WIDTH}_{WEIGHT_LM}.txt'),
                                                       os.path.join(result_dir, f'acc_train_{WIDTH}_{WEIGHT_LM}.txt'))

            # Evaluate for eval data
            eval_G_losses, eval_D_losses, eval_r_loss, eval_txt_r_loss, eval_g_loss, eval_pos_spk_loss, \
                eval_neg_spk_loss, eval_paired_loss, eval_d_loss, eval_gp_loss \
                = self.compute(test_data, 'test', result_file=os.path.join(result_dir, f'result_test_{epoch}.pkl'))
        
            self.logger.scalar_summary('eval_losses/G_losses', eval_G_losses, epoch)
            self.logger.scalar_summary('eval_losses/D_losses', eval_D_losses, epoch)
            self.logger.scalar_summary('eval_losses/r_loss', eval_r_loss, epoch)
            self.logger.scalar_summary('eval_losses/txt_r_loss', eval_txt_r_loss, epoch)
            self.logger.scalar_summary('eval_losses/g_loss', eval_g_loss, epoch)
            self.logger.scalar_summary('eval_losses/pos_spk_loss', eval_pos_spk_loss, epoch)
            self.logger.scalar_summary('eval_losses/neg_spk_loss', eval_neg_spk_loss, epoch)
            self.logger.scalar_summary('eval_losses/paired_loss', eval_paired_loss, epoch)
            self.logger.scalar_summary('eval_losses/d_loss', eval_d_loss, epoch)
            self.logger.scalar_summary('eval_losses/gp_loss', eval_gp_loss, epoch)

            print ('Eval -----> G_losses: ',eval_G_losses, ' D_losses: ', eval_D_losses, 
                   '\nr_loss:       ', eval_r_loss, '\ntxt_r_loss:   ', eval_txt_r_loss, '\ng_loss:       ',eval_g_loss, 
                   '\npos_spk_loss: ', eval_pos_spk_loss, '\nneg_spk_loss: ', eval_neg_spk_loss, 
                   '\npaired_loss:  ', eval_paired_loss, '\nd_loss:       ', eval_d_loss, '\ngp_loss:      ', eval_gp_loss)

            _, _ = self.score(test_data, 
                              os.path.join(result_dir, f'result_test_{epoch}.pkl'), 
                              os.path.join(result_dir, f'trans_test_{epoch}_{WIDTH}_{WEIGHT_LM}.txt'),
                              os.path.join(result_dir, f'acc_test_{WIDTH}_{WEIGHT_LM}.txt'),
                              True, train_txt_hiddens, train_wrds)

            # Save model
            state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict()
            }
            name = 'epoch_'+str(epoch)
            saver.save(state, name)


    ######################################################################
    # The whole evaluation process
    # ----------------------------
    #

    def evaluate(self, test_data, train_data, result_dir, print_every=1):
        # Evaluate for train data
        train_G_losses, train_D_losses, train_r_loss, train_txt_r_loss, train_g_loss, train_pos_spk_loss, \
            train_neg_spk_loss, train_paired_loss, train_d_loss, train_gp_loss \
            = self.compute(train_data, 'test', result_file=os.path.join(result_dir, f'result_train.pkl'))

        print ('Train -----> G_losses: ',train_G_losses, ' D_losses: ', train_D_losses, 
               '\nr_loss:       ', train_r_loss, '\ntxt_r_loss:   ', train_txt_r_loss, '\ng_loss:       ',train_g_loss, 
               '\npos_spk_loss: ', train_pos_spk_loss, '\nneg_spk_loss: ', train_neg_spk_loss, 
               '\npaired_loss:  ', train_paired_loss, '\nd_loss:       ', train_d_loss, '\ngp_loss:      ', train_gp_loss)

        train_txt_hiddens, train_wrds = self.score(train_data, 
                                                   os.path.join(result_dir, f'result_train.pkl'), 
                                                   os.path.join(result_dir, f'trans_train_{WIDTH}_{WEIGHT_LM}.txt'),
                                                   None)

        # Evaluate for test data
        eval_G_losses, eval_D_losses, eval_r_loss, eval_txt_r_loss, eval_g_loss, eval_pos_spk_loss, \
            eval_neg_spk_loss, eval_paired_loss, eval_d_loss, eval_gp_loss \
            = self.compute(test_data, 'test', result_file=os.path.join(result_dir, 'result_test.pkl'))

        print ('Eval -----> G_losses: ',eval_G_losses, ' D_losses: ', eval_D_losses, 
               '\nr_loss:       ', eval_r_loss, '\ntxt_r_loss:   ', eval_txt_r_loss, '\ng_loss:       ',eval_g_loss, 
               '\npos_spk_loss: ', eval_pos_spk_loss, '\nneg_spk_loss: ', eval_neg_spk_loss, 
               '\npaired_loss:  ', eval_paired_loss, '\nd_loss:       ', eval_d_loss, '\ngp_loss:      ', eval_gp_loss)

        _, _ = self.score(test_data, 
                          os.path.join(result_dir, f'result_test.pkl'), 
                          os.path.join(result_dir, f'trans_test_{WIDTH}_{WEIGHT_LM}.txt'),
                          None,
                          True, train_txt_hiddens, train_wrds)