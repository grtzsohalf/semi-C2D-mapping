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

# from WGAN_GP import FCDiscriminator
from model import Model
from seq2seq import EncoderRNN, DecoderRNN
from utils import weight_init, write_pkl, Logger, getNN, beam_search
from torch.optim.lr_scheduler import StepLR


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################
# Solver
# ======
#
# Set all parameters and construct a graph for the model.
#

class Solver:
    def __init__(self, init_lr, batch_size, seq_len, feat_dim, hidden_dim, 
                 enc_num_layers, dec_num_layers, dropout_rate, 
                 weight_r, weight_txt_ce, weight_x,
                 weight_pos_paired, weight_neg_paired,
                 top_NN, width, weight_LM, log_dir, mode, unit_type):

        self.init_lr = init_lr
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.enc_num_layers = enc_num_layers
        self.dec_num_layers = dec_num_layers
        # self.D_num_layers = D_num_layers
        self.dropout_rate = dropout_rate
        # self.iter_d = iter_d

        self.weight_r = weight_r
        self.weight_txt_ce = weight_txt_ce
        self.weight_x = weight_x
        # self.weight_g = weight_g
        # self.weight_pos_spk = weight_pos_spk
        # self.weight_neg_spk = weight_neg_spk
        self.weight_pos_paired = weight_pos_paired
        self.weight_neg_paired = weight_neg_paired
        # self.weight_d = weight_d
        # self.weight_gp = weight_gp

        self.top_NN = top_NN
        self.width = width
        self.weight_LM = weight_LM

        self.mode = mode
        if self.mode == 'train':
            self.logger = Logger(os.path.join(log_dir))
        self.unit_type = unit_type

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

    def build_model(self, neg_thres):

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
        aud_input_MLP = MLP([self.feat_dim, self.hidden_dim, self.hidden_dim])
        phn_encoder = EncoderRNN(self.hidden_dim, self.seq_len, self.hidden_dim, 
                                 input_dropout_p=self.dropout_rate, dropout_p=self.dropout_rate,
                                 n_layers=self.enc_num_layers, bidirectional=True, rnn_cell='gru', variable_lengths=True)
        spk_encoder = EncoderRNN(self.hidden_dim, self.seq_len, self.hidden_dim, 
                                 input_dropout_p=self.dropout_rate, dropout_p=self.dropout_rate,
                                 n_layers=self.enc_num_layers, bidirectional=True, rnn_cell='gru', variable_lengths=True)
        aud_decoder = DecoderRNN(self.hidden_dim*4, self.seq_len, self.hidden_dim*4, n_layers=self.dec_num_layers, 
                             rnn_cell='gru', bidirectional=True, input_dropout_p=self.dropout_rate, dropout_p=self.dropout_rate)
        aud_output_MLP = MLP([self.hidden_dim*4, self.hidden_dim, self.feat_dim])

        # a2v = A2VwD(input_MLP, phn_encoder, spk_encoder, decoder, output_MLP, self.dec_num_layers)

        # T2V
        if self.unit_type == 'char':
            txt_feat_dim = 27
        else:
            txt_feat_dim = 60
        txt_input_MLP = MLP([txt_feat_dim, self.hidden_dim])
        txt_encoder = EncoderRNN(self.hidden_dim, self.seq_len, self.hidden_dim, 
                                 n_layers=1, bidirectional=True, rnn_cell='gru', variable_lengths=True)
        txt_decoder = DecoderRNN(self.hidden_dim*2, self.seq_len, self.hidden_dim*2, n_layers=1, 
                             rnn_cell='gru', bidirectional=True)
        txt_output_MLP = MLP([self.hidden_dim*2, txt_feat_dim])

        # t2v = A2V(txt_input_MLP, txt_encoder, txt_decoder, txt_output_MLP, 1)

        # size of discriminator input = num_directions * p_hidden_dim * 2
        # discriminator = FCDiscriminator(2*self.hidden_dim*2, self.hidden_dim, self.D_num_layers)

        # the whole model
        self.model = Model(aud_input_MLP, phn_encoder, spk_encoder, aud_decoder, aud_output_MLP, self.dec_num_layers, 
                 txt_input_MLP, txt_encoder, txt_decoder, txt_output_MLP, 1, neg_thres) 

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

        total_losses = 0
        # total_G_losses = 0
        # total_D_losses = 0
        total_r_loss = 0
        total_txt_ce_loss = 0
        total_x_loss = 0
        # total_g_loss = 0
        # total_pos_spk_loss = 0
        # total_neg_spk_loss = 0
        total_pos_paired_loss = 0
        total_neg_paired_loss = 0
        # total_d_loss = 0
        # total_gp_loss = 0

        total_phn_hiddens = np.array([])
        total_txt_hiddens = np.array([])

        total_indices = np.arange(data.n_total_wrds)
        if mode == 'train':
            np.random.shuffle(total_indices)

        if data.num_paired == -1:
            total_indices_paired = np.arange(data.n_total_wrds)
        else:
            total_indices_paired = np.arange(min(data.num_paired, data.n_total_wrds))

        # for i in tqdm(range(data.n_batches-1, data.n_batches)):
        for i in tqdm(range(data.n_batches)):
            indices = total_indices[i*self.batch_size:(i+1)*self.batch_size]
            batch_size = len(indices)
            indices_paired = np.random.choice(total_indices_paired, batch_size, False)
            indices = np.concatenate((indices, indices_paired), axis=0)

            batch_data, batch_length, batch_order, \
                batch_txt, batch_txt_length, batch_txt_order, batch_txt_labels \
                = data.get_batch_data(indices, self.unit_type)

            if mode == 'train':
                ################
                # (1) Update D #
                ################
                # for p in self.model.discriminator.parameters(): # reset requires_grad
                    # p.requires_grad = True # set to False below in G update

                # for _ in range(self.iter_d):
                    # self.model.discriminator.zero_grad()

                    # phn_hiddens, spk_hiddens, txt_hiddens, r_loss, txt_r_loss, g_loss, d_loss, \
                        # gp_loss, pos_spk_loss, neg_spk_loss, pos_paired_loss, neg_paired_loss \
                        # = self.model(batch_size, batch_data, batch_length, batch_order, 
                                     # batch_txt, batch_txt_length, batch_txt_order, 'train')
                    # D_losses = self.weight_d * d_loss + self.weight_gp * gp_loss
                    # D_losses.backward()
                    # optimizer_D.step()

                ################
                # (2) Update G #
                ################
                # for p in self.model.discriminator.parameters():
                    # p.requires_grad = False # to avoid computation
                
                self.model.zero_grad()

                phn_hiddens, spk_hiddens, txt_hiddens, r_loss, txt_ce_loss, x_loss, \
                        pos_paired_loss, neg_paired_loss \
                        = self.model(batch_size, batch_data, batch_length, batch_order, 
                                     batch_txt, batch_txt_length, batch_txt_order, batch_txt_labels) 
                # G_losses = self.weight_r * r_loss + self.weight_txt_r * txt_r_loss + self.weight_g * g_loss \
                    # + self.weight_pos_spk * pos_spk_loss + self.weight_neg_spk * neg_spk_loss \
                    # + self.weight_pos_paired * pos_paired_loss + self.weight_neg_paired * neg_paired_loss
                losses = self.weight_r * r_loss + self.weight_txt_ce * txt_ce_loss + self.weight_x * x_loss \
                    + self.weight_pos_paired * pos_paired_loss + self.weight_neg_paired * neg_paired_loss
                losses.backward()
                optimizer_G.step()
            else:
                phn_hiddens, spk_hiddens, txt_hiddens, r_loss, txt_ce_loss, x_loss, \
                        pos_paired_loss, neg_paired_loss \
                        = self.model(batch_size, batch_data, batch_length, batch_order, 
                                     batch_txt, batch_txt_length, batch_txt_order, batch_txt_labels) 
                # D_losses = self.weight_d * d_loss + self.weight_gp * gp_loss
                # G_losses = self.weight_r * r_loss + self.weight_txt_r * txt_r_loss + self.weight_g * g_loss \
                    # + self.weight_pos_spk * pos_spk_loss + self.weight_neg_spk * neg_spk_loss \
                    # + self.weight_pos_paired * pos_paired_loss + self.weight_neg_paired * neg_paired_loss
                losses = self.weight_r * r_loss + self.weight_txt_ce * txt_ce_loss + self.weight_x * x_loss \
                    + self.weight_pos_paired * pos_paired_loss + self.weight_neg_paired * neg_paired_loss

            if mode == 'test' and result_file:
                write_pkl([phn_hiddens.cpu().detach().numpy(), 
                           spk_hiddens.cpu().detach().numpy(), 
                           txt_hiddens.cpu().detach().numpy()], 
                          result_file)
            if mode == 'test':
                if total_phn_hiddens.size:
                    total_phn_hiddens = np.concatenate((total_phn_hiddens, phn_hiddens.cpu().detach().numpy()), axis=0)
                    total_txt_hiddens = np.concatenate((total_txt_hiddens, txt_hiddens.cpu().detach().numpy()), axis=0)
                else:
                    total_phn_hiddens = phn_hiddens.cpu().detach().numpy()
                    total_txt_hiddens = txt_hiddens.cpu().detach().numpy()

            total_losses += losses.item()
            # total_G_losses += G_losses.item()
            # total_D_losses += D_losses.item()
            total_r_loss += r_loss.item()
            total_txt_ce_loss += txt_ce_loss.item()
            total_x_loss += x_loss.item()
            # total_g_loss += g_loss.item()
            total_pos_paired_loss += pos_paired_loss.item()
            total_neg_paired_loss += neg_paired_loss.item()
            # total_d_loss += d_loss.item()
            # total_gp_loss += gp_loss.item()

            # del loss, target_output

        return total_losses/data.n_batches, total_r_loss/data.n_batches, total_txt_ce_loss/data.n_batches, \
            total_x_loss/data.n_batches, total_pos_paired_loss/data.n_batches, total_neg_paired_loss/data.n_batches, \
            total_phn_hiddens, total_txt_hiddens


    ######################################################################
    # The scoring function:
    #

    def score(self, data, phn_hiddens, txt_hiddens, wrds, trans_file, acc_file=None):
        # print (phn_hiddens.shape)
        # print (txt_hiddens.shape)

        sim_values, sim_wrds = getNN(self.top_NN, phn_hiddens, txt_hiddens, wrds)

        # indices = []
        # for sim_w_utt, w in zip(sim_wrds, data.wrds):
            # for i, sim_w in enumerate(sim_w_utt):
                # if sim_w == w or i == len(sim_w_utt)-1:
                    # indices.append(i)
                    # break
        # print (indices)


        utt_lens = [len(u) for u in data.wrd_meta]
        # print (sum(utt_lens), len(sim_values), len(sim_wrds))
        start = 0
        sim_value_utts = []
        sim_wrd_utts = []
        for l in utt_lens:
            sim_value_utts.append(sim_values[start:start+l])
            sim_wrd_utts.append(sim_wrds[start:start+l])
            start += l
        # sim_value_utts = [sim_values[:utt_lens[0]]]
        # sim_word_utts = [sim_words[:utt_lens[0]]]

        trans = beam_search(sim_value_utts, sim_wrd_utts, data.LM, self.width, self.weight_LM, trans_file)

        acc = 0
        for w1, w2 in zip(trans, data.wrds):
            if w1 == w2:
                acc += 1
        print (acc / len(trans), acc, len(trans))
        if acc_file:
            with open(acc_file,'a') as f:
                f.write(str(acc / len(trans))+'\n')

        return


    ######################################################################
    # The whole training process
    # --------------------------
    #

    def train_iters(self, train_data, test_data, saver, n_epochs, global_step, result_dir, print_every=1):
        # optimizer = optim.Adam(self.model.parameters(), lr=self.init_lr, betas=(0.5, 0.9))
        # optimizer_D = optim.Adam(self.model.discriminator.parameters(), lr=self.init_lr, betas=(0.5, 0.9))
        optimizer_G = optim.Adam([{'params': self.model.parameters()}],
                                  lr=self.init_lr, betas=(0.5, 0.95))

        for epoch in range(global_step+1, global_step+1+n_epochs):
            start_time = time.time()
            print ('\nEpoch: ', epoch)
            # Train
            print (' ')
            train_losses, train_r_loss, train_txt_ce_loss, train_x_loss, \
                train_pos_paired_loss, train_neg_paired_loss, _, _ \
                = self.compute(train_data, 'train', optimizer_G=optimizer_G)
        
            self.logger.scalar_summary('train_losses/losses', train_losses, epoch)
            self.logger.scalar_summary('train_losses/r_loss', train_r_loss, epoch)
            self.logger.scalar_summary('train_losses/txt_ce_loss', train_txt_ce_loss, epoch)
            self.logger.scalar_summary('train_losses/x_loss', train_x_loss, epoch)
            self.logger.scalar_summary('train_losses/pos_paired_loss', train_pos_paired_loss, epoch)
            self.logger.scalar_summary('train_losses/neg_paired_loss', train_neg_paired_loss, epoch)

            print ('Train -----> losses: ',train_losses, 
                   '\nr_loss:          ', train_r_loss, '\ntxt_r_loss:      ', train_txt_ce_loss, '\nx_loss:          ', train_x_loss,
                   '\npos_paired_loss: ', train_pos_paired_loss, '\nneg_paired_loss: ', train_neg_paired_loss)

            # Evaluate for train data
            train_losses, train_r_loss, train_txt_ce_loss, train_x_loss, \
                train_pos_paired_loss, train_neg_paired_loss, train_phn_hiddens, train_txt_hiddens \
                = self.compute(train_data, 'test')#, result_file=os.path.join(result_dir, f'result_train_{epoch}.pkl'))
        
            unique_train_txt_hiddens = []
            unique_train_phn_wrds = []
            for phn_wrd, hidden in zip(train_data.phn_wrds, train_txt_hiddens):
                if not phn_wrd in unique_train_phn_wrds:
                    unique_train_phn_wrds.append(phn_wrd)
                    unique_train_txt_hiddens.append(hidden)
            unique_train_txt_hiddens = np.array(unique_train_txt_hiddens)
            unique_train_wrds = np.array([w[0] for w in unique_train_phn_wrds])

            self.score(train_data, train_phn_hiddens, unique_train_txt_hiddens, unique_train_wrds,
                       os.path.join(result_dir, f'trans_train_{epoch}_{self.top_NN}_{self.width}_{self.weight_LM}.txt'),
                       os.path.join(result_dir, f'acc_train_{self.top_NN}_{self.width}_{self.weight_LM}.txt'))


            # Evaluate for eval data
            print (' ')
            eval_losses, eval_r_loss, eval_txt_ce_loss, eval_x_loss, \
                eval_pos_paired_loss, eval_neg_paired_loss, eval_phn_hiddens, _ \
                = self.compute(test_data, 'test')#, result_file=os.path.join(result_dir, f'result_test_{epoch}.pkl'))
        
            self.logger.scalar_summary('eval_losses/losses', eval_losses, epoch)
            self.logger.scalar_summary('eval_losses/r_loss', eval_r_loss, epoch)
            self.logger.scalar_summary('eval_losses/txt_ce_loss', eval_txt_ce_loss, epoch)
            self.logger.scalar_summary('eval_losses/x_loss', eval_x_loss, epoch)
            self.logger.scalar_summary('eval_losses/pos_paired_loss', eval_pos_paired_loss, epoch)
            self.logger.scalar_summary('eval_losses/neg_paired_loss', eval_neg_paired_loss, epoch)

            print ('Eval -----> losses: ',eval_losses, 
                   '\nr_loss:          ', eval_r_loss, '\ntxt_ce_loss:      ', eval_txt_ce_loss, '\nx_loss:          ', eval_x_loss,
                   '\npos_paired_loss: ', eval_pos_paired_loss, '\nneg_paired_loss: ', eval_neg_paired_loss)

            self.score(test_data, eval_phn_hiddens, unique_train_txt_hiddens, unique_train_wrds,
                       os.path.join(result_dir, f'trans_test_{epoch}_{self.top_NN}_{self.width}_{self.weight_LM}.txt'),
                       os.path.join(result_dir, f'acc_test_{self.top_NN}_{self.width}_{self.weight_LM}.txt'))

            # Save model
            state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict()
            }
            name = 'epoch_'+str(epoch)
            saver.save(state, name)

            print (time.time() - start_time, '\n')


    ######################################################################
    # The whole evaluation process
    # ----------------------------
    #

    def evaluate(self, test_data, train_data, result_dir, model_name, print_every=1):
        # Evaluate for train data
        print (' ')
        train_losses, train_r_loss, train_txt_ce_loss, train_x_loss, \
            train_pos_paired_loss, train_neg_paired_loss, train_phn_hiddens, train_txt_hiddens \
            = self.compute(train_data, 'test', result_file=os.path.join(result_dir, f'result_train_{model_name}.pkl'))

        print ('Train -----> losses: ',train_losses, 
               '\nr_loss:          ', train_r_loss, '\ntxt_ce_loss:      ', train_txt_ce_loss, '\nx_loss:          ', train_x_loss,
               '\npos_paired_loss: ', train_pos_paired_loss, '\nneg_paired_loss: ', train_neg_paired_loss)
    
        unique_train_txt_hiddens = []
        unique_train_phn_wrds = []
        for phn_wrd, hidden in zip(train_data.phn_wrds, train_txt_hiddens):
            if not phn_wrd in unique_train_phn_wrds:
                unique_train_phn_wrds.append(phn_wrd)
                unique_train_txt_hiddens.append(hidden)
        unique_train_txt_hiddens = np.array(unique_train_txt_hiddens)
        unique_train_wrds = np.array([w[0] for w in unique_train_phn_wrds])

        self.score(train_data, train_phn_hiddens, unique_train_txt_hiddens, unique_train_wrds,
                   os.path.join(result_dir, f'trans_train_{self.top_NN}_{self.width}_{self.weight_LM}.txt'))

        # Evaluate for test data
        print (' ')
        eval_losses, eval_r_loss, eval_txt_ce_loss, eval_x_loss, \
            eval_pos_paired_loss, eval_neg_paired_loss, eval_phn_hiddens, _ \
            = self.compute(test_data, 'test', result_file=os.path.join(result_dir, f'result_test_{model_name}.pkl'))

        print ('Eval -----> losses: ',eval_losses, 
               '\nr_loss:          ', eval_r_loss, '\ntxt_ce_loss:      ', eval_txt_ce_loss, '\nx_loss:          ', eval_x_loss,
               '\npos_paired_loss: ', eval_pos_paired_loss, '\nneg_paired_loss: ', eval_neg_paired_loss)

        self.score(test_data, eval_phn_hiddens, unique_train_txt_hiddens, unique_train_wrds,
                   os.path.join(result_dir, f'trans_test_{self.top_NN}_{self.width}_{self.weight_LM}.txt'))
