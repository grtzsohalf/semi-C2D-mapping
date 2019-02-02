from __future__ import unicode_literals, print_function, division
import random
import os
import time
from datetime import datetime 
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch import autograd
import torch.nn.functional as F

from A2VwD import A2VwD, A2V, Model
from seq2seq import EncoderRNN, DecoderRNN
from utils import weight_init, repackage_hidden, write_pkl
from utils import thresh_dist_segment_eval, r_val_eval, print_progress
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
                 enc_num_layers, dec_num_layers, D_num_layers, dropout_rate, iter_d, log_dir):

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

        # self.mode = mode
        if self.mode == 'train':
            self.logger = Logger(os.path.join(log_dir, 'loss_'+loss_type))

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
                output = self.hidden[-1](x)
                return output

        # A2V
        input_MLP = MLP([self.feat_dim, self.hidden_dim, self.hidden_dim])
        phn_encoder = EncoderRNN(self.feat_dim, self.seq_len, self.hidden_dim, 
                                 input_dropout_p=self.dropout_rate, dropout_p=self.dropout_rate,
                                 n_layers=self.enc_num_layers, bidirectional=True, rnn_cell='gru', variable_lengths=True)
        spk_encoder = EncoderRNN(self.feat_dim, self.seq_len, self.hidden_dim, 
                                 input_dropout_p=self.dropout_rate, dropout_p=self.dropout_rate,
                                 n_layers=self.enc_num_layers, bidirectional=True, rnn_cell='gru', variable_lengths=True)
        decoder = DecoderRNN(self.hidden_dim, self.seq_len, self.hidden_dim, n_layers=self.dec_num_layers, 
                             rnn_cell='gru', input_dropout_p=self.dropout_rate, dropout_p=self.dropout_rate)
        output_MLP = MLP([self.hidden_dim, self.hidden_dim, self.feat_dim])

        a2v = A2VwD(input_MLP, phn_encoder, spk_encoder, decoder, output_MLPi, self.mode, self.decoder_num_layers)

        # T2V
        txt_input_MLP = MLP([self.feat_dim, self.hidden_dim])
        txt_encoder = EncoderRNN(self.feat_dim, self.seq_len, self.hidden_dim, 
                                 input_dropout_p=self.dropout_rate, dropout_p=self.dropout_rate,
                                 n_layers=1, bidirectional=True, rnn_cell='gru', variable_lengths=True)
        txt_decoder = DecoderRNN(self.hidden_dim, self.seq_len, self.hidden_dim, n_layers=1, 
                             rnn_cell='gru', input_dropout_p=self.dropout_rate, dropout_p=self.dropout_rate)
        txt_output_MLP = MLP([self.hidden_dim, self.feat_dim])

        t2v = A2V(txt_input_MLP, txt_encoder, txt_decoder, txt_output_MLP, self.mode, self.decoder_num_layers)

        # size of discriminator input = num_directions * p_hidden_dim * 2
        discriminator = Discriminator(2*self.hidden_dim*2, self.hidden_dim*2, self.D_num_layers)

        # the whole model
        self.model = Model(a2v, t2v, discriminator) 

        self.model.to(device)
        # print (next(self.model.parameters()).is_cuda)


    ######################################################################
    # The core compute function:
    #
    
    def compute(self, data, mode, optimizer_G=None, optimizer_D=None, result_dir=None):
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0  # Reset every print_every
        r_loss = 0  # Reset every print_every
        cpc_loss = 0  # Reset every print_every

        # Distance container for Evaluation
        distStack0   = []
        distStack1   = []
        batchLenStack = []

        total_indices = np.arange(data.n_utts)
        if mode == 'train':
            np.random.shuffle(total_indices)

        for i in tqdm(range(data.n_batches)):
            indices = total_indices[i*self.batch_size:(i+1)*self.batch_size]
            batch_size = len(indices)

            hidden = self.model.encoder.init_hidden(batch_size)
            hidden = repackage_hidden(hidden)

            batch_target, batch_length, neg_shift = data.get_batch_data(indices, self.pred_step, self.pred_neg_num)
            # print (batch_target.device, batch_length.device, neg_shift.device)

            if mode == 'train':
                optimizer.zero_grad()
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


            target_outputs, hidden, rnn_hs, dropped_rnn_hs, distances, reconstruction_loss \
                = self.model(batch_target, batch_length, neg_shift, hidden, self.loss_type)
            loss = reconstruction_loss + 

            if mode == 'test' and result_dir:
                write_pkl([target_outputs.cpu().detach().numpy(), 
                           distances[0].cpu().detach().numpy(), 
                           distances[1].cpu().detach().numpy()], 
                          os.path.join(result_dir, 'result.pkl'))

            if mode == 'evaluate' and result_dir:
                write_pkl([target_outputs.cpu().detach().numpy(), 
                           distances[0].cpu().detach().numpy(), 
                           distances[1].cpu().detach().numpy()], 
                          os.path.join(result_dir, 'result.pkl'))
                distStack0.append(distances[0])
                distStack1.append(distances[1])
                batchLenStack.append(batch_length)
                #print(batch_length)


            # Activiation Regularization
            loss = loss + sum(
                self.AR_scale * dropped_rnn_h.pow(2).mean()
                for dropped_rnn_h in dropped_rnn_hs[-1:]
            )

            total_loss += loss.item()
            r_loss += reconstruction_loss.item()

            # del loss, target_outputs, hidden, rnn_hs, dropped_rnn_hs, distances, reconstruction_loss, CPC_loss

        # return distances for test (Evaluation)
        if mode == 'evaluate' and result_dir:
            return total_loss/data.n_batches, r_loss/data.n_batches, cpc_loss/data.n_batches, (distStack0, distStack1), batchLenStack

        return total_loss/data.n_batches, r_loss/data.n_batches, 


    ######################################################################
    # The whole training process
    # --------------------------
    #

    def train_iters(self, train_data, eval_data, saver, n_epochs, global_step, print_every=1):
        # optimizer = optim.Adam(self.model.parameters(), lr=self.init_lr, betas=(0.5, 0.9))
        optimizer_D = optim.Adam(self.model.discriminator.parameters(), lr=self.init_lr, betas=(0.5, 0.9))
        optimizer_G = optim.Adam([self.model.phn_encoder.parameters(), self.model.spk_encoder.parameters(),
                                  self.model.decoder.parameters(), lr=self.init_lr, betas=(0.5, 0.9))

        for epoch in range(global_step+1, global_step+1+n_epochs):
            print ('Epoch: ', epoch)

            # Train
            train_total_loss, train_r_loss, train_cpc_loss = self.compute(train_data, 'train', 
                                                                          optimizer_D=optimizer_D, optimizer_G = optimizer_G)
        
            self.logger.scalar_summary('train_losses/total_loss', train_total_loss, epoch)
            self.logger.scalar_summary('train_losses/r_loss', train_r_loss, epoch)
            self.logger.scalar_summary('train_losses/cpc_loss', train_cpc_loss, epoch)
            #with open(self.log_train_file, 'a') as fo:
            #    fo.write(str(epoch)+' '+str(train_total_loss)+' '+str(train_r_loss)+' '+str(train_cpc_loss)+'\n')

            print ('Train -----> total_loss: ',train_total_loss,' r_loss: ',train_r_loss,' cpc_loss: ',train_cpc_loss)

            # Evaluate
            test_total_loss, test_r_loss, test_cpc_loss = self.compute(eval_data, 'test')
        
            self.logger.scalar_summary('test_losses/total_loss', test_total_loss, epoch)
            self.logger.scalar_summary('test_losses/r_loss', test_r_loss, epoch)
            self.logger.scalar_summary('test_losses/cpc_loss', test_cpc_loss, epoch)
            #with open(self.log_test_file, 'a') as fo:
            #    fo.write(str(epoch)+' '+str(test_total_loss)+' '+str(test_r_loss)+' '+str(test_cpc_loss)+'\n')

            print ('Eval -----> total_loss: ',test_total_loss,' r_loss: ', test_r_loss,' cpc_loss: ',test_cpc_loss,'\n')

            # Save model
            state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            name = 'epoch_'+str(epoch)
            saver.save(state, name)


    ######################################################################
    # The whole evaluation process
    # ----------------------------
    #

    def evaluate(self, eval_data, result_dir, th, criteria_idxs, tolerance_window=2, eval_mode='phn', print_every=1):
        test_total_loss, test_r_loss, test_cpc_loss, test_distances, test_batch_lens = self.compute(eval_data, 'evaluate', result_dir=result_dir)
        print ('Eval -----> total_loss: ',test_total_loss,' r_loss: ',test_r_loss,' cpc_loss: ',test_cpc_loss,'\n')

        # test_distances[0]: cforgetgate, [layer#, max seq_len, batch_size, hidden_size]
        # test_distances[1]: cingate, [layer#, max seq_len, batch_size, hidden_size]
        
        output_msg = ''
        recall_list = [ [ [] for i in range(len(th)) ] for i in range(len(criteria_idxs)) ]
        precision_list = [ [ [] for i in range(len(th)) ] for i in range(len(criteria_idxs)) ]
        
        for i in range(len(th)):
            recall_list.append([])
            precision_list.append([])

        if eval_mode == 'phn':
            meta_data = eval_data.phn_meta
        elif eval_mode == 'slb':
            meta_data = eval_data.slb_meta
        elif eval_mode == 'wrd':
            meta_data = eval_data.wrd_meta
        else:
            print("eval_mode should be phn|slb|wrd")
            exit()

        for i in range(eval_data.n_utts):
            utt_len = test_batch_lens[i // self.batch_size][i%self.batch_size]
            scores0 = test_distances[0][i // self.batch_size][:,:utt_len,i%self.batch_size,:] # (num_layers, length, batch_size, hidden_size)
            scores0 = scores0.cpu().detach().numpy()
            
            bounds_list = [[ info[2]-meta_data[i][0][2] for info in meta_data[i] ]]
            for c_idx, c_v in enumerate(criteria_idxs):
                for th_idx, t_f in enumerate(th):
                    batch_recall_list, batch_precision_list = \
                        thresh_dist_segment_eval(scores0, bounds_list, c_v, tolerance_window, t_f)
                    
                    if th_idx == 5:
                        print_progress(i+1, eval_data.n_utts, \
                                        batch_precision_list[0], \
                                        batch_recall_list[0], output_msg)

                    precision_list[c_idx][th_idx] += batch_precision_list
                    recall_list[c_idx][th_idx] += batch_recall_list

        r_val_list = []
        for c_idx, c_v in enumerate(criteria_idxs):
            for t_idx, t in enumerate(th):
                precision = sum(precision_list[c_idx][t_idx]) / len(precision_list[c_idx][t_idx])
                recall = sum(recall_list[c_idx][t_idx]) / len(recall_list[c_idx][t_idx])
                recall *= 100
                precision *= 100
                if recall == 0. or precision == 0.:
                    f_score = -1.
                    r_val = -1.
                else:
                    f_score = (2 * precision * recall) / (precision + recall)
                    r_val = r_val_eval(precision, recall)
                r_val_list.append(r_val)
                print('{:d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'. \
                    format(c_v, t, precision, recall, f_score, r_val))
        
        print('')
        print('The best r_val is: {:.4f}'.format(max(r_val_list)))
    
