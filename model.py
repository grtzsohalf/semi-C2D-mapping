import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import mask_with_length


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################
# Audio2Vec
# =========
#


class Model(nn.Module):

    def __init__(self, a2v, t2v, discriminator):
        super(Model, self).__init__()
        self.a2v = a2v
        self.t2v = t2v
        self.discriminator = discriminator

    def flatten_parameters(self):
        self.a2v.flatten_parameters()
        self.t2v.flatten_parameters()

    def compute_reconstruction_loss(self, x, y, lengths=None):
        # generate mask by lengths
        # mask = torch.zeros_like(target_feats)
        # for i, l in enumerate(target_lengths):
            # mask[i][:l] = torch.ones(l)
        if lengths:
            x, _ = mask_with_length(x, lengths)
            y, _ = mask_with_length(y, lengths)
        # criterion = nn.MSELoss(reduction='none')
        # MSE_loss = criterion(target_feats, reconstructed_target)
        MSE_loss = (x - y) ** 2
        return MSE_loss.mean()

    def compute_GAN_losses(self, target, pos, neg):
        # using WGAN-GP
        # pairs of (target, pos) -> real
        # pairs of (target, neg) -> fake
        target_first = target[:len(target)/2]
        target_last = target[len(target)/2:]
        pos = pos[:len(pos)/2]
        neg = neg[len(neg)/2:]

        pos_concat = torch.cat((target_first, pos), -1)
        neg_concat = torch.cat((target_last, neg), -1)
        pos_score = self.discriminator(pos_concat)
        neg_score = self.discriminator(neg_concat)
        generation_loss = ((pos_score - neg_score) ** 2).mean()
        discrimination_loss = - generation_loss

        GP_loss = calc_gradient_penalty(self.discriminator, pos_concat, neg_concat)
        return generation_loss, discrimination_loss, GP_loss

    def compute_speaker_losses(self, target, pos, neg):
        # pairs of (target, pos) -> as close as possible
        # pairs of (target, neg) -> far from each other to an extent
        pos_speaker_loss = ((target - pos) ** 2).mean()
        neg_speaker_loss = torch.mean(torch.max(0.01 - torch.norm(target - neg, dim=-1), 0.)) 
        return pos_speaker_loss, neg_speaker_loss

    def forward(self, batch_size, feats, lengths, orders, txt_feats, txt_lengths, txt_orders, mode):
        if mode == 'train':
            reconstructed, target_phn_hiddens, paired_phn_hiddens, pos_phn_hiddens, neg_phn_hiddens, \
                target_spk_hiddens, paired_spk_hiddens, pos_spk_hiddens, neg_spk_hiddens \
                = self.a2v(batch_size, feats, lengths, orders, mode)
            txt_reconstructed, txt_hiddens, txt_paired_hiddens \
                = self.t2v(batch_size, txt_feats, txt_lengths, txt_orders, mode)
        else:
            reconstructed, target_phn_hiddens, pos_phn_hiddens, neg_phn_hiddens, \
                target_spk_hiddens, pos_spk_hiddens, neg_spk_hiddens \
                = self.a2v(batch_size, feats, lengths, orders, mode)
            txt_reconstructed, txt_hiddens \
                = self.t2v(batch_size, txt_feats, txt_lengths, txt_orders, mode)

        feats = feats[orders]
        txt_feats = txt_feats[txt_orders]
        lengths = lengths[orders]
        txt_lengths = txt_lengths[txt_orders]
        reconstruction_loss = self.compute_reconstruction_loss(reconstructed, feats, lengths) 
        txt__loss = self.compute__loss(txt_reconstructed, txt_feats, txt_lengths) 
        generation_loss, discrimination_loss, GP_loss \
            = self.compute_GAN_losses(target_phn_hiddens, pos_phn_hiddens, neg_phn_hiddens)
        pos_speaker_loss, neg_speaker_loss = \
            self.compute_speaker_losses(target_spk_hiddens, pos_spk_hiddens, neg_spk_hiddens)
        if mode == 'train':
            paired_loss = self.compute_reconstruction_loss(paired_phn_hiddens, txt_paired_hiddens)
        else:
            paired_loss = self.compute_reconstruction_loss(target_phn_hiddens, txt_hiddens)
        return target_phn_hiddens, target_spk_hiddens, txt_hiddens, reconstruction_loss, generation_loss, discrimination_loss, \
            GP_loss, pos_speaker_loss, neg_speaker_loss, paired_loss


class A2VwD(nn.Module):

    def __init__(self, input_MLP, phn_encoder, spk_encoder, decoder, output_MLP, decoder_num_layers):
        super(A2VwD, self).__init__()
        self.input_MLP = input_MLP
        self.phn_encoder = phn_encoder
        self.spk_encoder = spk_encoder
        self.decoder = decoder
        self.output_MLP = output_MLP
        self.decoder_num_layers = decoder_num_layers

    def flatten_parameters(self):
        self.phn_encoder.rnn.flatten_parameters()
        self.spk_encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, batch_size, feats, lengths, orders, mode):
        # encoder
        emb_feats = self.input_MLP(feats)
        # split phn_hiddens
        _, phn_hiddens = self.phn_encoder(emb_feats, lengths)
        phn_hiddens = phn_hiddens[-2:, :, :].transpose(0, 1)
        phn_hiddens = phn_hiddens[orders]
        target_phn_hiddens = phn_hiddens[:batch_size, :, :]
        if mode == 'train':
            paired_phn_hiddens = phn_hiddens[batch_size:batch_size*2, :, :]
            pos_phn_hiddens = phn_hiddens[batch_size*2:batch_size*3, :, :]
            neg_phn_hiddens = phn_hiddens[batch_size*3:, :, :]
        else:
            pos_phn_hiddens = phn_hiddens[batch_size:batch_size*2, :, :]
            neg_phn_hiddens = phn_hiddens[batch_size*2:, :, :]

        # split spk_hiddens
        _, spk_hiddens = self.spk_encoder(emb_feats, lengths)
        spk_hiddens = spk_hiddens[-2:, :, :].transpose(0, 1)
        spk_hiddens = spk_hiddens[orders]
        target_spk_hiddens = spk_hiddens[:batch_size, :, :]
        if mode == 'train':
            paired_spk_hiddens = spk_hiddens[batch_size:batch_size*2, :, :]
            pos_spk_hiddens = spk_hiddens[batch_size*2:batch_size*3, :, :]
            neg_spk_hiddens = spk_hiddens[batch_size*3:, :, :]
        else:
            pos_spk_hiddens = spk_hiddens[batch_size:batch_size*2, :, :]
            neg_spk_hiddens = spk_hiddens[batch_size*2:, :, :]

        # construct decoder hiddens
        target_concat_hiddens = torch.cat((target_phn_hiddens, target_spk_hiddens), -1).transpose(0, 1)
        state_zeros = torch.zeros_like(target_concat_hiddens, device=device) 
        decoder_init_state = torch.tensor(target_concat_hiddens, device=device)
        for i in range(self.decoder_num_layers-1):
            decoder_init_state = torch.cat((decoder_init_state, state_zeros), 0)

        # decoder
        begin_zeros = torch.zeros_like(emb_feats[:batch_size, :, :]) 
        reconstructed, _, _ = self.decoder(inputs=begin_zeros, encoder_hidden=decoder_init_state, teacher_forcing_ratio=1.)
        # [seq_len, batch_size, hidden_dim] -> [batch_size, seq_len, hidden_dim]
        reconstructed = torch.stack(reconstructed).transpose(0, 1)
        reconstructed = self.output_MLP(reconstructed)

        if mode == 'train':
            return reconstructed, target_phn_hiddens, paired_phn_hiddens, pos_phn_hiddens, neg_phn_hiddens, \
                target_spk_hiddens, paired_spk_hiddens, pos_spk_hiddens, neg_spk_hiddens
        else:
            return reconstructed, target_phn_hiddens, pos_phn_hiddens, neg_phn_hiddens, \
                target_spk_hiddens, pos_spk_hiddens, neg_spk_hiddens


class A2V(nn.Module):

    def __init__(self, input_MLP, encoder, decoder, output_MLP, decoder_num_layers):
        super(A2V, self).__init__()
        self.input_MLP = input_MLP
        self.encoder = encoder
        self.decoder = decoder
        self.output_MLP = output_MLP
        self.decoder_num_layers = decoder_num_layers

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, batch_size, feats, lengths, orders):
        # encoder
        emb_feats = self.input_MLP(feats)
        # split phn_hiddens
        _, hiddens = self.phn_encoder(emb_feats, lengths)
        hiddens = hiddens[-2:, :, :].transpose(0, 1)
        hiddens = hiddens[orders]
        target_hiddens = hiddens[:batch_size, :, :]
        if mode == 'train':
            paired_hiddens = hiddens[batch_size:, :, :]

        # construct decoder hiddens
        state_zeros = torch.zeros_like(target_hiddens.transpose(0, 1), device=device) 
        decoder_init_state = torch.tensor(target_hiddens.transpose(0, 1), device=device)
        for i in range(len(self.decoder_num_layers)-1):
            decoder_init_state = torch.cat((decoder_init_state, state_zeros), 0)

        # decoder
        begin_zeros = torch.zeros_like(feats[:batch_size, :, :]) 
        reconstructed, _, _ = self.decoder(inputs=begin_zeros, encoder_hidden=decoder_init_state, teacher_forcing_ratio=1.)
        # [seq_len, batch_size, hidden_dim] -> [batch_size, seq_len, hidden_dim]
        reconstructed = torch.stack(reconstructed).transpose(0, 1)
        reconstructed = self.output_MLP(reconstructed)

        if mode == 'train':
            return reconstructed, target_hiddens, paired_hiddens
        else:
            return reconstructed, target_hiddens
