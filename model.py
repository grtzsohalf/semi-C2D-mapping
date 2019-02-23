import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from WGAN_GP import calc_gradient_penalty
from utils import mask_with_length


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################
# Audio2Vec
# =========
#


class Model(nn.Module):

    def __init__(self, aud_input_MLP, phn_encoder, spk_encoder, aud_decoder, aud_output_MLP, aud_decoder_num_layers, 
                 txt_input_MLP, txt_encoder, txt_decoder, txt_output_MLP, txt_decoder_num_layers, neg_thres, discriminator):
        super(Model, self).__init__()
        self.aud_input_MLP = aud_input_MLP
        self.phn_encoder = phn_encoder
        self.spk_encoder = spk_encoder
        self.aud_decoder = aud_decoder
        self.aud_output_MLP = aud_output_MLP
        self.aud_decoder_num_layers = aud_decoder_num_layers
        # self.a2v = a2v
        self.txt_input_MLP = txt_input_MLP
        self.txt_encoder = txt_encoder
        self.txt_decoder = txt_decoder
        self.txt_output_MLP = txt_output_MLP
        self.txt_decoder_num_layers = txt_decoder_num_layers
        # self.t2v = t2v
        # self.pos_thres = pos_thres
        self.neg_thres = neg_thres

        self.discriminator = discriminator
        self.other_parts = [self.aud_input_MLP, self.phn_encoder, self.spk_encoder, self.aud_decoder, self.aud_output_MLP,
                            self.txt_input_MLP, self.txt_encoder, self.txt_decoder, self.txt_output_MLP]

    def flatten_parameters(self):
        self.phn_encoder.flatten_parameters()
        self.spk_encoder.flatten_parameters()
        self.aud_decoder.flatten_parameters()
        self.txt_encoder.flatten_parameters()
        self.txt_decoder.flatten_parameters()

    def compute_reconstruction_loss(self, x, y, mask, lengths=None):
        # generate mask by lengths
        # mask = torch.zeros_like(target_feats)
        # for i, l in enumerate(target_lengths):
            # mask[i][:l] = torch.ones(l)
        # if mask:
            # x, _ = mask_with_length(x, lengths)
            # y, _ = mask_with_length(y, lengths)
        # criterion = nn.MSELoss()
        # MSE_loss = criterion(x, y)
        MSE_loss = (x - y) ** 2
        if mask:
            MSE_loss, _ = mask_with_length(MSE_loss, lengths)
        return MSE_loss.sum() / lengths.float().sum()

    # def compute_GAN_losses(self, batch_size, target, pos, neg):
        # # # using WGAN-GP
        # # # pairs of (target, pos) -> real
        # # # pairs of (target, neg) -> fake
        # target_first = target[:(batch_size//2)+1]
        # target_last = target[-(batch_size//2)-1:]
        # pos = pos[:(batch_size//2)+1]
        # neg = neg[-(batch_size//2)-1:]

        # pos_concat = torch.cat((target_first, pos), -1)
        # neg_concat = torch.cat((target_last, neg), -1)
        # pos_score = self.discriminator(pos_concat)
        # neg_score = self.discriminator(neg_concat)
        # # criterion = nn.MSELoss()
        # # generation_loss = criterion(pos_score, neg_score)
        # generation_loss = torch.mean((pos_score - neg_score) ** 2)
        # discrimination_loss = - generation_loss

        # GP_loss = calc_gradient_penalty((batch_size//2)+1, self.discriminator, pos_concat, neg_concat)
        # return generation_loss, discrimination_loss, GP_loss

    def compute_GAN_losses(self, batch_size, phn, txt):
        # # using WGAN-GP
        pos_score = self.discriminator(phn)
        neg_score = self.discriminator(txt)
        # criterion = nn.MSELoss()
        # generation_loss = criterion(pos_score, neg_score)
        generation_loss = torch.mean((pos_score - neg_score) ** 2)
        discrimination_loss = - generation_loss

        GP_loss = calc_gradient_penalty(batch_size, self.discriminator, phn, txt)
        return generation_loss, discrimination_loss, GP_loss

    def compute_hinge_losses(self, target, pos, neg, neg_thres):
        # pairs of (target, pos) -> as close as possible
        # pairs of (target, neg) -> far from each other to an extent
        # pos_speaker_loss = self.compute_reconstruction_loss(target, pos, False)
        # MSE_criterion = nn.MSELoss(reduction='none')
        # hinge_criterion = nn.HingeEmbeddingLoss(margin=0.01)
        # neg_speaker_loss = hinge_criterion(MSE_criterion(target, neg), 
                                           # -torch.ones(target.shape[0], device=device))
        pos_loss = torch.mean((target - pos) ** 2)
        neg_loss = torch.mean(torch.clamp(neg_thres - torch.mean((target - neg) ** 2, dim=-1), min=0.)) 
        return pos_loss, neg_loss

    def compute_CE_loss(self, x, y, mask, lengths=None):
        def _sequence_mask(sequence_length, max_len=None):
            if max_len is None:
                max_len = sequence_length.data.max()
            batch_size = sequence_length.size(0)
            seq_range = torch.arange(0, max_len, device=device).long()
            seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
            seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
            return seq_range_expand < seq_length_expand

        x_flat = x.view(-1, x.size(-1))
        log_probs_flat = F.log_softmax(x_flat, dim=1)
        y_flat = y.view(-1, 1)
        CE_loss_flat = -torch.gather(log_probs_flat, dim=1, index=y_flat)
        CE_loss = CE_loss_flat.view(*y.size())
        if mask:
            mask = _sequence_mask(sequence_length=lengths, max_len=y.size(1))
            CE_loss = CE_loss * mask.float()
        return CE_loss.sum() / lengths.float().sum()

    def forward(self, batch_size, aud_feats, aud_lengths, aud_orders, txt_feats, txt_lengths, txt_orders, txt_labels):
        aud_emb_feats = self.aud_input_MLP(aud_feats)
        ##########
        # encode #
        ##########

        # split phn_hiddens
        _, phn_hiddens = self.phn_encoder(aud_emb_feats, aud_lengths)
        phn_hiddens = phn_hiddens[-2:, :, :].transpose(0, 1)
        ordered_phn_hiddens = phn_hiddens[aud_orders]
        target_phn_hiddens = ordered_phn_hiddens[:batch_size, :, :].view(batch_size, -1)
        paired_phn_hiddens = ordered_phn_hiddens[batch_size:batch_size*2, :, :].view(batch_size, -1)
        # pos_phn_hiddens = phn_hiddens[batch_size*2:batch_size*3, :, :].view(batch_size, -1)
        # neg_phn_hiddens = phn_hiddens[batch_size*3:batch_size*4, :, :].view(batch_size, -1)
        neg_paired_phn_hiddens = ordered_phn_hiddens[batch_size*2:, :, :].view(batch_size, -1)

        # split spk_hiddens
        _, spk_hiddens = self.spk_encoder(aud_emb_feats, aud_lengths)
        spk_hiddens = spk_hiddens[-2:, :, :].transpose(0, 1)
        ordered_spk_hiddens = spk_hiddens[aud_orders]
        target_spk_hiddens = ordered_spk_hiddens[:batch_size, :, :].view(batch_size, -1)
        paired_spk_hiddens = ordered_spk_hiddens[batch_size:batch_size*2, :, :].view(batch_size, -1)
        # pos_spk_hiddens = spk_hiddens[batch_size*2:batch_size*3, :, :].view(batch_size, -1)
        # neg_spk_hiddens = spk_hiddens[batch_size*3:batch_size*4, :, :].view(batch_size, -1)
        neg_paired_spk_hiddens = ordered_spk_hiddens[batch_size*2:, :, :].view(batch_size, -1)

        txt_emb_feats = self.txt_input_MLP(txt_feats)

        # split txt_hiddens
        _, txt_hiddens = self.txt_encoder(txt_emb_feats, txt_lengths)
        txt_hiddens = txt_hiddens[-2:, :, :].transpose(0, 1)
        ordered_txt_hiddens = txt_hiddens[txt_orders]
        target_txt_hiddens = ordered_txt_hiddens[:batch_size, :, :].view(batch_size, -1)
        paired_txt_hiddens = ordered_txt_hiddens[batch_size:batch_size*2, :, :].view(batch_size, -1)

        ##########
        # decode #
        ##########

        # construct aud_decoder hiddens
        target_concat_hiddens = torch.cat((ordered_phn_hiddens[:batch_size, :, :], 
                                           ordered_spk_hiddens[:batch_size, :, :]), -1).transpose(0, 1)
        aud_state_zeros = torch.zeros_like(target_concat_hiddens, device=device) 
        target_aud_decoder_init_state = target_concat_hiddens.clone()
        for i in range(self.aud_decoder_num_layers-1):
            target_aud_decoder_init_state = torch.cat((target_aud_decoder_init_state, aud_state_zeros), 0)

        paired_concat_hiddens = torch.cat((ordered_txt_hiddens[batch_size:batch_size*2, :, :], 
                                           ordered_spk_hiddens[batch_size:batch_size*2, :, :]), -1).transpose(0, 1)
        paired_aud_decoder_init_state = paired_concat_hiddens.clone()
        for i in range(self.aud_decoder_num_layers-1):
            paired_aud_decoder_init_state = torch.cat((paired_aud_decoder_init_state, aud_state_zeros), 0)

        # aud_decoder
        aud_begin_zeros = torch.zeros_like(aud_emb_feats[:batch_size, :, :]) 
        aud_begin_zeros = torch.cat((aud_begin_zeros, aud_begin_zeros), -1)
        aud_begin_zeros = torch.cat((aud_begin_zeros, aud_begin_zeros), -1)
        aud_reconstructed_target, _, _ = self.aud_decoder(inputs=aud_begin_zeros, 
                                                          encoder_hidden=target_aud_decoder_init_state, teacher_forcing_ratio=1.)
        aud_reconstructed_paired, _, _ = self.aud_decoder(inputs=aud_begin_zeros, 
                                                          encoder_hidden=paired_aud_decoder_init_state, teacher_forcing_ratio=1.)
        # [seq_len, batch_size, hidden_dim] -> [batch_size, seq_len, hidden_dim]
        aud_reconstructed_target = torch.stack(aud_reconstructed_target).transpose(0, 1)
        aud_reconstructed_paired = torch.stack(aud_reconstructed_paired).transpose(0, 1)

        aud_reconstructed_target = self.aud_output_MLP(aud_reconstructed_target)
        aud_reconstructed_paired = self.aud_output_MLP(aud_reconstructed_paired)

        # construct txt_decoder hiddens
        txt_state_zeros = torch.zeros_like(ordered_txt_hiddens[:batch_size, :, :].transpose(0, 1), device=device) 
        target_txt_decoder_init_state = ordered_txt_hiddens[:batch_size, :, :].transpose(0, 1).clone()
        for i in range(self.txt_decoder_num_layers-1):
            target_txt_decoder_init_state = torch.cat((target_txt_decoder_init_state, txt_state_zeros), 0)
        paired_txt_decoder_init_state = ordered_phn_hiddens[batch_size:batch_size*2, :, :].transpose(0, 1).clone()
        for i in range(self.txt_decoder_num_layers-1):
            paired_txt_decoder_init_state = torch.cat((paired_txt_decoder_init_state, txt_state_zeros), 0)

        # txt_decoder
        txt_begin_zeros = torch.zeros_like(txt_emb_feats[:batch_size, :, :]) 
        txt_begin_zeros = torch.cat((txt_begin_zeros, txt_begin_zeros), -1)
        txt_reconstructed_target, _, _ = self.txt_decoder(inputs=txt_begin_zeros,
                                                          encoder_hidden=target_txt_decoder_init_state, teacher_forcing_ratio=1.)
        txt_reconstructed_paired, _, _ = self.txt_decoder(inputs=txt_begin_zeros,
                                                          encoder_hidden=paired_txt_decoder_init_state, teacher_forcing_ratio=1.)
        # [seq_len, batch_size, hidden_dim] -> [batch_size, seq_len, hidden_dim]
        txt_reconstructed_target = torch.stack(txt_reconstructed_target).transpose(0, 1)
        txt_reconstructed_paired = torch.stack(txt_reconstructed_paired).transpose(0, 1)

        txt_reconstructed_target = self.txt_output_MLP(txt_reconstructed_target)
        txt_reconstructed_paired = self.txt_output_MLP(txt_reconstructed_paired)

        # # x

        # # x_aud_reconstrcution
        # concat_hiddens = torch.cat((txt_hiddens, spk_hiddens), -1).transpose(0, 1)
        # aud_state_zeros = torch.zeros_like(concat_hiddens, device=device) 
        # aud_decoder_init_state = concat_hiddens.clone()
        # for i in range(self.aud_decoder_num_layers-1):
            # aud_decoder_init_state = torch.cat((aud_decoder_init_state, aud_state_zeros), 0)

        # aud_begin_zeros = torch.zeros_like(aud_emb_feats) 
        # aud_begin_zeros = torch.cat((aud_begin_zeros, aud_begin_zeros), -1)
        # aud_begin_zeros = torch.cat((aud_begin_zeros, aud_begin_zeros), -1)
        # aud_reconstructed, _, _ = self.aud_decoder(inputs=aud_begin_zeros, 
                                                   # encoder_hidden=aud_decoder_init_state, teacher_forcing_ratio=1.)

        # aud_reconstructed = torch.stack(aud_reconstructed).transpose(0, 1)
        # aud_reconstructed = self.aud_output_MLP(aud_reconstructed)

        # # x_txt_reconstrcution
        # txt_state_zeros = torch.zeros_like(phn_hiddens.transpose(0, 1), device=device) 
        # txt_decoder_init_state = phn_hiddens.transpose(0, 1).clone()
        # for i in range(self.txt_decoder_num_layers-1):
            # txt_decoder_init_state = torch.cat((txt_decoder_init_state, txt_state_zeros), 0)

        # txt_begin_zeros = torch.zeros_like(txt_emb_feats) 
        # txt_begin_zeros = torch.cat((txt_begin_zeros, txt_begin_zeros), -1)
        # txt_reconstructed, _, _ = self.txt_decoder(inputs=txt_begin_zeros,
                                                   # encoder_hidden=txt_decoder_init_state, teacher_forcing_ratio=1.)

        # txt_reconstructed = torch.stack(txt_reconstructed).transpose(0, 1)
        # txt_reconstructed = self.txt_output_MLP(txt_reconstructed)

        # # x_paired
        # x_aud_emb_feats = self.aud_input_MLP(aud_reconstructed)
        # _, x_phn_hiddens = self.phn_encoder(x_aud_emb_feats, aud_lengths)
        # x_phn_hiddens = x_phn_hiddens[-2:, :, :].transpose(0, 1)
        # x_ordered_phn_hiddens = x_phn_hiddens[txt_orders]
        # _, x_spk_hiddens = self.spk_encoder(x_aud_emb_feats, aud_lengths)
        # x_spk_hiddens = x_spk_hiddens[-2:, :, :].transpose(0, 1)
        # x_ordered_spk_hiddens = x_spk_hiddens[txt_orders]

        # x_txt_emb_feats = self.txt_input_MLP(txt_reconstructed)
        # _, x_txt_hiddens = self.txt_encoder(x_txt_emb_feats, txt_lengths)
        # x_txt_hiddens = x_txt_hiddens[-2:, :, :].transpose(0, 1)
        # x_ordered_txt_hiddens = x_txt_hiddens[aud_orders]

        # x_paired_concat_hiddens = torch.cat((x_ordered_txt_hiddens[batch_size:batch_size*2, :, :], 
                                             # x_ordered_spk_hiddens[batch_size:batch_size*2, :, :]), -1).transpose(0, 1)
        # aud_state_zeros = torch.zeros_like(x_paired_concat_hiddens, device=device) 
        # x_paired_aud_decoder_init_state = x_paired_concat_hiddens.clone()
        # for i in range(self.aud_decoder_num_layers-1):
            # x_paired_aud_decoder_init_state = torch.cat((x_paired_aud_decoder_init_state, aud_state_zeros), 0)

        # aud_begin_zeros = torch.zeros_like(aud_emb_feats[:batch_size, :, :]) 
        # aud_begin_zeros = torch.cat((aud_begin_zeros, aud_begin_zeros), -1)
        # aud_begin_zeros = torch.cat((aud_begin_zeros, aud_begin_zeros), -1)
        # x_aud_reconstructed_paired, _, _ = self.aud_decoder(inputs=aud_begin_zeros,
                                                            # encoder_hidden=x_paired_aud_decoder_init_state, teacher_forcing_ratio=1.)
        # x_aud_reconstructed_paired = torch.stack(x_aud_reconstructed_paired).transpose(0, 1)
        # x_aud_reconstructed_paired = self.aud_output_MLP(x_aud_reconstructed_paired)

        # txt_state_zeros = torch.zeros_like(x_ordered_phn_hiddens[:batch_size, :, :].transpose(0, 1), device=device) 
        # x_paired_txt_decoder_init_state = x_ordered_phn_hiddens[batch_size:batch_size*2, :, :].transpose(0, 1).clone()
        # for i in range(self.txt_decoder_num_layers-1):
            # x_paired_txt_decoder_init_state = torch.cat((x_paired_txt_decoder_init_state, txt_state_zeros), 0)

        # txt_begin_zeros = torch.zeros_like(txt_emb_feats[:batch_size, :, :]) 
        # txt_begin_zeros = torch.cat((txt_begin_zeros, txt_begin_zeros), -1)
        # x_txt_reconstructed_paired, _, _ = self.txt_decoder(inputs=txt_begin_zeros,
                                                            # encoder_hidden=x_paired_txt_decoder_init_state, teacher_forcing_ratio=1.)
        # x_txt_reconstructed_paired = torch.stack(x_txt_reconstructed_paired).transpose(0, 1)
        # x_txt_reconstructed_paired = self.txt_output_MLP(x_txt_reconstructed_paired)

        ##################
        # compute losses #
        ##################

        target_aud_feats = aud_feats[aud_orders][:batch_size]
        paired_aud_feats = aud_feats[aud_orders][batch_size:batch_size*2]
        target_txt_labels = txt_labels[txt_orders][:batch_size]
        paired_txt_labels = txt_labels[txt_orders][batch_size:batch_size*2]
        target_aud_lengths = aud_lengths[aud_orders][:batch_size]
        paired_aud_lengths = aud_lengths[aud_orders][batch_size:batch_size*2]
        target_txt_lengths = txt_lengths[txt_orders][:batch_size]
        paired_txt_lengths = txt_lengths[txt_orders][batch_size:batch_size*2]
        aud_reconstruction_loss = self.compute_reconstruction_loss(aud_reconstructed_target, target_aud_feats, True, target_aud_lengths) \
            + self.compute_reconstruction_loss(aud_reconstructed_paired, paired_aud_feats, True, paired_aud_lengths)
        txt_CE_loss = self.compute_CE_loss(txt_reconstructed_target, target_txt_labels, True, target_txt_lengths) \
            + self.compute_CE_loss(txt_reconstructed_paired, paired_txt_labels, True, paired_txt_lengths)
        # x_loss = self.compute_reconstruction_loss(x_aud_reconstructed_paired, paired_aud_feats, True, paired_aud_lengths) \
            # + self.compute_CE_loss(x_txt_reconstructed_paired, paired_txt_labels, True, paired_txt_lengths)
        generation_loss, discrimination_loss, GP_loss \
            = self.compute_GAN_losses(batch_size, target_phn_hiddens, target_txt_hiddens)
        # pos_speaker_loss, neg_speaker_loss = \
            # self.compute_hinge_losses(target_spk_hiddens, pos_spk_hiddens, neg_spk_hiddens, self.pos_thres, self.neg_thres)
        pos_paired_loss, neg_paired_loss = \
            self.compute_hinge_losses(paired_txt_hiddens, paired_phn_hiddens, neg_paired_phn_hiddens, self.neg_thres)

        return target_phn_hiddens, target_spk_hiddens, target_txt_hiddens, \
            aud_reconstruction_loss, txt_CE_loss, generation_loss, discrimination_loss, GP_loss, pos_paired_loss, neg_paired_loss


# class A2VwD(nn.Module):

    # def __init__(self, input_MLP, phn_encoder, spk_encoder, decoder, output_MLP, decoder_num_layers):
        # super(A2VwD, self).__init__()
        # self.input_MLP = input_MLP
        # self.phn_encoder = phn_encoder
        # self.spk_encoder = spk_encoder
        # self.decoder = decoder
        # self.output_MLP = output_MLP
        # self.decoder_num_layers = decoder_num_layers

    # def flatten_parameters(self):
        # self.phn_encoder.rnn.flatten_parameters()
        # self.spk_encoder.rnn.flatten_parameters()
        # self.decoder.rnn.flatten_parameters()

    # def forward(self, batch_size, feats, lengths, orders, mode):
        # # encoder
        # emb_feats = self.input_MLP(feats)
        # # split phn_hiddens
        # _, phn_hiddens = self.phn_encoder(emb_feats, lengths)
        # phn_hiddens = phn_hiddens[-2:, :, :].transpose(0, 1)
        # phn_hiddens = phn_hiddens[orders]
        # target_phn_hiddens = phn_hiddens[:batch_size, :, :].view(batch_size, -1)
        # if mode == 'train':
            # paired_phn_hiddens = phn_hiddens[batch_size:batch_size*2, :, :].view(batch_size, -1)
            # pos_phn_hiddens = phn_hiddens[batch_size*2:batch_size*3, :, :].view(batch_size, -1)
            # neg_phn_hiddens = phn_hiddens[batch_size*3:batch_size*4, :, :].view(batch_size, -1)
            # neg_paired_phn_hiddens = phn_hiddens[batch_size*4:, :, :].view(batch_size, -1)
        # else:
            # pos_phn_hiddens = phn_hiddens[batch_size:batch_size*2, :, :].view(batch_size, -1)
            # neg_phn_hiddens = phn_hiddens[batch_size*2:batch_size*3, :, :].view(batch_size, -1)
            # neg_paired_phn_hiddens = phn_hiddens[batch_size*3:, :, :].view(batch_size, -1)

        # # split spk_hiddens
        # _, spk_hiddens = self.spk_encoder(emb_feats, lengths)
        # spk_hiddens = spk_hiddens[-2:, :, :].transpose(0, 1)
        # spk_hiddens = spk_hiddens[orders]
        # target_spk_hiddens = spk_hiddens[:batch_size, :, :].view(batch_size, -1)
        # if mode == 'train':
            # paired_spk_hiddens = spk_hiddens[batch_size:batch_size*2, :, :].view(batch_size, -1)
            # pos_spk_hiddens = spk_hiddens[batch_size*2:batch_size*3, :, :].view(batch_size, -1)
            # neg_spk_hiddens = spk_hiddens[batch_size*3:batch_size*4, :, :].view(batch_size, -1)
            # neg_paired_spk_hiddens = spk_hiddens[batch_size*4:, :, :].view(batch_size, -1)
        # else:
            # pos_spk_hiddens = spk_hiddens[batch_size:batch_size*2, :, :].view(batch_size, -1)
            # neg_spk_hiddens = spk_hiddens[batch_size*2:batch_size*3, :, :].view(batch_size, -1)
            # neg_paired_spk_hiddens = spk_hiddens[batch_size*3:, :, :].view(batch_size, -1)

        # # construct decoder hiddens
        # target_concat_hiddens = torch.cat((phn_hiddens[:batch_size, :, :], spk_hiddens[:batch_size, :, :]), -1).transpose(0, 1)
        # state_zeros = torch.zeros_like(target_concat_hiddens, device=device) 
        # # decoder_init_state = torch.tensor(target_concat_hiddens, device=device)
        # decoder_init_state = target_concat_hiddens.clone()
        # for i in range(self.decoder_num_layers-1):
            # decoder_init_state = torch.cat((decoder_init_state, state_zeros), 0)

        # # decoder
        # begin_zeros = torch.zeros_like(emb_feats[:batch_size, :, :]) 
        # begin_zeros = torch.cat((begin_zeros, begin_zeros), -1)
        # begin_zeros = torch.cat((begin_zeros, begin_zeros), -1)
        # reconstructed, _, _ = self.decoder(inputs=begin_zeros, encoder_hidden=decoder_init_state, teacher_forcing_ratio=1.)
        # # [seq_len, batch_size, hidden_dim] -> [batch_size, seq_len, hidden_dim]
        # reconstructed = torch.stack(reconstructed).transpose(0, 1)
        # reconstructed = self.output_MLP(reconstructed)

        # if mode == 'train':
            # return reconstructed, target_phn_hiddens, paired_phn_hiddens, pos_phn_hiddens, neg_phn_hiddens, neg_paired_phn_hiddens, \
                # target_spk_hiddens, paired_spk_hiddens, pos_spk_hiddens, neg_spk_hiddens, neg_paired_spk_hiddens
        # else:
            # return reconstructed, target_phn_hiddens, pos_phn_hiddens, neg_phn_hiddens, neg_paired_phn_hiddens, \
                # target_spk_hiddens, pos_spk_hiddens, neg_spk_hiddens, neg_paired_spk_hiddens


# class A2V(nn.Module):

    # def __init__(self, input_MLP, encoder, decoder, output_MLP, decoder_num_layers):
        # super(A2V, self).__init__()
        # self.input_MLP = input_MLP
        # self.encoder = encoder
        # self.decoder = decoder
        # self.output_MLP = output_MLP
        # self.decoder_num_layers = decoder_num_layers

    # def flatten_parameters(self):
        # self.encoder.rnn.flatten_parameters()
        # self.decoder.rnn.flatten_parameters()

    # def forward(self, batch_size, feats, lengths, orders, mode):
        # # encoder
        # emb_feats = self.input_MLP(feats)
        # # split phn_hiddens
        # _, hiddens = self.encoder(emb_feats, lengths)
        # hiddens = hiddens[-2:, :, :].transpose(0, 1)
        # hiddens = hiddens[orders]
        # target_hiddens = hiddens[:batch_size, :, :].view(batch_size, -1)
        # if mode == 'train':
            # paired_hiddens = hiddens[batch_size:, :, :].view(batch_size, -1)

        # # construct decoder hiddens
        # state_zeros = torch.zeros_like(hiddens[:batch_size, :, :].transpose(0, 1), device=device) 
        # # decoder_init_state = torch.tensor(hiddens[:batch_size, :, :].transpose(0, 1), device=device)
        # decoder_init_state = hiddens[:batch_size, :, :].transpose(0, 1).clone()
        # for i in range(self.decoder_num_layers-1):
            # decoder_init_state = torch.cat((decoder_init_state, state_zeros), 0)

        # # decoder
        # begin_zeros = torch.zeros_like(emb_feats[:batch_size, :, :]) 
        # begin_zeros = torch.cat((begin_zeros, begin_zeros), -1)
        # reconstructed, _, _ = self.decoder(inputs=begin_zeros, encoder_hidden=decoder_init_state, teacher_forcing_ratio=1.)
        # # [seq_len, batch_size, hidden_dim] -> [batch_size, seq_len, hidden_dim]
        # reconstructed = torch.stack(reconstructed).transpose(0, 1)
        # reconstructed = self.output_MLP(reconstructed)

        # if mode == 'train':
            # return reconstructed, target_hiddens, paired_hiddens
        # else:
            # return reconstructed, target_hiddens
