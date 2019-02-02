import torch
import torch.nn as nn
import torch.nn.functional as F

from WGAN-GP import calc_gradient_penalty


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################
# Audio-Word2Vec with Disentanglement
# ===================================
#


class A2VwD(nn.Module):
    """ a sequence-to-sequence model with disentanglement mechanism.
    Args:
        phn_encoder: object of EncoderRNN for phonetic encoding
        spk_encoder: object of EncoderRNN for speaker encoding
        decoder: object of DecoderRNN for decoding
        discriminator: object of Discriminator
    Inputs: target_feats, pos_feats, neg_feats, target_lengths, pos_lengths, neg_lengths
        - **target_feats** (batch_size, seq_len, feat_size): list of sequences. The target features we want to encode.
        - **pos_feats** (batch_size, seq_len, feat_size): list of sequences. The features with the same speakers as target_feats.
        - **neg_feats** (batch_size, seq_len, feat_size): list of sequences. The features with different speakers with target_feats.
        - **target_lengths** (list of int): list that contains the lengths of target_feats
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
        - **pos_lengths** (list of int): list that contains the lengths of pos_feats
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
        - **neg_lengths** (list of int): list that contains the lengths of neg_feats
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
    Outputs: target_phn_hiddens, target_spk_hiddens, reconstruction_loss, generation_loss, 
             discrimination_loss, GP_loss, pos_speaker_loss, neg_speaker_loss
        - **target_phn_hiddens** (num_layers * num_directions, batch_size, hidden_size): tensor of phonetic vectors
        - **target_spk_hiddens** (num_layers * num_directions, batch_size, hidden_size): tensor of speaker vectors
        - **reconstruction_loss**: mean-square-error between target_feats and reconstructed_target
        - **generation_loss**: negative of discrimination_loss
        - **discrimination_loss**: mean-square-error between w-distances of positive pairs and negative pairs
        - **GP_loss**: gradient penalty loss
        - **pos_speaker_loss**: mean-square-error between speaker vectors of the same speakers
        - **neg_speaker_loss**: mean-square-error between speaker vectors of different speakers
    """

    def __init__(self, phn_encoder, spk_encoder, decoder, discriminator):
        super(A2VwD, self).__init__()
        self.phn_encoder = phn_encoder
        self.spk_encoder = spk_encoder
        self.decoder = decoder
        self.discriminator = discriminator

    def flatten_parameters(self):
        self.phn_encoder.rnn.flatten_parameters()
        self.spk_encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def compute_reconstruction_loss(self, target_feats, reconstructed_target, target_lengths):
        # generate mask by lengths
        mask = torch.zeros_like(target_feats)
        for i, l in enumerate(target_lengths):
            mask[i][:l] = torch.ones(l)
        return nn.MSELoss(target_feats*mask, reconstructed_target*mask)

    def compute_GAN_losses(self, target_phn_transpose, pos_phn_transpose, neg_phn_transpose):
        # using WGAN-GP
        # pairs of (target, pos) -> real
        # pairs of (target, neg) -> fake
        target_phn_first = target_transpose[:len(target_phn_transpose)/2]
        target_phn_last = target_transpose[len(target_phn_transpose)/2:]
        pos_phn_transpose = pos_transpose[:len(pos_phn_transpose)/2]
        neg_phn_transpose = neg_transpose[len(neg_phn_transpose)/2:]

        pos_phn_concat = torch.cat((target_phn_first, pos_phn_transpose), -1)
        neg_phn_concat = torch.cat((target_phn_last, neg_phn_transpose), -1)
        pos_score = self.discriminator(pos_phn_concat)
        neg_score = self.discriminator(neg_phn_concat)
        generation_loss = nn.MSELoss(pos_score, neg_score)
        discrimination_loss = - generation_loss

        GP_loss = calc_gradient_penalty(self.discriminator, pos_phn_concat, neg_phn_concat)
        return generation_loss, discrimination_loss, GP_loss

    def compute_speaker_losses(self, target_spk_transpose, pos_spk_transpose, neg_spk_transpose):
        # pairs of (target, pos) -> as close as possible
        # pairs of (target, neg) -> far from each other to an extent
        pos_speaker_loss = nn.MSELoss(target_spk_transpose, pos_spk_transpose)
        neg_speaker_loss = torch.mean(torch.max(0.01 - torch.norm(target_spk_hiddens - neg_spk_hiddens, dim=-1), 0.)) 
        return pos_speaker_loss, neg_speaker_loss

    def forward(self, target_feats, pos_feats, neg_feats, target_lengths, pos_lengths, neg_lengths):
        _, target_phn_hiddens = self.phn_encoder(target_feats, target_lengths)
        _, pos_phn_hiddens = self.phn_encoder(pos_feats, pos_lengths)
        _, neg_phn_hiddens = self.phn_encoder(neg_feats, neg_lengths)

        _, target_spk_hiddens = self.spk_encoder(target_feats, target_lengths)
        _, pos_spk_hiddens = self.spk_encoder(pos_feats, pos_lengths)
        _, neg_spk_hiddens = self.spk_encoder(neg_feats, neg_lengths)

        target_concat_hiddens = torch.cat((target_phn_hiddens, target_spk_hiddens), -1)
        pos_concat_hiddens = torch.cat((pos_phn_hiddens, pos_spk_hiddens), -1)
        neg_concat_hiddens = torch.cat((neg_phn_hiddens, neg_spk_hiddens), -1)

        begin_zeros = torch.zeros_like(target_feats) 

        reconstructed_target, _, _ = self.decoder(inputs=begin_zeros, encoder_hidden=target_concat_hiddens, teacher_forcing_ratio=1.)
        reconstructed_pos, _, _ = self.decoder(inputs=begin_zeros, encoder_hidden=pos_concat_hiddens, teacher_forcing_ratio=1.)
        reconstructed_neg, _, _ = self.decoder(inputs=begin_zeros, encoder_hidden=neg_concat_hiddens, teacher_forcing_ratio=1.)

        # (num_layers * num_directions, batch_size, hidden_size) -> (batch_size, -1)
        batch_size = target_phn_hiddens.size(1)
        target_phn_transpose = torch.transpose(target_phn_hiddens, 0, 1).view(batch_size, -1)
        pos_phn_transpose = torch.transpose(pos_phn_hiddens, 0, 1).view(batch_size, -1)
        neg_phn_transpose = torch.transpose(neg_phn_hiddens, 0, 1).view(batch_size, -1)
        target_spk_transpose = torch.transpose(target_spk_hiddens, 0, 1).view(batch_size, -1)
        pos_spk_transpose = torch.transpose(pos_spk_hiddens, 0, 1).view(batch_size, -1)
        neg_spk_transpose = torch.transpose(neg_spk_hiddens, 0, 1).view(batch_size, -1)

        reconstruction_loss = self.compute_reconstruction_loss(target_feats, reconstructed_target, target_lengths)
        generation_loss, discrimination_loss, GP_loss = self.compute_GAN_losses(target_phn_transpose, pos_phn_transpose, neg_phn_transpose)
        pos_speaker_loss, neg_speaker_loss = self.compute_speaker_losses(target_spk_transpose, pos_spk_transpose, neg_spk_transpose)

        return target_phn_hiddens, target_spk_hiddens, reconstruction_loss, generation_loss, 
               discrimination_loss, GP_loss, pos_speaker_loss, neg_speaker_loss
