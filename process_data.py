from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import pickle
from tqdm import tqdm
from collections import Counter
import random

from utils import read_pkl, pad_sequence

import numpy as np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################
# Loading data files
# ==================
#
# The data for this project is TIMIT.
#

sil_token = 0 # 'h#'


class Speech:
    def __init__(self, name, batch_size, num_paired):
        self.name = name
        self.batch_size = batch_size
        self.num_paired

        self.n_utts = 0
        self.n_batches = 0
        self.n_total_wrds = 0
        self.n_total_phns = 0
        # self.n_total_slbs = 0
        self.feat = []               # [num_of_wrds x num_of_frames x feat_dim]
        self.txt_feat = []           # [num_of_wrds x num_of_phns x txt_feat_dim]
        self.phn_in_wrd_feat = []
        self.phn_meta = []           # [num_of_utts x num_of_phns x (phn, utt_idx, start_frame, end_frame)]
        self.wrd_meta = []           # [num_of_utts x num_of_wrds x (wrd, utt_idx, start_frame, end_frame)]
        self.phn_in_wrd_meta = []
        # self.slb_meta = []           # [num_of_utts x num_of_slbs x (slb, utt_idx, start_frame, end_frame)]
        self.spk2utt_idx = {}
        self.utt_idx2spk = {}
        self.spk2wrd_idx = {}
        self.wrd_idx2spk = {}
        self.spks = []

        self.wrd2idx = {}
        self.wrd2cnt = Counter()
        self.idx2wrd = {}
        self.n_wrds = 0

        self.phn2idx = {}
        self.phn2cnt = Counter()
        self.idx2phn = {0: 'h#'}
        self.n_phns = 1              # Count 'h#'

        # self.slb2idx = {}
        # self.slb2cnt = Counter()
        # self.idx2slb = {}
        # self.n_slbs = 0


    ######################################################################
    # The full process for preparing the data.
    #

    def make_phn_in_wrd(phn_data, wrd_data):
        phn_in_wrd_data = []
        for wrd_utt, phn_utt in zip(wrd_data, phn_data):
            it = iter(wrd_utt)
            wrd_tuple = next(it)

            phn_in_wrd_utt = []
            phn_in_wrd = []
            for phn_tuple in phn_utt:
                if phn_tuple[2] > wrd_tuple[2]:
                    phn_in_wrd_utt.append(phn_in_wrd)
                    try:
                        wrd_tuple = next(it)
                        phn_in_wrd = []
                    except:
                        break
                if phn_tuple[1] >= wrd_tuple[1]:
                    phn_in_wrd.append(phn_tuple[0])
            phn_in_wrd_data.append(phn_in_wrd_utt)

        return phn_in_wrd_data


    def make_one_hot_feat(phn_list):
        phn_idx_array = np.array([self.phn2idx[phn] for phn in phn_list])
        f = np.zeros((len(phn_idx_array), self.n_phns-1))
        f[np.arange(len(phn_idx_array)), phn_idx_array] = 1.
        return f

    def process_data(self, meta_file, feat_file, phn_file, wrd_file, slb_file):
        meta_data = read_pkl(meta_file)     # {'prefix': [num_of_utts x drID_spkID_uttID]}                      
        feat = read_pkl(feat_file)          # [num_of_utts x num_of_frames x feat_dim]                                
        phn_data = read_pkl(phn_file)       # [num_of_utts x num_of_phns x [phn, start, end]] ** include 'h#' **
        wrd_data = read_pkl(wrd_file)       # [num_of_utts x num_of_wrds x [wrd, start, end]]           
        # slb_data = read_pkl(slb_file)       # [num_of_utts x num_of_slbs x [slb, start, end]]           

        phn_in_wrd_data = make_phn_in_wrd(phn_data, wrd_data)

        self.n_utts = len(self.feat)
        self.n_batches = self.n_utts // self.batch_size
        if self.n_utts % self.batch_size != 0:
            self.n_batches += 1
        print("Read %s utterances" % self.n_utts)

        for i, (utt, feat_utt, phn_utt, wrd_utt, phn_in_wrd_utt) \
                in enumerate(zip(meta_data['prefix'], feat, phn_data, wrd_data, phn_in_wrd_data)):
            spk = utt.split('_')[1]

            if not spk in self.spk2utt_idx:
                self.spks.append(spk)
                self.spk2utt_idx[spk] = []
                self.spk2wrd_idx[spk] = []
            self.spk2utt_idx[spk].append(i)
            self.utt_idx2spk[i] = spk

            # Process each phn in utt
            phn_meta_utt = []
            for j, (phn, phn_start, phn_end) in enumerate(phn_utt):
                self.n_total_phns += 1
                if not phn in self.phn2idx:
                    self.phn2idx[phn] = self.n_phns
                    self.idx2phn[self.n_phns] = phn
                    self.n_phns += 1
                self.phn2cnt[phn] += 1
                phn_meta_utt.append((phn, i, phn_start, phn_end))
            self.phn_meta.append(phn_meta_utt)

            # Process each wrd in utt
            wrd_meta_utt = []
            for j, (wrd, wrd_start, wrd_end) in enumerate(wrd_utt):
                self.n_total_wrds += 1
                if not wrd in self.wrd2idx:
                    self.wrd2idx[wrd] = self.n_wrds
                    self.idx2wrd[self.n_wrds] = wrd
                    self.n_wrds += 1
                self.wrd2cnt[wrd] += 1
                wrd_meta_utt.append((wrd, i, wrd_start, wrd_end))

                self.spk2wrd_idx[spk].append(self.n_total_wrds-1)
                self.wrd_idx2spk[self.n_tptal_wrds] = spk
                self.feat.append(feat[i, wrd_start:wrd_end, :])
                self.txt_feat.append(make_one_hot_feat(phn_in_wrd_utt[j]))

            self.wrd_meta.append(wrd_meta_utt)

            self.feat = np.array(self.feat)
            self.txt_feat = np.array(self.txt_feat)

            # Process each slb in utt
            # slb_meta_utt = []
            # for j, (slb, slb_start, slb_end) in enumerate(slb_utt):
                # if slb_start >= slb_utt[j-1][1] and slb_end <= slb_utt[j-1][2]:
                    # # print ('Syllables overlap:')
                    # # print (i, j-1, j, slb_utt[j-1], slb_utt[j])
                    # continue

                # self.n_total_slbs += 1
                # if not slb in self.slb2idx:
                    # self.slb2idx[slb] = self.n_slbs
                    # self.idx2slb[self.n_slbs] = slb
                    # self.n_slbs += 1
                # self.slb2cnt[slb] += 1

                # # slb_meta_utt update
                # slb_meta_utt.append((slb, i, slb_start, slb_end))

            # self.slb_meta.append(slb_meta_utt)

        return 

    def get_batch_data(self, indices, indices_paired):
        # target
        if indices_paired:
            batch_data = [self.feat[index] for index in np.concatenate((indices, indices_paired), axis=0)]
        else:
            batch_data = [self.feat[index] for index in indices]
        batch_txt_order = sorted(range(len(batch_data)), key=lambda k: len(batch_data[k]), reverse=True)

        # txt
        if indices_paired:
            batch_txt = np.array([self.txt_feat[index] for index in np.concatenate((indices, indices_paired), axis=0)])
        else:
            batch_txt = [self.txt_feat[index] for index in indices]
        batch_txt_length = torch.tensor([len(wrd) for wrd in batch_txt], device=device)
        batch_txt = batch_txt[batch_txt_order]
        batch_txt = pad_sequence(batch_txt).to(device)

        # randomly select pos & neg
        for idx in feat_indices:
            spk = self.wrd_idx2spk[idx]

            # feat_pos
            idx_pos = random.choice(self.spk2wrd_idx[spk])
            batch_data.append(self.feats[idx_pos])

        for idx in feat_indices:
            spk = self.wrd_idx2spk[idx]

            # feat_neg
            self.spks.remove(spk)
            rand_spk = random.choice(self.spks)
            self.spks.append(spk)
            idx_neg = random.choice(self.spk2wrd_idx[rand_spk])
            batch_data.append(self.feats[idx_neg])

        batch_length = torch.tensor([len(wrd) for wrd in batch_data], device=device)
        batch_order = sorted(range(len(batch_data)), key=lambda k: len(batch_data[k]), reverse=True)
        batch_data = batch_data[batch_order]
        batch_data = pad_sequence(batch_data).to(device)

        # invert indices for tracing
        batch_invert = np.zeros_like(batch_order)
        for i, j in enumerate(batch_order):
            batch_invert[j] = i
        batch_invert = torch.tensor(batch_invert, device=device)

        batch_txt_invert = np.zeros_like(batch_txt_order)
        for i, j in enumerate(batch_txt_order):
            batch_txt_invert[j] = i
        batch_txt_invert = torch.tensor(batch_txt_invert, device=device)

        # batch_data: target,( paired,) pos, neg
        # batch_txt: target,( paired)
        return batch_data, batch_length, batch_invert, \
               batch_txt, batch_txt_length, batch_txt_invert
