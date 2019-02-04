from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import pickle
from tqdm import tqdm
from collections import Counter
import random

from utils import read_pkl, pad_sequence, count_LM

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
        self.num_paired = num_paired

        self.n_utts = 0
        self.n_batches = 0
        self.n_total_wrds = 0
        self.n_total_phns = 0
        # self.n_total_slbs = 0

        self.feat = []               # [num_of_wrds x num_of_frames x feat_dim]
        self.txt_feat = []           # [num_of_wrds x num_of_phns x txt_feat_dim]

        self.phn_meta = []           # [num_of_utts x num_of_phns x (phn, utt_idx, start_frame, end_frame)]
        self.wrd_meta = []           # [num_of_utts x num_of_wrds x (wrd, utt_idx, start_frame, end_frame)]
        self.phn_wrd_meta = []    # [num_of_utts x num_of_wrds x ((wrd, phn_wrd), utt_idx, start_frame, end_frame)]
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
        self.wrds = []

        self.phn_wrd2idx = {}
        self.phn_wrd2cnt = Counter()
        self.idx2phn_wrd = {}
        self.n_phn_wrds = 0
        self.phn_wrds = []


        self.phn2idx = {'h#': 0}
        self.n_phns = 1
        with open('TIMIT_utils/phones.60-48-39.map.txt', 'r') as f:
            for line in f:
                phn = line.strip().split()[0]
                if phn != 'h#':
                    self.n_phns += 1
                    self.phn2idx[phn] = self.n_phns-1
        self.idx2phn = {}
        for phn, idx in self.phn2idx.items():
            self.idx2phn[idx] = phn
        self.phn2cnt = Counter()

        self.LM = None

        # self.slb2idx = {}
        # self.slb2cnt = Counter()
        # self.idx2slb = {}
        # self.n_slbs = 0


    ######################################################################
    # The full process for preparing the data.
    #

    def make_phn_wrd(self, phn_data, wrd_data):
        phn_wrd_data = []
        for i, (wrd_utt, phn_utt) in enumerate(zip(wrd_data, phn_data)):
            phn_wrd_utt = []
            for _ in range(len(wrd_utt)):
                phn_wrd_utt.append([])
            for phn_tuple in phn_utt:
                if phn_tuple[0] == 'h#':
                    continue
                for j, wrd_tuple in enumerate(wrd_utt):
                    if phn_tuple[1] >= wrd_tuple[1] and phn_tuple[2] <= wrd_tuple[2]:
                        phn_wrd_utt[j].append(phn_tuple[0])
            phn_wrd_data.append([tuple(w) for w in phn_wrd_utt])
            # for phn_wrd in phn_wrd_utt:
                # if phn_wrd == []:
                    # print (i)

        return phn_wrd_data


    def make_one_hot_feat(self, phn_tuple):
        phn_idx_array = np.array([self.phn2idx[phn]-1 for phn in phn_tuple])
        f = np.zeros((len(phn_idx_array), self.n_phns-1))
        f[np.arange(len(phn_idx_array)), phn_idx_array] = 1.
        return f

    def process_data(self, meta_file, feat_file, phn_file, wrd_file, slb_file):
        meta_data = read_pkl(meta_file)     # {'prefix': [num_of_utts x drID_spkID_uttID]}                      
        feat = read_pkl(feat_file)          # [num_of_utts x num_of_frames x feat_dim]                                
        phn_data = read_pkl(phn_file)       # [num_of_utts x num_of_phns x [phn, start, end]] ** include 'h#' **
        wrd_data = read_pkl(wrd_file)       # [num_of_utts x num_of_wrds x [wrd, start, end]]           
        # slb_data = read_pkl(slb_file)       # [num_of_utts x num_of_slbs x [slb, start, end]]           

        phn_wrd_data = self.make_phn_wrd(phn_data, wrd_data)

        self.n_utts = len(feat)
        print("Read %s utterances" % self.n_utts)

        for i, phn_utt in enumerate(phn_data):
            # Process each phn in utt
            phn_meta_utt = []
            for j, (phn, phn_start, phn_end) in enumerate(phn_utt):
                self.n_total_phns += 1
                self.phn2cnt[phn] += 1
                phn_meta_utt.append((phn, i, phn_start, phn_end))
            self.phn_meta.append(phn_meta_utt)

        for i, (utt, feat_utt, wrd_utt, phn_wrd_utt) \
                in enumerate(zip(meta_data['prefix'], feat, wrd_data, phn_wrd_data)):
            spk = utt.split('_')[1]

            if not spk in self.spk2utt_idx:
                self.spks.append(spk)
                self.spk2utt_idx[spk] = []
                self.spk2wrd_idx[spk] = []
            self.spk2utt_idx[spk].append(i)
            self.utt_idx2spk[i] = spk

            # Process each wrd in utt
            wrd_meta_utt = []
            phn_wrd_meta_utt = []
            for j, ((wrd, wrd_start, wrd_end), phn_wrd) in enumerate(zip(wrd_utt, phn_wrd_utt)):
                wrd = wrd.lower()
                if j != 0 and wrd_start >= wrd_utt[j-1][1] and wrd_end <= wrd_utt[j-1][2]:
                    # print ('Words overlap:')
                    # print (i, j-1, j, wrd_utt[j-1], wrd_utt[j])
                    continue
                if j != len(wrd_utt)-1 and wrd_start >= wrd_utt[j+1][1] and wrd_end <= wrd_utt[j+1][2]:
                    # print ('Words overlap:')
                    # print (i, j, j+1, wrd_utt[j], wrd_utt[j+1])
                    continue
                if wrd_end - wrd_start == 0:
                    # print (i, wrd)
                    continue
                if not len(phn_wrd):
                    # print (i, j)
                    continue
                self.n_total_wrds += 1
                if not wrd in self.wrd2idx:
                    self.wrd2idx[wrd] = self.n_wrds
                    self.idx2wrd[self.n_wrds] = wrd
                    self.n_wrds += 1
                self.wrd2cnt[wrd] += 1
                wrd_meta_utt.append((wrd, i, wrd_start, wrd_end))

                if not (wrd, phn_wrd) in self.phn_wrd2idx:
                    self.phn_wrd2idx[(wrd, phn_wrd)] = self.n_phn_wrds
                    self.idx2phn_wrd[self.n_phn_wrds] = (wrd, phn_wrd)
                    self.n_phn_wrds += 1
                self.phn_wrd2cnt[(wrd, phn_wrd)] += 1
                phn_wrd_meta_utt.append(((wrd, phn_wrd), i, wrd_start, wrd_end))

                self.spk2wrd_idx[spk].append(self.n_total_wrds-1)
                self.wrd_idx2spk[self.n_total_wrds-1] = spk
                self.feat.append(feat[i][wrd_start:wrd_end])
                self.txt_feat.append(self.make_one_hot_feat(phn_wrd))

            self.wrd_meta.append(wrd_meta_utt)
            self.phn_wrd_meta.append(phn_wrd_meta_utt)

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

        self.feat = np.array(self.feat)
        self.txt_feat = np.array(self.txt_feat)
        self.n_batches = len(self.feat) // self.batch_size
        if len(self.feat) % self.batch_size != 0:
            self.n_batches += 1

        for wrd_meta_utt in self.wrd_meta:
            self.wrds.extend([w[0] for w in wrd_meta_utt])
        for phn_wrd_meta_utt in self.phn_wrd_meta:
            self.phn_wrds.extend([w[0] for w in phn_wrd_meta_utt])

        print ('Num of total words: ', len(self.feat), len(self.txt_feat), self.n_total_wrds)
        print ('Num of distinct words (with different phonemes): ', self.n_phn_wrds)
        print ('Num of distinct words: ', self.n_wrds)
        print ('Num of batches: ', self.n_batches)

        return 

    def get_batch_data(self, indices, mode):
        # txt
        batch_txt = [self.txt_feat[index] for index in indices]
        batch_txt_length = torch.tensor([len(wrd) for wrd in batch_txt], device=device)
        batch_txt_order = np.array(sorted(range(len(batch_txt)), key=lambda k: len(batch_txt[k]), reverse=True))
        batch_txt = np.array(batch_txt)[batch_txt_order]
        batch_txt_length = batch_txt_length[batch_txt_order]
        batch_txt = pad_sequence(batch_txt).to(device)

        # target
        batch_data = [self.feat[index] for index in indices]

        # randomly select pos & neg
        if mode == 'train':
            pos_neg_indices = indices[:len(batch_data)//2]
        else:
            pos_neg_indices = indices
        for idx in pos_neg_indices:
            spk = self.wrd_idx2spk[idx]

            # feat_pos
            idx_pos = random.choice(self.spk2wrd_idx[spk])
            batch_data.append(self.feat[idx_pos])

        for idx in pos_neg_indices:
            spk = self.wrd_idx2spk[idx]

            # feat_neg
            self.spks.remove(spk)
            rand_spk = random.choice(self.spks)
            self.spks.append(spk)
            idx_neg = random.choice(self.spk2wrd_idx[rand_spk])
            batch_data.append(self.feat[idx_neg])

        batch_length = torch.tensor([len(wrd) for wrd in batch_data], device=device)
        batch_order = np.array(sorted(range(len(batch_data)), key=lambda k: len(batch_data[k]), reverse=True))
        batch_data = np.array(batch_data)[batch_order]
        batch_length = batch_length[batch_order]
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
