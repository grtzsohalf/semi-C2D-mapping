import time
import math
import pickle
from collections import Counter
from tqdm import tqdm
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

import os
import sys

from tensorboardX import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################
# Read and write pickle files. 
#

def read_pkl(pkl_file):
    return pickle.load(open(pkl_file, 'rb'))

def write_pkl(data, pkl_file): 
    with open(pkl_file, 'ab') as f:
        pickle.dump(data, f)


######################################################################
# Helper functions for pytorch
# ----------------------------
#

def onehot(input_x, encode_dim=None):
    if encode_dim is None:
        encode_dim = torch.max(input_x) + 1
    input_x = input_x.type(torch.LongTensor).unsqueeze(2)
    return input_x.new_zeros(input_x.size(0), input_x.size(1), encode_dim).scatter_(-1, input_x, 1)

def sample_gumbel(size, eps=1e-20):
    u = torch.rand(size, device=device)
    sample = -torch.log(-torch.log(u + eps) + eps)
    return sample

def gumbel_softmax_sample(logits, temperature=1.):
    y = logits + sample_gumbel(logits.size(), device=device)
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=1., hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        _, max_ind = torch.max(y, dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y).scatter_(-1, max_ind, 1.0)
        y = (y_hard - y).detach() + y
    return y


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
                #init.normal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
                #init.normal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def _inflate(tensor, times, dim):
    repeat_dims = [1] * tensor.dim()
    repeat_dims[dim] = times
    return tensor.repeat(*repeat_dims)

def _inflate_np(np_array, times, dim):
    repeat_dims = [1] * np_array.ndim
    repeat_dims[dim] = times
    return np_array.repeat(repeat_dims)

def adjust_learning_rate(optimizer, lr):
    state_dict = optimizer.state_dict()
    for param_group in state_dict['param_groups']:
        param_group['lr'] = lr
    optimizer.load_state_dict(state_dict)
    return lr

def normalize_judge_scores(judge_scores, lengths):
    # judge_scores: [batch_size x frame_length]
    # lengths: [len_1, len_2, ..., len_n]
    for i in range(judge_scores.size(0)):
        miu = torch.mean(judge_scores[i, :lengths[i]])
        std = torch.std(judge_scores[i, :lengths[i]]) + 1e-9
        judge_scores[i, :lengths[i]] = (judge_scores[i, :lengths[i]] - miu) / std
    return

def pad_sequence(np_list, padding_value=0, length=None):
    tensor_list = [torch.from_numpy(np_array) for np_array in np_list]
    #print('tensor length: ', len(tensor_list))
    #for tensor in tensor_list:
    #    print('shape', tensor.size())
    pad_tensor = torch.nn.utils.rnn.pad_sequence(tensor_list, batch_first=True, padding_value=padding_value)
    if length is None:
        return pad_tensor
    else:
        pad_length = pad_tensor.size()[1]
        if pad_length >= length:
            return pad_tensor[:, :length, :]
        else:
            pad = torch.zeros([pad_tensor.size()[0], length-pad_length, pad_tensor.size()[2]])
            return torch.cat([pad_tensor, pad], 1)

def mask_with_length(tensor, length):
    '''
    mask the last two dimensin of the tensor with length
    tensor: rank>2 [... length x feat_dim]
    length: 1D array
    the first dimension of the tensor and length must be the same
    '''
    assert (len(tensor.size()) >= 2)
    assert (tensor.size()[0] == length.size()[0])

    tensor_length = tensor.size()[-2]
    rank = len(tensor.size())

    a = torch.arange(tensor_length, device=device).unsqueeze(-1).expand_as(tensor).int()
    b = length.view(-1, *([1]*(rank-1))).int()

    mask = torch.lt(a, b).float().to(device)
    return tensor.float()*mask, mask


######################################################################
# A helper function for RNN
# -------------------------
#

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


######################################################################
# A helper class for tensorboardX
# ---------------------
#

class Logger(object):

    def __init__(self, logdir='./log'):
        self.writer = SummaryWriter(logdir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def text_summary(self, tag, value, step):
        self.writer.add_text(tag, value, step)


######################################################################
# Helper functions for sequences
# ------------------------------
#

def remove_pad_eos(sequences, eos=2):
    sub_sequences = []
    for sequence in sequences:
        try:
            eos_index = next(i for i, v in enumerate(sequence) if v == eos)
        except StopIteration:
            eos_index = len(sequence)
        sub_sequence = sequence[:eos_index]
        sub_sequences.append(sub_sequence)
    return sub_sequences

def remove_pad_eos_batch(sequences, eos=2):
    length = sequences.size(1)
    indices = [length for _ in range(sequences.size(0))]
    for i, elements in enumerate(zip(*sequences)):
        indicators = [element == eos for element in elements]
        indices = [i if index == length and indicator else index for index, indicator in zip(indices, indicators)]
    sub_sequences = [sequence[:index] for index, sequence in zip(indices, sequences)]
    return sub_sequences

def ind2character(sequences, non_lang_syms, vocab):
    inv_vocab = {v: k for k, v in vocab.items()}
    non_lang_syms_ind = [vocab[sym] for sym in non_lang_syms]
    char_seqs = []
    for sequence in sequences:
        char_seq = [inv_vocab[ind] for ind in sequence if ind not in non_lang_syms_ind]
        #char_seq = [inv_vocab[ind] for ind in sequence]
        char_seqs.append(char_seq)
    return char_seqs

def to_sents(ind_seq, vocab, non_lang_syms):
    char_list = ind2character(ind_seq, non_lang_syms, vocab)
    sents = char_list_to_str(char_list)
    return sents

def char_list_to_str(char_lists):
    sents = []
    for char_list in char_lists:
        sent = ''.join([char if char != '<space>' else ' ' for char in char_list])
        sents.append(sent)
    return sents


######################################################################
# These are helper functions to turn Unicode characters to ASCII, 
# make everything lowercase, and trim most punctuation.
#

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicode_to_Ascii(self, s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(self, s):
    s = unicode_to_Ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


######################################################################
# A helper function for infinite iterations
# -----------------------------------------
#

def infinite_iter(iterable):
    it = iter(iterable)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(iterable)


######################################################################
# Helper functions for time
# -------------------------
#

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


############################################################################
# Helper functions for tolerance precision, recall, R score and segmentation
# --------------------------------------------------------------------------
#

def tolerance_precision(bounds, seg_bnds, tolerance_window):
    #Precision
    if len(seg_bnds) == 0:
        return 0

    hit = 0.0
    for bound in seg_bnds:
        for l in range(tolerance_window + 1):
            if (bound + l in bounds) and (bound + l > 0):
                hit += 1
                break
            elif (bound - l in bounds) and (bound - l > 0):
                hit += 1
                break
    return (hit / (len(seg_bnds)))


def tolerance_recall(bounds, seg_bnds, tolerance_window):
    #Recall
    hit = 0.0
    for bound in bounds:
        for l in range(tolerance_window + 1):
            if (bound + l in seg_bnds) and (bound + l > 0):
                hit += 1
                break
            elif (bound - l in seg_bnds) and (bound - l > 0):
                hit += 1
                break

    return (hit / (len(bounds)))

def thresh_dist_segment_eval(scores, bounds_list, criteria_idx, tolerance_window, diff_thresh):
    recall_rate_list = []
    precision_rate_list = []

    seg_bnds = []

    for i in range(len(bounds_list)):
        bnd = []

        for frame in range(scores.shape[1]):
            if scores[i][frame][criteria_idx] > diff_thresh:
                bnd.append(frame)

        seg_bnds.append(bnd)

    #Recall
    for i in range(len(bounds_list)):
        single_utt_recall = tolerance_recall(bounds_list[i], \
                                seg_bnds[i], tolerance_window)
        recall_rate_list.append(single_utt_recall)

    #Precision
    for i in range(len(seg_bnds)):
        single_utt_precision = tolerance_precision(bounds_list[i], \
                                seg_bnds[i], tolerance_window)
        precision_rate_list.append(single_utt_precision)

    return recall_rate_list, precision_rate_list

def thresh_segmentation_eval(scores, bounds_list, tolerance_window, diff_thresh_factor):

    recall_rate_list = []
    precision_rate_list = []

    seg_bnds = []

    for i in range(len(bounds_list)):
        bnd = []
        bnd_set = set()
        bnd_set.add(0)
        diff_sorted_idx = scores[i].argsort()[::-1]

        iter_idx = 0
        #diff_thresh = diff_thresh_factor * np.fabs(np.mean(scores[i]))
        diff_thresh = diff_thresh_factor #* np.fabs(np.mean(scores[i]))
        while True:
            if iter_idx >=len(diff_sorted_idx): break

            if diff_sorted_idx[iter_idx] in bnd_set:
                iter_idx += 1
                continue

            if scores[i, diff_sorted_idx[iter_idx]] < diff_thresh:
                if len(bnd) == 0:
                    bnd.append(-1)
                break

            bnd.append(diff_sorted_idx[iter_idx])
            bnd_set.add(diff_sorted_idx[iter_idx])

            #local maximum
            bnd_set.add(diff_sorted_idx[iter_idx] + 1)
            bnd_set.add(diff_sorted_idx[iter_idx] - 1)

            iter_idx += 1

        seg_bnds.append(bnd)

    #Recall
    for i in range(len(bounds_list)):
        single_utt_recall = tolerance_recall(bounds_list[i], \
                                seg_bnds[i], tolerance_window)
        recall_rate_list.append(single_utt_recall)

    #Precision
    for i in range(len(seg_bnds)):
        single_utt_precision = tolerance_precision(bounds_list[i], \
                                seg_bnds[i], tolerance_window)
        precision_rate_list.append(single_utt_precision)

    return recall_rate_list, precision_rate_list


def r_val_eval(u_p, u_r):
    if u_r == 0 or u_p == 0:
        u_f = -1.
        u_r_val = -1.
    else:
        u_f = 2 * u_p * u_r / (u_p + u_r)
        u_os = (u_r/u_p - 1) * 100
        u_r_val = 1 - (math.fabs(math.sqrt((100-u_r)*(100-u_r) + \
         math.pow(u_os, 2))) + math.fabs( (u_r - 100 - u_os)/math.sqrt(2))) / 200
    return u_r_val * 100


# Add for evaluation showing on stdout
def print_progress(progress, total, precision, recall, output_msg):
    output_msg = 'Progress: {}/{} Precision:{:.4f}, Recall:{:.4f}'.format(progress, total, precision, recall)
    sys.stdout.write('\b'*len(output_msg))
    sys.stdout.write(output_msg)
    sys.stdout.flush()


def count_LM(word_utts, n_grams=3, LM_file=None):
    if LM_file and os.path.isfile(LM_file):
        with open(LM_file, 'rb') as f:
            return pickle.load(f)

    counter = Counter()
    for word_utt in word_utts:
        counter[('<BOS>', '<BOS>', word_utt[0])] += 1
        counter[('<BOS>', word_utt[0], word_utt[1])] += 1
        for i, word in enumerate(word_utt[:-n_grams+1]):
            counter[(word, word_utt[i+1], word_utt[i+2])] += 1
        counter[(word_utt[-2], word_utt[-1], '<EOS>')] += 1
        counter[(word_utt[-1], '<EOS>', '<EOS>')] += 1

    num_pairs = len(counter)
    print ('Total num of pairs: ', num_pairs)

    for pair, counts in counter.items():
        counter[pair] = math.log(counts/num_pairs)

    if LM_file:
        with open(LM_file, 'wb') as f:
            pickle.dump(counter, f)

    return counter


def getNN(top_N, emb1, emb2, words):
    start_time = time.time()
    emb1 = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
    emb2 = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
    # [num_1 x num_2]
    cos_sim = np.dot(emb1, emb2.T)

    top_N_index = np.flip(np.argsort(cos_sim, axis=1), axis=1)[:, :top_N]
    print (f'Time cost of getNN: {time.time() - start_time}')

    return np.array([cos_sim[x, y] for x, y in zip(np.arange(len(cos_sim)), top_N_index)]), \
        np.array([words[z] for z in top_N_index])


def beam_search(sim_values, sim_words, LM_probs, width, weight_LM, result_file, n_grams=3):
    '''
    sim_values -> [n_utts x n_time_steps x n_log_values(from high to low)]
    sim_words -> [n_utts x n_time_steps x n_words]
    LM_probs -> {pair: log_prob}
    '''
    trans = []
    with open(result_file, 'w') as f:
        for u in tqdm(range(len(sim_values))):
            paths = [(0., ['<BOS>', '<BOS>'])]

            # Initial three time steps
            probs_at_t = {}
            for i, (v, w) in enumerate(zip(sim_values[u][0], sim_words[u][0])):
                v = math.log(v)
                temp_path_words = list(paths[0][1])
                temp_path_words.append(w)
                if not ('<BOS>', '<BOS>', w) in LM_probs:
                    probs_at_t[v - 10000. + random.random()/10000] = temp_path_words
                else:
                    probs_at_t[v + \
                               weight_LM * LM_probs[('<BOS>', '<BOS>', w)] + \
                               random.random()/10000] = temp_path_words
                    # probs_at_t[paths[j][0] + p] = temp_path_words
            sorted_probs_at_t = sorted(probs_at_t.items(), key=lambda kv: kv[0], reverse=True)
            paths = sorted_probs_at_t[:width]

            # At each time step
            for t in range(1, len(sim_values[u])):
                probs_at_t = {}
                # For each of previous K paths
                for j in range(width):
                    for i, (v, w) in enumerate(zip(sim_values[u][t], sim_words[u][t])):
                        v = math.log(v)
                        temp_path_words = list(paths[j][1])
                        temp_path_words.append(w)
                        if not (paths[j][1][-2], paths[j][1][-1], w) in LM_probs:
                            probs_at_t[paths[j][0] + v - 10000. + random.random()/10000] = temp_path_words
                        else:
                            probs_at_t[paths[j][0] + v + \
                                       weight_LM * LM_probs[(paths[j][1][-2], paths[j][1][-1], w)] + \
                                       random.random()/10000] = temp_path_words
                            # probs_at_t[paths[j][0] + p] = temp_path_words
                sorted_probs_at_t = sorted(probs_at_t.items(), key=lambda kv: kv[0], reverse=True)
                paths = sorted_probs_at_t[:width]
                # print (paths[0])

            # Last two time steps
            for _ in range(2):
                probs_at_t = {}
                # For each of previous K paths
                for j in range(width):
                    try:
                        temp_path_words = list(paths[j][1])
                    except:
                        print (len(paths))
                        print (paths)
                        exit()
                    temp_path_words.append('<EOS>')
                    if not (paths[j][1][-2], paths[j][1][-1], '<EOS>') in LM_probs:
                        probs_at_t[paths[j][0] + 0. - 10000. + random.random()/10000] = temp_path_words
                    else:
                        probs_at_t[paths[j][0] + 0. + \
                                   weight_LM * LM_probs[(paths[j][1][-2], paths[j][1][-1], '<EOS>')] + \
                                   random.random()/10000] = temp_path_words
                        # probs_at_t[paths[j][0] + p] = temp_path_words
                sorted_probs_at_t = sorted(probs_at_t.items(), key=lambda kv: kv[0], reverse=True)
                paths = sorted_probs_at_t[:width]

            prob = paths[0][0]
            words = paths[0][1][2:-2]
            f.write(' '.join(words) + '\n')
            trans.extend(words)
    return np.array(trans)
