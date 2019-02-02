import os
import pickle

read_train_phn_txt = 'processed/timit-train-phn-caption.txt'
write_train_phn_txt = 'processed/timit-train-phn-caption-39.txt'
read_train_phn_pkl = 'processed/timit-train-phn.pkl'
write_train_phn_pkl = 'processed/timit-train-phn-39.pkl'

read_test_phn_txt = 'processed/timit-test-phn-caption.txt'
write_test_phn_txt = 'processed/timit-test-phn-caption-39.txt'
read_test_phn_pkl = 'processed/timit-test-phn.pkl'
write_test_phn_pkl = 'processed/timit-test-phn-39.pkl'

mapping = {'ao': 'aa',
           'ax': 'ah',
           'ax-h': 'ah',
           'axr': 'er',
           'hv': 'hh',
           'ix': 'ih',
           'el': 'l',
           'em': 'm',
           'en': 'n',
           'nx': 'n',
           'eng': 'ng',
           'zh': 'sh',
           'ux': 'uw',
           'pau': 'h#',
           'epi': 'h#',
           'q': 't'}

closures = {'bcl': ['b'], 
            'dcl': ['d', 'jh'], 
            'pcl': ['p'], 
            'gcl': ['g'], 
            'tcl': ['t', 'ch'], 
            'kcl': ['k']}


# 1. txt
def map_txt(read_file, write_file):
    with open(read_file, 'r') as f_in:
        with open(write_file, 'w') as f_out:
            for line in f_in:
                phns = line.strip().split(' ')
                for i, phn in enumerate(phns):
                    # 61 -> 39 types
                    if phn in mapping:
                        phns[i] = mapping[phn]

                    # check closure in ['bcl', 'dcl', 'gcl', 'pcl', 'tcl', 'kcl']
                    elif phn in closures:
                        if i != len(phns)-1 and phns[i+1] in closures[phn]:
                            phns[i] = 'h#'
                        else:
                            phns[i] = closures[phn][0]

                    # check for 'dx' -> 't'
                    elif phn == 'dx':
                        phns[i] = 't'
                f_out.write(' '.join(phns) + '\n')


# 2. pkl
def map_pkl(read_file, write_file):
    pkl_list = []
    with open(read_file, 'rb') as f_in:
        pkl_list = pickle.load(f_in)

    for i, utt in enumerate(pkl_list):
        for j, phn_tuple in enumerate(utt):
            # 61 -> 39 types
            if phn_tuple[0] in mapping:
                pkl_list[i][j][0] = mapping[phn_tuple[0]]

            # check closure in ['bcl', 'dcl', 'gcl', 'pcl', 'tcl', 'kcl']
            elif phn_tuple[0] in closures:
                if j != len(utt)-1 and utt[j+1][0] in closures[phn_tuple[0]]:
                    pkl_list[i][j][0] = 'h#'
                else:
                    pkl_list[i][j][0] = closures[phn_tuple[0]][0]

            # check for 'dx' -> 't'
            elif phn_tuple[0] == 'dx':
                pkl_list[i][j][0] = 't'

    with open(write_file, 'wb') as f_out:
        pickle.dump(pkl_list, f_out)


###
map_txt(read_train_phn_txt, write_train_phn_txt)
map_txt(read_test_phn_txt, write_test_phn_txt)
map_pkl(read_train_phn_pkl, write_train_phn_pkl)
map_pkl(read_test_phn_pkl, write_test_phn_pkl)
