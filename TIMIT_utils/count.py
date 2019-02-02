import os
import pickle

train_mfcc = 'processed/timit-train-mfcc-nor.pkl'
test_mfcc = 'processed/timit-test-mfcc-nor.pkl'

train_phn = 'processed/timit-train-phn.pkl'
test_phn = 'processed/timit-test-phn.pkl'

train_wrd = 'processed/timit-train-wrd.pkl'
test_wrd = 'processed/timit-test-wrd.pkl'


''' 1. min/max lengths of an utterance '''
def print_min_max_utt_len(file_name):
    min_len = 1000000
    max_len = 0

    mfcc_list = pickle.load(open(file_name, 'rb'))
    for utt in mfcc_list:
        utt_len = len(utt)
        if utt_len < min_len:
            min_len = utt_len 
        if utt_len > max_len:
            max_len = utt_len 
    print (f'num_utts: {len(mfcc_list)}')
    print (f'min_utt: {min_len}')
    print (f'max_utt: {max_len}\n')


''' 2. min/max lengths of a word/syllable/phoneme '''
def print_min_max_unit_len(file_name, unit_type):
    min_len = 1000000
    max_len = 0
    num_units = 0

    unit_dict = {}

    unit_list = pickle.load(open(file_name, 'rb'))
    # unit_list = unit_list[1:-1] if unit_type == 'phoneme'

    for i, utt in enumerate(unit_list):
        for unit in utt:
            num_units += 1
            unit_name = unit[0]
            unit_len = unit[2] - unit[1]
            # if unit_len == 0:
                # print (i, unit)
            if not unit_name in unit_dict:
                unit_dict[unit_name] = [unit_len, unit_len]
            else:
                if unit_len < unit_dict[unit_name][0]:
                    unit_dict[unit_name][0] = unit_len 
                if unit_len > unit_dict[unit_name][0]:
                    unit_dict[unit_name][1] = unit_len 
            if unit_len < min_len:
                min_len = unit_len 
            if unit_len > max_len:
                max_len = unit_len 

    # print (f'num_units: {num_units}')
    # print (f'min_unit: {min_len}')
    # print (f'max_unit: {max_len}')
    sorted_dict = sorted(unit_dict.items(), key=lambda kv: kv[0])
    # sorted_dict = sorted(unit_dict.items(), key=lambda kv: kv[1][0])
    # if unit_type == 'phn':
        # print ('For each kind of unit:')
        # for unit_min_max in sorted_dict:
            # print (unit_min_max) 
    # print ('\n')
    print ([u[0] for u in sorted_dict])
    print (len(unit_dict), '\n')


###
# print_min_max_utt_len(train_mfcc)
# print_min_max_utt_len(test_mfcc)
print_min_max_unit_len(train_phn, 'phn')
print_min_max_unit_len(test_phn, 'phn')
# print_min_max_unit_len(train_wrd, 'wrd')
# print_min_max_unit_len(test_wrd, 'wrd')
