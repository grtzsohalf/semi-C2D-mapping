import os 
import pickle
import syllabifier 

train_phn_pkl = 'processed/timit-train-phn-39.pkl'
train_wrd_pkl = 'processed/timit-train-wrd.pkl'
train_phn_in_wrd_pkl = 'processed/timit-train-phn-in-wrd.pkl'
train_slb_pkl = 'processed/timit-train-slb.pkl'

test_phn_pkl = 'processed/timit-test-phn-39.pkl'
test_wrd_pkl = 'processed/timit-test-wrd.pkl'
test_phn_in_wrd_pkl = 'processed/timit-test-phn-in-wrd.pkl'
test_slb_pkl = 'processed/timit-test-slb.pkl'


# Currently only for 'DX' -> 'T'
# def _ambig(phns):
    # phns = phns.split(' ')
    # phns_list = []
    # new_phns = ''
    # for i, phn in enumerate(phns):
        # if phn == 'DX':
            # phn = 'T'
        # if i == 0:
            # new_phns += phn
        # else:
            # new_phns += (' ' + phn)
    # phns_list.append(new_phns)

    # return phns_list


def load_pkl(file_name):
    return pickle.load(open(file_name, 'rb'))

def dump_pkl(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


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
                phn_in_wrd.append(phn_tuple)
        phn_in_wrd_data.append(phn_in_wrd_utt)

    return phn_in_wrd_data


def wrd_to_slbs(phn_in_wrd, wrd_tuple):
    lang = syllabifier.English
    # phns = ' '.join([phn_tuple[0] for phn_tuple in phn_in_wrd]).upper()
    phns = ''
    for i, phn_tuple in enumerate(phn_in_wrd):
        phn_token = phn_tuple[0]
        # check 'h#'
        if phn_token == 'h#':
            continue
        if i == 0:
            phns += phn_token
        else:
            phns += (' ' + phn_token)

    phns = phns.upper()
    # try:
    slbs = syllabifier.syllabify(lang, phns)
    slbs = syllabifier.stringify(slbs)
    # except:
        # check ambiguous phn
        # phns_list = _ambig(phns)
        # for j, phns in enumerate(phns_list):
            # try:
                # slbs = syllabifier.syllabify(lang, phns)
                # slbs = syllabifier.stringify(slbs)
                # break
            # except:
                # if j == len(phns_list)-1:
                    # print ('Exception!')
                    # print (wrd_tuple, phns)
                    # raise
        # except:
            # try:
                # seg = _exception(wrd_tuple[0])
                # slbs = ' . '.join([' '.join(phns[seg[k]: seg[k+1]]) for k in range(len(seg)-1)])
                # # print (slbs)
            # except:
                # print (wrd_tuple, phns)
    return slbs.lower()


def make_syllable_data(phn_in_wrd_data, wrd_data):
    slb_data = []
    for n_utt, (phn_in_wrd_utt, wrd_utt) in enumerate(zip(phn_in_wrd_data, wrd_data)):
        slb_utt = []
        for n_wrd, (phn_in_wrd, wrd_tuple) in enumerate(zip(phn_in_wrd_utt, wrd_utt)):
            slbs = wrd_to_slbs(phn_in_wrd, wrd_tuple).split(' . ')
            slbs = [s.strip().split(' ') for s in slbs]

            phn_count = 0
            for slb in slbs:
                if not slb[0]:
                    print (n_utt, n_wrd, wrd_tuple, slbs, phn_in_wrd)
                    continue
                # print (slb)
                slb_tuple = ['', -1, -1]
                for i, phn_token in enumerate(slb):
                    while True:
                        phn_tuple = phn_in_wrd[phn_count]
                        phn_count += 1
                        # check if they are the same
                        if phn_tuple[0] == phn_token:
                            break
                    slb_tuple[0] += (' ' + phn_token)
                    if i == 0:
                        slb_tuple[1] = phn_tuple[1]
                    if i == len(slb)-1:
                        slb_tuple[2] = phn_tuple[2]
                        break
                # if slb_tuple[2] == -1:
                    # print (n_utt, n_wrd, wrd_tuple, slbs, slb, phn_in_wrd, slb_tuple, phn_tuple, i, len(slb)-1)
                slb_utt.append(slb_tuple)
        slb_data.append(slb_utt)
    return slb_data



###
train_phn_data = load_pkl(train_phn_pkl)
train_wrd_data = load_pkl(train_wrd_pkl)
train_phn_in_wrd_data = make_phn_in_wrd(train_phn_data, train_wrd_data)
train_slb_data = make_syllable_data(train_phn_in_wrd_data, train_wrd_data)
dump_pkl(train_slb_data, train_slb_pkl)

test_phn_data = load_pkl(test_phn_pkl)
test_wrd_data = load_pkl(test_wrd_pkl)
test_phn_in_wrd_data = make_phn_in_wrd(test_phn_data, test_wrd_data)
test_slb_data = make_syllable_data(test_phn_in_wrd_data, test_wrd_data)
dump_pkl(test_slb_data, test_slb_pkl)
