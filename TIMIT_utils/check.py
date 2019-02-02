import pickle
from collections import Counter


closures = {'bcl': [], 
            'dcl': [], 
            'pcl': [], 
            'gcl': [], 
            'tcl': [], 
            'kcl': []}


# filename = 'processed/timit-train-meta.pkl'
# data = []
# with open(filename, 'rb') as f:
    # data = pickle.load(f)
# print (len(data))
# print (len(data['prefix']))
# print (len(data['mfcc']['mean']))
# print (len(data['fbank']['mean']))
# print (len(data['spec']['mean']))
# print (len(data['fbank-80']['mean']))


# filename = 'processed/timit-train-mfcc-nor.pkl'
# data = []
# with open(filename, 'rb') as f:
    # data = pickle.load(f)
# print (len(data))
# print (len(data[0]))
# print (len(data[0][0]))
# print (len(data[1]))
# print (len(data[1][0]))
# print (len(data[2]))
# print (len(data[2][0]))


filename = 'processed/timit-train-phn.pkl'
# filename = 'processed/timit-train-phn-39.pkl'
data = []
with open(filename, 'rb') as f:
    data = pickle.load(f)
# print (len(data), '\n')
# print (data[0], '\n')
print (data[1], '\n')
# print (data[230], '\n')
# print (data[313], '\n')
# print (data[963], '\n', '\n')
# print (' '.join([phn[0] for phn in data[0]]))
# print (len(data[0]))
# print (len(data[1]))


# filename = 'processed/timit-train-phn.pkl'
filename = 'processed/timit-train-slb.pkl'
data = []
with open(filename, 'rb') as f:
    data = pickle.load(f)
# print (len(data), '\n')
# print (data[0], '\n')
print (data[1], '\n')
# print (data[230], '\n')
# print (data[313], '\n')
# print (data[963], '\n', '\n')
# print (' '.join([phn[0] for phn in data[0]]))
# print (len(data[0]))
# print (len(data[1]))


# filename = 'processed/timit-train-phn-caption.txt'
# filename = 'processed/timit-test-phn-caption.txt'
# data = []
# with open(filename, 'r') as f:
    # data = f.readlines()
# print (len(data[0].strip().split()))
# print (len(data[1].strip().split()))
# counter = Counter()
# total = 0
# for seq in data:
    # phns = seq.strip().split()
    # for i, phn in enumerate(phns):
        # if phn in closures and i != len(phns)-1:
            # if not phns[i+1] in closures[phn]:
                # closures[phn].append(phns[i+1])
    # counter[len(phns)] += 1
    # total += len(phns)
# print (closures)
# sorted_counter = sorted(counter.items(), key=lambda kv: kv[0])
# for i in sorted_counter:
    # print (i)
# print (total)


# filename = 'processed/timit-train-self-lexicon.pkl'
# data = []
# with open(filename, 'rb') as f:
    # data = pickle.load(f)
# print (len(data))
# print (data['he'])


# filename = 'processed/timit-train-transcript.pkl'
# data = []
# with open(filename, 'rb') as f:
    # data = pickle.load(f)
# print (len(data))
# print (data[0])


filename = 'processed/timit-train-wrd.pkl'
# filename = 'processed/timit-test-wrd.pkl'
data = []
with open(filename, 'rb') as f:
    data = pickle.load(f)
# print (len(data))
# print (data[0], '\n')
print (data[1], '\n')
# print (data[230], '\n')
# print (data[313], '\n')
# print (data[963], '\n')
# print (data[1])
# counter = Counter()
# total = 0
# for seq in data:
    # counter[len(seq)] += 1
    # total += len(seq)
# sorted_counter = sorted(counter.items(), key=lambda kv: kv[0])
# for i in sorted_counter:
    # print (i)
# print (total)
