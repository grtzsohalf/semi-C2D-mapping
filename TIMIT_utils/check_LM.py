from collections import Counter

LM_file = 'LM.txt'

counter = Counter()
with open(LM_file, 'r') as f:
    for line in f:
        counter[float(line.strip().split()[-1])] += 1
for v, c in sorted(counter.items(), key=lambda kv: kv[0], reverse=True):
    print (v, c)
