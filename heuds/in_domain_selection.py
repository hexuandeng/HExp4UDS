import kenlm
import math
base = kenlm.Model('datasets/base.bin')
uds = kenlm.Model('datasets/uds.bin')
difs = []
with open('datasets/news.2021.en.shuffled.deduped', 'r', encoding='utf-8') as f:
    for line in f:
        if len(line.strip().split()) <= 5:
            dif = 100
        else:
            score1 = base.score(line.strip())
            score2 = uds.score(line.strip())
            dif = (score1 - score2) / len(line.strip().split())
        difs.append(dif)

idx = sorted(range(len(difs)), key=lambda i: difs[i])
idx = sorted(idx[: int(1.5e5)])

with open('datasets/news.2021.en.shuffled.deduped', 'r', encoding='utf-8') as f,\
    open('datasets/news.2021.en.shuffled.uds.clean', 'w', encoding='utf-8') as w:
    for id, line in enumerate(f):
        if id == idx[0]:
            w.write(line)
            idx.pop(0)
