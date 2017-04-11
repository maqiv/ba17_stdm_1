with open('/home/patman/pa/1_Code/data/speaker_lists/speakers_100_50w_50m_not_reynolds.txt', 'rb') as f:
    v2 = f.readlines()

with open('/home/patman/pa/1_Code/data/speaker_lists/speakers_80_clustering.txt', 'rb') as f:
    v3 = f.readlines()

c = 0
for s in v3:
    print s in v2
    if s in v2:
        c += 1

print c
