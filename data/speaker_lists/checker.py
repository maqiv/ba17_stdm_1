with open('speakers_590_clustering_without_raynolds.txt', 'rb') as f:
    v2 = f.readlines()

with open('speakers_40_clustering_vs_reynolds.txt', 'rb') as f:
    v3 = f.readlines()

c = 0
for s in v3:
    print s in v2
    if s in v2:
        c += 1

print c