with open('/home/patman/pa/1_Code/data/training/TIMIT/crp/timit-test.crp', 'rb') as f:
    v2 = f.readlines()

def rem_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


l = []
for v in v2:
	l.append(v[9:14])

print len(l)

l = rem_duplicates(l)

print len(l)


thefile = open('test_speakers.txt', 'w')
for item in l:
  thefile.write("%s\n" % item)

