with open('test_speakers.txt', 'rb') as f:
    v2 = f.readlines()


nl = []
for i in range(100):
	nl.append(v2[i])


print len(nl)


thefile = open('speakers_100_clustering.txt', 'w')
for item in nl:
  thefile.write("%s\n" % item)
