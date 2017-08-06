import random

SPEAKER_LIST = '/home/patman/pa/1_Code/data/speaker_lists/speakers_190_clustering.txt'
#SPEAKER_LIST = 'speakers_all.txt'
#SPEAKER_LIST = '/home/patman/pa/data/speaker_lists/speakers_40_clustering_vs_reynolds.txt'
valid_speakers = []
with open(SPEAKER_LIST, 'rb') as f:
    for line in f:
        valid_speakers.append(line.replace('\n', ''))


f = 0.0
m = 0.0
for s in valid_speakers:
	if s[0] == 'F':
		f += 1
	if s[0] == 'M':
		m += 1


ratio = f/len(valid_speakers)
print m
print f
print ratio
print int(80*ratio)
print int(80*(1-ratio))
fl = []
ml = []
for s in valid_speakers:
	if s[0] == 'F':
		fl.append(s)
	if s[0] == 'M':
		ml.append(s)

new_speaker_list = []
new_speaker_list.extend(fl[:int(20*ratio)])
print len(new_speaker_list)
new_speaker_list.extend(ml[:int(20*(1-ratio))])

clust = [] 
clust.extend(fl[(int(80*ratio)):])
clust.extend(ml[int(80*(1-ratio)):])

random.shuffle(new_speaker_list)
thefile = open('train_speakers_20_clustering.txt', 'w')
for item in new_speaker_list:
  thefile.write("%s\n" % item)

#thefile.close()
#random.shuffle(clust)
#thefile = open('speakers_190_clustering.txt', 'w')
#for item in clust:
#  thefile.write("%s\n" % item)

print new_speaker_list
print len(new_speaker_list)




