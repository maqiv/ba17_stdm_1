from ConfigParser import SafeConfigParser



def getParameters():
	parser = SafeConfigParser()
	parser.read('params.ini')
	iterations = parser.getint('Params', 'ITERATIONS')
	layers = parser.getint('Params', 'LAYERS')
	batch_size = parser.getint('Params', 'BATCH_SIZE')
	l_rate = parser.getfloat('Params', 'LEARNING_RATE')
	hidden = parser.getint('Params', 'HIDDEN')
	return iterations, layers, l_rate , hidden, batch_size

print getParameters()
