import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import core.settings as settings





def save_accuracy_plot(history, name):
	sav = settings.PLOT_PATH+name+"_acc.png"
	fig = plt.figure()
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train_acc', 'val_acc'], loc='lower right')
	plt.grid()
	plt.savefig(sav)

def save_loss_plot(history, name):
	sav = settings.PLOT_PATH+name+"_loss.png"
	fig = plt.figure()
	ax = fig.gca()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train_loss', 'val_loss'], loc='upper right')
	plt.grid()
	plt.savefig(sav)

