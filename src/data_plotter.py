import csv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import re
from scipy.interpolate import UnivariateSpline
from shutil import copyfile

SaveFolderGraphs = '/home/sebastian/Dokumente/uni/BT/write/graphs/'
SaveFolderImages = '/home/sebastian/Dokumente/uni/BT/write/thesis/images/'
#SaveFile = 'lernfaehigkeit_cnn_gru_train_loss_0.png'
SaveFile = ''
#Path =  '../data/experiments/logs/sess_1494417541/'

#file0 = 'csv_2017-05-10_11-57-22.csv'
file0 = '../data/experiments/logs/sess_1496395752/csv_2017-06-02_09-27-24.csv'
file1 = '../data/experiments/logs/sess_1496329423/csv_2017-06-01_15-03-21.csv'
file2 = '../data/experiments/logs/sess_1496538409/csv_2017-06-04_01-05-07.csv'
# file1 = 'keras_BLSTM64_2Layer_100sp.csv'
# file2 = 'keras_BLSTM32_2Layer_100sp.csv'
# file3 = 'BiLSTM256_l2_sp100.csv'


#df = pd.read_csv(Path+file0)
df = pd.read_csv(file0, header=None)
acc0 =  df[1]
loss0 = df[2]
val_acc0 = df[3]
val_loss0 = df[4]

#df = pd.read_csv(Path+file1)
df = pd.read_csv(file1, header=None)
acc1 =  df[1]
loss1 = df[2]
val_acc1 = df[3]
val_loss1 = df[4]

#df = pd.read_csv(Path+file2)
df = pd.read_csv(file2, header=None)
acc2 =  df[1]
loss2 = df[2]
val_acc2 = df[3]
val_loss2 = df[4]
#
#df = pd.read_csv(Path+file3)
#acc3 =  df['acc']
#loss3 = df['loss']
#val_acc3 = df['val_acc']
#val_loss3 = df['val_loss']



#val_acc = np.array(val.items(), dtype=dtype)
#print val_acc0.shape
v_acc_arr0 = np.array(acc0)
v_acc_arr1 = np.array(acc1)
v_acc_arr2 = np.array(acc2)
#v_acc_arr = np.array(val_acc)
#v_acc_arr1 = np.array(val_acc1)
#v_acc_arr2 = np.array(val_acc2)
#v_acc_arr3 = np.array(val_acc3)

#x_acc = np.linspace(0, len(val_acc), num=len(val_acc), endpoint=False)
x_acc0 = np.linspace(0, len(acc0), num=len(acc0), endpoint=False)
spl_acc0 = UnivariateSpline(x_acc0, v_acc_arr0)
spl_acc0.set_smoothing_factor(1.8)

#x_acc1 = np.linspace(0, len(val_acc1), num=len(val_acc1), endpoint=False)
x_acc1 = np.linspace(0, len(acc1), num=len(acc1), endpoint=False)
spl_acc1 = UnivariateSpline(x_acc1, v_acc_arr1)
spl_acc1.set_smoothing_factor(1.8)

#x_acc2 = np.linspace(0, len(val_acc2), num=len(val_acc2), endpoint=False) 
x_acc2 = np.linspace(0, len(acc2), num=len(acc2), endpoint=False) 
spl_acc2 = UnivariateSpline(x_acc2, v_acc_arr2)
spl_acc2.set_smoothing_factor(1.8)
#
#x_acc3 = np.linspace(0, len(val_acc), num=len(val_acc), endpoint=False)
#spl_acc3 = UnivariateSpline(x_acc3,v_acc_arr3)
#spl_acc3.set_smoothing_factor(1.8)

#x_train_loss = np.linspace(0, len(loss), num=len(loss), endpoint=False)
#spl_train = UnivariateSpline(x_train_loss, loss)
#spl_train.set_smoothing_factor(0.2)
#
#x_valid_loss = np.linspace(0, len(val_loss), num=len(val_loss), endpoint=False)
#spl_valid = UnivariateSpline(x_valid_loss, val_loss)
#spl_valid.set_smoothing_factor(20)

## Training Loss
x_loss0 = np.linspace(0, len(loss0), num=len(loss0), endpoint=False)
spl_loss0 = UnivariateSpline(x_loss0, loss0)
spl_loss0.set_smoothing_factor(20)

x_loss1 = np.linspace(0, len(loss1), num=len(loss1), endpoint=False)
spl_loss1 = UnivariateSpline(x_loss1, loss1)
spl_loss1.set_smoothing_factor(20)

x_loss2 = np.linspace(0, len(loss2), num=len(loss2), endpoint=False)
spl_loss2 = UnivariateSpline(x_loss2, loss2)
spl_loss2.set_smoothing_factor(20)

## Validation Loss
#x_val_loss0 = np.linspace(0, len(val_loss0), num=len(val_loss0), endpoint=False)
#spl_val_loss0 = UnivariateSpline(x_loss0, val_loss0)
#spl_val_loss0.set_smoothing_factor(20)
#
#x_val_loss1 = np.linspace(0, len(val_loss1), num=len(val_loss1), endpoint=False)
#spl_val_loss1 = UnivariateSpline(x_loss1, val_loss1)
#spl_val_loss1.set_smoothing_factor(20)
#
#x_val_loss2 = np.linspace(2, len(val_loss2), num=len(val_loss2), endpoint=False)
#spl_val_loss2 = UnivariateSpline(x_loss2, val_loss2)
#spl_val_loss2.set_smoothing_factor(20)


type = "loss"

#sav = "keras_biLSTM_1Layer_128_40sp.png"
if type == "acc":
    #plt.plot( spl_acc3(x_acc3), label='BLSTM 256', color='r')
    plt.plot( spl_acc(x_acc), label='CNN-GRU 128 #0', color='b')
    plt.plot( spl_acc1(x_acc1), label='CNN-GRU 128 #1', color='g')
    plt.plot( spl_acc2(x_acc2), label='CNN-GRU 128 #2', color='y')
    #plt.plot(val_acc, color='b', alpha = 0.2)
    #plt.plot(val_acc1, color='g', alpha = 0.2)
    #plt.plot(val_acc2, color='y', alpha = 0.2)
    plt.plot(acc, color='b', alpha = 0.2)
    plt.plot(acc1, color='g', alpha = 0.2)
    plt.plot(acc2, color='y', alpha = 0.2)
    #plt.plot(val_acc3, color='r', alpha = 0.2)
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    #plt.legend(['BLSTM 256', 'BLSTM 128', 'BLSTM64', 'BLSTM32'] , loc='lower right')
    plt.legend(['CNN-GRU 128 #0','CNN-GRU 128 #1','CNN-GRU 128 #2'] , loc='lower right')

#loss plot
if type == "loss":
    plt.plot(spl_loss0(x_loss0), label='CNN-GRU 512 Train', color='g')
    plt.plot(spl_loss1(x_loss1), label='CNN-GRU 128 #1', color='b')
    plt.plot(spl_loss2(x_loss1), label='CNN-GRU 128 #2', color='r')
    #plt.plot(spl_val_loss0(x_val_loss0), label='CNN-GRU 512 Val', color='b')
    plt.plot(loss0, color='g', alpha = 0.2)
    plt.plot(loss1, color='b', alpha = 0.2)
    plt.plot(loss2, color='r', alpha = 0.2)
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    #plt.legend(['CNN-GRU 512 Train', 'CNN-GRU 512 Val'],loc='upper right')


plt.grid()

if len(SaveFile) > 0:
    plt.savefig(SaveFolderGraphs+SaveFile, dpi=170)
    copyfile(SaveFolderGraphs+SaveFile, SaveFolderImages+SaveFile)

plt.show()
