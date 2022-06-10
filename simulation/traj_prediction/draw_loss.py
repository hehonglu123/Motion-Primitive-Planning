import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf

folder = 'result_lr0001/'
with open(folder+'train_loss_save.npy','rb') as f:
    train_loss_epoch = np.load(f)
with open(folder+'test_loss_save.npy','rb') as f:
    test_loss_epoch = np.load(f)

# print(train_loss_epoch)

for ji in range(6):
    plt.clf()
    plt.plot(np.vstack((train_loss_epoch[ji],test_loss_epoch[ji])).T)
    plt.legend(['Training','Testing'])
    plt.title('Training/Testing Loss across Epoch, Joint '+str(ji+1))
    plt.xticks(range(0,100,10))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('result_loss'+str(ji+1)+'png')