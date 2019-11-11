import librosa
import numpy as np
import os
from glob import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sr = 16000
frame = int(16000 * 0.032)
overlap = int(16000 * 0.016)

num = str(250)
ckpt = str(21)
source = np.load('./result/{}_np_file/before_predict_epoch={}.npy'.format(ckpt,num))
target = np.load('./result/{}_np_file/y_real_epoch={}.npy'.format(ckpt,num))
train = np.load('./result/{}_np_file/after_predict_epoch={}.npy'.format(ckpt,num))
source = source[:, 1:]
target = target[:, 1:]
print(np.shape(source), np.shape(target), np.shape(train))

length = len(source.T)
#print(length)


pic_dir = './pic'

for i in range(length):
    #fig = plt.plot(figsize=(30,10))
    fig = plt.figure(figsize=(30,10))
    
    #plt.figure(figsize=(30,10))
    
    ax1 = fig.add_subplot(1, 3, 1)
    #fig1 = plt.subplot(1, 3, 1)
    plt.title('source')
    src = source[:,i]
    
    x = range(len(src))
    
    plt.bar(x, src, color="red")
    plt.xlabel('frequency')
    plt.ylabel('magnitude')
    
    tar = target[:,i]
    
    ax2 = fig.add_subplot(1, 3, 2)
    #fig2 = plt.subplot(1, 3, 2)
    plt.title('target')
    plt.bar(x, tar, color="green")
    plt.xlabel('frequency')
    plt.ylabel('magnitude')
    
    res = train[:, i]
    
    ax3 = fig.add_subplot(1, 3, 3)
    #fig3 = plt.subplot(1, 3, 3)
    plt.title('train')
    plt.bar(x, res, color="blue")
    plt.xlabel('frequency')
    plt.ylabel('magnitude')
    
    plt.tight_layout()
    concat = 'ckpt={},epoch={}'.format(ckpt, num)
    save_dir = os.path.join(pic_dir,concat)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir+'/{}th_freq.png'.format(i))
    
    plt.cla()
    plt.close()
    #plt.close(fig1)
    
    #plt.close(fig2)
    #plt.close(fig3)
    
    print("{}th done".format(i))

