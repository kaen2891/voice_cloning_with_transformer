from glob import glob
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')

def append_array_3(array1, array2, array3, save_dir1, save_dir2, save_dir3):
    for i in range(len(array1)):
        if i == 0:
            x_final = np.load(array1[0])
            y_final_dec = np.load(array2[0])
            y_final_tar = np.load(array3[0])
            continue
        x_item = np.load(array1[i])
        y_item_dec = np.load(array2[i])
        y_item_tar = np.load(array3[i])

        x_final = np.concatenate((x_final, x_item), axis=0)
        y_final_dec = np.concatenate((y_final_dec, y_item_dec), axis=0)
        y_final_tar = np.concatenate((y_final_tar, y_item_tar), axis=0)
        print('{}th finished'.format(i))

    np.save(save_dir1, x_final)
    np.save(save_dir2, y_final_dec)
    np.save(save_dir3, y_final_tar)

def append_array_2(array1, array2, save_dir1, save_dir2):
    for i in range(len(array1)):
        if i == 0:
            x_final = np.load(array1[0])
            y_final_dec = np.load(array2[0])
            # y_final_tar = np.load(array3[0])
            continue
        x_item = np.load(array1[i])
        y_item_dec = np.load(array2[i])
        # y_item_tar = np.load(array3[i])

        x_final = np.concatenate((x_final, x_item), axis=0)
        y_final_dec = np.concatenate((y_final_dec, y_item_dec), axis=0)
        # y_final_tar = np.concatenate((y_final_tar, y_item_tar), axis=0)
        print('{}th finished'.format(i))

    np.save(save_dir1, x_final)
    np.save(save_dir2, y_final_dec)
    # np.save(save_dir3, y_final_tar)

# now only use origin
#x_train_ori_all = sorted(glob('/mnt/junewoo/speech/cmu1018/train_vad/256/*_male*/origin_zeropadding.npy'))
nfft = int(512)
over = int(256)
id = 'cmu1120/dtw'
x_train_ori_all = sorted(glob('/mnt/junewoo/speech/'+id+'/train_vad/nfft={}_hop={}/*_male*/origin_zeropadding.npy'.format(nfft,over)))
y_train_ori_dec_all = sorted(glob('/mnt/junewoo/speech/'+id+'/train_vad/nfft={}_hop={}/*_female*/origin_zeropadding_dec_inp.npy'.format(nfft,over)))
y_train_ori_tar_all = sorted(glob('/mnt/junewoo/speech/'+id+'/train_vad/nfft={}_hop={}/*_female*/origin_zeropadding_tar_inp.npy'.format(nfft,over)))

x_test_ori_all = sorted(glob('/mnt/junewoo/speech/'+id+'/test_vad/nfft={}_hop={}/*_male*/origin_zeropadding.npy'.format(nfft,over)))
y_test_ori_dec_all = sorted(glob('/mnt/junewoo/speech/'+id+'/test_vad/nfft={}_hop={}/*_female*/origin_zeropadding_dec_inp.npy'.format(nfft,over)))
y_test_ori_tar_all = sorted(glob('/mnt/junewoo/speech/'+id+'/test_vad/nfft={}_hop={}/*_female*/origin_zeropadding_tar_inp.npy'.format(nfft,over)))


x_train_phase_all = sorted(glob('/mnt/junewoo/speech/'+id+'/train_vad/nfft={}_hop={}/*_male*/phase_zeropadding.npy'.format(nfft,over)))
y_train_phase_all = sorted(glob('/mnt/junewoo/speech/'+id+'/train_vad/nfft={}_hop={}/*_female*/phase_zeropadding.npy'.format(nfft,over)))

x_test_phase_all = sorted(glob('/mnt/junewoo/speech/'+id+'/test_vad/nfft={}_hop={}/*_male*/phase_zeropadding.npy'.format(nfft,over)))
y_test_phase_all = sorted(glob('/mnt/junewoo/speech/'+id+'/test_vad/nfft={}_hop={}/*_female*/phase_zeropadding.npy'.format(nfft,over)))


x_train_save_dir = '/mnt/junewoo/speech/'+id+'/x_train_ori_all_{}_{}'.format(nfft,over)
y_train_dec_save_dir = '/mnt/junewoo/speech/'+id+'/y_train_dec_all_{}_{}'.format(nfft,over)
y_train_tar_save_dir = '/mnt/junewoo/speech/'+id+'/y_train_tar_all_{}_{}'.format(nfft,over)
x_test_save_dir = '/mnt/junewoo/speech/'+id+'/x_test_ori_all_{}_{}'.format(nfft,over)
y_test_save_dir = '/mnt/junewoo/speech/'+id+'/y_test_ori_all_{}_{}'.format(nfft,over)
y_test_dec_save_dir = '/mnt/junewoo/speech/'+id+'/y_test_dec_all_{}_{}'.format(nfft,over)
y_test_tar_save_dir = '/mnt/junewoo/speech/'+id+'/y_test_tar_all_{}_{}'.format(nfft,over)


x_train_phase_save_dir = '/mnt/junewoo/speech/'+id+'/x_train_phase_all_{}_{}'.format(nfft,over)
y_train_phase_save_dir = '/mnt/junewoo/speech/'+id+'/y_train_phase_all_{}_{}'.format(nfft,over)
x_test_phase_save_dir = '/mnt/junewoo/speech/'+id+'/x_test_phase_all_{}_{}'.format(nfft,over)
y_test_phase_save_dir = '/mnt/junewoo/speech/'+id+'/y_test_phase_all_{}_{}'.format(nfft,over)

append_array_3(x_train_ori_all, y_train_ori_dec_all, y_train_ori_tar_all, x_train_save_dir, y_train_dec_save_dir, y_train_tar_save_dir)
append_array_3(x_test_ori_all, y_test_ori_dec_all, y_test_ori_tar_all, x_test_save_dir, y_test_dec_save_dir, y_test_tar_save_dir)

append_array_2(x_train_phase_all, y_train_phase_all, x_train_phase_save_dir, y_train_phase_save_dir)
append_array_2(x_test_phase_all, y_test_phase_all, x_test_phase_save_dir, y_test_phase_save_dir)
