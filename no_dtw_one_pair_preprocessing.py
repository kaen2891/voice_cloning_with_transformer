import os
import pickle
import numpy as np
import librosa
# from ezdtw import dtw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf
import librosa.display
from pyvad import vad,trim
import os
from glob import glob
import shutil
from utils import *

from scipy.spatial.distance import cdist

nfft=512
hop=256

# vad all file

def do_vad(glob_list):
    for i in range(len(glob_list)):
        dir, id = os.path.split(glob_list[i])
        dir_dir, speaker_name = os.path.split(dir)
        mode = 'vad'

        save_dir = os.path.join(dir_dir, mode, speaker_name)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        file_dir = os.path.join(dir_dir, mode, speaker_name,id)
        cmu_vad(glob_list[i], file_dir)


# find max sequence for zero padding
def find_max(one_speaker_vad):
    max_all = []
    for i in range(len(one_speaker_vad)):
        dir, id = os.path.split(one_speaker_vad[i])
        dir_dir, speaker = os.path.split(dir)
        read_all_utt = sorted(glob(dir_dir+'/*/{}'.format(id)))
        length = []
        for x in range(len(read_all_utt)):
            y, sr = librosa.load(read_all_utt[x], sr=16000)
            wav_length = librosa.core.get_duration(y=y, sr=sr, n_fft=nfft, hop_length=hop)
            length.append(wav_length)
        np_length = np.asarray(length)
        max_len = np_length.max()
        max_idx, idx = find_nearest(np_length, max_len)  # find nearset wav from average
        max_all.append(read_all_utt[idx])

    each_max = []
    for ii in range(len(max_all)):
        x, sr = librosa.load(max_all[ii], sr=16000)
        length = librosa.core.get_duration(y=x, sr=sr, n_fft=nfft, hop_length=hop)
        each_max.append(length)
    np_each_max = np.asarray(each_max)
    max_len = np_each_max.max()
    max_idx, idx = find_nearest(np_each_max, max_len)
    whole_max = max_all[idx]

    z, sr = librosa.load(whole_max, sr=16000)
    z_len = librosa.core.get_duration(y=z, sr=16000, n_fft=nfft, hop_length=hop)
    max_len_spec = librosa.stft(z, n_fft=nfft, hop_length=hop)
    max_len_mag, max_len_pha = return_mag_pha(max_len_spec)
    return whole_max, z_len

def find_min_max(glob_list):
    all_spec = []
    for i in range(len(glob_list)):
        y, sr = librosa.load(glob_list[i], sr=16000)
        spec = librosa.stft(y, n_fft=nfft, hop_length=hop)
        mag, _ = return_mag_pha(spec)
        all_spec.append(mag)
    mag = np.asarray(mag)
    min, max = get_min_max(mag)

    return min, max

def find_min_max_log(glob_list):
    all_spec = []
    for i in range(len(glob_list)):
        y, sr = librosa.load(glob_list[i], sr=16000)
        spec = librosa.stft(y, n_fft=nfft, hop_length=hop)
        mag, _ = return_mag_pha(spec)
        mag = np.log10(mag)
        all_spec.append(mag)
    mag = np.asarray(mag)
    min, max = get_min_max(mag)

    return min, max

def dtw2save(x_set, max_wav, sos_token, eos_token):
    #glob_list = train set
    # sos_token = sos_token.T
    # eos_token = eos_token.T
    sos_token = np.load('/mnt/junewoo/speech/cmu1113/sos_token.npy')
    eos_token = np.load('/mnt/junewoo/speech/cmu1113/eos_token.npy')
    # sos_token = sos_token.T
    # eos_token = eos_token.T

    # np.save('/mnt/junewoo/speech/cmu1120/sos_token', sos_token)
    # np.save('/mnt/junewoo/speech/cmu1120/eos_token', eos_token)

    # avg_all = []
    #hyper = np.load('/mnt/junewoo/speech/min_max.npy')

    # for zeropadding whole max
    max_y, sr = librosa.load(max_wav, sr=16000)
    max_spec = librosa.stft(max_y, n_fft=nfft, hop_length=hop)
    max_mag, max_phase = return_mag_pha(max_spec)  # whole max ==> 0.016 * 437 ==> 6.992 seconds
    for i in range(len(x_set)):
        dir, id = os.path.split(x_set[i])
        dir_dir, set_mode = os.path.split(dir)
        read_all_utt = sorted(glob(dir_dir+'/*/{}'.format(id)))
        length = []
        for x in range(len(read_all_utt)):
            y, sr = librosa.load(read_all_utt[x], sr=16000)
            wav_length = librosa.core.get_duration(y=y, sr=sr, n_fft=nfft, hop_length=hop)
            length.append(wav_length)
        np_length = np.asarray(length)
        max_len = np_length.max()
        avg_idx, idx = find_nearest(np_length, max_len)  # find nearset wav from average

        max_wav = read_all_utt[idx]
        # stand, sr = librosa.load(max_wav, sr=16000)  # reference => average
        # ref_spec = librosa.stft(stand, n_fft=nfft, hop_length=hop)  # reference stft for dtw

        mode_name = set_mode[2:]
        set_dir = '/mnt/junewoo/speech/cmu1120/origin/{}_vad/nfft={}_hop={}'.format(mode_name,nfft,hop)
        num = id[-9:-4]
        x_data_name = num + '_male_utt'
        y_data_name = num + '_female_utt'

        female_dir = os.path.join(set_dir, y_data_name)
        if not os.path.isdir(female_dir):
            os.makedirs(female_dir)
        male_dir = os.path.join(set_dir, x_data_name)
        if not os.path.isdir(male_dir):
            os.makedirs(male_dir)

        male_infor_array = []
        male_phase_array = []
        female_dec_inp = []
        female_tar_inp = []
        female_phase_array = []

        for k in range(len(read_all_utt)):
            dir, id = os.path.split(read_all_utt[k])
            dir2, set_name = os.path.split(dir)

            if set_name[0] == 'y':
                #y = female (1016)
                y, sr = librosa.load(read_all_utt[k], sr=16000)
                id = id[-9:-4]
                final_output = librosa.stft(y, n_fft=nfft, hop_length=hop)
                '''
                compare_stft = librosa.stft(y, n_fft=nfft, hop_length=hop)  # DTW compare source
                output = my_dtw(ref_spec.T, compare_stft.T, len(ref_spec.T),
                                len(compare_stft.T))  # using dtw, output is path
                print("id = {}, set_name = {}, female compare_stft len = {}, output len = {}".format(id, set_name,
                                                                                              len(compare_stft.T),
                                                                                              len(output)))
                '''
                print("id = {}, set_name = {}, output len = {}".format(id, set_name, len(final_output)))


                # final_output = compare_stft[:, output]  # compare source -> adapt path

                origin_mag, origin_pha = return_mag_pha(final_output)
                origin_mag_dec_inp = np.concatenate((sos_token, origin_mag), axis=1)
                origin_mag_tar_inp = np.concatenate((origin_mag, eos_token), axis=1)

                origin_mag_dec_inp = zero_padding(origin_mag_dec_inp, max_mag, 1)
                origin_mag_tar_inp = zero_padding(origin_mag_tar_inp, max_mag, 1)
                # origin_mag4 = zero_padding(origin_mag3, max_mag, 2) # save as origin padding
                #origin_mag_padding = np.concatenate((sos_token, origin_mag2), axis=1)

                origin_phase_padding = zero_padding_complex(origin_pha, max_phase, 1)


                mode = 'phase_zeropadding'
                phase_female_savedir = os.path.join(female_dir, mode)
                # if not os.path.isdir(phase_female_savedir):
                #     os.makedirs(phase_female_savedir)

                mode = 'origin_zeropadding_dec_inp'
                original_female_dec_inp_dir = os.path.join(female_dir, mode)

                mode = 'origin_zeropadding_tar_inp'
                original_female_tar_inp_dir = os.path.join(female_dir, mode)
                # if not os.path.isdir(original_female_savedir):
                #     os.makedirs(original_female_savedir)

                female_dec_inp.append(origin_mag_dec_inp)
                female_tar_inp.append(origin_mag_tar_inp)
                female_phase_array.append(origin_phase_padding)

                #for inverse
                #de_norm_with_log = de_normalized(min_log, max_log, norm_after_log)
                #inverse = 10**(de_norm_with_log)

                print('done')

            else:
                y, sr = librosa.load(read_all_utt[k], sr=16000)
                id = id[-9:-4]
                final_output = librosa.stft(y, n_fft=nfft, hop_length=hop)
                '''
                compare_stft = librosa.stft(y, n_fft=nfft, hop_length=hop)  # DTW compare source
                output = my_dtw(ref_spec.T, compare_stft.T, len(ref_spec.T),
                                len(compare_stft.T))  # using dtw, output is path
                print("id = {}, set_name = {}, female compare_stft len = {}, output len = {}".format(id, set_name,
                                                                                              len(compare_stft.T),
                                                                                              len(output)))
                '''
                print("id = {}, set_name = {}, output len = {}".format(id, set_name, len(final_output)))

                origin_mag, origin_pha = return_mag_pha(final_output)
                origin_mag_padding = zero_padding(origin_mag, max_mag, 0)  # save as origin padding

                origin_phase_padding = zero_padding_complex(origin_pha, max_phase, 0)

                mode = 'phase_zeropadding'
                phase_male_savedir = os.path.join(male_dir, mode)
                # if not os.path.isdir(phase_male_savedir):
                #     os.makedirs(phase_male_savedir)

                mode = 'origin_zeropadding'
                original_male_savedir = os.path.join(male_dir, mode)
                # if not os.path.isdir(original_male_savedir):
                #     os.makedirs(original_male_savedir)


                male_infor_array.append(origin_mag_padding)
                male_phase_array.append(origin_phase_padding)
            print("done in {} and set_name={}".format(k, set_name))

        np_female_dec_inp = np.asarray(female_dec_inp)
        np_female_tar_inp = np.asarray(female_tar_inp)
        np_male_final_array = np.asarray(male_infor_array)

        np_female_phase_array = np.asarray(female_phase_array)
        np_male_phase_array = np.asarray(male_phase_array)

        np.save(original_female_dec_inp_dir, np_female_dec_inp)
        np.save(original_female_tar_inp_dir, np_female_tar_inp)
        np.save(original_male_savedir, np_male_final_array) # male original padding

        np.save(phase_female_savedir, np_female_phase_array)  # female phase
        np.save(phase_male_savedir, np_male_phase_array)  # male phase

        print('{}th original, normalized, log, norm_log done'.format(i))


'''
def save_set(glob_list, mode):
    root = '/mnt/junewoo/speech/'
    save_dir = os.path.join(root, mode)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    for i in range(len(glob_list)):
        shutil.copy(glob_list[i], save_dir)
'''



if __name__ == '__main__':

    x_train = sorted(glob('/mnt/junewoo/speech/x_train/*.wav'))
    y_train = sorted(glob('/mnt/junewoo/speech/y_train/*.wav'))
    x_test = sorted(glob('/mnt/junewoo/speech/x_test/*.wav'))
    y_test = sorted(glob('/mnt/junewoo/speech/y_test/*.wav'))



    #all_vad = sorted(glob('/mnt/junewoo/speech/cmu_all_dataset/cmu_a592/vad/*/*.wav'))
    #one_speaker_vad = sorted(glob('/mnt/junewoo/speech/cmu_all_dataset/cmu_a592/vad/cmu_us_aew_arctic/*.wav'))
    all_train = x_train + y_train
    max_wav, max_length = find_max(all_train)
    print("whole maximum file is {} and the second is {}".format(max_wav, max_length))


    # hyper = np.array([min, max])
    # np.save('/mnt/junewoo/speech/cmu_all_dataset/cmu_a592/vad/min_max', hyper)
    sos_nfft = int((nfft/2)+1)
    eos_nfft = int((nfft / 2) + 1)
    sos_token = np.random.rand(1, sos_nfft)
    eos_token = np.random.rand(1, eos_nfft)

    dtw2save(x_train, max_wav, sos_token, eos_token)
    print('train all finish')

    all_test = x_test + y_test

    dtw2save(x_test, max_wav, sos_token, eos_token)
    #print(len(vad_all))
