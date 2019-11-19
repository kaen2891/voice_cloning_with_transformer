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

nfft=512
hop=256
from scipy.spatial.distance import cdist
def dtw_distance(distances):
    DTW = np.empty_like(distances)
    # DTW[:, 0] = np.inf
    # DTW[0, :] = np.inf
    DTW[0, 0] = 0
    print(DTW.shape[0], DTW.shape[1]) # 10, 8
    for i in range(0, DTW.shape[0]):
        for j in range(0, DTW.shape[1]):
            if i ==0 and j==0:
                DTW[i, j] = distances[i, j]
            elif i == 0:
                DTW[i, j] = distances[i, j] + DTW[i, j-1]
            elif j == 0:
                DTW[i, j] = distances[i, j] + DTW[i-1, j]
            else:
                DTW[i, j] = distances[i, j] + min(DTW[i-1, j],  # insertion
                                              DTW[i, j-1],  # deletion
                                              DTW[i-1, j-1] #match
                                             )
    #print("done")
    return DTW

def backtrack(DTW):
    i, j = DTW.shape[0] - 1, DTW.shape[1] - 1
    output_x = []
    output_y = []
    output_x.append(i) # last scalar in reference spectrogram
    output_y.append(j) # last scalar in compare spectrogram
    while i > 0 and j > 0:
        local_min = np.argmin((DTW[i - 1, j - 1], DTW[i, j - 1], DTW[i - 1, j]))
        if local_min == 0:
            i -= 1
            j -= 1
        elif local_min == 1:
            j -= 1
        else:
            i -= 1
        output_x.append(i)
        output_y.append(j)

    output_x.append(0)
    output_y.append(0)
    output_x.reverse()
    output_y.reverse()
    return np.array(output_x), np.array(output_y) # this output is dtw path. Longer than reference

def multi_DTW(a,b,len_ref,len_tar): #a is ref, b is compare. a, b lengths are same.
    cnt = []
    for x in range(len(a)):
        if a[x-1] == a[x]:
            cnt.append(x)
    target = np.delete(b, cnt) # 470, len_ref = 473
    if len(target) < len_ref:
        differ =  len_ref - len(target) # 3
        target = np.pad(target,(0,differ), 'constant', constant_values=(1, len_tar-1))
    return target
    # print("done")
    # print("done")

def my_dtw(a, b, len_ref, len_tar, distance_metric='euclidean'):
    distance = cdist(a, b, distance_metric)
    cum_min_dist = dtw_distance(distance)
    x, y = backtrack(cum_min_dist)
    final_y = multi_DTW(x,y,len_ref,len_tar)
    #print(final_y, len(final_y))
    # print("here, done")
    return final_y

def cut_small_value(magnitude):
    mask = (magnitude >= 1e-2).astype(np.float32)
    new_mag = magnitude * mask
    new_mag[new_mag <= 1e-2] = 1e-3

    return new_mag

def plot_wav(spectrogram, speaker_name, mode, save_dir):
    #input_istft = librosa.istft(spectrogram, hop_length=hop)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), y_axis='hz', x_axis='time', sr=16000, hop_length=hop)
    name = speaker_name
    plt.title(name)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(save_dir + name + '.png')

def make_wav(spectrogram, speaker_name, mode, save_dir):
    name = speaker_name
    input_istft = librosa.istft(spectrogram, hop_length=hop)
    sf.write(save_dir + name + '.wav', input_istft, 16000, format='WAV', endian='LITTLE', subtype='PCM_16')

def use_trim(y, frame, hop):
    test = librosa.effects.trim(y, frame_length=512, hop_length=256)
    return test

def return_mag_pha(input_stft):
    mag, pha = librosa.magphase(input_stft)
    return mag, pha

def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx
#
# def zero_padding_y(array, max_length):
#     differ = len(max_length.T) - len(array.T)
#     npad = ((0, 0), (0, differ))
#     target = np.pad(array, npad, 'constant')
#     return target

def zero_padding(array, max_length, num):
    differ = len(max_length.T) + num - len(array.T)
    npad = ((0, 0), (0, differ))
    target = np.pad(array, npad, 'constant')
    return target
#
# def zero_padding_complex(complex_array, max_length):
#     differ = len(max_length.T) - len(complex_array.T)
#     complex_npad = ((0,0),(0,differ))
#     complex_target = np.pad(complex_array, complex_npad, 'constant')
#     return complex_target

def zero_padding_complex(complex_array, max_length, num):
    differ = len(max_length.T) + num - len(complex_array.T)
    complex_npad = ((0,0),(0,differ))
    complex_target = np.pad(complex_array, complex_npad, 'constant')
    return complex_target

'''
def save_vad_wav(glob_list, save_dir):
    for i in range(len(glob_list)):
        (dir, file_num) = os.path.split(glob_list[i])  # dir + filename
        (dir_dir, speaker_id) = os.path.split(dir)
        (_, speaker_name) = os.path.split(dir_dir)
        print("file_num is {} and speaker_name is {}".format(file_num, speaker_name))
        file_num = file_num[7:12]
        speaker_name = speaker_name[7:10]
        #print(file_num, speaker_name)
        cmu_vad(save_dir, file_num, speaker_name, glob_list[i])
'''
def cmu_vad(wav, save_dir):
    #name = '/data/Ses01F_script01_1_F012.wav'
    (_, file_id) = os.path.split(wav)
    print(file_id)

    y, fs = librosa.load(wav, sr=16000)

    trimed = trim(y, fs, fs_vad = fs, hoplength = 30, vad_mode=3)

    if isinstance(trimed, type(None)):
        print("It can't VAD")
        data = y
    else:
        print("VAD Finished...")
        data = trimed

    #now_dir = os.getcwd()
    #name = speaker_id+'_'+file_num+'.wav'


    #print(out_put_dir)
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    '''
    sf.write(save_dir, data, 16000, format='WAV', endian='LITTLE', subtype='PCM_16')
    print("{} is saved".format(save_dir))

def get_min_max(array):
    max = array.max()
    min = array.min()
    return min, max

def normalized(min, max, array):
    x = array
    normalized_array = (x-min)/(max-min)
    return normalized_array

def de_normalized(min, max, array):
    x = array
    de_normalized_array = (x * (max - min) + min)
    return de_normalized_array
