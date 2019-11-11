import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import os
#X_phase = 

def de_normalized(min, max, array):
    x = array
    de_normalized_array = (x * (max - min) + min)
    return de_normalized_array

name = 'result_teacher_forcing'
ckpt_path = '1'

result = np.load('./output_teacher_forcing.npy')
result = np.transpose(result, (1, 0))

hyper_log = np.load('./cmu1016/min_max_log.npy')
min_log, max_log = hyper_log[0], hyper_log[1]
de_norm = de_normalized(min_log, max_log, result)
inverse = 10**(de_norm)

plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.amplitude_to_db(result, ref=np.max), y_axis='hz', x_axis='time', sr=16000, hop_length=256)
plt.title(name)
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
fig_save_dir = './result/'+ckpt_path+'_fig/'
if not os.path.exists(fig_save_dir):
    os.makedirs(fig_save_dir)
    
        #fig_save_dir = '/mnt/junewoo/workspace/transform/tf_transformer/result/0925/one_figure'
plt.savefig(fig_save_dir+name+'.png')
        
make_wav = librosa.istft(result,hop_length=256)
#print(np.shape(make_wav))

        #wav_save_dir = '/mnt/junewoo/workspace/transform/tf_transformer/result/0925/one_wav/'
wav_save_dir = './result/'+ckpt_path+'_wav/'
if not os.path.exists(wav_save_dir):
    os.makedirs(wav_save_dir)
sf.write(wav_save_dir+name+'.wav', make_wav, 16000, format='WAV', endian='LITTLE', subtype='PCM_16')
 