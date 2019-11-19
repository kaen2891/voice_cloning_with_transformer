import os
from glob import glob
import shutil
import sentencepiece as spm
import pickle
import numpy as np

def zero_padding(array, num):
    differ = 38 - len(array.T) + num
    npad = ((0, 0), (0, differ))
    target = np.pad(array, npad, 'constant')
    return target

# train_script = sorted(glob('/mnt/junewoo/speech/cmu1113/test_vad/nfft=512_hop=256/*_male*/*.txt'))

sp = spm.SentencePieceProcessor()
# sp.Load("/mnt/junewoo/speech/cmu1113/train_size_1000.model")
sp.Load("/mnt/junewoo/speech/cmu1113/train_1000.model")

txt_file = '/mnt/junewoo/speech/cmu1113/train_all.txt'
# write_dir_id = '/mnt/junewoo/speech/cmu1113/test_id.txt'
# write_dir_sp = '/mnt/junewoo/speech/cmu1113/test_sp.txt'
f = open(txt_file, 'r')
# w_id = open(write_dir_id, 'w', encoding='utf-8')
# w_sp = open(write_dir_sp, 'w', encoding='utf-8')
i = 0
# extra_options = 'bos:eos'
extra_options = 'eos'
sp.SetEncodeExtraOptions(extra_options)
while True:
    line = f.readline().strip()
    #print(line)
    i += 1
    if not line: break

    # ts = sp.EncodeAsPieces(line)
    # ts = np.array(ts)
    # ts = np.expand_dims((ts), axis=0)
    # ts = zero_padding(ts, 2)

    # print(ts)
    # w_sp.write(str(ts)+'\n')
    ts_id = sp.EncodeAsIds(line)
    ts_id = np.array(ts_id)
    ts_id = np.expand_dims((ts_id), axis=0)
    ts_id = zero_padding(ts_id, 1)
    if i == 1:
        # ts_cat = np.array(ts)
        # ts_cat = zero_padding(ts_cat)
        # ts_cat = np.expand_dims((ts_cat), axis=0)
        ts_id_cat = np.array(ts_id)
        # ts_id_cat = zero_padding(ts_cat)
        # ts_id_cat = np.expand_dims((ts_id_cat), axis=0)
        continue

    # ts_cat = np.concatenate((ts_cat, ts), axis=0)
    ts_id_cat = np.concatenate((ts_id_cat, ts_id), axis=0)
    print('{}th done'.format(i))
    print('done')

# np.save('/mnt/junewoo/speech/cmu1113/test_sp', ts_cat)
# np.save('/mnt/junewoo/speech/cmu1120/test_id_enc', ts_id_cat)
np.save('/mnt/junewoo/speech/cmu1120/train_id_tar', ts_id_cat)
print('done')
exit()