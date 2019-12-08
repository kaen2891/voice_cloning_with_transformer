import os
from glob import glob
import utils

slt = sorted(glob('/mnt/junewoo/speech/cmu_all_dataset/cmu_us_slt_arctic/wav/*.wav'))

for i in range(len(slt)):
    slt_dir, wav_name = os.path.split(slt[i])
    real_dir, _ = os.path.split(slt_dir)
    save_dir = os.path.join(real_dir, 'vad')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = save_dir+'/'
    utils.cmu_vad(slt[i], save_dir)

print('{} done'.format(real_dir))


aew = sorted(glob('/mnt/junewoo/speech/cmu_all_dataset/cmu_us_aew_arctic/wav/*.wav'))

for i in range(len(aew)):
    aew_dir, wav_name = os.path.split(aew[i])
    real_dir, _ = os.path.split(aew_dir)
    save_dir = os.path.join(real_dir, 'vad')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = save_dir+'/'
    utils.cmu_vad(aew[i], save_dir)

print('{} done'.format(real_dir))

ahw = sorted(glob('/mnt/junewoo/speech/cmu_all_dataset/cmu_us_ahw_arctic/wav/*.wav'))

for i in range(len(ahw)):
    ahw_dir, wav_name = os.path.split(ahw[i])
    real_dir, _ = os.path.split(ahw_dir)
    save_dir = os.path.join(real_dir, 'vad')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = save_dir + '/'
    utils.cmu_vad(ahw[i], save_dir)

print('{} done'.format(real_dir))


aup = sorted(glob('/mnt/junewoo/speech/cmu_all_dataset/cmu_us_aup_arctic/wav/*.wav'))

for i in range(len(aup)):
    aup_dir, wav_name = os.path.split(aup[i])
    real_dir, _ = os.path.split(aup_dir)
    save_dir = os.path.join(real_dir, 'vad')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = save_dir + '/'
    utils.cmu_vad(aup[i], save_dir)

print('{} done'.format(real_dir))

awb = sorted(glob('/mnt/junewoo/speech/cmu_all_dataset/cmu_us_awb_arctic/wav/*.wav'))

for i in range(len(awb)):
    awb_dir, wav_name = os.path.split(awb[i])
    real_dir, _ = os.path.split(awb_dir)
    save_dir = os.path.join(real_dir, 'vad')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = save_dir + '/'
    utils.cmu_vad(awb[i], save_dir)

print('{} done'.format(real_dir))

bdl = sorted(glob('/mnt/junewoo/speech/cmu_all_dataset/cmu_us_bdl_arctic/wav/*.wav'))

for i in range(len(bdl)):
    bdl_dir, wav_name = os.path.split(bdl[i])
    real_dir, _ = os.path.split(bdl_dir)
    save_dir = os.path.join(real_dir, 'vad')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = save_dir + '/'
    utils.cmu_vad(bdl[i], save_dir)

print('{} done'.format(real_dir))

fem = sorted(glob('/mnt/junewoo/speech/cmu_all_dataset/cmu_us_fem_arctic/wav/*.wav'))

for i in range(len(fem)):
    fem_dir, wav_name = os.path.split(fem[i])
    real_dir, _ = os.path.split(fem_dir)
    save_dir = os.path.join(real_dir, 'vad')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = save_dir + '/'
    utils.cmu_vad(fem[i], save_dir)

print('{} done'.format(real_dir))

jmk = sorted(glob('/mnt/junewoo/speech/cmu_all_dataset/cmu_us_jmk_arctic/wav/*.wav'))

for i in range(len(jmk)):
    jmk_dir, wav_name = os.path.split(jmk[i])
    real_dir, _ = os.path.split(jmk_dir)
    save_dir = os.path.join(real_dir, 'vad')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = save_dir + '/'
    utils.cmu_vad(jmk[i], save_dir)

print('{} done'.format(real_dir))

ksp = sorted(glob('/mnt/junewoo/speech/cmu_all_dataset/cmu_us_ksp_arctic/wav/*.wav'))

for i in range(len(ksp)):
    ksp_dir, wav_name = os.path.split(ksp[i])
    real_dir, _ = os.path.split(ksp_dir)
    save_dir = os.path.join(real_dir, 'vad')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = save_dir + '/'
    utils.cmu_vad(ksp[i], save_dir)

print('{} done'.format(real_dir))


rms = sorted(glob('/mnt/junewoo/speech/cmu_all_dataset/cmu_us_rms_arctic/wav/*.wav'))

for i in range(len(rms)):
    rms_dir, wav_name = os.path.split(rms[i])
    real_dir, _ = os.path.split(rms_dir)
    save_dir = os.path.join(real_dir, 'vad')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = save_dir + '/'
    utils.cmu_vad(rms[i], save_dir)

print('{} done'.format(real_dir))


rxr = sorted(glob('/mnt/junewoo/speech/cmu_all_dataset/cmu_us_rxr_arctic/wav/*.wav'))

for i in range(len(rxr)):
    rxr_dir, wav_name = os.path.split(rxr[i])
    real_dir, _ = os.path.split(rxr_dir)
    save_dir = os.path.join(real_dir, 'vad')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = save_dir + '/'
    utils.cmu_vad(rxr[i], save_dir)

print('{} done'.format(real_dir))

print('done')