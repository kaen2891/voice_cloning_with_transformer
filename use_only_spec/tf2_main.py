import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import pickle
import matplotlib.pyplot as plt

from datetime import datetime
import tf2_model
import time
from datetime import datetime

import librosa
import librosa.display
import soundfile as sf
import matplotlib

matplotlib.use('Agg')

parser = argparse.ArgumentParser()

parser.add_argument('--num_layers', type=int, default='6', help='number of encoder and decoder layers')
parser.add_argument('--d_model', type=int, default='256', help='number of hidden size(frequency sizes)')
parser.add_argument('--num_heads', type=int, default='8', help='number of multihead attention')
parser.add_argument('--dff', type=int, default='1024', help='number of feed forward network size')
parser.add_argument('--max_sequence_length', type=int, default='256', help='number of max sequence size')
parser.add_argument('--dropout_rate', type=float, default='0.1', help='number of max sequence size')
parser.add_argument('--lr', type=float, default='1e-5', help='initial learning rate')
parser.add_argument('--nfft', type=int, default='512', help='number of fft')
parser.add_argument('--hop', type=int, default='256', help='number of noverlap')
parser.add_argument('--ckpt', default='1000', help='check point path')
parser.add_argument('--batch_size', type=int, default='5', help='number of batch')
parser.add_argument('--epochs', type=int, default='10000', help='number of epochs')
parser.add_argument('--gpus', type=str, default='0', help='using gpu number')
parser.add_argument('--infor', type=str, default='what', help='option')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

# for use tf ver 1.0
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


def de_normalized(min, max, array):
    x = array
    de_normalized_array = (x * (max - min) + min)
    return de_normalized_array


def create_padding_mask(seq):
    seq = tf.cast(tf.not_equal(seq, 0), tf.float32)
    seq = tf.cast(tf.reduce_max(seq, axis=-1), tf.float32)
    seq = tf.cast(tf.not_equal(seq, 1), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)

    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask


loss_object_l1 = tf.keras.losses.MeanAbsoluteError(reduction='none')
loss_object_mse = tf.keras.losses.MeanSquaredError(reduction='none')


def loss_function(real, pred):
    mask = tf.cast(tf.math.equal(real, 0), tf.float32)
    mask = tf.cast(tf.logical_not(tf.cast(tf.reduce_min(mask, axis=-1), tf.bool)), tf.float32)
    loss = loss_object_mse(real, pred, sample_weight=mask)
    loss2 = loss_object_l1(real, pred, sample_weight=mask)
    final_loss = (loss * 0.5) + (loss2 * 0.5)
    # return loss
    return tf.reduce_mean(final_loss)


def input_fn(enc_inp, dec_inp, tar_inp, BATCH_SIZE, BUFFER_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((enc_inp, dec_inp, tar_inp))
    print("dataset slide", dataset)
    dataset = dataset.cache()

    train_dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    # print("train_dataset shuffle",train_dataset)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    # print("train_dataset",train_dataset)
    # print("prefetch", train_dataset)
    # return next(iter(train_data))
    return train_dataset


def main():
    # load dataset here

    enc_inp = np.load('./cmu1120/origin/x_train_ori_all_512_256.npy')  # source man, ori

    enc_inp = enc_inp.astype('float32')

    # tar = np.load('./cmu1107/y_train_ori_all_2048_1024.npy') # 2048, 1024
    # tar = np.load('./cmu1107/y_train_ori_all_1024_512.npy') # 1024, 512
    dec_inp = np.load('./cmu1120/origin/y_train_dec_all_512_256.npy')  # source man, ori
    dec_inp = dec_inp.astype('float32')

    tar_inp = np.load('./cmu1120/origin/y_train_tar_all_512_256.npy')  # source man, ori
    tar_inp = tar_inp.astype('float32')

    enc_inp = enc_inp[:, :-1, :]
    dec_inp = dec_inp[:, :-1, :]
    tar_inp = tar_inp[:, :-1, :]

    # transpose as batch, seq, fft
    enc_inp = np.transpose(enc_inp, (0, 2, 1))
    dec_inp = np.transpose(dec_inp, (0, 2, 1))
    tar_inp = np.transpose(tar_inp, (0, 2, 1))

    ckpt_path = args.ckpt

    batch_size = args.batch_size
    buffer_size = 1500
    EPOCHS = args.epochs

    train_dataset = input_fn(enc_inp, dec_inp, tar_inp, batch_size, buffer_size)
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    transformer = tf2_model.Transformer(args.num_layers, args.d_model, args.num_heads, args.dff,
                                        args.max_sequence_length, rate=args.dropout_rate)

    # learning_rate = tf2_model.CustomSchedule(args.d_model)
    initial_learning_rate = args.lr
    # decay_rate = 0.96
    # decay_steps = 100000
    # lr = de

    # use decay 
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=4000,
        decay_rate=0.96,
        staircase=True)

    # initial_learning_rate = 1e-5

    optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    checkpoint_path = "./checkpoints{}/train".format(args.ckpt)
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=None)

    # writer = tf.summary.create_file_writer("/tmp/mylogs/eager")
    logdir = "logs/scalars{}/".format(args.ckpt) + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    train_step_signature = [
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(enc_inp, dec_inp, tar_inp):

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(enc_inp, dec_inp)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(enc_inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
            loss = loss_function(tar_inp, predictions)
        # if batch%
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)

        return predictions

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        # train_accuracy.reset_states()

        # inp -> man, tar -> woman
        for (batch, (enc_inp, dec_inp, tar_inp)) in enumerate(train_dataset):

            epc_before = int(epoch)
            name_before = 'before_predict_epoch={}'.format(epc_before)
            result_before = enc_inp[0]
            result_before = np.transpose(result_before, (1, 0))
            result = train_step(enc_inp, dec_inp, tar_inp)

            if batch % 20 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(
                    epoch + 1, batch, train_loss.result()))

        if (epoch + 1) % 20 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Epoch {} Loss {:.4f}'.format(epoch + 1, train_loss.result()))

        tf.summary.scalar('loss', data=train_loss.result(), step=epoch)

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

        if epoch % 5 == 0:
            epc = int(epoch)
            name_after = 'after_predict_epoch={}'.format(epc)
            result_after = result[0]
            result_after = np.transpose(result_after, (1, 0))

            # train before (original input)
            plt.figure(figsize=(10, 4))
            # plt.figure(figsize=(10, 4))
            librosa.display.specshow(librosa.amplitude_to_db(result_before, ref=np.max), y_axis='hz', x_axis='time',
                                     sr=16000, hop_length=args.hop)
            plt.title(name_before)
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            fig_save_dir = './result/' + ckpt_path + '_fig/'
            if not os.path.exists(fig_save_dir):
                os.makedirs(fig_save_dir)
            plt.savefig(fig_save_dir + name_before + '.png')
            plt.cla()
            plt.close()

            make_wav = librosa.istft(result_before, hop_length=args.hop)
            wav_save_dir = './result/' + ckpt_path + '_wav/'
            if not os.path.exists(wav_save_dir):
                os.makedirs(wav_save_dir)
            sf.write(wav_save_dir + name_before + '.wav', make_wav, 16000, format='WAV', endian='LITTLE',
                     subtype='PCM_16')

            # train after (y_hat)
            plt.figure(figsize=(10, 4))
            # plt.figure(figsize=(10, 4))
            librosa.display.specshow(librosa.amplitude_to_db(result_after, ref=np.max), y_axis='hz', x_axis='time',
                                     sr=16000, hop_length=args.hop)
            plt.title(name_after)
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plt.savefig(fig_save_dir + name_after + '.png')
            plt.cla()
            plt.close()

            make_wav = librosa.istft(result_after, hop_length=args.hop)
            sf.write(wav_save_dir + name_after + '.wav', make_wav, 16000, format='WAV', endian='LITTLE',
                     subtype='PCM_16')

            # real input (source)
            save_tar = tar_inp[0]
            save_tar = np.transpose(save_tar, (1, 0))
            plt.figure(figsize=(10, 4))
            # plt.figure(figsize=(10, 4))
            librosa.display.specshow(librosa.amplitude_to_db(save_tar, ref=np.max), y_axis='hz', x_axis='time',
                                     sr=16000, hop_length=args.hop)
            real_name = 'real_epoch={}'.format(epc_before)
            plt.title(real_name)
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            fig_save_dir = './result/' + ckpt_path + '_fig/'
            if not os.path.exists(fig_save_dir):
                os.makedirs(fig_save_dir)
            plt.savefig(fig_save_dir + real_name + '.png')
            plt.cla()
            plt.close()

            make_wav = librosa.istft(save_tar, hop_length=args.hop)

            wav_save_dir = './result/' + ckpt_path + '_wav/'
            if not os.path.exists(wav_save_dir):
                os.makedirs(wav_save_dir)
            sf.write(wav_save_dir + real_name + '.wav', make_wav, 16000, format='WAV', endian='LITTLE',
                     subtype='PCM_16')

            # train before np file
            np_save_dir = './result/' + ckpt_path + '_np_file/'
            if not os.path.exists(np_save_dir):
                os.makedirs(np_save_dir)
            np.save(np_save_dir + name_before, result_before)

            # train after np file
            np_save_dir = './result/' + ckpt_path + '_np_file/'
            if not os.path.exists(np_save_dir):
                os.makedirs(np_save_dir)
            np.save(np_save_dir + name_after, result_after)

            # real
            np_save_dir = './result/' + ckpt_path + '_np_file/'
            if not os.path.exists(np_save_dir):
                os.makedirs(np_save_dir)
            real_name = 'y_real_epoch={}'.format(epc)

            np.save(np_save_dir + real_name, save_tar)


if __name__ == '__main__':
    main()


