# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import pickle
import matplotlib.pyplot as plt

from datetime import datetime
import text_spec_model
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
parser.add_argument('--max_sequence_length', type=int, default='477', help='number of max sequence size')
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


def create_padding_mask_text(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_padding_mask_spec(seq):
    seq = tf.cast(tf.not_equal(seq, 0), tf.float32)
    seq = tf.cast(tf.reduce_max(seq, axis=-1), tf.float32)
    seq = tf.cast(tf.not_equal(seq, 1), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(inp_txt, inp_spec, tar_txt, tar_spec):
    # Encoder padding mask
    #print("inp_txt size", inp_txt)
    #print("inp_spec size", inp_spec)
    enc_padding_mask_text = create_padding_mask_text(inp_txt)  # --> 5dim
    enc_padding_mask_spec = create_padding_mask_spec(inp_spec)
    enc_padding_mask = tf.concat([enc_padding_mask_text, enc_padding_mask_spec], axis=3)  # concat with text and spec
    
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask_text = create_padding_mask_text(inp_txt)
    dec_padding_mask_spec = create_padding_mask_spec(inp_spec)
    dec_padding_mask = tf.concat([dec_padding_mask_text, dec_padding_mask_spec], axis=3)  # concat with text and spec
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(args.max_sequence_length)
    # look_ahead_mask_txt = create_look_ahead_mask(tf.shape(tar_txt)[1])
    # print("look_ahead mask txt shape is", look_ahead_mask_txt)
    # look_ahead_mask_spec = create_look_ahead_mask(tf.shape(tar_spec)[1])
    #print("look_ahead_mask spec shape is", look_ahead_mask)
    # exit()
    # look_ahead_mask = tf.concat([look_ahead_mask_txt, look_ahead_mask_spec], axis=3)
    dec_target_padding_mask_text = create_padding_mask_text(tar_txt)
    dec_target_padding_mask_spec = create_padding_mask_spec(tar_spec)
    dec_target_padding_mask = tf.concat([dec_target_padding_mask_text, dec_target_padding_mask_spec], axis=3)

    combined_mask_text = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    # combined_mask_spec = tf.maximum(dec_target_padding_mask_spec, look_ahead_mask)
    # combined_mask = tf.concat([combined_mask_text, combined_mask_spec], axis=3) # concat with text and spec

    return enc_padding_mask, combined_mask_text, dec_padding_mask


loss_object_l1 = tf.keras.losses.MeanAbsoluteError(reduction='none')
loss_object_mse = tf.keras.losses.MeanSquaredError(reduction='none')
loss_object_sparse = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function_spec(real, pred):
    mask = tf.cast(tf.math.equal(real, 0), tf.float32)
    mask = tf.cast(tf.logical_not(tf.cast(tf.reduce_min(mask, axis=-1), tf.bool)), tf.float32)
    loss = loss_object_mse(real, pred, sample_weight=mask)
    loss2 = loss_object_l1(real, pred, sample_weight=mask)
    final_loss = (loss * 0.5) + (loss2 * 0.5)
    # return loss
    return tf.reduce_mean(final_loss)


def loss_function_text(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object_sparse(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def input_fn(txt_inp, spec_inp, txt_tar, spec_tar, BATCH_SIZE, BUFFER_SIZE):
    # txt_dataset = tf.data.Dataset.from_tensor_slices((txt_inp, txt_inp)).map(tf_encode)
    # print(txt_dataset)
    dataset = tf.data.Dataset.from_tensor_slices((txt_inp, spec_inp, txt_tar, spec_tar))
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

    txt_inp = np.load('./cmu1116/train_id.npy')
    inp_txt = txt_inp[:, 1:-1]
    tar_txt = txt_inp

    spec_inp = np.load('./cmu1116/x_train_ori_all_512_256.npy')  # source man, ori
    spec_inp = spec_inp.astype('float32')

    spec_tar = np.load('./cmu1116/y_train_ori_all_512_256.npy')  # source man, ori
    spec_tar = spec_tar.astype('float32')

    print("spec_inp", np.shape(spec_inp))  # batch, fft, time
    print("spec_tar", np.shape(spec_tar))
    print("inp_txt", np.shape(inp_txt))
    print("tar_txt", np.shape(tar_txt))

    new_X = spec_inp[:, 1:, :]
    new_Y = spec_tar[:, 1:, :]

    print("new inp", np.shape(new_X))
    # exit()

    # transpose as batch, seq, fft
    # inp_txt = np.transpose(inp_txt, (0, 2, 1))
    inp_spec = np.transpose(new_X, (0, 2, 1))
    # tar_txt = np.transpose(tar_txt, (0, 2, 1))
    tar_spec = np.transpose(new_Y, (0, 2, 1))

    ckpt_path = args.ckpt

    batch_size = args.batch_size
    buffer_size = 1500
    EPOCHS = args.epochs
    vocab_size = 1000

    train_dataset = input_fn(inp_txt, inp_spec, tar_txt, tar_spec, batch_size, buffer_size)
    print(train_dataset)  # ok

    train_loss_text = tf.keras.metrics.Mean(name='train_loss_text')
    print(train_loss_text)
    train_loss_spec = tf.keras.metrics.Mean(name='train_loss_spec')
    print(train_loss_spec)

    transformer = text_spec_model.Transformer(args.num_layers, args.d_model, args.num_heads, args.dff, vocab_size,
                                              args.max_sequence_length, rate=args.dropout_rate)
    print(transformer)
    # exit()

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

    optimizer_text = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    optimizer_spec = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    ## okay

    # exit()

    checkpoint_path = "./checkpoints{}/train".format(args.ckpt)
    #ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer_t)
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer_text=optimizer_text, optimizer_spec=optimizer_spec)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=None)

    # writer = tf.summary.create_file_writer("/tmp/mylogs/eager")
    logdir = "logs/scalars{}/".format(args.ckpt) + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp_txt, inp_spec, tar_txt, tar_spec):
        tar_txt_inp = tar_txt[:, :-1]
        tar_txt_real = tar_txt[:, 1:]
        tar_spec_inp = tar_spec[:, :-1]
        tar_spec_real = tar_spec[:, 1:]
        # return enc_padding_mask_text, enc_padding_mask_spec, combined_mask_text, combined_mask_spec, dec_padding_mask_text, dec_padding_mask_spec
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp_txt, inp_spec, tar_txt_inp, tar_spec_inp)
        # combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as text_tape, tf.GradientTape() as spec_tape:
            # print("inp_txt",inp_txt)
            # print("inp_spec",inp_spec)
            # print("tar_txt_inp",tar_txt_inp)
            # print("tar_spec_inp",tar_spec_inp)
            predict_txt, predict_spec, attention_weights = transformer(inp_txt, inp_spec, tar_txt_inp, tar_spec_inp,
                                                                       True,
                                                                       enc_padding_mask,
                                                                       combined_mask,
                                                                       dec_padding_mask)
            loss_text = loss_function_text(tar_txt_real, predict_txt)
            loss_spec = loss_function_spec(tar_spec_real, predict_spec)
        # if batch%

        gradients_text = text_tape.gradient(loss_text, transformer.trainable_variables)
        gradients_spec = spec_tape.gradient(loss_spec, transformer.trainable_variables)
        # gradients_spec = tape.gradient(loss_spec, transformer.trainable_variables)

        optimizer_text.apply_gradients(zip(gradients_text, transformer.trainable_variables))
        optimizer_spec.apply_gradients(zip(gradients_spec, transformer.trainable_variables))

        train_loss_text(loss_text)
        train_loss_spec(loss_spec)

        return predict_txt, predict_spec

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss_text.reset_states()
        train_loss_spec.reset_states()
        # train_accuracy.reset_states()

        # inp -> man, tar -> woman
        for (batch, (inp_txt, inp_spec, tar_txt, tar_spec)) in enumerate(train_dataset):

            # inp_txt = inp_txt[:,:,0]
            # tar_txt = tar_txt[:,:,0]

            epc_before = int(epoch)
            name_before = 'before_predict_epoch={}'.format(epc_before)
            result_before = inp_spec[0]
            result_before = np.transpose(result_before, (1, 0))
            result_txt, result = train_step(inp_txt, inp_spec, tar_txt, tar_spec)

            if batch % 20 == 0:
                print('Epoch {} Batch {} Text_Loss {:.4f} Spec_Loss {:.4f}'.format(
                    epoch + 1, batch, train_loss_text.result(), train_loss_spec.result()))

        if (epoch + 1) % 20 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Epoch {} Text_Loss {:.4f} Spec_Loss {:.4f}'.format(epoch + 1, train_loss_text.result(),
                                                                  train_loss_spec.result()))

        tf.summary.scalar('text_loss', data=train_loss_text.result(), step=epoch)
        tf.summary.scalar('spec_loss', data=train_loss_spec.result(), step=epoch)

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
            save_tar = tar_spec[0]
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


