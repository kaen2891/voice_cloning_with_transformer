# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import pickle
import matplotlib.pyplot as plt
# from keras.models import Model
# from tensorflow.python.layers import base
# from keras.models import Model
from datetime import datetime
import model_2dec
import time
from datetime import datetime

import librosa
import librosa.display
import soundfile as sf
import matplotlib

matplotlib.use('Agg')

parser = argparse.ArgumentParser()

parser.add_argument('--num_enc', type=int, default='6', help='number of encoderlayers')
parser.add_argument('--num_dec', type=int, default='6', help='number of decoder layers')
parser.add_argument('--d_model', type=int, default='256', help='number of hidden size(frequency sizes)')
parser.add_argument('--num_heads', type=int, default='8', help='number of multihead attention')
parser.add_argument('--dff', type=int, default='1024', help='number of feed forward network size')
parser.add_argument('--max_sequence_length', type=int, default='438', help='number of max sequence size')
parser.add_argument('--dropout_rate', type=float, default='0.1', help='number of max sequence size')
parser.add_argument('--max_text_length', type=int, default='39', help='number of text max sequence size')
parser.add_argument('--lr', type=float, default='1e-5', help='initial learning rate')
parser.add_argument('--nfft', type=int, default='512', help='number of fft')
parser.add_argument('--hop', type=int, default='256', help='number of noverlap')
parser.add_argument('--ckpt', default='0', help='check point path')
parser.add_argument('--batch_size', type=int, default='64', help='number of batch')
parser.add_argument('--epochs', type=int, default='10000', help='number of epochs')
parser.add_argument('--gpus', type=str, default='0', help='using gpu number')
parser.add_argument('--infor', type=str, default='what', help='option')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

# for use tf ver 1.0
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


def plot_attention_weights_spec(attention, layer, cnt, find_zero, find_zero_tar):
    fig = plt.figure(figsize=(16, 8))
    # print("first attention", np.shape(attention))

    # print("attention[layer]", attention[layer])
    # print("shape_tensor", attention[layer].shape)
    # print("shape_np", np.shape(attention[layer].shape))

    attention = attention[layer]
    # print("shape of attention[layer]", attention.shape)
    attention = attention[0]
    # print("shape of attention[0]", attention.shape)
    # attention = tf.squeeze(attention, axis=0)

    # attention = tf.squeeze(attention[layer], axis=0)

    # print("final attention shape", attention.shape)
    attention = attention[:, :find_zero_tar, :find_zero]
    # print("attention shape after find_zero", attention.shape)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        # plot the attention weights
        ax.matshow(attention[head], cmap='viridis')  # for now

        fontdict = {'fontsize': 12}

        ax.set_title(' Encoder time step ', fontdict=fontdict)
        ax.set_ylabel(' Decoder time step', fontdict=fontdict)
        # ax.set_xlabel('Head {}'.format(head + 1))
        # plt.title('Head {}'.format(head + 1))
        ax.set_xlabel('Head {}'.format(head + 1))

    plt.tight_layout()

    cnt = str(cnt)
    save_dir = './attn_map/train/ckpt={}/'.format(args.ckpt)
    others = 'spec,epoch={}'.format(cnt)
    save_dir = os.path.join(save_dir, others)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + '/' + layer + '.png')
    plt.cla()
    plt.close()
    # plt.show()


def plot_attention_weights(attention, layer, cnt, find_zero, find_zero_asr):
    fig = plt.figure(figsize=(16, 8))
    # print("first attention", np.shape(attention))

    attention = attention[layer]
    attention = attention[0]
    # attention = tf.squeeze(attention, axis=0)

    # attention = tf.squeeze(attention[layer], axis=0)
    # print('original input attention shape', attention.shape)
    attention = attention[:, :find_zero_asr, :find_zero]
    # print("find_zero {} find_zero_asr {} attention {}".format(find_zero, find_zero_asr, attention.shape))

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        # plot the attention weights
        ax.matshow(attention[head], cmap='viridis')  # for now

        fontdict = {'fontsize': 12}

        ax.set_title(' Encoder time step ', fontdict=fontdict)
        ax.set_ylabel(' Decoder time step', fontdict=fontdict)
        # ax.set_xlabel('Head {}'.format(head + 1))
        # plt.title('Head {}'.format(head + 1))
        ax.set_xlabel('Head {}'.format(head + 1))

    plt.tight_layout()

    cnt = str(cnt)
    save_dir = './attn_map/train/ckpt={}/'.format(args.ckpt)
    others = 'spec,epoch={}'.format(cnt)
    save_dir = os.path.join(save_dir, others)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + '/' + layer + '.png')
    plt.cla()
    plt.close()
    # plt.show()


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


def create_masks(inp_spec, tar_txt, tar_spec):
    # Encoder padding mask

    enc_padding_mask = create_padding_mask_spec(inp_spec)
    print("enc_padding_mask", enc_padding_mask)
    print("enc_padding mask[0]", enc_padding_mask[0])

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    # dec_padding_mask_asr = create_padding_mask_text(inp_txt)
    dec_padding_mask = create_padding_mask_spec(inp_spec)
    # dec_padding_mask = tf.concat([dec_padding_mask_text, dec_padding_mask_spec], axis=3)  # concat with text and spec
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask_asr = create_look_ahead_mask(tf.shape(tar_txt)[1])  # 39, 39
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar_spec)[1])  # 438, 438
    # look_ahead_mask_txt = create_look_ahead_mask(tf.shape(tar_txt)[1])
    # print("look_ahead mask txt shape is", look_ahead_mask_txt)
    # look_ahead_mask_spec = create_look_ahead_mask(tf.shape(tar_spec)[1])
    # print("look_ahead_mask spec shape is", look_ahead_mask)

    # look_ahead_mask = tf.concat([look_ahead_mask_txt, look_ahead_mask_spec], axis=3)
    dec_target_padding_mask_asr = create_padding_mask_text(tar_txt)  # batch_size, 1, 1, 39
    dec_target_padding_mask = create_padding_mask_spec(tar_spec)  # batch_size, 1, 1, 438
    # dec_target_padding_mask = tf.concat([dec_target_padding_mask_text, dec_target_padding_mask_spec], axis=3)

    combined_mask_asr = tf.maximum(dec_target_padding_mask_asr, look_ahead_mask_asr)  # batch_size, 1, 1, 39
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)  # batch_size, 1, 1, 438
    # combined_mask_spec = tf.maximum(dec_target_padding_mask_spec, look_ahead_mask)
    # combined_mask = tf.concat([combined_mask_text, combined_mask_spec], axis=3) # concat with text and spec

    return enc_padding_mask, combined_mask_asr, combined_mask, dec_padding_mask


loss_object_l1 = tf.keras.losses.MeanAbsoluteError(reduction='none')
loss_object_mse = tf.keras.losses.MeanSquaredError(reduction='none')
loss_object_sparse = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


# def loss_function(real_asr, real_spec, pred_asr, pred_spec):
#     if real_a

def loss_function_spec(real, pred):
    mask = tf.cast(tf.math.equal(real, 0), tf.float32)
    mask = tf.cast(tf.logical_not(tf.cast(tf.reduce_min(mask, axis=-1), tf.bool)), tf.float32)
    # mse = loss_object_mse(real, pred, sample_weight=mask)
    l1 = loss_object_l1(real, pred, sample_weight=mask)
    # final_loss = (loss * 0.5) + (loss2 * 0.5)

    # return loss
    return tf.reduce_mean(l1)


def loss_function_text(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))

    loss_ = loss_object_sparse(real, pred)
    # loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(real, pred, from_logits=True))

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    # mask = tf.cast(mask, dtype=loss.dtype)
    # loss *= mask

    # return loss

    return tf.reduce_mean(loss_)


def input_fn(spec_inp, txt_dec, spec_dec, txt_tar, spec_tar, BATCH_SIZE, BUFFER_SIZE):
    # txt_dataset = tf.data.Dataset.from_tensor_slices((txt_inp, txt_inp)).map(tf_encode)
    # print(txt_dataset)
    dataset = tf.data.Dataset.from_tensor_slices((spec_inp, txt_dec, spec_dec, txt_tar, spec_tar))
    print("dataset slide", dataset)
    dataset = dataset.cache()

    train_dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    # train_dataset = dataset.batch(BATCH_SIZE)

    # print("train_dataset shuffle",train_dataset)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    # print("train_dataset",train_dataset)
    # print("prefetch", train_dataset)
    # return next(iter(train_data))
    return train_dataset


def main():
    # load dataset here

    asr_dec_inp = np.load('./cmu1120/train_id_dec.npy')
    asr_tar_inp = np.load('./cmu1120/train_id_tar.npy')

    spec_enc_inp = np.load('./cmu1120/origin/x_train_ori_all_512_256.npy')  # source man, ori
    spec_enc_inp = spec_enc_inp.astype('float32')

    spec_dec_inp = np.load('./cmu1120/origin/y_train_dec_all_512_256.npy')  # source man, ori
    spec_dec_inp = spec_dec_inp.astype('float32')

    spec_tar_inp = np.load('./cmu1120/origin/y_train_tar_all_512_256.npy')  # source man, ori
    spec_tar_inp = spec_tar_inp.astype('float32')

    spec_enc_inp = spec_enc_inp[:, :-1, :]
    spec_dec_inp = spec_dec_inp[:, :-1, :]
    spec_tar_inp = spec_tar_inp[:, :-1, :]

    enc_inp_spec = np.transpose(spec_enc_inp, (0, 2, 1))
    dec_inp_spec = np.transpose(spec_dec_inp, (0, 2, 1))
    tar_inp_spec = np.transpose(spec_tar_inp, (0, 2, 1))

    ckpt_path = args.ckpt

    batch_size = args.batch_size
    buffer_size = 1500
    EPOCHS = args.epochs
    vocab_size = 1000

    train_dataset = input_fn(enc_inp_spec, asr_dec_inp, dec_inp_spec, asr_tar_inp, tar_inp_spec, batch_size,
                             buffer_size)
    print(train_dataset)  # ok

    train_loss_text = tf.keras.metrics.Mean(name='train_loss_text')
    print(train_loss_text)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    train_loss_spec = tf.keras.metrics.Mean(name='train_loss_spec')
    print(train_loss_spec)
    transformer = model_2dec.Transformer(args.num_enc, args.num_dec, args.d_model, args.num_heads, args.dff, vocab_size,
                                         args.max_sequence_length, args.max_text_length, rate=args.dropout_rate)
    print(transformer)
    # my_model = model_2dec.Transformer
    # model_summary()

    lr_schedule = model_2dec.CustomSchedule(args.d_model)

    # initial_learning_rate = args.lr

    # use decay

    '''
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=4000,
        decay_rate=0.96,
        staircase=True)
    '''

    # initial_learning_rate = 1e-5

    # optimizer_text = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.9, beta_2=0.98,
    #                                           epsilon=1e-9)
    # optimizer_spec = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.9, beta_2=0.98,
    #                                           epsilon=1e-9)
    optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    ## okay

    checkpoint_path = "./checkpoints{}/train".format(args.ckpt)

    # ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer_t)
    # ckpt = tf.train.Checkpoint(optimizer_text=optimizer_text, optimizer_spec=optimizer_spec, transformer=transformer)
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=None)

    # merged_summary = tf.summary.merge_all()

    # writer = tf.summary.create_file_writer("/tmp/mylogs/eager")
    logdir = "logs/scalars{}/".format(args.ckpt) + datetime.now().strftime("%Y%m%d-%H%M%S")

    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()

    # write_op = tf.summary.merge_all()
    # merged_summary = tf.compat.v1.contrib.summary.merge_all()

    # tf.compat.v1.summary.all_v2_summary_ops()
    # writer = tf.compat.v1.summary.FileWriter(logdir + "/metrices", sess.graph)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    train_step_signature = [
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),

    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp_spec, asr_txt, dec_spec, tar_txt, tar_spec):
        # not key
        enc_padding_mask, combined_mask_asr, combined_mask, dec_padding_mask = create_masks(inp_spec, asr_txt, dec_spec)

        with tf.GradientTape() as tape:
            predict_text, predict_spec, attention_weight_asr, attention_weight = transformer(inp_spec, asr_txt,
                                                                                             dec_spec,
                                                                                             True,
                                                                                             enc_padding_mask,
                                                                                             combined_mask_asr,
                                                                                             combined_mask,
                                                                                             dec_padding_mask)
            loss_text = loss_function_text(tar_txt, predict_text)
            loss_spec = loss_function_spec(tar_spec, predict_spec)
            final_loss = loss_text + loss_spec

        # if batch%

        graident = tape.gradient(final_loss, transformer.trainable_variables)
        # gradients_text = text_tape.gradient(loss_text, transformer.trainable_variables)
        # gradients_spec = spec_tape.gradient(loss_spec, transformer.trainable_variables)
        # print("gradients_text {}, gradients_spec {}".format(gradients_text, gradients_spec))
        # gradients_spec = tape.gradient(loss_spec, transformer.trainable_variables)

        optimizer.apply_gradients(zip(graident, transformer.trainable_variables))
        # optimizer_text.apply_gradients(zip(gradients_text, transformer.trainable_variables))
        # optimizer_spec.apply_gradients(zip(gradients_spec, transformer.trainable_variables))
        # print("optimizer_text {}, gradients_spec {}".format(gradients_text, gradients_spec))

        train_loss_text(loss_text)
        train_loss_spec(loss_spec)
        train_accuracy(tar_txt, predict_text)

        return predict_text, predict_spec, attention_weight_asr, attention_weight

    # tf.summary.trace_on(graph=True, profiler=True)

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss_text.reset_states()
        train_loss_spec.reset_states()
        train_accuracy.reset_states()

        # inp -> man, tar -> woman
        for (batch, (inp_spec, dec_txt, dec_spec, tar_txt, tar_spec)) in enumerate(train_dataset):

            epc_before = int(epoch)
            name_before = 'before_predict_epoch={}'.format(epc_before)
            result_before = inp_spec[0]
            result_before = np.transpose(result_before, (1, 0))

            result_txt, result, attention_weight_asr, attention_weight = train_step(inp_spec, dec_txt, dec_spec,
                                                                                    tar_txt, tar_spec)

            # profiler_outdir=logdir + "/metrics")

            if batch % 20 == 0:
                print('Epoch {} Batch {} Text_Loss {:.4f} Spec_Loss {:.4f}, Text_Acc {:.4f}'.format(
                    epoch + 1, batch, train_loss_text.result(), train_loss_spec.result(), train_accuracy.result()))

        if (epoch + 1) % 20 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))
            '''                                                    
            spec_t = inp_spec[0]
            spec_t = spec_t.numpy()
            spec_t = spec_t.T
            print("spec_t shape is", np.shape(spec_t))
            # spec_t = tf.transpose(spec_t)
            idx_spec = np.argwhere(np.diff(np.r_[False, spec_t[0], False]))
            find_zero_spec = np.squeeze(idx_spec)
            zero_cnt = find_zero_spec[-1]

            print("zero_cnt", zero_cnt)

            for x in range(6):
                plot = 'decoder_layer{}_block2'.format(x + 1)
                # plot_asr = 'asr_decoder_layer{}_block2'.format(x + 1)

                ###################### check ######################

                # plot_attention_weights_spec(attention_weights, plot, i + 1, find_zero, set_name)  # only spec??
                plot_attention_weights_spec(attention_weight, plot, epoch, zero_cnt)  # spec plot
                # plot_attention_weights(attention_weights_asr, plot_asr, i + 1, set_name)  # asr plot
            '''

        print('Epoch {} Text_Loss {:.4f} Spec_Loss {:.4f}, Text_Acc {:.4f}'.format(epoch + 1, train_loss_text.result(),
                                                                                   train_loss_spec.result(),
                                                                                   train_accuracy.result()))

        tf.summary.scalar('text_loss', data=train_loss_text.result(), step=epoch)
        tf.summary.scalar('spec_loss', data=train_loss_spec.result(), step=epoch)
        tf.summary.scalar('text_accuracy', data=train_accuracy.result(), step=epoch)

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

        if epoch % 20 == 0:
            # print("attention weight", np.shape(attention_weight))
            # print("for tensor", attention_weight)
            # print("shape", attention_weight.shape)
            spec_t = inp_spec[0]

            spec_tar = dec_spec[0]
            spec_tar = spec_tar.numpy()
            # print("spec_tar", np.shape(spec_tar))
            spec_tar_t = spec_tar.T

            # print("attn weight type", type(attention_weight))
            # attention_weight = attention_weight[]
            # attention_weight = attention_weight[0]
            spec_t = spec_t.numpy()
            spec_t = spec_t.T

            text = dec_txt[0]
            idx_text_inp = np.argwhere(np.diff(np.r_[False, text, False]))
            idex_text_inp = np.squeeze(idx_text_inp)
            zero_cnt_text = idex_text_inp[-1]
            # print("spec_t shape is", np.shape(spec_t))
            # spec_t = tf.transpose(spec_t)
            idx_spec = np.argwhere(np.diff(np.r_[False, spec_t[0], False]))
            idx_tar = np.argwhere(np.diff(np.r_[False, spec_tar_t[0], False]))
            find_zero_spec = np.squeeze(idx_spec)
            find_zero_spec_tar = np.squeeze(idx_tar)
            zero_cnt = find_zero_spec[-1]
            zero_cnt_tar = find_zero_spec_tar[-1]

            # print("zero_cnt", zero_cnt)

            for x in range(6):
                plot = 'decoder_layer{}_block2'.format(x + 1)
                plot_asr = 'asr_decoder_layer{}_block2'.format(x + 1)

                ###################### check ######################

                # plot_attention_weights_spec(attention_weights, plot, i + 1, find_zero, set_name)  # only spec??
                plot_attention_weights_spec(attention_weight, plot, epoch, zero_cnt, zero_cnt_tar)  # spec plot
                plot_attention_weights(attention_weight_asr, plot_asr, epoch, zero_cnt, zero_cnt_text)  # asr plot

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