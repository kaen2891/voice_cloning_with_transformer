import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import pickle
import matplotlib.pyplot as plt
import librosa
import librosa.display
import new_model
import time
from datetime import datetime
import matplotlib

matplotlib.use('Agg')
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import os

parser = argparse.ArgumentParser()

parser.add_argument('--num_enc', type=int, default='6', help='number of encoder layers')
parser.add_argument('--num_dec', type=int, default='6', help='number of decoder layers')
parser.add_argument('--d_model', type=int, default='256', help='number of hidden size(frequency sizes)')
parser.add_argument('--num_heads', type=int, default='8', help='number of multihead attention')
parser.add_argument('--dff', type=int, default='1024', help='number of feed forward network size')
parser.add_argument('--max_sequence_length', type=int, default='444', help='number of max sequence size')
parser.add_argument('--max_text_length', type=int, default='39', help='number of max text size')
parser.add_argument('--max_spec_length', type=int, default='438', help='number of max spec size')
parser.add_argument('--dropout_rate', type=float, default='0.1', help='number of max sequence size')
parser.add_argument('--nfft', type=int, default='512', help='number of fft')
parser.add_argument('--hop', type=int, default='256', help='number of noverlap')
parser.add_argument('--ckpt', type=str, default='1', help='check point path')
parser.add_argument('--infor', type=str, default='hello transformer', help='option')
parser.add_argument('--gpus', type=str, default='0', help='using gpu number')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


def recover(concat, for_save, name):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(concat, ref=np.max), y_axis='hz', x_axis='time', sr=16000,
                             hop_length=args.hop)
    plt.title(name)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    print("dir", for_save)
    fig_save_dir = for_save + '/fig/'
    if not os.path.exists(fig_save_dir):
        os.makedirs(fig_save_dir)
    plt.savefig(fig_save_dir + name + '.png')

    make_wav = librosa.istft(concat, hop_length=args.hop)
    wav_save_dir = for_save + '/wav/'
    if not os.path.exists(wav_save_dir):
        os.makedirs(wav_save_dir)
    sf.write(wav_save_dir + name + '.wav', make_wav, 16000, format='WAV', endian='LITTLE', subtype='PCM_16')
    print("{}th done", name)
    plt.cla()
    plt.close()


def plot_attention_weights_spec(attention, layer, cnt, max_seq_len, max_text_length, set):
    fig = plt.figure(figsize=(16, 8))

    attention = tf.squeeze(attention[layer], axis=0)
    attention = attention[:, max_text_length:max_text_length + max_seq_len + 1,
                max_text_length:max_text_length + max_seq_len - 1]
    print('attention shape in spec', attention.shape)

    # print("attention shape", attention.shape[0])

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
    save_dir = './attn_map/ckpt={}/{}'.format(args.ckpt, set)
    others = 'spec,cnt={}'.format(cnt)
    save_dir = os.path.join(save_dir, others)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + '/' + layer + '.png')
    plt.cla()
    plt.close()
    # plt.show()


def plot_attention_weights_text(attention, layer, cnt, inp_length, tar_legnth, set):
    fig = plt.figure(figsize=(16, 8))

    attention = tf.squeeze(attention[layer], axis=0)
    # (num_heads, tar, inp)

    # print("attention shape np", np.shape(attention))  # (8, max, inp)
    # print('attention type', type(attention))
    attention = attention[:, :tar_legnth, :inp_length]
    print('attention shape in text', attention.shape)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        # plot the attention weights
        # ax.matshow(attention[head][:-1, :], cmap='viridis')
        ax.matshow(attention[head], cmap='viridis')  # for now
        # print(attention[head][:-1, :])

        fontdict = {'fontsize': 12}
        # small_font = {'fontsize': 5}

        ax.set_xticks(range(0, inp_length, 5))
        ax.set_yticks(range(0, tar_legnth, 5))
        # ax.set_xticks(range(0, 38, 4))
        # ax.set_yticks(range(0, 38, 4))

        # ax.set_xticklabels(range(0,201,20), rotation=90)
        # ax.set_yticklabels(range(0,201,20))

        # ax.set_ylim(len(result) - 1.5, -0.5)

        # ax.set_xticklabels(fontsize=7, rotation=90)
        # ax.set_yticklabels(fontsize=7)

        '''
        ax.set_xticklabels(
            ['input'] + [[i] for i in inp], fontdict=fontdict, rotation=90)

        ax.set_yticklabels(['output'] for i in result
                            if i < args.max_sequence_length],
                           fontdict=fontdict)
        '''
        #
        # ax.set_xlabel('input', fontdict=fontdict)
        ax.set_title('Head {}'.format(head + 1))
        ax.set_xlabel(' Encoder time step ', fontdict=fontdict)
        ax.set_ylabel(' Decoder time step', fontdict=fontdict)

        # ax.set_xlabel('Head {}'.format(head + 1))
        # plt.title('Head {}'.format(head + 1))

    plt.tight_layout()

    cnt = str(cnt)
    save_dir = './attn_map/ckpt={}/{}'.format(args.ckpt, set)
    others = 'text,cnt={}'.format(cnt)
    save_dir = os.path.join(save_dir, others)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + '/' + layer + '.png')
    plt.cla()
    plt.close()


def plot_attention_weights_all(attention, layer, cnt, inp_text_length, inp_spec_length, max_text_length, set):
    fig = plt.figure(figsize=(16, 8))

    attention = tf.squeeze(attention[layer], axis=0)
    attention = attention.numpy()

    for i in range(attention.shape[0]):
        for k in range(max_text_length):
            if k <= inp_text_length - 1:
                for m in range(attention.shape[2]):
                    if m <= inp_text_length - 1:
                        continue
                    else:
                        attention[i][k][m] = 0
                continue
            else:
                attention[i][k] = 0

    for ii in range(attention.shape[0]):
        for kk in range(max_text_length, attention.shape[2]):
            if kk <= max_text_length + inp_spec_length - 1:
                for mm in range(max_text_length, attention.shape[2]):
                    if mm <= max_text_length + inp_spec_length - 1:
                        continue
                    else:
                        attention[ii][kk][mm] = 0
                continue
            else:
                attention[ii][kk] = 0
    # print("concat_attention shape", concat_attention.shape)

    # print("attention shape", attention.shape[0])

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
    save_dir = './attn_map/ckpt={}/{}'.format(args.ckpt, set)
    others = 'text+spec,cnt={}'.format(cnt)
    save_dir = os.path.join(save_dir, others)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + '/' + layer + '.png')
    np.save(save_dir + '/' + layer, attention)
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


def create_masks(inp_text, inp_spec, tar_text, tar):
    # Encoder padding mask
    enc_padding_mask_text = create_padding_mask_text(inp_text)
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    enc_padding_mask_spec = create_padding_mask_spec(inp_spec)
    enc_padding_mask = tf.concat([enc_padding_mask_text, enc_padding_mask_spec], axis=3)  # concat with text and spec

    dec_padding_mask_text = create_padding_mask_text(inp_text)
    dec_padding_mask_spec = create_padding_mask_spec(inp_spec)
    dec_padding_mask = tf.concat([dec_padding_mask_text, dec_padding_mask_spec], axis=3)  # concat with text and spec

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar_text)[1] + tf.shape(tar)[1])

    dec_target_padding_mask_text = create_padding_mask_text(tar_text)
    dec_target_padding_mask_spec = create_padding_mask_spec(tar)
    dec_target_padding_mask = tf.concat([dec_target_padding_mask_text, dec_target_padding_mask_spec], axis=3)

    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


def evaluate(inp_text, inp_spectrogram, tar_text, transformer):
    # encoder input text
    encoder_input_text = tf.expand_dims(inp_text, 0)  # (1, 39)
    # decoder input text
    # sos_num = [1]
    # eos_num = [2]
    # decoder_input_text = np.array(tar_text)
    decoder_input_text = tf.expand_dims(tar_text, 0)  # (1, 1)

    # encoder input spec
    encoder_input_spec = tf.expand_dims(inp_spectrogram, 0)  # seq_len, d_model
    # decoder input spec
    decoder_input_spec = np.load('./cmu1120/sos_token.npy')  # [257, 1]
    decoder_input_spec = decoder_input_spec[1:, :]  # [256, 1]
    decoder_input_spec = np.transpose(decoder_input_spec, (1, 0))  # [1, 256]
    decoder_input_spec = tf.cast(decoder_input_spec, dtype=tf.float32)
    print("decoder_input_spec is", np.shape(decoder_input_spec))
    # print("decoder_input shape", np.shape(decoder_input))
    # eos_token = np.load('./cmu1116/end_token.npy') # check

    # decoder_input = np.transpose(decoder_input, (1, 0))

    # output_text = tf.expand_dims(decoder_input_text, 0)  # (1, 39, 256)
    # output_spec = tf.expand_dims(decoder_input_spec, 0)  # (1, 1, 256)
    output_text = tf.cast(decoder_input_text, dtype=tf.int64)
    # output_spec = tf.cast(output_spec, dtype=tf.float32)

    tar_shape = np.zeros((1, 438, 256))
    zero_spec = np.zeros_like(tar_shape)
    print("zero_spec is", np.shape(zero_spec))

    zero_spec[:, 0, :] = decoder_input_spec

    input_spec = zero_spec
    output_spec = zero_spec

    # trainable = True
    # iter_length = tf.shape(inp_spectrogram)[0]
    print("encoder_input_text shape {}".format(np.shape(encoder_input_text)))
    print("encoder_input_spec shape {}".format(np.shape(encoder_input_spec)))
    print("output_text shape {}".format(np.shape(output_text)))
    print("1. output_spec shape {}".format(np.shape(output_spec)))
    print("zero_spec shape {}".format(np.shape(zero_spec)))
    # print("zero_spec shape {}".format(np.shape(zero_spec)))
    # print("zero_spec shape {}".format(np.shape(zero_spec)))

    # exit()
    # for i in range(args.max_sequence_length-1):
    spec_max_seq_length = 438
    for i in range(spec_max_seq_length - 1):  # for now
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input_text, encoder_input_spec,
                                                                         output_text, input_spec)  # check

        _, predict_spec, attention_weights = transformer(encoder_input_text, encoder_input_spec, output_text,
                                                         input_spec, False,
                                                         enc_padding_mask, combined_mask, dec_padding_mask)
        # predict_text = predict_text[:, -1:, :] # (batci_size, 1, vocab_size)
        # predict_text = tf.cast(tf.argmax(predict_text, axis=-1), tf.int64)

        predict_spec = predict_spec[:, i, :]  # (batch_size, 1, frequency size)
        '''
        if encoder_input_spec[i] == eos_token:
            output_spec[:, i, :] = predict_spec
            return tf.squeeze(output_spec, axis=0), attention_weights
        '''

        output_spec[:, i + 1, :] = predict_spec
        input_spec = output_spec

        # output_spec = tf.concat([output_spec, predict_spec], axis=1)
        # output = tf.compat.v1.concat([output_spec, predict_spec], axis=1)
        print("{}th iter, output_spec {}".format(i, output_spec.shape))

    return tf.squeeze(output_spec, axis=0), attention_weights


def main():
    checkpoint_path = "./checkpoints{}/train".format(args.ckpt)
    vocab_size = 1000
    transformer = new_model.Transformer(args.num_enc, args.num_dec, args.d_model, args.num_heads, args.dff, vocab_size,
                                        args.max_sequence_length, args.max_text_length, args.max_spec_length,
                                        rate=args.dropout_rate)
    ckpt = tf.train.Checkpoint(transformer=transformer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    model = ckpt.restore(ckpt_manager.latest_checkpoint)

    # test mag
    # X_test = np.load('./cmu1016/x_test_ori_all.npy') # ori
    X_sen = np.load('./cmu1120/test_id_enc.npy')
    Y_sen = np.load('./cmu1120/test_id_dec.npy')
    #X_sen = np.load('./cmu1120/train_id_enc.npy')
    #Y_sen = np.load('./cmu1120/train_id_dec.npy')

    X_spec = np.load('./cmu1120/dtw/x_test_ori_all_512_256.npy')  # test
    #X_spec = np.load('./cmu1120/dtw/x_train_ori_all_512_256.npy')  # tes
    X_spec = np.transpose(X_spec, (0, 2, 1))  # (batch, seq_len, d_model)
    X_spec = X_spec[:, :, 1:]  # (batch, seq_len, d_model - 1)

    #Y_spec = np.load('./cmu1120/dtw/y_train_tar_all_512_256.npy')  # test
    Y_spec = np.load('./cmu1120/dtw/y_test_tar_all_512_256.npy')  # test

    Y_spec = Y_spec[:, 1:, :]  # (batch, seq_len, d_model)

    # test phase
    #X_phase = np.load('./cmu1120/dtw/x_train_phase_all_512_256.npy')  # test
    X_phase = np.load('./cmu1120/dtw/x_test_phase_all_512_256.npy')  # test
    X_phase = X_phase[:, 1:, :]  # (batch, seq_len, d_model)

    #Y_phase = np.load('./cmu1120/dtw/y_train_phase_all_512_256.npy')  # train
    Y_phase = np.load('./cmu1120/dtw/y_test_phase_all_512_256.npy')  # train
    Y_phase = Y_phase[:, 1:, :]  # (batch, seq_len, d_model)

    set_name = 'testset'
    name = 'ckpt={},{}'.format(
        args.ckpt, set_name)  # ckpt 12

    save_dir = './result/1123/' + set_name + '/'
    for i in range(len(X_spec)):
        inp_text = X_sen[i]
        inp_spec = X_spec[i]  # 437 256
        inp_pha = X_phase[i]  # 256 437
        tar_text = Y_sen[i]

        # tar = Y_test[i]
        predict_spec, attention_weights = evaluate(inp_text, inp_spec, tar_text, transformer)
        print("after predict, spec shape {}".format(np.shape(predict_spec)))

        idx_text_inp = np.argwhere(np.diff(np.r_[False, inp_text, False]))
        find_zero_text_inp = len((np.squeeze(idx_text_inp, axis=1))) - 1
        # cut zeropadding place for plot attention weight, target
        idx_text_tar = np.argwhere(np.diff(np.r_[False, tar_text, False]))
        find_zero_text_tar = len((np.squeeze(idx_text_tar, axis=1))) - 1
        blow = inp_spec.T  # 256, 437

        idx_spec = np.argwhere(np.diff(np.r_[False, blow[0], False]))
        find_zero = len((np.squeeze(idx_spec, axis=1))) - 1
        '''
        for x in range(6):
            plot = 'decoder_layer{}_block2'.format(x + 1)
            ###################### check ######################
            plot_attention_weights_text(attention_weights, plot, i + 1, find_zero_text_inp, find_zero_text_tar,
                                        set_name)  # only text
            plot_attention_weights_spec(attention_weights, plot, i + 1, find_zero, args.max_text_length,
                                        set_name)  # only spec??
            plot_attention_weights_all(attention_weights, plot, i + 1, find_zero_text_inp, find_zero,
                                       args.max_text_length, set_name)
        '''

        predict_spec = predict_spec[1:, :]  # (seq_len, d_model)
        predict_spec = np.transpose(predict_spec, (1, 0))  # (d_model, seq_len)

        # y_hat wav, fig save
        concat = predict_spec * inp_pha
        save_name = name + '_{}th'.format(i)
        for_save = os.path.join(save_dir, name)
        if not os.path.exists(for_save):
            os.makedirs(for_save)
        recover(concat, for_save, save_name)

        np_save_dir = 'np_file'
        np_dir = os.path.join(for_save, np_save_dir)
        if not os.path.exists(np_dir):
            os.makedirs(np_dir)
        # y_hat np file save
        save_np = '{}th_predict.result'.format(i)
        np_final_predict = os.path.join(np_dir, save_np)
        np.save(np_final_predict, concat)

        ########### check #######, x_real plot
        # x_real np file save
        x_real = inp_spec.T * X_phase[i]
        save_np_x_real = '{}th_x_real.result'.format(i)
        np_final_x_real = os.path.join(np_dir, save_np_x_real)
        np.save(np_final_x_real, x_real)

        # x_real wav, fig file save
        save_name_real = 'x_real_' + name + '_{}th'.format(i)
        for_save_real = os.path.join(save_dir, name)
        if not os.path.exists(for_save_real):
            os.makedirs(for_save_real)
        # np.save(for_save_real, real)
        recover(x_real, for_save_real, save_name_real)

        # y_real np file save
        y_real = Y_spec[i] * Y_phase[i]
        save_np_y_real = '{}th_y_real.result'.format(i)
        np_final_y_real = os.path.join(np_dir, save_np_y_real)
        np.save(np_final_y_real, y_real)

        # y_real wav, fig file save
        save_name_real = 'y_real_' + name + '_{}th'.format(i)
        for_save_real = os.path.join(save_dir, name)
        if not os.path.exists(for_save_real):
            os.makedirs(for_save_real)
        recover(y_real, for_save_real, save_name_real)


if __name__ == '__main__':
    main()
