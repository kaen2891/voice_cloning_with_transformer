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

parser.add_argument('--num_enc', type=int, default='6', help='number of encoder layers')
parser.add_argument('--num_dec', type=int, default='6', help='number of decoder layers')
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


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
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


loss_object_sparse = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function_text(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object_sparse(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

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

    enc_inp = np.load('/data1/junewoo/workspace/voice_cloning_with_tranformer/cmu1120/train_id_enc.npy')  # source man, ori

    enc_inp = enc_inp.astype('int64')

    # tar = np.load('./cmu1107/y_train_ori_all_2048_1024.npy') # 2048, 1024
    # tar = np.load('./cmu1107/y_train_ori_all_1024_512.npy') # 1024, 512
    dec_inp = np.load('/data1/junewoo/workspace/voice_cloning_with_tranformer/cmu1120/train_id_dec.npy')  # source man, ori
    dec_inp = dec_inp.astype('int64')

    tar_inp = np.load('/data1/junewoo/workspace/voice_cloning_with_tranformer/cmu1120/train_id_tar.npy')  # source man, ori
    tar_inp = tar_inp.astype('int64')


    # ckpt_path = args.ckpt

    batch_size = args.batch_size
    buffer_size = 1500
    EPOCHS = args.epochs
    vocab_size = 1000

    train_dataset = input_fn(enc_inp, dec_inp, tar_inp, batch_size, buffer_size)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    

    transformer = tf2_model.Transformer(args.num_enc, args.num_dec, args.d_model, args.num_heads, args.dff, vocab_size,
                                        args.max_sequence_length, rate=args.dropout_rate)

    # learning_rate = tf2_model.CustomSchedule(args.d_model)
    initial_learning_rate = args.lr

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
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(enc_inp, dec_inp, tar_inp):

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(enc_inp, dec_inp)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(enc_inp, dec_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
            loss = loss_function_text(tar_inp, predictions)
        # if batch%
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_inp, predictions)

        return predictions

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> man, tar -> woman
        for (batch, (enc_inp, dec_inp, tar_inp)) in enumerate(train_dataset):

            result = train_step(enc_inp, dec_inp, tar_inp)

            if batch % 20 == 0:
                print('Epoch {} Batch {} Text_Loss {:.4f}, Text_Acc {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        if (epoch + 1) % 20 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Epoch {} Text_Loss {:.4f}, Text_Acc {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))

        tf.summary.scalar('text_loss', data=train_loss.result(), step=epoch)
        tf.summary.scalar('text_accuracy', data=train_accuracy.result(), step=epoch)

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


if __name__ == '__main__':
    main()


