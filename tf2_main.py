import tensorflow as tf
'''
import tf2_model as ml
import few_sample_data as data
'''
import numpy as np
import argparse
import os
import sys
import pickle
#from tf2_config import DEFINES
import matplotlib.pyplot as plt
from datetime import datetime
import tf2_model
import time
from datetime import datetime


parser = argparse.ArgumentParser()

parser.add_argument('--num_layers', type=int, default='6', help='number of encoder and decoder layers')
parser.add_argument('--d_model', type=int, default='256', help='number of hidden size(frequency sizes)')
parser.add_argument('--num_heads', type=int, default='8', help='number of multihead attention')
parser.add_argument('--dff', type=int, default='1024', help='number of feed forward network size')
parser.add_argument('--max_sequence_length', type=int, default='256', help='number of max sequence size')
parser.add_argument('--dropout_rate', type=float, default='0.1', help='number of max sequence size')
parser.add_argument('--nfft', default='512', help='number of fft')
parser.add_argument('--hop', default='256', help='number of noverlap')
parser.add_argument('--ckpt', default='1000', help='check point path')
parser.add_argument('--batch_size', type=int, default='5', help='number of batch')
parser.add_argument('--epochs', type=int, default='10000', help='number of epochs')
parser.add_argument('--gpus', type=str, default='0', help='using gpu number')
parser.add_argument('--infor', type=str, default='what', help='option')
args = parser.parse_args()

# for use tf ver 1.0
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

os.environ['CUDA_VISIBLE_DEVICES']=args.gpus

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

loss_object = tf.keras.losses.MeanAbsoluteError(reduction='none')

def loss_function(real, pred):
    mask = tf.cast(tf.math.equal(real, 0), tf.float32)
    mask = tf.cast(tf.logical_not(tf.cast(tf.reduce_min(mask, axis=-1), tf.bool)), tf.float32)
    loss = loss_object(real, pred, sample_weight=mask)
    
    return tf.reduce_mean(loss)

def input_fn(inp, out, BATCH_SIZE, BUFFER_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((inp, out))
    print("dataset slide",dataset)
    dataset = dataset.cache()
    
    train_dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    print("train_dataset shuffle",train_dataset)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    print("train_dataset",train_dataset)
    #print("prefetch", train_dataset)
    return train_dataset

def main():
    # load dataset here
    inp = np.load('./cmu1016/x_final_all.npy')
    tar = np.load('./cmu1016/y_final_all.npy')
    print("inp", np.shape(inp)) # batch, fft, time
    
    new_X = inp[:, 1:, :]
    new_Y = inp[:, 1:, :]
    
    print("new inp", np.shape(new_X))
    
    # transpose as batch, seq, fft
    inp = np.transpose(new_X, (0, 2, 1))
    tar = np.transpose(new_Y, (0, 2, 1))
    print("transpose inp",np.shape(inp))
    
    batch_size = args.batch_size
    buffer_size = 700
    num_layers = args.num_layers
    d_model = args.d_model
    num_heads = args.num_heads
    dff = args.dff
    dropout_rate = args.dropout_rate
    max_sequence_length = args.max_sequence_length
    EPOCHS = args.epochs

    train_dataset = input_fn(inp, tar, batch_size, buffer_size)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    
    transformer = tf2_model.Transformer(args.num_layers, args.d_model, args.num_heads, args.dff, args.max_sequence_length, rate=args.dropout_rate)

    learning_rate = tf2_model.CustomSchedule(args.d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)
                                     
    checkpoint_path = "./checkpoints{}/train".format(args.ckpt)
    ckpt = tf.train.Checkpoint(transformer=transformer,optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    
    #writer = tf.summary.create_file_writer("/tmp/mylogs/eager")
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()
    
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')
    
    train_step_signature = [
    tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    ]
    
    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp,
                                         True,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
        
        train_loss(loss)
        
    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        #train_accuracy.reset_states()

        # inp -> man, tar -> woman
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)
            print("batch is", batch)

            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(
                    epoch + 1, batch, train_loss.result()))

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Epoch {} Loss {:.4f}'.format(epoch + 1, train_loss.result()))
        
        #tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
        
        tf.summary.scalar('loss', data=train_loss.result(), step=epoch)
        #tf.summary.scalar('optimizer', data=optimizer, step=epoch)
        #tf.summary.scalar('learning rate', data=learning_rate, step=epoch)

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))




if __name__ == '__main__':
    main()


