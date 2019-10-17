import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import pickle
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tf2_model
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

parser.add_argument('--num_layers', type=int, default='6', help='number of encoder and decoder layers')
parser.add_argument('--d_model', type=int, default='256', help='number of hidden size(frequency sizes)')
parser.add_argument('--num_heads', type=int, default='8', help='number of multihead attention')
parser.add_argument('--dff', type=int, default='1024', help='number of feed forward network size')
parser.add_argument('--max_sequence_length', type=int, default='438', help='number of max sequence size')
parser.add_argument('--dropout_rate', type=float, default='0.1', help='number of max sequence size')
parser.add_argument('--nfft', default='512', help='number of fft')
parser.add_argument('--hop', default='256', help='number of noverlap')
parser.add_argument('--ckpt', default='./1', help='check point path')
parser.add_argument('--infor', type=str, default='hello transformer', help='option')

args = parser.parse_args()



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

def evaluate(inp_spectrogram, transformer):

    encoder_input = tf.expand_dims(inp_spectrogram, 0)
    #decoder_input = np.load('/data1/junewoo/workspace/voice_cloning_with_tranformer/cmu1016/sos_token.npy')
    #decoder_input = decoder_input[1:, ]
    
    decoder_input = np.load('/data1/junewoo/workspace/voice_cloning_with_tranformer/cmu1016/y_final_all.npy')
    decoder_input = decoder_input[0]
    decoder_input = np.transpose(decoder_input, (1, 0))
    decoder_input = decoder_input[:, 1:] 
    
    #decoder_input = np.transpose(decoder_input, (1, 0)) 
    
    output = tf.expand_dims(decoder_input, 0) # (1, T, 256)
    
    #transformer = tf2_model.Transformer(args.num_layers, args.d_model, args.num_heads, args.dff, args.max_sequence_length, rate=args.dropout_rate)
    
    
    for i in range(args.max_sequence_length):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
        
        predictions, attention_weights = transformer(encoder_input, output, False, enc_padding_mask, combined_mask, dec_padding_mask)
        output = tf.cast(output, dtype=tf.float32)
        print("output shape is", type(output))
        print("output dtype", output.dtype)
        print("output", output.shape)
        #predictions = predictions[:, -1:, :]  # (batch_size, 1, frequency size)
        #output = tf.compat.v1.concat([output, predictions], axis=1)
    
    #print("done")
        #return tf.squeeze(output, axis=0)
        return tf.squeeze(predictions, axis=0)
    
def main():
    
    checkpoint_path = "./checkpoints5/train"
    transformer = tf2_model.Transformer(args.num_layers, args.d_model, args.num_heads, args.dff, args.max_sequence_length, rate=args.dropout_rate)
    ckpt = tf.train.Checkpoint(transformer=transformer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    
    model = ckpt.restore(ckpt_manager.latest_checkpoint)
    
    #print(model)
    #exit()
    X_test = np.load('./cmu1016/x_final_all.npy')
    X_test = X_test[0] # 257 437
    X_test = np.transpose(X_test, (1, 0)) # 437 257
    X_test = X_test[:, 1:] # 437 256
    #inp_spectrogram = tf.data.Dataset.from_tensor_slices(X_test)
    
    result = evaluate(X_test, transformer)
    np.save('./output_teacher_forcing', result) 

    
if __name__ == '__main__':
    main()
