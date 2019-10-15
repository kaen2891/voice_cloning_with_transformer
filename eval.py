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
import librosa
import librosa.display
from datetime import datetime
import data
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
parser.add_argument('--ckpt', default='./1', help='check point path')
parser.add_argument('--infor', type=str, default='hello transformer', help='option')

args = parser.parse_args()

def evaluate(inp_spectrograms):

    encoder_input = tf.expand_dims(inp_spectrogram, 0)
    decoder_input = np.load('random1.npz')
    decoder_input = np.transpose(decoder_input, (0, 1))
    output = tf.expand_dims(decoder_input, 0) # (1, 1, 256)
    
    transformer = tf2_model.Transformer(args.num_layers, args.d_model, args.num_heads, args.dff, args.max_sequence_length, rate=args.dropout_rate)
    
    for i in range(args.max_seq_length):
        predictions, attention_weights = transformer(encoder_input, output, False, enc_padding_mask, combined_mask, dec_padding_mask)
        predictions = predictions[:, -1:, :]  # (batch_size, 1, frequency size)
        output = tf.concat([output, predictions], axis=-1)
    return tf.squeeze(output, axis=0)
    
def main():
    imported = tf.saved_model.load("./checkpoints2/train")
    print(imported)
    exit()
    X_test = np.load('.npz')
    inp_spectrogram = tf.data.Dataset.from_tensor_slices(X_test)
    
    evaluate(inp_spectrogram) 

    
if __name__ == '__main__':
    main()
