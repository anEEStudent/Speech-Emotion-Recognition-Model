#!/usr/bin/env python3

import numpy as np
from scipy.stats import zscore
import librosa
from pathlib import Path
import numpy as np
import librosa
from scipy.stats import zscore
from processingFuncs import *

MAX_PAD_LENGTH = 48000

def mel_spectrogram(y, sr=16000, n_fft=512, win_length=256, hop_length=128, window='hamming', n_mels=128, fmax=4000): 
    # Compute spectogram
    mel_spect = np.abs(librosa.stft(y, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)) ** 2
    # Compute mel spectrogram
    mel_spect = librosa.feature.melspectrogram(S=mel_spect, sr=sr, n_mels=n_mels, fmax=fmax)    
    # Compute log-mel spectrogram
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max) 
    return mel_spect

# Split spectrogram into frames
def frame(x, win_step=128, win_size=128):
    nb_frames = 1 + int((x.shape[2] - win_size) / win_step)
    frames = np.zeros((x.shape[0], nb_frames, x.shape[1], win_size)).astype(np.float32)
    for t in range(nb_frames):
        frames[:,t,:,:] = np.copy(x[:,:,(t * win_step):(t * win_step + win_size)]).astype(np.float32)
    return frames

def preprocess(dir, label, signals, labels): 
    # Sample rate (16.0 kHz)
    sample_rate = 16000     
    # Max pad length (5.0 sec)
    max_pad_len = MAX_PAD_LENGTH
    walker = sorted(str(p) for p in Path(dir).glob(f'*.wav'))

    for i, path in enumerate(walker): 
        y, sr = librosa.core.load(path, sr=sample_rate, offset=0.5)
        # Z-normalization
        y = zscore(y)
        # Padding or truncated signal 
        if len(y) < max_pad_len:    
            y_padded = np.zeros(max_pad_len)
            y_padded[:len(y)] = y
            y = y_padded
        elif len(y) > max_pad_len:
            y = np.asarray(y[:max_pad_len])
        signals.append(y)
        labels.append(label) 
    return signals, labels 