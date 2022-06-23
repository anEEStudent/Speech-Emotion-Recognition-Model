### General imports ###
import os
from glob import glob
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

### Audio import ###
import librosa
from IPython.display import Audio
from pathlib import Path

## Basics ##
import os
import numpy as np

## Audio Preprocessing ##
import librosa
from scipy.stats import zscore

## Time Distributed CNN ##
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, TimeDistributed
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.layers import LSTM


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

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



from keras.models import load_model
best_model = load_model('/home/dh/Downloads/Trained_Model_FINAL.hdf5')
filenames = sorted(os.listdir('/home/dh/Downloads/test_audio'))
print(f'Number of files: {len(filenames)}')
test_signals = []
sample_rate = 16000     
# Max pad length (5.0 sec)
max_pad_len = MAX_PAD_LENGTH
for file in filenames: 
  y, sr = librosa.core.load(Path('/home/dh/Downloads/test_audio/' + file), sr=sample_rate, offset=0.5)
  # Z-normalization
  y = zscore(y)
  # Padding or truncated signal 
  if len(y) < max_pad_len:    
    y_padded = np.zeros(max_pad_len)
    y_padded[:len(y)] = y
    y = y_padded
  elif len(y) > max_pad_len:
    y = np.asarray(y[:max_pad_len])
  test_signals.append(y)

mel_spect_test = np.asarray(list(map(mel_spectrogram, test_signals)))
X_test = frame(mel_spect_test)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] , X_test.shape[2], X_test.shape[3], 1)
print(X_test[0].shape)
class_map = ['angry', 'fear', 'happy', 'neutral', 'sad'] 
pred = best_model.predict(np.expand_dims(X_test[1], axis = 0))
pred_list = [] 
for i in range(len(X_test)):
  pred = best_model.predict(np.expand_dims(X_test[i], axis = 0))
  pred_list.append(class_map[pred[0].argmax(0)])

import pandas as pd
import datetime
result_tuple = list(zip(filenames, pred_list))
submission = pd.DataFrame(result_tuple, columns=['filename', 'label'])
now = datetime.datetime.now() 
submission_name = 'finals_submission_new.csv'
submission[['filename', 'label']].to_csv(submission_name, header=None, index=None)