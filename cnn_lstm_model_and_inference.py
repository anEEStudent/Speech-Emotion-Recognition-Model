### General imports ###
import os
from glob import glob
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
#!/usr/bin/env python3

### Audio import ###
import librosa
from pathlib import Path

## Basics ##
import time
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

signals = [] 
labels = []
signals, labels = preprocess('/home/teppei/Desktop/nlp_training_dataset/train_dataset/angry', 'angry', signals, labels)
signals, labels = preprocess('/home/teppei/Desktop/nlp_training_dataset/train_dataset/fear', 'fear', signals, labels)
signals, labels = preprocess('/home/teppei/Desktop/nlp_training_dataset/train_dataset/happy', 'happy', signals, labels)
signals, labels = preprocess('/home/teppei/Desktop/nlp_training_dataset/train_dataset/neutral', 'neutral', signals, labels)
signals, labels = preprocess('/home/teppei/Desktop/nlp_training_dataset/train_dataset/sad', 'sad', signals, labels)
mel_spect = np.asarray(list(map(mel_spectrogram, signals)))

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(np.ravel(labels)))

from sklearn.model_selection import train_test_split
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(mel_spect, y_train, test_size=0.2, random_state=42, shuffle=True)

X_train_split = frame(X_train_split)
X_test_split = frame(X_test_split)

# Reshape for convolution
X_train_split = X_train_split.reshape(X_train_split.shape[0], X_train_split.shape[1] ,X_train_split.shape[2], X_train_split.shape[3], 1)
X_test_split = X_test_split.reshape(X_test_split.shape[0], X_test_split.shape[1] , X_test_split.shape[2], X_test_split.shape[3], 1)

K.clear_session()

# Define input
input_y = Input(shape=X_train_split.shape[1:], name='Input_MELSPECT')

# First LFLB (local feature learning block)
y = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_1_MELSPECT')(input_y)
y = TimeDistributed(BatchNormalization(), name='BatchNorm_1_MELSPECT')(y)
y = TimeDistributed(Activation('elu'), name='Activ_1_MELSPECT')(y)
y = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'), name='MaxPool_1_MELSPECT')(y)
y = TimeDistributed(Dropout(0.2), name='Drop_1_MELSPECT')(y)

# Second LFLB (local feature learning block)
y = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_2_MELSPECT')(y)
y = TimeDistributed(BatchNormalization(), name='BatchNorm_2_MELSPECT')(y)
y = TimeDistributed(Activation('elu'), name='Activ_2_MELSPECT')(y)
y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_2_MELSPECT')(y)
y = TimeDistributed(Dropout(0.2), name='Drop_2_MELSPECT')(y)

# Third LFLB (local feature learning block)
y = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_3_MELSPECT')(y)
y = TimeDistributed(BatchNormalization(), name='BatchNorm_3_MELSPECT')(y)
y = TimeDistributed(Activation('elu'), name='Activ_3_MELSPECT')(y)
y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_3_MELSPECT')(y)
y = TimeDistributed(Dropout(0.2), name='Drop_3_MELSPECT')(y)

# Fourth LFLB (local feature learning block)
y = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_4_MELSPECT')(y)
y = TimeDistributed(BatchNormalization(), name='BatchNorm_4_MELSPECT')(y)
y = TimeDistributed(Activation('elu'), name='Activ_4_MELSPECT')(y)
y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_4_MELSPECT')(y)
y = TimeDistributed(Dropout(0.2), name='Drop_4_MELSPECT')(y)

# Flat
y = TimeDistributed(Flatten(), name='Flat_MELSPECT')(y)

# LSTM layer
y = LSTM(256, return_sequences=False, dropout=0.2, name='LSTM_1')(y)

# Fully connected
y = Dense(5, activation='softmax', name='FC')(y)

# Build final model
model = Model(inputs=input_y, outputs=y)

model.summary()

class_names = ['angry', 'fear', 'happy', 'neutral', 'sad']

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# Compile model
model.compile(optimizer=SGD(learning_rate=0.01, decay=1e-6, momentum=0.8), loss='categorical_crossentropy', metrics=['accuracy'])

# Save best model
best_model_save = ModelCheckpoint('Trained_Model_extra.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')

# Early stopping
# early_stopping = EarlyStopping(monitor='val_accuracy', patience=30, verbose=1, mode='max')

NUM_EPOCHS = 300
# Fit model
history = model.fit(X_train_split, y_train_split, batch_size=64, epochs=NUM_EPOCHS, validation_data=(X_test_split, y_test_split), callbacks=[best_model_save])

from keras.models import load_model
best_model = load_model('Trained_Model_extra.hdf5')
filenames = sorted(os.listdir('test_dataset'))
test_signals = []
sample_rate = 16000     
# Max pad length (5.0 sec)
max_pad_len = MAX_PAD_LENGTH
for file in filenames: 
  y, sr = librosa.core.load(Path('test_dataset/' + file), sr=sample_rate, offset=0.5)
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
best_model.predict(np.expand_dims(X_test[1], axis = 0))
pred_list = [] 
for i in range(len(X_test)):
  pred = best_model.predict(np.expand_dims(X_test[i], axis = 0))
  pred_list.append(class_map[pred[0].argmax(0)])

import pandas as pd
import datetime
result_tuple = list(zip(filenames, pred_list))
submission = pd.DataFrame(result_tuple, columns=['filename', 'label'])
now = datetime.datetime.now() 
submission_name = 'KL_model_' + now.strftime("%m%d%Y %H:%M:%S") + '.csv'
submission[['filename', 'label']].to_csv(submission_name, header=None, index=None)


