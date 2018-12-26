'''Trains a Bidirectional LSTM on the IMDB sentiment classification task.

Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''

from __future__ import print_function

import json
import keras
import matplotlib.pyplot as plt
import numpy
import numpy as np
from keras.datasets import imdb
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.models import Sequential
from keras.preprocessing import sequence


class GaussianNoise(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        # print(len(self.model.layers))
        hidden_layer = self.model.layers[1]
        params = hidden_layer.get_weights()
        weights = params[0]
        signs = numpy.sign(weights)
        guassian_noise = numpy.random.uniform(0, 1, weights.shape) * 0.001
        guassian_noise = guassian_noise * signs
        guassain_weights = numpy.add(weights, guassian_noise)
        params[0] = guassain_weights
        hidden_layer.set_weights(params)


max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 500
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(max_features, 64, input_length=maxlen))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

gaussian_callback = GaussianNoise()
# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=10,
                    validation_data=(x_test, y_test),
                    validation_split=0.2)

model.summary()
# Plot training & validation accuracy values
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.savefig('imdb_bidirectional_lstm_no_noise.png')
#
# with open('imdb_bidirectional_lstm_no_noise.json', 'w') as f:
#     json.dump(history.history, f)
