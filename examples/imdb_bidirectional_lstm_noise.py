'''Trains a Bidirectional LSTM on the IMDB sentiment classification task.

Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''

from __future__ import print_function

import json
from keras import optimizers

import keras
import matplotlib.pyplot as plt
import numpy
import numpy as np
from keras.datasets import imdb
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.models import Sequential
from keras.preprocessing import sequence


class GaussianNoise(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        hidden_layer = self.model.layers[1]
        params = hidden_layer.get_weights()
        weights = params[0]
        biases = params[1]

        weight_signs = numpy.sign(weights)
        bias_signs = numpy.sign(biases)

        weight_noise = numpy.random.uniform(0, 1, weights.shape) * 0.0001
        weight_noise = weight_noise * weight_signs
        weight_noise = numpy.add(weight_noise, weights)
        params[0] = weight_noise

        bias_noise = numpy.random.uniform(0, 1, biases.shape) * 0.0001
        bias_noise = bias_noise * bias_signs
        bias_noise = numpy.add(bias_noise, biases)
        params[1] = bias_noise

        hidden_layer.set_weights(params)

max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 100
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
# adam = optimizers.adam(lr=0.0001)
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer="adam")

print('Train...')
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=10,
                    validation_data=(x_test, y_test))

model.summary()
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print(history.history)
print('Test score:', score)
print('Test accuracy:', acc)

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
