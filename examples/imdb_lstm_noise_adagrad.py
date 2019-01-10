'''Trains an LSTM model on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.

# Notes

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function

import json
from keras import optimizers

import keras
import matplotlib.pyplot as plt
import numpy
from keras.datasets import imdb
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import plot_model


class Gaussian(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        hidden_layers = self.model.layers[1:-1]

        for hidden_layer in hidden_layers:
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
# cut texts after this number of words (among top max_features most common words)
maxlen = 80
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

print('Build model...')
model = Sequential()
# model.add(Embedding(max_features, 128))
model.add(Embedding(max_features, 64, input_length=maxlen))
# model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

# adagrad = optimizers.adagrad(lr=0.0001)
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer="adagrad",
              metrics=['accuracy'])

print('Train...')

guassian_noise = Gaussian()
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=10,
                    callbacks=[guassian_noise],
                    validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
model.summary()

print('Test score:', score)
print('Test accuracy:', acc)
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.savefig('imdb_lstm_noise.png')
#
# with open('imdb_lstm_noise.json', 'w') as f:
#     json.dump(history.history, f)
#
# plot_model(model, to_file='imdb_lstm_architecture.png')
