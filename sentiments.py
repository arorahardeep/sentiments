#!/usr/bin/env python

"""
Market sentiment analysis on top news feeds data
"""

from load_data_amz import SentimentsData
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


def checkpoints():
    filepath="models/weights-improvement-{epoch:02d}-{loss:.4f}-sentiments.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    return callbacks_list

sd = SentimentsData()

print('Loading data...')
(x_train, y_train), (x_test, y_test) = sd.load()

max_features = sd.corpus_size
maxlen = 1000 #sd.max_size  # cut texts after this number of words (among top max_features most common words)
batch_size = 64
sd.create_embeddings_matrix()

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print("Max features %i, Maxlen %i"%(max_features,maxlen))

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print(sd.embedding_matrix)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features+1, sd.EMBEDDING_DIM, weights=[sd.embedding_matrix], input_length=maxlen, trainable=True))
model.add(LSTM(128, recurrent_dropout=0.4, return_sequences = True, dropout=0.2 ))
model.add(LSTM(128, recurrent_dropout = 0.3, dropout=0.2 ))
#model.add(LSTM(128, recurrent_dropout = 0.3 ))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.summary()

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

model.save("models/sentiments_full_glove_embeddings.hdf5")
