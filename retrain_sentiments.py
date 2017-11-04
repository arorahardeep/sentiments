#!/usr/bin/env python

from load_data_amz import SentimentsData
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing import sequence

print("Loading model...")
model = load_model("sentiments_full_glove_embeddings.hdf5")
model.compile(loss='binary_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.summary()

print("Get the data to predict on...")
sd = SentimentsData()
print('Loading data...')
(x_train, y_train), (x_test, y_test) = sd.load()

max_features = sd.corpus_size
maxlen = 1000 #sd.max_size  # cut texts after this number of words (among top max_features most common words)
batch_size = 64
sd.create_embeddings_matrix()

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=5,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

model.save("models/sentiments_full_glove_embeddings_1.hdf5")