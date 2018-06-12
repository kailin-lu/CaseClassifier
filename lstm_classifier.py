import os
from collections import Counter

import numpy as np
from numpy.random import seed
seed(0)
from tensorflow import set_random_seed
set_random_seed(0)

from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Concatenate, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

class TextLSTM():
    def __init__(self, vocab_size=100, embedding_size=300, seq_length=2000,
                dropout=0.5, num_units=512, num_classes=2, lr=1e-5):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.seq_length = seq_length
        self.dropout = dropout
        self.num_units = num_units
        self.num_classes = num_classes
        self.lr = lr

    def _build_model(self):
        main_input = Input(batch_shape=(None, self.seq_length), name='main_input')
        aux_input = Input(batch_shape=(None, 7), name='aux_input')
        embed = Embedding(self.vocab_size, self.embedding_size,
                          mask_zero=True, input_length=self.seq_length)(main_input)
        lstm1 = LSTM(self.num_units, return_sequences=True)(embed)
        lstm2 = LSTM(self.num_units)(lstm1)
        concat = Concatenate()([lstm2, aux_input])
        dropout = Dropout(self.dropout)(concat) 
        dense = Dense(self.num_classes, activation='sigmoid')(dropout)
        model = Model(inputs=[main_input, aux_input], outputs=dense)

        adam = Adam(lr=self.lr)
        model.compile(optimizer=adam,
                      loss='binary_crossentropy',
                      metrics=['accuracy', categorical_accuracy])
        print(model.summary())
        return model

    def train(self, train_x, train_y, train_add_x, val_add_x,
              val_x, val_y, geniss, embed_metadata,
              epochs=10, batch_size=32, model=None):

        labels = [np.argmax(label) for label in train_y]
        # Calculate class weight
        class_weight = Counter(labels)
        for k,v in class_weight.items():
            class_weight[k] = 1 - (v / len(labels))
        print('Class Weight Liberal: {} Conservative {}'.format(class_weight[0],
                                                                class_weight[1]))
        # Build model
        if not model: 
            model = self._build_model()

        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001,
                                  patience=3, verbose=1, mode='auto')

        tensorboard = TensorBoard(log_dir='tf_logs', embeddings_freq=2,
                                  embeddings_metadata=embed_metadata)

        callbacks_list = [earlystop, tensorboard]

        model.fit({'main_input': train_x, 'aux_input': train_add_x}, train_y,
                  validation_data=({'main_input': val_x, 'aux_input': val_add_x}, val_y), 
                  class_weight=class_weight,
                  epochs=epochs, batch_size=batch_size,
                  callbacks=callbacks_list)
        score = model.evaluate({'main_input': val_x, 'aux_input': val_add_x}, val_y, batch_size=batch_size)
        print('Validation Accuracy:', score[1])
        save_path = 'final_lstm_model/geniss{}-acc{:2f}-lr{:2f}-units{}.h5'.format(geniss,
                                                                                   score[1], 
                                                                                   self.lr, self.num_units) 
        model_json = model.to_json()
        model_name = 'final_lstm_model/model-{}-{}.json'.format(self.num_units, self.lr)
        with open(model_name, 'w') as json_file:
            json_file.write(model_json)
        model.save(save_path)

        val_pred = model.predict({'main_input': val_x, 'aux_input': val_add_x})
        return val_pred
