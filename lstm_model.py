# lstm_model.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer

def load_data(filepath="data/ciphertexts.csv"):
    """
    Loads the dataset and prepares it for LSTM training.
    """
    df = pd.read_csv(filepath)
    return df['plaintext'], df['ciphertext']

def preprocess_data(plaintexts, ciphertexts, max_len=10):
    """
    Preprocesses the text data by tokenizing and padding sequences.
    """
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(plaintexts + ciphertexts)

    plaintext_seq = tokenizer.texts_to_sequences(plaintexts)
    ciphertext_seq = tokenizer.texts_to_sequences(ciphertexts)

    X = pad_sequences(ciphertext_seq, maxlen=max_len, padding='post')
    y = pad_sequences(plaintext_seq, maxlen=max_len, padding='post')
    y = to_categorical(y, num_classes=len(tokenizer.word_index) + 1)

    return X, y, tokenizer

def build_lstm_model(input_shape, num_classes):
    """
    Builds and compiles the LSTM model.
    """
    model = Sequential()
    model.add(Embedding(input_dim=num_classes, output_dim=128, input_length=input_shape[1]))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(X, y, tokenizer, epochs=10):
    """
    Trains the LSTM model on the given data.
    """
    num_classes = len(tokenizer.word_index) + 1
    model = build_lstm_model(X.shape, num_classes)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)

    model.save("lstm_cryptanalysis_model.h5")
    return model
