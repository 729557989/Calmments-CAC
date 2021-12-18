# importing essential libraries ----------------------------------------------------------------------------------------------
import tensorflow as tf
import pandas as pd
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from string import ascii_letters
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
import pickle
import re
import nltk
from bs4 import BeautifulSoup

# make sure training is done on GPU
tf.config.list_physical_devices('GPU')

# read our training data
data = pd.read_csv(r'jigsaw-toxic-comment-classification-challenge\train.csv') # your data file path

# define features and labels
X = data['comment_text']
y = data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]

# data cleaning functions + setup --------------------------------------------------------------------------------------------------------------------------------------------

def lower_text(text): #1
    return text.lower()
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}
def change_contraction(text): # 2
    return ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in text.split(" ")])

def remove_html(text): # 3
    return BeautifulSoup(text, "lxml").text

def text_clean(string): # 1
    temp = re.sub(r'\W',' ',string)
    return re.sub(r'\s+',' ',temp)
    
def remove_urls(text): # 2
    return re.sub('https?://\S+|www\.\S+', '', text)

allowed = set(ascii_letters + ' ')
def remove_punctuation(text): # 4
    return ''.join([c if c in allowed else ' ' for c in text])

stemmer = PorterStemmer() # 6
def stem_words(text):
    return [stemmer.stem(word) for word in text]


# NOTE: text preprocessing function -----------------------------------------------------------------------------------------------------------

def clean_txt(X):
    X = [lower_text(sentence) for sentence in X]
    X = [change_contraction(sentence) for sentence in X]
    X = [text_clean(sentence) for sentence in X]
    X = [remove_urls(sentence) for sentence in X]
    X = [remove_html(sentence) for sentence in X]
    X = [nltk.word_tokenize(sentence) for sentence in X]
    X = [stem_words(sentence) for sentence in X]
    return X

# NOTE: apparently... our data is too much, I only have 16gb of ram so gotta cut the training size --------------------------------
X = clean_txt(X[:110000])
y = y[:110000]

# NOTE: train test split, split some for testing so our model don't overfit

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=5)

# NOTE: train tokenizer model (an NLP technique that maps string to integar) ----------------------------------------------------
maxlen = 120

max_vocab = 7000
tokenizer = Tokenizer(num_words=max_vocab)
tokenizer.fit_on_texts(X_train)
tokenizer.fit_on_texts(X_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

# NOTE: tokenize our data set ------------------------------------------------------------------------------------------------
def tokenize(text):
    text = tokenizer.texts_to_sequences(text)
    return pad_sequences(text, padding='post', maxlen=maxlen)

X_train = tokenize(X_train)
X_test = tokenize(X_test)

X_train = np.asarray(X_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)

# NOTE: build our Long Short Term Memory neural network model --------------------------------------------------------------------------
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 120, input_length=maxlen, trainable=True))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(60, activation='relu'))
model.add(tf.keras.layers.Dense(30, activation='relu'))
model.add(tf.keras.layers.Dense(6, activation='sigmoid'))
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# NOTE: define backpropagating metrics
callbacks_ = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8,
             restore_best_weights=True, verbose=1), ModelCheckpoint(filepath='LSTM_detector-{epoch:02d}.h5',
             monitor='accuracy', save_best_only=True, 
             save_weights_only=False, mode='auto', save_freq='epoch', verbose=1)]

# NOTE: training !!!
history = model.fit(X_train, y_train,
                    epochs=30,
                    batch_size=500,
                    shuffle=True,
                    validation_data=(X_test, y_test),
                    verbose=2,
                    callbacks=[callbacks_])

loss, accuracy = model.evaluate(X_train, y_train, verbose=1)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print("Testing Accuracy:  {:.4f}".format(accuracy))

# NOTE: save tokenizer, we already saved our AI model as it was programmed some from "callbacks_" ----------------------------------

# saving tokenizer
with open('tokenizer2_LSTM.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)