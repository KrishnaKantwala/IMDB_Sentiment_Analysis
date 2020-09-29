# import required packages
from glob import glob
import numpy as np
import pandas as pd
import os,re,string
import re
import pickle
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import GlobalMaxPooling1D
from keras.layers import Dense,Dropout,Embedding,MaxPool1D
from keras.layers.convolutional import Conv1D
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.initializers import Constant
from nltk.stem import WordNetLemmatizer
from keras.layers import LSTM
import nltk

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

# saving the training data from files to labels
def organize_data(path,folders):
    texts,labels = [],[]
    for idx,label in enumerate(folders):
        for fname in glob(os.path.join(path, label, '*.*')):
            texts.append(open(fname, 'r',encoding="utf8").read())
            labels.append(idx)
    # stored as np.int8 to save space 
    return texts, np.array(labels).astype(np.int8)


def cleaning_reviews(train_x):
    
    cleanWords = []
    for i in range(len(train_x)):
    
        ps = PorterStemmer() 
        
        # Removing the html tags
        train_remove_html = re.sub(r'<[^<>]+>', " ", train_x[i])
         # Converting numbers to "NUMBER"
        train_num = re.sub(r'[0-9]+', 'number', train_remove_html)
         # Converting review to lower case.
        train_lowercase = train_num.lower()    
        # removing string punctuation         
        train_remove_punctuation = re.sub(r"[^a-zA-Z]", " ", train_lowercase )
           
        # using porter stemming.
        #print("Processing dataset with porter stemming...")
        train_stemmedWords = [ps.stem(word) for word in re.findall(r"\w+", train_remove_punctuation)]
        clean_words = " ".join(train_stemmedWords)
        cleanWords.append(clean_words)
        
    return cleanWords

def perform_tokenization(x_train):
    # Generate the text sequence for RNN model
    np.random.seed(1000)
    most_freq_words = 5000
    max_length = 500           
    
    tokenizer = Tokenizer(num_words = most_freq_words)
    tokenizer.fit_on_texts(x_train)
    
    #tokenising train data
    train_reviews_tokenized = tokenizer.texts_to_sequences(x_train)      
    x_train = pad_sequences(train_reviews_tokenized, maxlen = max_length) 
    
    # Save the tokenizer so we used in testing to predict any reviews.
    with open('models\\tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return x_train, most_freq_words, max_length

def run_model(x_train,train_data, most_freq_words, max_length):
    # running model
    embedding_vector_length = 32
    model = Sequential()
    model.add(Embedding(input_dim = most_freq_words, 
                                    output_dim = embedding_vector_length,
                                    input_length = max_length))
        
    model.add(Dropout(0.2))
    model.add(Conv1D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))
    model.add(MaxPool1D(pool_size = 2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid'))             
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    history = model.fit(x_train, train_data, batch_size=64, epochs=3)
    
    return model,history

if __name__ == "__main__": 
	# 1. load your training data
    
    PATH='data\\aclImdb\\'
    label = ['neg','pos']
    
    train_x,train_data = organize_data(f'{PATH}train',label)
    len(train_x),len(train_data)
    
    x_train = cleaning_reviews(train_x)
    print(len(train_x))
    
    # tokenizing data
    x_train, most_freq_words, max_length  = perform_tokenization(x_train)

	# 2. Train your network
	# 	Make sure to print your training loss and accuracy within training to show progress
    model, history = run_model(x_train,train_data, most_freq_words, max_length)
	# 	Make sure you print the final training accuracy
    accuracy = history.history['accuracy'][-1]
    print("Final Training accuracy:" , accuracy*100)

	# 3. Save your model
    model.save("models\\NLP_model.h5")
