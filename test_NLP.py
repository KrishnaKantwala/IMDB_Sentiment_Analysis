# import required packages
from glob import glob
import numpy as np
import pandas as pd
import os,re,string
import re
import pickle
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
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
from keras.models import load_model

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

# saving the testing data from files to labels
def organize_data(path,folders):
    texts,labels = [],[]
    for idx,label in enumerate(folders):
        for fname in glob(os.path.join(path, label, '*.*')):
            texts.append(open(fname, 'r',encoding="utf8").read())
            labels.append(idx)
    # stored as np.int8 to save space 
    return texts, np.array(labels).astype(np.int8)

def cleaning_reviews(test_x):
    
    testcleanWords = []
    for i in range(len(test_x)):
    
        ps = PorterStemmer() 
        
         # Removing the html tags
        test_remove_html = re.sub(r'<[^<>]+>', " ", test_x[i]) 
         # Converting numbers to "NUMBER"
        test_num = re.sub(r'[0-9]+', 'number', test_remove_html) 
         # Converting entire review to lower case.
        test_lowercase = test_num.lower()     
        # removing string punctuation
        test_remove_punctuation = re.sub(r"[^a-zA-Z]", " ", test_lowercase )
           
        # using porter stemming.
        #print("Processing dataset with porter stemming...")
        test_stemmedWords = [ps.stem(word) for word in re.findall(r"\w+", test_remove_punctuation)]
        clean_words = " ".join(test_stemmedWords)
        testcleanWords.append(clean_words)
        
    return testcleanWords

def perform_tokenization(x_test):
    
    with open('models\\tokenizer.pickle', 'rb') as file:
        tokenizer = pickle.load(file)
    
    max_length = 500  
    
    #tokenising Test data
    test_review_tokenized = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(test_review_tokenized, maxlen = max_length)
    
    return x_test


if __name__ == "__main__": 
    
	# 1. Load your saved model
    model = load_model('models\\NLP_model.h5')

	# 2. Load your testing data
    PATH='data\\aclImdb\\'
    label = ['neg','pos']
    
    # saving data from files into testing lables
    test_x,test_data = organize_data(f'{PATH}test',label)
    len(test_x),len(test_data)
    
    # cleaning test data
    x_test = cleaning_reviews(test_x)
    print(len(test_x))
    
    # tokenizing data
    x_test  = perform_tokenization(x_test)
    
    
	# 3. Run prediction on the test data and print the test accuracy
    loss, accuracy = model.evaluate(x_test,test_data, batch_size=128)
    print('Accuracy: %f' % (accuracy*100))
    
    
    
    
    
    
    
    
    
    
    
    
    
