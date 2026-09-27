import pandas as pd
import re
import string
import pickle
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
from silence_tensorflow import silence_tensorflow
silence_tensorflow()   

# Parametri per la tokenizzazione
max_features = 10000
maxlen = 500

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def main():
    print("--- Caricamento e pulizia dei dati ---")
    df = pd.read_csv("/home/agno34/fakeNews/WELFake_Dataset/WELFake_Dataset.csv")
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df.dropna(inplace=True)
    df['text'] = df['title'] + " " + df['text']
    df.drop(columns=['title'], inplace=True)
    df['text'] = df['text'].apply(clean_text)

    print("--- Tokenizzazione e vettorizzazione ---")
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(df['text'])
    X = tokenizer.texts_to_sequences(df['text'])
    X = pad_sequences(X, maxlen=maxlen)
    y = df['label'].values

    print("--- Divisione e salvataggio dei dataset ---")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    with open('processed_data.pkl', 'wb') as f:
        pickle.dump({
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'tokenizer': tokenizer
        }, f)

    print("Dati processati e salvati in 'processed_data.pkl'")

if __name__ == "__main__":
    main()
