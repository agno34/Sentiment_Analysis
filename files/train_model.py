import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Parametri del modello (devono essere gli stessi di prepare_data.py)
max_features = 10000
maxlen = 500

def main():
    print("--- Caricamento dei dati pre-elaborati ---")
    with open('processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']

    print("--- Creazione e addestramento del modello ---")
    embedding_dim = 128
    lstm_units = 64
    dropout_rate = 0.5
    
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=maxlen))
    model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=True)))
    model.add(Bidirectional(LSTM(units=lstm_units)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)
    
    print("Inizio addestramento...")
    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint]
    )
    print("Addestramento completato. Il modello migliore è stato salvato come 'best_model.keras'")

if __name__ == "__main__":
    main()
