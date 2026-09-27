# Sentiment Analysis su Fake News

Questo progetto implementa un modello di deep learning progettato per distinguere tra notizie vere e notizie false. Il modello esegue un compito di classificazione binaria per identificare se un articolo è "real" o "fake". Il progetto utilizza il dataset WELFake, che comprende 72.134 articoli.

## Architettura del Modello
L'architettura del modello è sviluppata in Keras e si basa su reti neurali ricorrenti bidirezionali (Bidirectional LSTM). I livelli principali includono:
* **Embedding Layer**: Proietta le sequenze di parole in uno spazio vettoriale denso a 128 dimensioni.
* **Primo strato BLSTM**: Composto da 64 unità, restituisce le intere sequenze per catturare il contesto locale parola per parola.
* **Secondo strato BLSTM**: Composto da 64 unità, comprime le informazioni in un'unica rappresentazione finale.
* **Dropout**: Un livello con un tasso di 0.5 è impiegato per ridurre il rischio di overfitting disattivando casualmente i neuroni.
* **Dense Layer**: Un singolo neurone finale con funzione di attivazione sigmoide produce un output interpretabile come probabilità.

## Struttura del Progetto

* `files/prepare_data.py`: Script che carica i dati, pulisce il testo rimuovendo punteggiatura e link, e applica la tokenizzazione limitata alle 10.000 parole più frequenti. Uniforma le sequenze con un padding di 500 token, divide i dati in set di training (70%), validation (15%) e test (15%), e salva tutto in `processed_data.pkl`.
* `files/train_model.py`: Script che costruisce la rete neurale e la addestra utilizzando l'ottimizzatore Adam e la binary cross-entropy. Utilizza l'Early Stopping per interrompere l'addestramento in caso di mancato miglioramento e salva la versione migliore della rete nel file `best_model.keras`.
* `files/evaluate_model.py`: Script che carica i dati di test salvati e il modello pre-addestrato per effettuare le predizioni. Calcola le metriche di Accuracy, Precision, Recall e F1-Score, e genera a schermo la matrice di confusione utilizzando le librerie matplotlib e seaborn.
* `files/complete_execution.sh`: uno script Bash che esegue in sequenza automatica e silenziosa la preparazione dei dati, l'addestramento e la valutazione del modello.
* `files/only_evaluation.sh`: uno script per eseguire unicamente la fase di valutazione.
* `Sentiment_Analysis.pdf`: Documento accademico dell'Università di Parma a cura di Andrea Agnetti contenente l'analisi dettagliata, lo studio teorico sulle reti LSTM e le valutazioni del modello.

## Requisiti

Per eseguire gli script è necessario installare le dipendenze Python:

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib seaborn silence-tensorflow
