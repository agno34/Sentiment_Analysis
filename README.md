# Sentiment Analysis su Fake News

Questo progetto implementa un modello di deep learning progettato per distinguere tra notizie vere e notizie false[cite: 5]. Il modello esegue un compito di classificazione binaria per identificare se un articolo di cronaca è "real" o "fake"[cite: 5]. Il progetto utilizza il dataset WELFake, che comprende 72.134 articoli pre-processati[cite: 5].

## Architettura del Modello
L'architettura del modello è sviluppata in Keras e si basa su reti neurali ricorrenti bidirezionali (Bidirectional LSTM)[cite: 4, 5]. I livelli principali includono:
* **Embedding Layer**: Proietta le sequenze di parole in uno spazio vettoriale denso a 128 dimensioni[cite: 4, 5].
* **Primo strato BLSTM**: Composto da 64 unità, restituisce le intere sequenze per catturare il contesto locale parola per parola[cite: 4, 5].
* **Secondo strato BLSTM**: Composto da 64 unità, comprime le informazioni in un'unica rappresentazione finale[cite: 4, 5].
* **Dropout**: Un livello con un tasso di 0.5 è impiegato per ridurre il rischio di overfitting disattivando casualmente i neuroni[cite: 4, 5].
* **Dense Layer**: Un singolo neurone finale con funzione di attivazione sigmoide produce un output interpretabile come probabilità[cite: 4, 5].

## Struttura del Progetto

* `files/prepare_data.py`: Script che carica i dati, pulisce il testo rimuovendo punteggiatura e link, e applica la tokenizzazione limitata alle 10.000 parole più frequenti[cite: 3, 5]. Uniforma le sequenze con un padding di 500 token, divide i dati in set di training (70%), validation (15%) e test (15%), e salva tutto in `processed_data.pkl`[cite: 3, 5].
* `files/train_model.py`: Script che costruisce la rete neurale e la addestra utilizzando l'ottimizzatore Adam e la binary cross-entropy[cite: 4, 5]. Utilizza l'Early Stopping per interrompere l'addestramento in caso di mancato miglioramento e salva la versione migliore della rete nel file `best_model.keras`[cite: 4, 5].
* `files/evaluate_model.py`: Script che carica i dati di test salvati e il modello pre-addestrato per effettuare le predizioni[cite: 2]. Calcola le metriche di Accuracy, Precision, Recall e F1-Score, e genera a schermo la matrice di confusione utilizzando le librerie matplotlib e seaborn[cite: 2].
* `files/complete_execution.sh`: uno script Bash che esegue in sequenza automatica e silenziosa la preparazione dei dati, l'addestramento e la valutazione del modello[cite: 1].
* `files/only_evaluation.sh`: uno script per eseguire unicamente la fase di valutazione.
* `Sentiment_Analysis.pdf`: Documento accademico dell'Università di Parma a cura di Andrea Agnetti contenente l'analisi dettagliata, lo studio teorico sulle reti LSTM e le valutazioni del modello[cite: 5].

## Requisiti

Per eseguire gli script è necessario installare le dipendenze Python:

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib seaborn silence-tensorflow
