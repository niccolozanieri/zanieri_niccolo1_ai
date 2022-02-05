# Addestramento e test di una cascata di classificatori per la localizzazione di volti in immagini

Per quanto riguarda la descrizione del progetto e del funzionamento delle classi e funzioni utilizzate rimando alla lettura del file pdf che è presente nella cartella zip consegnata.
Questo file ha il solo scopo di descrivere come replicare i risultati da me ottenuti.

## Come utilizzare il progetto

All'interno della cartella src presente nella repository in cui è situato questo file sono presenti i file cc_dump.py e cc_load.py.

L'esecuzione del primo file permette di addestrare una cascata di classificatori a partire dalle immagini contenute ai path specificati dalle variabili faces_path e non_faces_path presenti nel file 
src/weak_classifiers_learning.
Attualmente le variabili sono inizializzate per andare ad utilizzare le immagini presenti all'interno della cartella data/train_data fornita all'interno della repository.
Poiché sul mio calcolatore la fase di addestramento ha richiesto circa tre ore, è possibile evitare di effettuare direttamente l'addestramento in quanto il risultato dell'ultima esecuzione del file è caricato nel file pickled_classifiers/classifiers_cascade.pickle ed automaticamente utilizzato dal file cc_load.py.

Il file cc_load.py va a caricare il classificatore come descritto sopra e lo utilizza per ottenere le predizioni per le immagini contenute ai path descritti nelle variabili test_faces_path e test_non_faces_path dello stesso file. Infine, il codice va a stampare i risultati ottenuti sulla console.
In caso si eseguisse cc_dump.py senza raggiungere e terminare l'esecuzione del dump di pickle il classificatore usato sarà quello caricato nell'ultima esecuzione andata a buon fine.

In quanto il virtual environment da me usato è stato escluso dalla repository, ai fini di garantire il funzionamento dei file è necessario installare le dipendenze del progetto tramite pip ed il file requirements.txt contenuto nella repository.

Per eventuali problemi è possibile contattarmi alla mail: niccolo.zanieri1@stud.unifi.it

## Eventuali variazioni nell'uso del progetto

Nel caso in cui si volessero modificare le immagini utilizzate in fase di addestramento e/o di test è necessario tenere in conto che potrebbe esserci bisogno di prendere alcune precauzioni:

    1. Le immagini usate possono essere sia a colori che in scala di grigi ma devono avere dimensione 24x24 pixel
    
    2. I path a cui le immagini vengono cercate sono descritti di sopra e potrebbe esserci bisogno di cambiare il valore delle variabili in questione
    
    3. Il codice richiede che quando si usa una lista o array di immagini queste siano rappresentate come numpy.ndarray di dimensione 24x24 e contenenti valori interi
    
In caso di modifica del codice o di mancata osservazione delle informazioni indicate non posso garantire il corretto funzionamento del progetto.



