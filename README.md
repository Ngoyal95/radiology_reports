# radiology_reports

The following packages are necessary:
pandas
numpy
scikit-learn
nltk
gensim
biobert-embeddings

To install biobert-embeddings, navigate to: radiology_reports/biobert_embeddings/, and run the command -python setup.py install-

You also need to download the GloVe embeddings Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download)
https://nlp.stanford.edu/projects/glove/

Extract these to the folder: radiology_reports/data/glove_models/


---

If you have multiple .csv files with radiology report data (id, scantype, scandesc, report), put them all in the /data/raw_data/ folder, and run the script: combine_files.py
