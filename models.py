from transformers import pipeline

import pickle

#-------------------model="facebook/bart-large-cnn"--------------------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

#-------------------model="google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
#-------------------model = "deepset/roberta-base-squad2"
qadata = pipeline('question-answering', model="google-bert/bert-large-uncased-whole-word-masking-finetuned-squad")

with open('summarizer.pkl', 'wb') as pkl:
    pickle.dump(summarizer, pkl, pickle.HIGHEST_PROTOCOL)

with open('qamodel.pkl', 'wb') as p:
    pickle.dump(qadata, p, pickle.HIGHEST_PROTOCOL)