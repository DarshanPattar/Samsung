from transformers import pipeline
import pdfplumber
import streamlit as st

#-------------------model="facebook/bart-large-cnn"--------------------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

#-------------------model="google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
#-------------------model = "deepset/roberta-base-squad2"
qadata = pipeline('question-answering', model="google-bert/bert-large-uncased-whole-word-masking-finetuned-squad")


st.title("Summary")

file = st.file_uploader("Choose a file.", type=('pdf'))

if file:

    reader = pdfplumber.open(file)
    #----PDF read ---------------------------
    #reader = PdfReader('Profile.pdf')
    data = ""
    for page in reader.pages[:2]:
        data += page.extract_text()

    #-----------------------------------------



    input = {
        'question': 'What are the skills?',
        'context': data
    }
    expertise = qadata(input)['answer']

    input = {
        'question': 'What is the expertise?',
        'context': data
    }
    expertise += qadata(input)['answer']


    article = data
    summary = summarizer(article)[0]['summary_text']

    print('\nSummary: ',summary)
    print('\nExpertise: ',expertise)

    st.write(summary)
    st.write(expertise)
