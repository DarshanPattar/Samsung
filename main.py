import PyPDF2
import docx2txt
import nltk
import google.generativeai as genai
import streamlit as st
from io import BytesIO

genai.configure(api_key="AIzaSyAWOVVIs-6LXraOG7RIvimx_rZqzKWx4CE")

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192*2,
  "response_mime_type": "text/plain",
}

def parse_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
    text = ''
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text


def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_words = [w for w in tokens if w not in stop_words]
    return " ".join(filtered_words)

def llm_process(text):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config,)
    chat_session = model.start_chat(history=[])
    prompt = f"In the following profile information \n'{text}'\n\nIdentify the top three areas of expertise or skills (key as expertise) mentioned in the text, the email and college from the profile in json format."
    response = chat_session.send_message(prompt)
    return response.text

def summarize_and_extract(pdf_file):
    text = parse_pdf(pdf_file)
    processed_text = preprocess_text(text)
    response = llm_process(processed_text)
    return response


def summarize_and_extract_doc(doc_file):
    text = docx2txt.process(doc_file)
    processed_text = preprocess_text(text)
    response = llm_process(processed_text)
    return response


st.title("Profile analysis and Expertise Extractor")

option = st.selectbox(
    "Choose the type",('pdf', 'doc', 'url')
)

if option == 'pdf':
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner("Processing..."):
            response = summarize_and_extract(uploaded_file)
            st.write(response)


if option == 'doc':
    uploaded_file = st.file_uploader("Choose a DOC file")
    if uploaded_file is not None:
        with st.spinner("Processing..."):
            response = summarize_and_extract_doc(uploaded_file)
            st.write(response)


if option == 'url':
    path = st.text_input(label='Url')
    model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config,)
    chat_session = model.start_chat(history=[])
    prompt = f"In the following url of a profile \n'{path}'\n\nIdentify the top three areas of expertise or skills (key as expertise) mentioned in the text, the email and college from the profile in json format."
    response = chat_session.send_message(prompt)
    st.write(response)