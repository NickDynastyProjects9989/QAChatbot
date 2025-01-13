import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

import time

from dotenv import load_dotenv

load_dotenv()

os.environ['GROQ_API_KEY'] = "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"
groq_api =  os.getenv('GROQ_API_KEY')

llm = ChatGroq(groq_api_key= groq_api,model_name = 'Llama3-8b-8192')

prompt = ChatPromptTemplate.from_template(
    """
    Answer questions based on provided context only
    please provide most accurate answer based on question
    <context>
     {context} 
    <context>
    Question: {input}
"""


)

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings =  OllamaEmbeddings()
        st.session_state.loader= PyPDFDirectoryLoader("research_papers")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors= FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)



user_prompt = st.text_input("Enter youe query from research paper")

if st.button("Document embedding"):
    create_vector_embeddings()
    st.write("Vector Database is ready")


