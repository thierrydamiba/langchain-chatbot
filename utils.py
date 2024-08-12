import streamlit as st
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import numpy as np
import PyPDF2
import docx
import random
import pandas as pd

# ... (keep all previous functions)

def extract_text_from_file(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'txt':
        return uploaded_file.getvalue().decode('utf-8')
    elif file_extension == 'pdf':
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        return ' '.join(page.extract_text() for page in pdf_reader.pages)
    elif file_extension in ['docx', 'doc']:
        doc = docx.Document(uploaded_file)
        return ' '.join(paragraph.text for paragraph in doc.paragraphs)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(text)

def analyze_and_sample_text(text, llm):
    prompt = ChatPromptTemplate.from_template(
        "Analyze the following text and determine its type (e.g., academic paper, news article, fiction, etc.). "
        "Based on the type, suggest an appropriate number of samples (between 2 and 5) and a sampling strategy. "
        "Respond in the format: 'Type: [type], Samples: [number], Strategy: [strategy]'\n\nText: {text}"
    )
    
    response = llm(prompt.format_messages(text=text[:1000]))  # Use first 1000 chars to save tokens
    
    analysis = response.content.split(', ')
    text_type = analysis[0].split(': ')[1]
    num_samples = int(analysis[1].split(': ')[1])
    strategy = analysis[2].split(': ')[1]
    
    chunks = chunk_text(text)
    
    if strategy.lower() == 'random':
        samples = random.sample(chunks, min(num_samples, len(chunks)))
    elif strategy.lower() == 'evenly spaced':
        step = max(1, len(chunks) // num_samples)
        samples = chunks[::step][:num_samples]
    else:  # Default to random if strategy is not recognized
        samples = random.sample(chunks, min(num_samples, len(chunks)))
    
    return samples, text_type

def configure_llm():
    api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
    if api_key:
        return ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)
    else:
        st.sidebar.warning("Please enter your OpenAI API Key to enable LLM-based text analysis.")
        return None
