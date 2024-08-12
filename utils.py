import streamlit as st
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import PyPDF2
import docx
import random
import pandas as pd
import io

# ... (keep other functions unchanged)

def extract_text_from_file(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'txt':
        # Read the file as bytes and decode
        text = uploaded_file.getvalue().decode('utf-8', errors='ignore')
        # Remove any non-printable characters
        return ''.join(char for char in text if char.isprintable() or char.isspace())
    elif file_extension == 'pdf':
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
        text = ' '.join(page.extract_text() for page in pdf_reader.pages)
        return ' '.join(text.split())  # Remove extra whitespace
    elif file_extension in ['docx', 'doc']:
        doc = docx.Document(io.BytesIO(uploaded_file.getvalue()))
        text = ' '.join(paragraph.text for paragraph in doc.paragraphs)
        return ' '.join(text.split())  # Remove extra whitespace
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    # Ensure each chunk is a proper sentence or paragraph
    return [chunk.strip() for chunk in chunks if chunk.strip()]

# ... (keep other functions unchanged)
