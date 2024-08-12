import streamlit as st
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import PyPDF2
import docx
import random
import matplotlib.pyplot as plt
import seaborn as sns

EMBEDDING_MODELS = [
    {"name": "BAAI/bge-small-en-v1.5", "dim": 384, "description": "Fast and Default English model", "size": 0.067},
    {"name": "BAAI/bge-base-en-v1.5", "dim": 768, "description": "Base English model, v1.5", "size": 0.210},
    {"name": "sentence-transformers/all-MiniLM-L6-v2", "dim": 384, "description": "Sentence Transformer model, MiniLM-L6-v2", "size": 0.090},
    {"name": "thenlper/gte-large", "dim": 1024, "description": "Large general text embeddings model", "size": 1.200}
]

def configure_embedding_models():
    try:
        st.sidebar.write("### Select Embedding Models")
        selected_models = st.sidebar.multiselect(
            "Choose models to compare",
            options=[model["name"] for model in EMBEDDING_MODELS],
            default=[EMBEDDING_MODELS[0]["name"]],
            format_func=lambda x: f"{x.split('/')[-1]} ({next(model['dim'] for model in EMBEDDING_MODELS if model['name'] == x)}d, {next(model['size'] for model in EMBEDDING_MODELS if model['name'] == x)}GB)"
        )

        st.sidebar.write("### Selected Models:")
        for model in selected_models:
            st.sidebar.write(f"- {model.split('/')[-1]}")

        return selected_models
    except Exception as e:
        st.error(f"An error occurred while configuring embedding models: {str(e)}")
        st.error("Using default model: BAAI/bge-small-en-v1.5")
        return ["BAAI/bge-small-en-v1.5"]

# ... (keep other functions unchanged)
