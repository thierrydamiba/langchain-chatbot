import streamlit as st
from langchain_community.embeddings import FastEmbedEmbeddings
import numpy as np

def configure_embedding_models():
    available_embedding_models = {
        "BAAI/bge-small-en-v1.5": "English",
        "BAAI/bge-small-zh-v1.5": "Chinese",
        "BAAI/bge-base-en-v1.5": "English (Base)",
        "BAAI/bge-base-zh-v1.5": "Chinese (Base)"
    }
    selected_models = st.sidebar.multiselect(
        label="Select Embedding Models",
        options=list(available_embedding_models.keys()),
        default=["BAAI/bge-small-en-v1.5"],
        format_func=lambda x: f"{x} ({available_embedding_models[x]})"
    )
    return selected_models

def get_embedding_model(model_name):
    return FastEmbedEmbeddings(model_name=model_name)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def process_texts(texts, embedding_models):
    results = {}
    for model_name in embedding_models:
        model = get_embedding_model(model_name)
        embeddings = model.embed_documents(texts)
        
        similarity_matrix = np.zeros((len(texts), len(texts)))
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                similarity = cosine_similarity(embeddings[i], embeddings[j])
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        results[model_name] = similarity_matrix
    return results

def display_results(results, texts):
    for model_name, similarity_matrix in results.items():
        st.subheader(f"Results for {model_name}")
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                st.write(f"Similarity between Text {i+1} and Text {j+1}: {similarity_matrix[i][j]:.4f}")
        st.write("---")
