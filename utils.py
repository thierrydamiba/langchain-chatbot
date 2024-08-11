import streamlit as st
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import PyPDF2
import docx
import random

def get_fastembed_models():
    return {
        "Text Models": [
            "BAAI/bge-small-en-v1.5", "BAAI/bge-small-zh-v1.5", "snowflake/snowflake-arctic-embed-xs",
            "sentence-transformers/all-MiniLM-L6-v2", "jinaai/jina-embeddings-v2-small-en",
            "BAAI/bge-small-en", "snowflake/snowflake-arctic-embed-s", "nomic-ai/nomic-embed-text-v1.5-Q",
            "BAAI/bge-base-en-v1.5", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "Qdrant/clip-ViT-B-32-text", "jinaai/jina-embeddings-v2-base-de", "BAAI/bge-base-en",
            "snowflake/snowflake-arctic-embed-m", "nomic-ai/nomic-embed-text-v1.5",
            "jinaai/jina-embeddings-v2-base-en", "nomic-ai/nomic-embed-text-v1",
            "snowflake/snowflake-arctic-embed-m-long", "mixedbread-ai/mxbai-embed-large-v1",
            "jinaai/jina-embeddings-v2-base-code", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "snowflake/snowflake-arctic-embed-l", "thenlper/gte-large", "BAAI/bge-large-en-v1.5",
            "intfloat/multilingual-e5-large"
        ],
        "Sparse Text Models": [
            "Qdrant/bm25", "Qdrant/bm42-all-minilm-l6-v2-attentions",
            "prithvida/Splade_PP_en_v1", "prithivida/Splade_PP_en_v1"
        ],
        "Late Interaction Models": ["colbert-ir/colbertv2.0"],
        "Image Models": [
            "Qdrant/resnet50-onnx", "Qdrant/clip-ViT-B-32-vision",
            "Qdrant/Unicom-ViT-B-32", "Qdrant/Unicom-ViT-B-16"
        ]
    }

def configure_embedding_models():
    available_models = get_fastembed_models()
    all_models = [model for models in available_models.values() for model in models]
    
    st.sidebar.write("### Select Embedding Models")
    selected_models = st.sidebar.multiselect(
        "Choose models to compare",
        options=all_models,
        default=all_models[:2],
        format_func=lambda x: x.split('/')[-1]
    )

    st.sidebar.write("### Selected Models:")
    for model in selected_models:
        st.sidebar.write(f"- {model.split('/')[-1]}")

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
        st.subheader(f"Results for {model_name.split('/')[-1]}")
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                st.write(f"Similarity between Text {i+1} and Text {j+1}: {similarity_matrix[i][j]:.4f}")
        st.write("---")

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

def sample_chunks(chunks, num_samples=5):
    return random.sample(chunks, min(num_samples, len(chunks)))
