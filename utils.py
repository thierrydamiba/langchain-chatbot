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
    {"name": "BAAI/bge-small-zh-v1.5", "dim": 512, "description": "Fast and recommended Chinese model", "size": 0.090},
    {"name": "snowflake/snowflake-arctic-embed-xs", "dim": 384, "description": "Based on all-MiniLM-L6-v2 model with only 22m ...", "size": 0.090},
    {"name": "sentence-transformers/all-MiniLM-L6-v2", "dim": 384, "description": "Sentence Transformer model, MiniLM-L6-v2", "size": 0.090},
    {"name": "jinaai/jina-embeddings-v2-small-en", "dim": 512, "description": "English embedding model supporting 8192 sequen...", "size": 0.120},
    {"name": "BAAI/bge-small-en", "dim": 384, "description": "Fast English model", "size": 0.130},
    {"name": "snowflake/snowflake-arctic-embed-s", "dim": 384, "description": "Based on infloat/e5-small-unsupervised, does n...", "size": 0.130},
    {"name": "nomic-ai/nomic-embed-text-v1.5-Q", "dim": 768, "description": "Quantized 8192 context length english model", "size": 0.130},
    {"name": "BAAI/bge-base-en-v1.5", "dim": 768, "description": "Base English model, v1.5", "size": 0.210},
    {"name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "dim": 384, "description": "Sentence Transformer model, paraphrase-multili...", "size": 0.220},
    {"name": "Qdrant/clip-ViT-B-32-text", "dim": 512, "description": "CLIP text encoder", "size": 0.250},
    {"name": "jinaai/jina-embeddings-v2-base-de", "dim": 768, "description": "German embedding model supporting 8192 sequenc...", "size": 0.320},
    {"name": "BAAI/bge-base-en", "dim": 768, "description": "Base English model", "size": 0.420},
    {"name": "snowflake/snowflake-arctic-embed-m", "dim": 768, "description": "Based on intfloat/e5-base-unsupervised model, ...", "size": 0.430},
    {"name": "nomic-ai/nomic-embed-text-v1.5", "dim": 768, "description": "8192 context length english model", "size": 0.520},
    {"name": "jinaai/jina-embeddings-v2-base-en", "dim": 768, "description": "English embedding model supporting 8192 sequen...", "size": 0.520},
    {"name": "nomic-ai/nomic-embed-text-v1", "dim": 768, "description": "8192 context length english model", "size": 0.520},
    {"name": "snowflake/snowflake-arctic-embed-m-long", "dim": 768, "description": "Based on nomic-ai/nomic-embed-text-v1-unsuperv...", "size": 0.540},
    {"name": "mixedbread-ai/mxbai-embed-large-v1", "dim": 1024, "description": "MixedBread Base sentence embedding model, does...", "size": 0.640},
    {"name": "jinaai/jina-embeddings-v2-base-code", "dim": 768, "description": "Source code embedding model supporting 8192 se...", "size": 0.640},
    {"name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "dim": 768, "description": "Sentence-transformers model for tasks like clu...", "size": 1.000},
    {"name": "snowflake/snowflake-arctic-embed-l", "dim": 1024, "description": "Based on intfloat/e5-large-unsupervised, large...", "size": 1.020},
    {"name": "thenlper/gte-large", "dim": 1024, "description": "Large general text embeddings model", "size": 1.200},
    {"name": "BAAI/bge-large-en-v1.5", "dim": 1024, "description": "Large English model, v1.5", "size": 1.200},
    {"name": "intfloat/multilingual-e5-large", "dim": 1024, "description": "Multilingual model, e5-large. Recommend using ...", "size": 2.240}
]

def configure_embedding_models():
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

def get_embedding_model(model_name):
    try:
        return FastEmbedEmbeddings(model_name=model_name)
    except Exception as e:
        st.error(f"An error occurred while loading the model '{model_name}': {str(e)}")
        return None

def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def process_texts(texts, embedding_models):
    results = {}
    for model_name in embedding_models:
        model = get_embedding_model(model_name)
        if model is None:
            continue
        
        embeddings = model.embed_documents(texts)
        
        similarity_matrix = np.zeros((len(texts), len(texts)))
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = cosine_similarity(embeddings[i], embeddings[j])
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        results[model_name] = similarity_matrix
    return results

def display_results(results, texts):
    if not results:
        st.error("No results to display. All models failed to process the texts.")
        return

    avg_similarities = {model: np.mean(matrix[np.triu_indices_from(matrix, k=1)]) for model, matrix in results.items()}
    ranked_models = sorted(avg_similarities.items(), key=lambda x: x[1], reverse=True)

    st.subheader("Summary of Average Similarities")
    for rank, (model_name, avg_similarity) in enumerate(ranked_models, 1):
        color = get_color(avg_similarity, min(avg_similarities.values()), max(avg_similarities.values()))
        circle = f'<svg width="20" height="20"><circle cx="10" cy="10" r="8" fill="{color}" /></svg>'
        st.markdown(f"{rank}. {circle} **{model_name.split('/')[-1]}**: {avg_similarity:.4f}", unsafe_allow_html=True)
    
    st.write("---")

    st.subheader("Model Comparisons")
    for rank, (model_name, avg_similarity) in enumerate(ranked_models, 1):
        st.markdown(f"### Model {rank}: {model_name.split('/')[-1]}")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"**Average Similarity**: {circle} {avg_similarity:.4f}", unsafe_allow_html=True)
            st.write("#### Individual Chunk Similarity Scores")
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    score = results[model_name][i][j]
                    st.write(f"Similarity between Chunk {i + 1} and Chunk {j + 1}: {score:.4f}")

        with col2:
            st.write("#### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(results[model_name], annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar=True)
            ax.set_title(f"Confusion Matrix for {model_name.split('/')[-1]}")
            ax.set_xlabel("Document Index")
            ax.set_ylabel("Document Index")
            st.pyplot(fig)

    st.write("---")

def get_color(score, min_score, max_score):
    normalized_score = (score - min_score) / (max_score - min_score)
    r = max(0, min(255, int(255 * (1 - normalized_score))))
    g = max(0, min(255, int(255 * normalized_score)))
    return f'rgb({r},{g},0)'

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

def sample_chunks(chunks, num_samples):
    return random.sample(chunks, min(num_samples, len(chunks)))
