import streamlit as st
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import PyPDF2
import docx
import random
import matplotlib.pyplot as plt
import seaborn as sns

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

def cosine_similarity(vec_a, vec_b):
    """Compute the cosine similarity between two vectors."""
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def process_texts(texts, embedding_models):
    results = {}
    
    for model_name in embedding_models:
        model = get_embedding_model(model_name)
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
    # Calculate average similarity for each model
    avg_similarities = {}
    for model_name, similarity_matrix in results.items():
        avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
        avg_similarities[model_name] = avg_similarity
    
    # Rank models based on average similarity
    ranked_models = sorted(avg_similarities.items(), key=lambda x: x[1], reverse=True)
    
    # Get min and max scores for color scaling
    min_score = min(avg_similarities.values())
    max_score = max(avg_similarities.values())
    
    st.subheader("Ranked Models by Average Similarity")
    for rank, (model_name, avg_similarity) in enumerate(ranked_models, 1):
        # Get color based on comparative score
        color = get_color(avg_similarity, min_score, max_score)
        
        # Create colored circle and model name with score
        circle = f'<svg width="20" height="20"><circle cx="10" cy="10" r="8" fill="{color}" /></svg>'
        model_info = f"{model_name.split('/')[-1]}: {avg_similarity:.4f}"
        
        # Display ranking with colored circle
        st.markdown(f"{rank}. {circle} {model_info}", unsafe_allow_html=True)
    
    st.subheader("Detailed Similarity Scores")
    for model_name, similarity_matrix in results.items():
        st.write(f"Results for {model_name.split('/')[-1]}")
        scores = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        min_score = np.min(scores)
        max_score = np.max(scores)
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                score = similarity_matrix[i][j]
                color = get_color(score, min_score, max_score)
                circle = f'<svg width="15" height="15"><circle cx="7.5" cy="7.5" r="6" fill="{color}" /></svg>'
                st.markdown(f"{circle} Similarity between Text {i + 1} and Text {j + 1}: {score:.4f}", unsafe_allow_html=True)
        st.write("---")

    st.subheader("Confusion Matrix")
    for model_name, similarity_matrix in results.items():
        st.write(f"Confusion Matrix for {model_name.split('/')[-1]}")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar=True)
        ax.set_title(f"Confusion Matrix for {model_name.split('/')[-1]}")
        ax.set_xlabel("Document Index")
        ax.set_ylabel("Document Index")
        st.pyplot(fig)

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
