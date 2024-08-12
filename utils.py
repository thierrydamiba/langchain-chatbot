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

    # Summary of average similarities
    st.subheader("Summary of Average Similarities")
    for rank, (model_name, avg_similarity) in enumerate(ranked_models, 1):
        color = get_color(avg_similarity, min_score=min(avg_similarities.values()), max_score=max(avg_similarities.values()))
        circle = f'<svg width="20" height="20"><circle cx="10" cy="10" r="8" fill="{color}" /></svg>'
        st.markdown(f"{rank}. {circle} **{model_name.split('/')[-1]}**: {avg_similarity:.4f}", unsafe_allow_html=True)
    
    st.write("---")

    st.subheader("Model Comparisons")

    # Display each model's results
    for rank, (model_name, avg_similarity) in enumerate(ranked_models, 1):
        st.markdown(f"### Model {rank}: {model_name.split('/')[-1]}")
        
        # Create columns for chunk similarity and confusion matrix
        col1, col2 = st.columns([1, 2])

        with col1:
            # Display average similarity with color
            st.markdown(f"**Average Similarity**: {circle} {avg_similarity:.4f}", unsafe_allow_html=True)
            
            st.write("#### Individual Chunk Similarity Scores")
            # Display each chunk similarity
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
