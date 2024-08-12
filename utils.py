import streamlit as st
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import PyPDF2
import docx
import random
import matplotlib.pyplot as plt
import seaborn as sns

# ... (keep the EMBEDDING_MODELS list and other functions unchanged)

def get_embedding_model(model_name):
    try:
        return FastEmbedEmbeddings(model_name=model_name)
    except Exception as e:
        st.error(f"An error occurred while loading the model '{model_name}': {str(e)}")
        return None

def process_texts(texts, embedding_models):
    results = {}
    for model_name in embedding_models:
        model = get_embedding_model(model_name)
        if model is None:
            continue
        
        try:
            embeddings = model.embed_documents(texts)
            
            similarity_matrix = np.zeros((len(texts), len(texts)))
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    similarity = cosine_similarity(embeddings[i], embeddings[j])
                    similarity_matrix[i][j] = similarity
                    similarity_matrix[j][i] = similarity
            
            results[model_name] = similarity_matrix
        except Exception as e:
            st.error(f"Error processing texts with model {model_name}: {str(e)}")
    return results

def display_results(results, texts):
    if not results:
        st.error("No results to display. All models failed to process the texts.")
        return

    avg_similarities = {}
    for model_name, similarity_matrix in results.items():
        avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
        avg_similarities[model_name] = avg_similarity
    
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

# ... (keep other functions unchanged)
