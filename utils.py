import streamlit as st
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import PyPDF2
import docx
import random

# ... (keep all previous functions unchanged)

def display_results(results, texts):
    # Calculate average similarity for each model
    avg_similarities = {}
    for model_name, similarity_matrix in results.items():
        avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
        avg_similarities[model_name] = avg_similarity
    
    # Rank models based on average similarity
    ranked_models = sorted(avg_similarities.items(), key=lambda x: x[1], reverse=True)
    
    st.subheader("Ranked Models by Average Similarity")
    for rank, (model_name, avg_similarity) in enumerate(ranked_models, 1):
        # Calculate color based on similarity score
        r = max(0, min(255, int(510 * (1 - avg_similarity))))
        g = max(0, min(255, int(510 * avg_similarity)))
        color = f'rgb({r},{g},0)'
        
        # Create colored circle and model name with score
        circle = f'<svg width="20" height="20"><circle cx="10" cy="10" r="8" fill="{color}" /></svg>'
        model_info = f"{model_name.split('/')[-1]}: {avg_similarity:.4f}"
        
        # Display ranking with colored circle
        st.markdown(f"{rank}. {circle} {model_info}", unsafe_allow_html=True)
    
    st.subheader("Detailed Similarity Scores")
    for model_name, similarity_matrix in results.items():
        st.write(f"Results for {model_name.split('/')[-1]}")
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                score = similarity_matrix[i][j]
                # Calculate color for each score
                r = max(0, min(255, int(510 * (1 - score))))
                g = max(0, min(255, int(510 * score)))
                color = f'rgb({r},{g},0)'
                circle = f'<svg width="15" height="15"><circle cx="7.5" cy="7.5" r="6" fill="{color}" /></svg>'
                st.markdown(f"{circle} Similarity between Text {i+1} and Text {j+1}: {score:.4f}", unsafe_allow_html=True)
        st.write("---")

# ... (keep all other functions unchanged)
