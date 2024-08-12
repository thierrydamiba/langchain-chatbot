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
import plotly.figure_factory as ff

# ... (keep all existing functions)

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
        st.write(f"{rank}. {model_name.split('/')[-1]}: {avg_similarity:.4f}")
    
    st.subheader("Similarity Heatmaps")
    for model_name, similarity_matrix in results.items():
        fig = ff.create_annotated_heatmap(
            z=similarity_matrix,
            x=[f'Text {i+1}' for i in range(len(texts))],
            y=[f'Text {i+1}' for i in range(len(texts))],
            colorscale='RdYlGn',
            zmin=0, zmax=1
        )
        fig.update_layout(title=f"Similarity Heatmap for {model_name.split('/')[-1]}")
        st.plotly_chart(fig)
    
    st.subheader("Detailed Similarity Scores")
    for model_name, similarity_matrix in results.items():
        st.write(f"Results for {model_name.split('/')[-1]}")
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                score = similarity_matrix[i][j]
                color = f'rgb({int(255*(1-score))},{int(255*score)},0)'
                st.markdown(f"Similarity between Text {i+1} and Text {j+1}: <span style='color:{color}'>{score:.4f}</span>", unsafe_allow_html=True)
        st.write("---")

# ... (keep all other functions unchanged)
