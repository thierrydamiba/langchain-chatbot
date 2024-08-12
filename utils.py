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

def configure_embedding_models():
    available_models = [
        "BAAI/bge-small-en-v1.5",
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-base-en-v1.5",
        "thenlper/gte-large"
    ]
    
    st.sidebar.write("### Select Embedding Models")
    selected_models = st.sidebar.multiselect(
        "Choose models to compare",
        options=available_models,
        default=available_models[:2],
        format_func=lambda x: x.split('/')[-1]
    )

    st.sidebar.write("### Selected Models:")
    for model in selected_models:
        st.sidebar.write(f"- {model.split('/')[-1]}")

    return selected_models

def get_embedding_model(model_name):
    try:
        return FastEmbedEmbeddings(model_name=model_name)
    except Exception as e:
        st.error(f"Error loading model {model_name}: {str(e)}")
        st.error("This could be due to network issues or restrictions in the current environment.")
        st.error("Please try again later or consider using a different model.")
        return None

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

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
                for j in range(i+1, len(texts)):
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
        st.write(f"Heatmap for {model_name.split('/')[-1]}")
        df = pd.DataFrame(similarity_matrix, 
                          columns=[f'Text {i+1}' for i in range(len(texts))],
                          index=[f'Text {i+1}' for i in range(len(texts))])
        st.dataframe(df.style.background_gradient(cmap='RdYlGn', vmin=0, vmax=1))
    
    st.subheader("Detailed Similarity Scores")
    for model_name, similarity_matrix in results.items():
        st.write(f"Results for {model_name.split('/')[-1]}")
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                score = similarity_matrix[i][j]
                color = f'rgb({int(255*(1-score))},{int(255*score)},0)'
                st.markdown(f"Similarity between Text {i+1} and Text {j+1}: <span style='color:{color}'>{score:.4f}</span>", unsafe_allow_html=True)
        st.write("---")

# ... (keep other functions unchanged)

def configure_llm():
    api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
    if api_key:
        return ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)
    else:
        st.sidebar.warning("Please enter your OpenAI API Key to enable LLM-based text analysis.")
        return None
