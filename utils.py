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

def analyze_and_sample_text(text, llm):
    prompt = ChatPromptTemplate.from_template(
        "Analyze the following text and determine its type (e.g., academic paper, news article, fiction, etc.). "
        "Based on the type, suggest an appropriate number of samples (between 2 and 5) and a sampling strategy. "
        "Respond in the format: 'Type: [type], Samples: [number], Strategy: [strategy]'\n\nText: {text}"
    )
    
    response = llm(prompt.format_messages(text=text[:1000]))  # Use first 1000 chars to save tokens
    
    analysis = response.content.split(', ')
    text_type = analysis[0].split(': ')[1]
    num_samples = int(analysis[1].split(': ')[1])
    strategy = analysis[2].split(': ')[1]
    
    chunks = chunk_text(text)
    
    if strategy.lower() == 'random':
        samples = random.sample(chunks, min(num_samples, len(chunks)))
    elif strategy.lower() == 'evenly spaced':
        step = max(1, len(chunks) // num_samples)
        samples = chunks[::step][:num_samples]
    else:  # Default to random if strategy is not recognized
        samples = random.sample(chunks, min(num_samples, len(chunks)))
    
    return samples, text_type

def configure_llm():
    api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
    if api_key:
        return ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)
    else:
        st.sidebar.warning("Please enter your OpenAI API Key to enable LLM-based text analysis.")
        return None
