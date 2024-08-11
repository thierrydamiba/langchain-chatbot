import os
import openai
import streamlit as st
from datetime import datetime
from streamlit.logger import get_logger
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

logger = get_logger('Langchain-Chatbot')

# Add this line at the beginning of each function
logger.info("Entering function: [function_name]")

# ... (keep all other functions the same, but add the logging line at the start of each)

def select_embedding_model():
    logger.info("Entering function: select_embedding_model")
    available_embedding_models = ["BAAI/bge-small-en-v1.5", "BAAI/bge-small-zh-v1.5"]
    try:
        selected_model = st.sidebar.selectbox(
            label="Embedding Model",
            options=available_embedding_models,
            key="SELECTED_EMBEDDING_MODEL"
        )
        logger.info(f"Selected embedding model: {selected_model}")
        return selected_model
    except Exception as e:
        logger.error(f"Error in select_embedding_model: {str(e)}")
        return "BAAI/bge-small-en-v1.5"  # Default to English model if there's an error

@st.cache_resource
def get_embedding_model(model_name):
    logger.info(f"Entering function: get_embedding_model with model_name: {model_name}")
    try:
        embedding_model = FastEmbedEmbeddings(model_name=model_name)
        logger.info("Embedding model created successfully")
        return embedding_model
    except Exception as e:
        logger.error(f"Error in get_embedding_model: {str(e)}")
        return None

def configure_embedding_model():
    logger.info("Entering function: configure_embedding_model")
    selected_model = select_embedding_model()
    return get_embedding_model(selected_model)

# ... (keep all other functions the same, but add the logging line at the start of each)
