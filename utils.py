import os
import openai
import streamlit as st
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings

def enable_chat_history(func):
    if os.environ.get("OPENAI_API_KEY"):
        current_page = func.__qualname__
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = current_page
        if st.session_state["current_page"] != current_page:
            try:
                st.cache_resource.clear()
                del st.session_state["current_page"]
                del st.session_state["messages"]
            except:
                pass
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])
    def execute(*args, **kwargs):
        func(*args, **kwargs)
    return execute

def display_msg(msg, author):
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)

def configure_llm():
    available_llms = ["gpt-4o-mini","llama3.1:8b","use your openai api key"]
    llm_opt = st.sidebar.radio(
        label="LLM",
        options=available_llms,
        key="SELECTED_LLM"
    )
    if llm_opt == "llama3.1:8b":
        llm = ChatOllama(model="llama3.1", base_url=st.secrets["OLLAMA_ENDPOINT"])
    elif llm_opt == "gpt-4o-mini":
        llm = ChatOpenAI(model_name=llm_opt, temperature=0, streaming=True, api_key=st.secrets["OPENAI_API_KEY"])
    else:
        model, openai_api_key = choose_custom_openai_key()
        llm = ChatOpenAI(model_name=model, temperature=0, streaming=True, api_key=openai_api_key)
    return llm

def choose_custom_openai_key():
    openai_api_key = st.sidebar.text_input(
        label="OpenAI API Key",
        type="password",
        placeholder="sk-...",
        key="SELECTED_OPENAI_API_KEY"
    )
    if not openai_api_key:
        st.error("Please add your OpenAI API key to continue.")
        st.info("Obtain your key from this link: https://platform.openai.com/account/api-keys")
        st.stop()
    model = "gpt-4o-mini"
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        available_models = [{"id": i.id, "created":datetime.fromtimestamp(i.created)} for i in client.models.list() if str(i.id).startswith("gpt")]
        available_models = sorted(available_models, key=lambda x: x["created"])
        available_models = [i["id"] for i in available_models]
        model = st.sidebar.selectbox(
            label="Model",
            options=available_models,
            key="SELECTED_OPENAI_MODEL"
        )
    except openai.AuthenticationError as e:
        st.error(e.__cause__)
        st.stop()
    except Exception as e:
        st.error("Something went wrong. Please try again later.")
        st.stop()
    return model, openai_api_key

def configure_embedding_model():
    available_embedding_models = ["BAAI/bge-small-en-v1.5", "BAAI/bge-small-zh-v1.5"]
    selected_model = st.sidebar.selectbox(
        label="Embedding Model",
        options=available_embedding_models,
        key="SELECTED_EMBEDDING_MODEL"
    )
    embedding_model = FastEmbedEmbeddings(model_name=selected_model)
    return embedding_model

def print_qa(cls, question, answer):
    print(f"\nUsecase: {cls.__name__}\nQuestion: {question}\nAnswer: {answer}\n{'------'*10}")

def sync_st_session():
    for k, v in st.session_state.items():
        st.session_state[k] = v
