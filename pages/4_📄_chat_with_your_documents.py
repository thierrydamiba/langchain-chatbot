import streamlit as st
import utils

st.set_page_config(page_title="Text Similarity Comparison", page_icon="📄", layout="wide")
st.header('Compare Text Similarity with Multiple Embedding Models')
st.write('Upload a document or enter text manually to compare similarity using different embedding models.')
st.write('[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/4_%F0%9F%93%84_chat_with_your_documents.py)')

class TextSimilarityComparison:
    def __init__(self):
        self.embedding_models = utils.configure_embedding_models()
        self.llm = utils.configure_llm()

    def main(self):
        st.sidebar.write("### Instructions")
        st.sidebar.write("1. Select one or more embedding models from the dropdown in the sidebar.")
        st.sidebar.write("2. Choose to upload a file or enter text manually.")
        st.sidebar.write("3. The app will analyze the text and determine appropriate sampling.")
        st.sidebar.write("4. Click 'Compare Texts' to see similarity scores and visualizations.")

        input_method = st.radio("Choose input method:", ("Upload File", "Enter Text Manually"))

        text = ""
        if input_method == "Upload File":
            uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf', 'docx', 'doc'])
            if uploaded_file is not None:
                try:
                    text = utils.extract_text_from_file(uploaded_file)
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        else:
            text = st.text_area("Enter your text here", height=300)

        if st.button("Compare Texts") and text and self.embedding_models and self.llm:
            with st.spinner('Analyzing and processing text...'):
                texts, text_type = utils.analyze_and_sample_text(text, self.llm)
                st.write(f"Detected text type: {text_type}")
                st.write(f"Number of samples: {len(texts)}")
                for i, chunk in enumerate(texts):
                    st.text_area(f"Text Chunk {i+1}", value=chunk, height=150, key=f"chunk_{i}")
                
                results = utils.process_texts(texts, self.embedding_models)
                utils.display_results(results, texts)
        elif not self.llm:
            st.error("Please enter your OpenAI API Key in the sidebar to enable text analysis.")
        elif not text:
            st.error("Please enter some text or upload a file.")
        elif not self.embedding_models:
            st.error("Please select at least one embedding model.")

if __name__ == "__main__":
    obj = TextSimilarityComparison()
    obj.main()
