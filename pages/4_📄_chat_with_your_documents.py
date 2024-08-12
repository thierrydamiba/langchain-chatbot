import streamlit as st
import utils

st.set_page_config(page_title="Text Similarity Comparison", page_icon="📄", layout="wide")
st.header('Compare Text Similarity with Multiple Embedding Models')
st.write('Upload a document or enter text manually to compare similarity using different embedding models.')
st.write('[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/4_%F0%9F%93%84_chat_with_your_documents.py)')

class TextSimilarityComparison:
    def __init__(self):
        self.embedding_models = utils.configure_embedding_models()

    def main(self):
        st.sidebar.write("### Instructions")
        st.sidebar.write("1. Select one or more embedding models from the dropdown in the sidebar.")
        st.sidebar.write("2. Choose to upload a file or enter text manually.")
        st.sidebar.write("3. If uploading a file, select the number of text chunks to compare.")
        st.sidebar.write("4. Click 'Compare Texts' to see similarity scores.")

        input_method = st.radio("Choose input method:", ("Upload File", "Enter Text Manually"))

        texts = []
        if input_method == "Upload File":
            uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf', 'docx', 'doc'])
            if uploaded_file is not None:
                try:
                    text = utils.extract_text_from_file(uploaded_file)
                    chunks = utils.chunk_text(text)
                    num_samples = st.number_input("Number of text chunks to compare", min_value=2, max_value=5, value=3)
                    texts = utils.sample_chunks(chunks, num_samples)
                    for i, chunk in enumerate(texts):
                        st.text_area(f"Text Chunk {i+1}", value=chunk, height=150, key=f"chunk_{i}")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        else:
            num_texts = st.number_input("Number of text pieces to compare", min_value=2, max_value=5, value=2)
            for i in range(num_texts):
                text = st.text_area(f"Text {i+1}", height=150, key=f"text_{i}")
                texts.append(text)

        if st.button("Compare Texts"):
            if all(texts) and self.embedding_models:
                with st.spinner('Processing texts...'):
                    results = utils.process_texts(texts, self.embedding_models)
                utils.display_results(results, texts)
            else:
                st.error("Please ensure all texts are entered and at least one embedding model is selected.")

if __name__ == "__main__":
    obj = TextSimilarityComparison()
    obj.main()
