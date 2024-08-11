import streamlit as st
import utils

st.set_page_config(page_title="Text Similarity Comparison", page_icon="📄", layout="wide")
st.header('Compare Text Similarity with Multiple Embedding Models')
st.write('Upload multiple pieces of text and compare their similarity using different embedding models.')
st.write('[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/4_%F0%9F%93%84_chat_with_your_documents.py)')

class TextSimilarityComparison:
    def __init__(self):
        self.embedding_models = utils.configure_embedding_models()

    def main(self):
        st.sidebar.write("### Instructions")
        st.sidebar.write("1. Select one or more embedding models using the buttons in the sidebar.")
        st.sidebar.write("2. Enter 2 to 5 pieces of text in the main window.")
        st.sidebar.write("3. Click 'Compare Texts' to see similarity scores.")

        # Text input
        num_texts = st.number_input("Number of text pieces to compare", min_value=2, max_value=5, value=2)
        texts = []
        for i in range(num_texts):
            text = st.text_area(f"Text {i+1}", height=100, key=f"text_{i}")
            texts.append(text)

        if st.button("Compare Texts"):
            if all(texts) and self.embedding_models:
                with st.spinner('Processing texts...'):
                    results = utils.process_texts(texts, self.embedding_models)
                utils.display_results(results, texts)
            else:
                st.error("Please enter all texts and select at least one embedding model.")

if __name__ == "__main__":
    obj = TextSimilarityComparison()
    obj.main()
