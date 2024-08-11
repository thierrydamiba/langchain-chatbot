class CustomDocChatbot:

    def __init__(self):
        try:
            utils.sync_st_session()
            self.llm = utils.configure_llm()
            self.embedding_model = utils.configure_embedding_model()
            if self.embedding_model is None:
                st.error("Failed to initialize embedding model. Please check your configuration.")
                st.stop()
        except Exception as e:
            st.error(f"An error occurred during initialization: {str(e)}")
            st.stop()

    # ... (keep all other methods the same)

    @st.spinner('Analyzing documents..')
    def setup_qa_chain(self, uploaded_files):
        try:
            # Load documents
            docs = []
            for file in uploaded_files:
                file_path = self.save_file(file)
                loader = PyPDFLoader(file_path)
                docs.extend(loader.load())
            
            # Split documents and store in vector db
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(docs)
            vectordb = DocArrayInMemorySearch.from_documents(splits, self.embedding_model)

            # ... (rest of the method remains the same)
        except Exception as e:
            st.error(f"An error occurred while setting up the QA chain: {str(e)}")
            return None

    # ... (keep all other methods the same)

    @utils.enable_chat_history
    def main(self):
        try:
            # User Inputs
            uploaded_files = st.sidebar.file_uploader(label='Upload PDF files', type=['pdf'], accept_multiple_files=True)
            if not uploaded_files:
                st.info("Please upload PDF documents to continue!")
                return

            user_query = st.chat_input(placeholder="Ask me anything!")

            if uploaded_files and user_query:
                qa_chain = self.setup_qa_chain(uploaded_files)
                if qa_chain is None:
                    return

                utils.display_msg(user_query, 'user')

                with st.chat_message("assistant"):
                    st_cb = StreamHandler(st.empty())
                    result = qa_chain.invoke(
                        {"question":user_query},
                        {"callbacks": [st_cb]}
                    )
                    response = result["answer"]
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    utils.print_qa(CustomDocChatbot, user_query, response)

                    # to show references
                    for idx, doc in enumerate(result['source_documents'],1):
                        filename = os.path.basename(doc.metadata['source'])
                        page_num = doc.metadata['page']
                        ref_title = f":blue[Reference {idx}: *{filename} - page.{page_num}*]"
                        with st.popover(ref_title):
                            st.caption(doc.page_content)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    try:
        obj = CustomDocChatbot()
        obj.main()
    except Exception as e:
        st.error(f"An error occurred while running the application: {str(e)}")
