import os
import tempfile
import streamlit as st

from config import VECT_STORE_DIR, MODELS_DIR, EMB_MODEL_DIR, EMB_MODEL_NAME_BGE_M3, LLM_MODEL_NAME, template, RANDOM, MARKDOWN_SEPARATORS

from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_community.document_loaders import PyPDFLoader #, TextLoader, DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

embeddings = HuggingFaceEmbeddings(
        model_name=EMB_MODEL_NAME_BGE_M3,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        cache_folder=EMB_MODEL_DIR)

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # The maximum number of characters in a chunk: we selected this value arbitrarily
    chunk_overlap=100,  # The number of characters to overlap between chunks
    strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
    separators=MARKDOWN_SEPARATORS,
)

model = LlamaCpp(model_path=os.path.join(MODELS_DIR, LLM_MODEL_NAME),
                       temperature=0.0,
                       max_tokens=2048,
                       n_ctx=4096,
                       seed=RANDOM,
                       callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                       verbose=False)

st.subheader("LangChain RAG LLM Chatbot with FAISS VectorStore")
# st.markdown(f"""
#             <style>
#             .stApp {{background-image: url("https://images.unsplash.com/photo-1509537257950-20f875b03669?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1469&q=80");
#                      background-attachment: fixed;
#                      background-size: cover}}
#          </style>
#          """, unsafe_allow_html=True)


with st.sidebar:
    source_doc = st.file_uploader("Документ для RAG", type="pdf")

col1, col2 = st.columns([4,1])
query = col1.text_input("Query", label_visibility="collapsed")

# Session state initialization for documents and retrievers
if 'retriever' not in st.session_state or 'loaded_doc' not in st.session_state:
    st.session_state.retriever = None
    st.session_state.loaded_doc = None

submit = col2.button("Жми!")

if submit:
    # Validate inputs
    if not query:
        st.warning("Поле с вопросом должно быть заполнено.")
    elif not source_doc:
        st.warning("Загрузите документ.")
    else:
        with st.spinner("Ожидайте..."):
            if st.session_state.loaded_doc != source_doc:
                try:
                    # Save uploaded file temporarily to disk, load and split the file into pages, delete temp file
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(source_doc.read())
                    loader = PyPDFLoader(tmp_file.name)
                    loader = loader.load()
                    document = []
                    for page in loader:
                        document.append(page)
                    os.remove(tmp_file.name)
                    docs = parent_splitter.split_documents(document)
                    # Generate embeddings for the pages, and store in Chroma vector database
                    db = FAISS.from_documents(docs, embeddings)

                    # Configure Chroma as a retriever with top_k=5
                    st.session_state.retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 1})

                    # Store the uploaded file in session state to prevent reloading
                    st.session_state.loaded_doc = source_doc
                except Exception as e:
                    st.error(f"An error occurred: {e}")

            try:
                # Initialize the ChatGoogleGenerativeAI module, create and invoke the retrieval chain
                prompt = ChatPromptTemplate.from_template(template)

                chain = (
                            {"page_content": st.session_state.retriever | format_docs, "quest": RunnablePassthrough()}
                            | ChatPromptTemplate.from_template(template)
                            | model
                            | StrOutputParser()
                        )
                response = chain.invoke(query)

                st.success(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")


