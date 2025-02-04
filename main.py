import os
import time
import streamlit as st
import nltk
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI  


nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")


load_dotenv()


api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ OpenAI API key is missing! Set it as an environment variable.")
    st.stop()


st.title("📰 InsightAI 🔍 – AI-driven insights from news sources 📈")
st.sidebar.title("🔗 Enter News Article URLs")


urls = [st.sidebar.text_input(f"URL {i+1}").strip() for i in range(3)]
urls = [url for url in urls if url]  # Remove empty inputs

process_url_clicked = st.sidebar.button("🚀 Process URLs")
faiss_index_path = "faiss_index"
main_placeholder = st.empty()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9, max_tokens=500)

def process_urls(urls):
    """Processes the input URLs, extracts content, and creates embeddings."""
    if not urls:
        st.error("⚠️ Please enter at least one valid URL.")
        return
    
    try:
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("📡 Loading data from URLs...")
        data = loader.load()
        if not data:
            st.error("⚠️ No data retrieved! Check the URLs and try again.")
            return
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return
    

    main_placeholder.text("✂️ Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
    docs = text_splitter.split_documents(data)
    if not docs:
        st.error("⚠️ No documents created after splitting! Check input data.")
        return
    

    try:
        embeddings = OpenAIEmbeddings()
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("📊 Building FAISS embedding vector...✅")
        time.sleep(2)
        vectorstore_openai.save_local(faiss_index_path)
    except Exception as e:
        st.error(f"❌ Error creating embeddings: {e}")
        return


if process_url_clicked:
    process_urls(urls)


query = main_placeholder.text_input("Ask a Question ❓")
if query:
    if os.path.exists(faiss_index_path):
        try:
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            
            st.header("✅ Answer")
            st.write(result.get("answer", "⚠️ No answer found."))
            
            sources = result.get("sources", "").strip()
            if sources:
                st.subheader("🔗 Sources:")
                for source in sources.split("\n"):
                    st.write(f"🔹 {source}")
        except Exception as e:
            st.error(f"❌ Error retrieving answer: {e}")
    else:
        st.error("⚠️ FAISS index not found! Process URLs first.")
