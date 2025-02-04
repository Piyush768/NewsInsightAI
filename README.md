
# **InsightAI – AI-powered Insights from Web Article**

**InsightAI** is a web application that extracts insights and answers questions based on articles. The app allows users to input article URLs, processes the content, and provides AI-driven responses powered by **OpenAI's GPT-3.5**.

### **How it works:**
1. **URL Input**: Users enter URLs of news articles.
2. **Content Extraction**: The app extracts content from the URLs using **LangChain**.
3. **Text Processing**: The content is split into smaller chunks for better handling.
4. **Embedding & Indexing**: **FAISS** is used to index the content and create embeddings for fast similarity search.
5. **Question Answering**: Users can ask questions about the article, and the app retrieves answers from the indexed content using **OpenAI’s GPT-3.5**.

### **Technologies Used:**
- **Streamlit**: For creating the interactive user interface.
- **LangChain**: For document loading, text splitting, and embeddings.
- **OpenAI API** (GPT-3.5): For natural language processing and question answering.
- **FAISS**: For fast document retrieval and indexing.

