import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# ====== 1. Set your Groq API Key ======
load_dotenv()

# Get the API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ====== 2. Load PDF ======
pdf_path = "D:\\PRJCT\\ChatwithPDF\\Neha Biju.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# ====== 3. Split into chunks ======
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(pages)

# ====== 4. Generate embeddings (locally) ======
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ====== 5. Store vectors using FAISS ======
vector_store = FAISS.from_documents(documents, embeddings)

# ====== 6. Create retriever ======
retriever = vector_store.as_retriever()

# ====== 7. Setup Groq LLM (Mixtral model) ======
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-8b-8192"  # or use "llama3-8b-8192"
)

# ====== 8. Define QA Chain ======
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# ====== 9. Query the chatbot ======
while True:
    query = input("\nAsk your question (or type 'exit'): ")
    if query.lower() in ["exit", "quit"]:
        break

    result = qa_chain.invoke(query)
    answer = result['result']
    print(f"\n📘 Answer: {answer}")
