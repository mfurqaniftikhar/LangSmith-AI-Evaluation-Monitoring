# pip install -U langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv langsmith langchain-groq unstructured

import os
from dotenv import load_dotenv

from langsmith import traceable
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

PDF_PATH = r"D:\langsmith\langsmith-masterclass\islr.pdf"  # absolute path

# ---------- traced setup steps ----------
@traceable(name="load_pdf")
def load_pdf(path: str):
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
    except Exception as e:
        print(f"[!] PyPDFLoader failed: {e}\nTrying UnstructuredPDFLoader instead...")
        loader = UnstructuredPDFLoader(path)
        docs = loader.load()
    return docs

@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

@traceable(name="build_vectorstore")
def build_vectorstore(splits):
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 
    vs = FAISS.from_documents(splits, emb)
    return vs

@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path: str):
    print("[1] Loading PDF...")
    docs = load_pdf(pdf_path)
    print(f"   ✅ Loaded {len(docs)} pages/chunks")

    print("[2] Splitting text...")
    splits = split_documents(docs)
    print(f"   ✅ Created {len(splits)} chunks")

    print("[3] Building FAISS vectorstore...")
    vs = build_vectorstore(splits)
    print("   ✅ Vectorstore ready")
    return vs

# ---------- pipeline ----------
llm = ChatGroq(model='llama-3.1-8b-instant', api_key=os.getenv("GROQ_API_KEY"))

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer only from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# Build the index
vectorstore = setup_pipeline(PDF_PATH)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})

chain = parallel | prompt | llm | StrOutputParser()

# ---------- run a query ----------
print("\n✅ PDF RAG system ready. Ask a question about your PDF content (Ctrl+C to exit).")

while True:
    q = input("\nQ: ").strip()
    if not q:
        break
    config = {"run_name": "pdf_rag_query"}
    ans = chain.invoke(q, config=config)
    print("\nA:", ans)
