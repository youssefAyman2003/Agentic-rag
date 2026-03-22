from pypdf import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
 
 
# ── Embedding model (runs locally) ──────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
 
 
def load_pdfs(uploaded_files) -> list[Document]:
    """
    Extract text from a list of Streamlit UploadedFile objects.
    Returns a list of LangChain Documents (one per page).
    """
    documents = []
    for file in uploaded_files:
        reader = PdfReader(file)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue
            documents.append(Document(
                page_content=text,
                metadata={
                    "page": page_num + 1,
                    "source": file.name,
                }
            ))
    return documents
 
 
def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into smaller overlapping chunks for better retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=400,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks
 
 
def build_vectorstore(chunks: list[Document]) -> Chroma:
    """
    Create a Chroma vectorstore from document chunks using HuggingFace embeddings.
    Uses in-memory Chroma (no persistence needed for Streamlit sessions).
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
 
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="self_rag_docs",
    )
    return vectorstore
 
 
def get_retriever(vectorstore: Chroma):
    """
    Return a retriever from the vectorstore.
    Fetches top-4 most relevant chunks per query.
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
 
 
def ingest_pdfs(uploaded_files):
    """
    Full pipeline: PDFs → Documents → Chunks → Vectorstore → Retriever.
    This is the single entry point called from app.py.
 
    Returns:
        retriever: LangChain retriever ready for use in the RAG graph
        num_chunks: int — number of chunks created (for UI display)
    """
    documents = load_pdfs(uploaded_files)
    chunks = chunk_documents(documents)
    vectorstore = build_vectorstore(chunks)
    retriever = get_retriever(vectorstore)
    return retriever, len(chunks)