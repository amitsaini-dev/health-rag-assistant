import os

# To read text files ppt docs etc 
from langchain_community.document_loaders import TextLoader,DirectoryLoader 

# For chunking
from langchain_text_splitters import CharacterTextSplitter

# Embedding model
from langchain_huggingface import HuggingFaceEmbeddings

# Chroma vector database we can host locally
from langchain_chroma import Chroma

# Loading environment variable
from dotenv import load_dotenv

load_dotenv()
# print("API Key:", os.getenv("OPENAI_API_KEY")[:15])

# function to load the files
def load_documents(docs_path="docs"):
    """Load all text files form docs Directory"""
    print(f"Looking for documents form {docs_path}...")

    # Checks if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all text file from the docs directory 
    loader=DirectoryLoader(
    path=docs_path,
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
    )
    
    documents=loader.load()
    
    if(len(documents)==0):
        raise FileNotFoundError(f"No .txt file found in {docs_path}. Please add your company documents.")
    
    for i, doc in enumerate(documents[:2]): # show first two documents
        print(f"\nDocument {i+1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} Characters")
        print(f"Content Preview:{doc.page_content[:100]}...")
        print(f"metadata:{doc.metadata}")
    
    return documents 


# function for chunking
def split_documents(documents,chunk_size=800,chunk_overlap=0):
    """Splitting Documents into smaller chunks with overlap"""
    print("Splitting Documents into chunk...")

    text_splitter=CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks=text_splitter.split_documents(documents)

    if chunks:

        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks


# Funciton for chunking 
def create_vector_store(chunks,presist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store """
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create ChromaDB Vector Store
    print("Creating Vector Store")
    
    vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=presist_directory,
    collection_metadata={"hnsw:space": "cosine"} #passes metadata to the ChromaDB collection when it is created.
    # HNSW stands for Hierarchical Navigable Small World. It is the indexing algorithm ChromaDB uses to efficiently search through vector embeddings.
    )

    print("finished creating vecotr store")

    print(f"Vector store create and saved to{presist_directory}")

    return vectorstore

def main():
    # 1. Loading the files
    documents=load_documents(docs_path="docs")
    # 2. Chunking the files
    chunks=split_documents(documents) 
    # 3. Emdedding and Storing in Vector DB
    vector=create_vector_store(chunks)

if __name__=="__main__":
    main()