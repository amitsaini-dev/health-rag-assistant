from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
# from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,SystemMessage
load_dotenv()

persistent_directory="db/chroma_db"

# Load embeddings and vector store
embedding_model=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db=Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

#seach for relevant docs
query="How does utilitarian thinking influence triage decisions?"

# Synthetic Questions: 
# What is an electronic health record (EHR)?
# What types of data are included in an EHR?
# How do EHR systems help in reducing medical errors?
# How do EHR systems both improve and increase risks in healthcare?
# How does continuity of care by PCPs affect patient outcomes?
# Why are PCPs often the first point of contact for patients?
# What is medical triage?
# What is the origin of the term “triage”?
# Who introduced the modern concept of battlefield triage?
# What are the main categories used in triage systems?
# What does the ABCDE assessment stand for in triage?
# What is the difference between simple triage and advanced triage?
# What is reverse triage and when is it used?
# What are the differences between START and JumpSTART triage systems?
# What steps are involved in the ABCDE assessment?
# How did the Korean War influence modern triage systems?
# How does utilitarian thinking influence triage decisions?

# retriver=db.as_retriever(search_kwargs={"k":3})

retriver=db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k":5,
        "score_threshold":0.5 #only return chunks with cosine similarity >=0.3
    }
)


relevant_docs=retriver.invoke(query)

print(f"User Query:{query}")

# Display Results
print("Content")
for i, docs in enumerate(relevant_docs,1):
    print(f"Document {i}:\n{docs.page_content}\n")



# Combine the query and the relevant document contents
combined_input = f"""Answer the question using ONLY the documents below. 
The answer is somewhere in these documents — read carefully.

Question: {query}

Documents:
{chr(10).join([f"Document {i+1}: {doc.page_content}" for i, doc in enumerate(relevant_docs)])}

Answer directly and concisely. Do not say you cannot find the answer if it exists in the documents above.
"""

# Create a ChatOpenAI model

# model = ChatOllama(
#     model="llama3.2",
#     temperature=0
# )
model = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)

