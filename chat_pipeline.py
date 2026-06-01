from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

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

model = ChatOllama(model="llama3.2", temperature=0)

retriver=db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k":5,
        "score_threshold":0.3 #only return chunks with cosine similarity >=0.3
    }
)


#Store our conversation as message
chat_history=[]

def ask_question(user_question):
    print(f"\n You asked: {user_question}")

    #step 1: Rewrite question using chat history
    if chat_history:
        #ask ai to make question standalone
        messages=[
           SystemMessage(content="You are a helpful health assistant. Answer using ONLY the documents provided. Be concise and accurate.")
        ]+chat_history+[
            HumanMessage(content=f"New Question: {user_question}")
        ]

        result=model.invoke(messages)
        search_question=result.content.strip()
        print(f"Searching for :{search_question}")
    else:
        search_question=user_question
    
    #step 2:find the relevent docs
    relevant_docs=retriver.invoke(search_question)
    if not relevant_docs:
        print("No relevant documents found.")
        return
        
    # Step 3 build prompt
    # Combine the query and the relevant document contents
    combined_input = f"""Answer the question using ONLY the documents below. 
    The answer is somewhere in these documents — read carefully.
    Question: {search_question}
    Documents:
    {chr(10).join([f"Document {i+1}: {doc.page_content}" for i, doc in enumerate(relevant_docs)])}
    Answer directly and concisely. Do not say you cannot find the answer if it exists in the documents above.
    """
    # step 4:Get answer
    messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
    ]

    result = model.invoke(messages)
    answer = result.content

    print(f"Question Searched: {search_question}")
    print(f"\n--- Answer ---\n{answer}")
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

#simple chat loop
def start_chat():
    print("Ask me question or Type quit to exit.")
    while True:
        question=input("\n Your Question: ")

        if question.lower()=="quit":
            print("Goodbye")
            break
        
        ask_question(question)

if __name__ == "__main__":
    start_chat()