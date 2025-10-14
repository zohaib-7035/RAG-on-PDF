import os
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI  

# ✅ Set your Gemini API key directly (no need for .env)
os.environ["GOOGLE_API_KEY"] = "AIzaSyCKP6b_abFXIh9Yo2xkplWY-iWBrnjeEy0"

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Define the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the existing vector store
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Define the user's question
query = "How can I learn more about LangChain?"

# Retrieve relevant documents
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
relevant_docs = retriever.invoke(query)

# Display relevant docs
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# Combine query and docs
combined_input = (
    f"Here are some documents that might help answer the question: {query}\n\n"
    + "Relevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide an answer based only on the provided documents. "
      "If the answer is not found in the documents, respond with 'I'm not sure'."
)

# Create the Gemini model (Flash = faster, Pro = smarter)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# Define messages
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

#Invoke Gemini
result = model.invoke(messages)

# Display output
print("\n--- Generated Response ---")
print(result.content)
