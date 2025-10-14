import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# Define the embedding model
embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ) # Update to a valid embedding model if needed
# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# Define the user's question
query = "What two extremes did Holmes alternate between while staying in Baker Street?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.1},
)
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    print(f"Source: {doc.metadata['source']}\n")



# --- Prepare short, safe context (avoid token overflow) ---
context = "\n\n".join([doc.page_content[:400] for doc in relevant_docs])

# --- Prompt setup ---
system_prompt = (
    "You are a precise and concise assistant. "
    "Answer strictly based on the context provided. "
    "If the context does not contain enough information, say: 'Not enough information in context.' "
    "Give your answer in one short sentence only."
)

user_prompt = f"Question: {query}"

final_prompt = f"{system_prompt}\n\nContext:\n{context}\n\n{user_prompt}"

# --- Load a Hugging Face model for generation ---
print("\n--- Generating Answer with Hugging Face Model ---")
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",  
    max_new_tokens=60,
)

# --- Generate response ---
response = generator(final_prompt)
answer = response[0]["generated_text"].strip()

print("\n--- Model Answer ---")
print(answer)
