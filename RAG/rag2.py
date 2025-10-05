import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import pipeline
from dotenv import load_dotenv

# --- Load environment ---
load_dotenv()

# --- Define persistent directory ---
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# --- Embedding model ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Load Chroma DB ---
print(f"Loading Chroma DB from: {persistent_directory}")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# --- Query ---
query = "What two extremes did Holmes alternate between while staying in Baker Street?"

# --- Retrieve documents ---
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.0},
)
relevant_docs = retriever.invoke(query)

# --- Display retrieved docs ---
print("\n--- Relevant Documents ---")
if not relevant_docs:
    print("No relevant documents found.")
else:
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content[:200]}...\n")

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
    model="google/flan-t5-base",  # ✅ Correct pipeline for T5
    max_new_tokens=60,
)

# --- Generate response ---
response = generator(final_prompt)
answer = response[0]["generated_text"].strip()

print("\n--- Model Answer ---")
print(answer)
