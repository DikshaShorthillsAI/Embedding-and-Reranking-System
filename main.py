import os
from dotenv import load_dotenv
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import PyPDF2
from urllib.parse import quote_plus

# Load environment variables
load_dotenv()

# Load the embedding model
def load_embedding_model():
    """Initialize and return the embedding model."""
    return SentenceTransformer('sentence-transformers/gtr-t5-base')

# MongoDB connection
def get_mongo_collection():
    """Connect to MongoDB and return the embeddings collection."""
    mongo_url = os.getenv("MONGO_URL")
    if not mongo_url:
        raise ValueError("MongoDB URL is not set in the environment variables.")
    client = MongoClient(mongo_url,  serverSelectionTimeoutMS=30000)
    db = client["embedding_database"]
    return db["embeddings"]

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    extracted_text = []
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text.append(text.strip())
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")
    return extracted_text

# Generate and store embeddings in MongoDB
def generate_and_store_embeddings(chunks, collection):
    """Generate embeddings for text chunks and store them in MongoDB."""
    model = load_embedding_model()
    embeddings = model.encode(chunks, convert_to_tensor=True).tolist()
    
    docs = [{"chunk": chunk, "embedding": embedding} for chunk, embedding in zip(chunks, embeddings)]
    try:
        collection.insert_many(docs)
        print(f"Inserted {len(docs)} embeddings into MongoDB.")
    except Exception as e:
        raise Exception(f"Failed to insert embeddings into MongoDB: {e}")

# Get top N chunks before reranking
def get_top_chunks_before_reranking(chunks, top_n):
    """Return the top N chunks based on their original order."""
    return chunks[:top_n]

# Rerank chunks based on query
def rerank_chunks(chunks, embeddings, query, top_n):
    """Rerank text chunks based on a query using cosine similarity."""
    model = load_embedding_model()
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Normalize embeddings for cosine similarity calculation
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Calculate cosine similarities
    scores = np.dot(embeddings, query_embedding)

    # Sort and retrieve top N chunks
    ranked_indices = np.argsort(scores)[::-1][:top_n]
    ranked_chunks = [(chunks[i], scores[i]) for i in ranked_indices]

    return ranked_chunks

# Save text chunks with ranking to a file
def save_to_file(file_path, chunks_with_ranking):
    """Save ranked text chunks to a file."""
    try:
        with open(file_path, 'w') as file:
            for rank, (chunk, score) in enumerate(chunks_with_ranking, start=1):
                file.write(f"Rank {rank}:\n")
                file.write(f"{chunk}\n")
                file.write(f"Score: {score:.4f}\n\n")
    except Exception as e:
        raise Exception(f"Failed to save to file {file_path}: {e}")

# Main script
if __name__ == "__main__":
    pdf_path = "/home/shtlp_0010/Desktop/Embedding_Reranking/assets/genai.pdf"

    # Step 1: Extract text chunks from PDF
    chunks = extract_text_from_pdf(pdf_path)
    if not chunks:
        raise ValueError("No text extracted from the PDF.")

    # Step 2: Connect to MongoDB
    collection = get_mongo_collection()

    # Step 3: Generate and store embeddings
    generate_and_store_embeddings(chunks, collection)

    # Step 4: Retrieve embeddings from MongoDB
    records = list(collection.find())
    embeddings = [np.array(record["embedding"]) for record in records]

    # Ensure alignment between chunks and records
    chunks = chunks[:len(records)]

    # Step 5: Save top 5 chunks before reranking
    top_5_chunks = get_top_chunks_before_reranking(chunks, top_n=5)
    before_reranking_path = "/home/shtlp_0010/Desktop/Embedding_Reranking/assets/before_reranking.txt"
    save_to_file(before_reranking_path, [(chunk, 0) for chunk in top_5_chunks])
    print(f"Top 5 chunks before reranking saved to {before_reranking_path}.")

    # Step 6: Rerank chunks based on a query
    query = "What factors have contributed to the rapid advancement of Generative AI tools to human-level performance?"
    ranked_chunks = rerank_chunks(chunks, np.array(embeddings), query, top_n=5)

    # Save top 5 chunks after reranking
    after_reranking_path = "/home/shtlp_0010/Desktop/Embedding_Reranking/assets/after_reranking.txt"
    save_to_file(after_reranking_path, ranked_chunks)
    print(f"Top 5 chunks after reranking saved to {after_reranking_path}.")

    # Print results
    print("\nTop 5 chunks before reranking:")
    for i, chunk in enumerate(top_5_chunks, start=1):
        print(f"Rank {i}: {chunk}")

    print("\nTop 5 chunks after reranking:")
    for i, (chunk, score) in enumerate(ranked_chunks, start=1):
        print(f"Rank {i}: {chunk} (Score: {score:.4f})")
