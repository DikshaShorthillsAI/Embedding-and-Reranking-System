import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import PyPDF2


def load_embedding_model():
    return SentenceTransformer('sentence-transformers/gtr-t5-base')

def get_mongo_collection():
    client = MongoClient("mongodb://localhost:27017/") 
    db = client["embedding_database"]
    return db["embeddings"]

def extract_text_from_pdf(pdf_path):
    extracted_text = []
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            extracted_text.append(page.extract_text())
    return extracted_text

def generate_and_store_embeddings(chunks, collection):
    model = load_embedding_model()
    embeddings = model.encode(chunks, convert_to_tensor=True).tolist()
    
    docs = [
        {"chunk": chunk, "embedding": embedding}
        for chunk, embedding in zip(chunks, embeddings)
    ]

    collection.insert_many(docs)
    print(f"Inserted {len(docs)} embeddings into MongoDB.")

def get_top_chunks_before_reranking(chunks, top_n):
    return chunks[:top_n]

def rerank_chunks(chunks, embeddings, query, top_n):
    model = load_embedding_model()
    query_embedding = model.encode(query, convert_to_tensor=True)

    scores = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    top_n = min(len(chunks), top_n) 
    ranked_indices = np.argsort(scores)[::-1][:top_n]
    ranked_chunks = [(chunks[i], scores[i]) for i in ranked_indices]

    return ranked_chunks

def save_to_file(file_path, chunks_with_ranking):
    with open(file_path, 'w') as file:
        for rank, (chunk, score) in enumerate(chunks_with_ranking, start=1):
            file.write(f"Rank {rank}:\n")
            file.write(f"{chunk}\n")
            file.write(f"Score: {score:.4f}\n\n")

if __name__ == "__main__":
    
    pdf_path = "/home/shtlp_0010/Desktop/Embedding_Reranking/assets/genai.pdf" 

    chunks = extract_text_from_pdf(pdf_path)

    collection = get_mongo_collection()

    generate_and_store_embeddings(chunks, collection)

    records = list(collection.find())
   
    embeddings = []
    for record in records:
        embedding = np.array(record["embedding"])
        print(f"Embedding shape: {embedding.shape}") 
        embeddings.append(embedding)

    max_len = max(len(embed) for embed in embeddings)
    embeddings = [np.pad(embed, (0, max_len - len(embed)), mode='constant') if len(embed) < max_len else embed[:max_len] for embed in embeddings]

    embeddings = np.array(embeddings)

    chunks = chunks[:len(records)]

    top_5_chunks = get_top_chunks_before_reranking(chunks, top_n=5)

    before_reranking_path = "/home/shtlp_0010/Desktop/Embedding_Reranking/assets/before_reranking.txt"
    save_to_file(before_reranking_path, [(chunk, 0) for chunk in top_5_chunks]) 
    print(f"Top 5 chunks before reranking saved to {before_reranking_path}.")

    top_20_chunks = chunks[:20]

    query = "What factors have contributed to the rapid advancement of Generative AI tools to human-level performance?"

    ranked_chunks = rerank_chunks(top_20_chunks, embeddings[:len(top_20_chunks)], query, top_n=5)

    after_reranking_path = "/home/shtlp_0010/Desktop/Embedding_Reranking/assets/after_reranking.txt"
    save_to_file(after_reranking_path, ranked_chunks)
    print(f"Top 5 chunks after reranking saved to {after_reranking_path}.")

    print("\nTop 5 chunks before reranking:")
    for i, chunk in enumerate(top_5_chunks, start=1):
        print(f"Rank {i}: {chunk}")

    print("\nTop 5 chunks after reranking:")
    for i, (chunk, score) in enumerate(ranked_chunks[:5], start=1):
        print(f"Rank {i}: {chunk} (Score: {score:.4f})")
