import os
import json
import re
from dotenv import load_dotenv
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sentence_transformers import CrossEncoder
from docling.document_converter import DocumentConverter  # type: ignore
from langchain.text_splitter import MarkdownHeaderTextSplitter

load_dotenv()

def load_embedding_model():
    """Initialize and return the embedding model."""
    return SentenceTransformer('thenlper/gte-base')

def get_mongo_collection():
    """Connect to MongoDB and return the embeddings collection."""
    mongo_url = os.getenv("MONGO_URL")
    if not mongo_url:
        raise ValueError("MongoDB URL is not set in the environment variables.")
    client = MongoClient(mongo_url, serverSelectionTimeoutMS=30000)
    db = client["embedding_database"]
    return db["embeddings"]

def export_to_markdown(pdf_path):
    """Convert the PDF content to Markdown format."""
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    return result.document.export_to_markdown()

def split_chunks(pdf_path):
    """Convert PDF to markdown and split it into chunks."""
    markdown_text = export_to_markdown(pdf_path=pdf_path)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    docs = markdown_splitter.split_text(markdown_text)
    chunks = []

    for i, doc in enumerate(docs):
        try:
            chunk = doc.page_content.strip()
            if len(chunk) >= 100:
                chunks.append(chunk)
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")

    return chunks

def preprocess_text(text):
    """Preprocess text by stripping and cleaning."""
    return text.strip()

def process_pdfs_for_chunks(pdf_paths):
    """Process a list of PDF paths to generate chunks."""
    all_chunks = []
    for path in tqdm(pdf_paths, desc="Processing PDFs"):
        try:
            chunks = split_chunks(path)
            for chunk in chunks:
                chunk = preprocess_text(chunk)
                if len(chunk) >= 100:
                    all_chunks.append({"pdf_path": path, "text": chunk})
        except FileNotFoundError:
            print(f"PDF file not found: {path}")
        except Exception as e:
            print(f"Error processing PDF {path}: {e}")
    return all_chunks

def generate_and_store_embeddings(chunks, collection):
    """Generate embeddings for text chunks and store them in MongoDB."""
    model = load_embedding_model()
    embeddings = model.encode([chunk["text"] for chunk in chunks], convert_to_tensor=False).tolist()

    docs = [
        {
            "pdf_path": chunk["pdf_path"],
            "chunk": chunk["text"],
            "embedding": embedding
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]

    try:
        collection.insert_many(docs)
        print(f"Inserted {len(docs)} embeddings into MongoDB.")
    except Exception as e:
        raise Exception(f"Failed to insert embeddings into MongoDB: {e}")

def extract_queries_from_text_file(query_text_file_path):
    """Extract queries from a text file containing the list of queries."""
    try:
        with open(query_text_file_path, 'r') as file:
            queries = file.readlines()
            return [preprocess_text(query) for query in queries if query.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Query text file not found at path: {query_text_file_path}")

def rerank_chunks(chunks, embeddings, query, top_n=5):
    """
    Rerank text chunks based on a query using embeddings for initial filtering 
    and Cross-Encoder for reranking.
    """
    model = load_embedding_model()
    query_embedding = model.encode(preprocess_text(query), convert_to_tensor=False)
    query_embedding /= np.linalg.norm(query_embedding)

    embeddings = np.array(embeddings)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    scores = np.dot(embeddings, query_embedding)

    top_20_indices = np.argsort(scores)[::-1][:20]
    top_20_chunks = [chunks[i] for i in top_20_indices]

    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    inputs = [(preprocess_text(query), preprocess_text(chunk["text"])) for chunk in top_20_chunks]

    rerank_scores = cross_encoder.predict(inputs)

    reranked_chunks = [{"text": chunk["text"], "score": score} for chunk, score in zip(top_20_chunks, rerank_scores)]
    reranked_chunks = sorted(reranked_chunks, key=lambda x: x["score"], reverse=True)

    return reranked_chunks[:top_n]

def save_to_json(before_file_path, after_file_path, queries, top_chunks_list, reranked_chunks_list):
    """Save chunks before and after reranking to separate JSON files."""
    try:
        if not (len(queries) == len(top_chunks_list) == len(reranked_chunks_list)):
            raise ValueError(
                "Mismatch in lengths of queries, top_chunks_list, and reranked_chunks_list. "
                f"Lengths: queries={len(queries)}, top_chunks_list={len(top_chunks_list)}, "
                f"reranked_chunks_list={len(reranked_chunks_list)}"
            )
        before_data = [
            {
                "query": queries[i],
                "top_chunks_before_reranking": [
                    {
                        "chunk": chunk["text"],
                        "score": 0 
                    }
                    if isinstance(chunk, dict) else {"chunk": chunk, "score": 0} 
                    for chunk in top_chunks_list[i]
                ]
            }
            for i in range(len(queries))
        ]
        after_data = [
            {
                "query": queries[i],
                "top_chunks_after_reranking": [
                    {
                        "chunk": chunk["text"],
                        "score": float(chunk["score"])
                    }
                    if isinstance(chunk, dict) else {"chunk": chunk, "score": 0}  
                    for chunk in reranked_chunks_list[i]
                ]
            }
            for i in range(len(queries))
        ]
        with open(before_file_path, 'w') as before_file:
            json.dump(before_data, before_file, indent=4)

        with open(after_file_path, 'w') as after_file:
            json.dump(after_data, after_file, indent=4)

    except Exception as e:
        raise Exception(f"Failed to save to files {before_file_path} and {after_file_path}: {e}")


if __name__ == "__main__":
    pdf_paths = ["/home/shtlp_0010/Desktop/Embedding_Reranking/assets/FinancialManagement.pdf"]
    query_text_file_path = "/home/shtlp_0010/Desktop/Embedding_Reranking/assets/queries.txt"
    before_results_path = "/home/shtlp_0010/Desktop/Embedding_Reranking/assets/before_reranking.json"
    after_results_path = "/home/shtlp_0010/Desktop/Embedding_Reranking/assets/after_reranking.json"

    chunks = process_pdfs_for_chunks(pdf_paths)
    if not chunks:
        raise ValueError("No text extracted from the PDF files.")

    collection = get_mongo_collection()

    if collection.count_documents({}) == 0:
        generate_and_store_embeddings(chunks, collection)

    records = list(collection.find())
    embeddings = [np.array(record["embedding"]) for record in records]
    chunks = [{"text": record["chunk"], "pdf_path": record["pdf_path"]} for record in records]
    queries = extract_queries_from_text_file(query_text_file_path)

    top_chunks_list = []
    reranked_chunks_list = []

    for query in queries:
        reranked_chunks = rerank_chunks(chunks, embeddings, query, top_n=5)
        top_chunks_list.append([{"text": chunk["text"], "score": 0} for chunk in reranked_chunks])
        reranked_chunks_list.append(reranked_chunks)

    save_to_json(before_results_path, after_results_path, queries, top_chunks_list, reranked_chunks_list)

    print(f"Results saved to {before_results_path} and {after_results_path}.")
