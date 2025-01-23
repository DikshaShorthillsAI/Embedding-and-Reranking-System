import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def extract_strings(data):
    """
    Recursively extract strings from a nested structure (list/dict).
    
    Args:
        data: The JSON data (list, dict, or other structure).
    
    Returns:
        list: A flat list of all strings found.
    """
    strings = []
    if isinstance(data, str):
        strings.append(data.strip())
    elif isinstance(data, list):
        for item in data:
            strings.extend(extract_strings(item))
    elif isinstance(data, dict):
        for value in data.values():
            strings.extend(extract_strings(value))
    return strings

def compute_cosine_similarity(text1, text2):
    """
    Compute the cosine similarity between two strings.
    
    Args:
        text1 (str): The first string.
        text2 (str): The second string.
    
    Returns:
        float: Cosine similarity score between 0 and 1.
    """
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(vectors[0], vectors[1])
    return similarity[0][0]

def compare_chunks(provided_file, generated_file, similarity_threshold=0.7):
    """
    Compare chunks in two JSON files using cosine similarity.
    
    Args:
        provided_file (str): Path to the provided JSON file.
        generated_file (str): Path to the generated JSON file.
        similarity_threshold (float): Minimum similarity to consider a match (default: 0.7).

    Returns:
        dict: A dictionary containing similarities, unmatched chunks, and statistics.
    """
    try:
        with open(provided_file, 'r') as file1:
            provided_data = json.load(file1)
        with open(generated_file, 'r') as file2:
            generated_data = json.load(file2)

        provided_chunks = extract_strings(provided_data)
        generated_chunks = extract_strings(generated_data)

        matched_chunks = []
        unmatched_chunks = []
        similarities = []

        for provided in provided_chunks:
            max_similarity = 0
            best_match = None
            for generated in generated_chunks:
                similarity = compute_cosine_similarity(provided, generated)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = generated

            if max_similarity >= similarity_threshold:
                matched_chunks.append({
                    "provided_chunk": provided,
                    "matched_chunk": best_match,
                    "similarity": max_similarity
                })
            else:
                unmatched_chunks.append({
                    "provided_chunk": provided,
                    "best_attempt": best_match,
                    "max_similarity": max_similarity
                })

            similarities.append(max_similarity)

        similarity_percentage = (len(matched_chunks) / len(provided_chunks)) * 100 if provided_chunks else 0

        return {
            "similarity_percentage": similarity_percentage,
            "total_chunks_provided": len(provided_chunks),
            "total_chunks_generated": len(generated_chunks),
            "matched_chunks_count": len(matched_chunks),
            "matched_chunks": matched_chunks,
            "unmatched_chunks": unmatched_chunks,
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        return {
            "similarity_percentage": 0,
            "total_chunks_provided": 0,
            "total_chunks_generated": 0,
            "matched_chunks_count": 0,
            "matched_chunks": [],
            "unmatched_chunks": []
        }

if __name__ == "__main__":
    provided_file = "/home/shtlp_0010/Desktop/Embedding_Reranking/assets/before_reranking.json"
    generated_file = "/home/shtlp_0010/Desktop/Embedding_Reranking/assets/provided_chunks.json"

    result = compare_chunks(provided_file, generated_file, similarity_threshold=0.7)

    print(f"Similarity: {result['similarity_percentage']:.2f}%")
    print(f"Total chunks in provided file: {result['total_chunks_provided']}")
    print(f"Total chunks in generated file: {result['total_chunks_generated']}")
    print(f"Matched chunks count: {result['matched_chunks_count']}")
    # Print detailed matches and unmatched chunks if needed
    # print(json.dumps(result['matched_chunks'], indent=4))
    # print(json.dumps(result['unmatched_chunks'], indent=4))
