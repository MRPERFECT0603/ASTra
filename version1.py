import os
import numpy as np
import pandas as pd  # Added pandas for DataFrame support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Function to read .cpp files from a directory
def read_cpp_files(directory):
    cpp_files = []
    file_names = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".cpp"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    content = f.read()
                    cpp_files.append(content)
                    file_names.append(file)
                # Check if the file is empty
                if not content.strip():
                    print(f"Warning: {file_path} is empty.")
    return cpp_files, file_names


# Function to preprocess .cpp files
def preprocess_cpp_files(cpp_files):
    return cpp_files


# Function to compute TF-IDF vectors
def compute_tfidf_vectors(cpp_files):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cpp_files)
    return tfidf_matrix, vectorizer


# Function to compute similarity matrix for a list of .cpp files
def compute_similarity_matrix(tfidf_matrix):
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix


# Main function
def main():
    # Step 1: Read .cpp files
    cpp_directory = "/Users/vivek/Desktop/ASTra/content/"
    cpp_files, file_names = read_cpp_files(cpp_directory)
    print("Read the following files:")
    for name in file_names:
        print(f"- {name}")

    # Check if any files were read
    if not cpp_files:
        print("Error: No .cpp files found in the directory.")
        return  # Exit if no files found

    # Step 2: Preprocess .cpp files
    preprocessed_cpp_files = preprocess_cpp_files(cpp_files)
    print("\nPreprocessed contents of the files:")
    for i, file_content in enumerate(preprocessed_cpp_files):
        print(f"File: {file_names[i]}")
        print("------------------------")
        print(file_content)
        print()

    # Step 3: Compute TF-IDF vectors
    tfidf_matrix, vectorizer = compute_tfidf_vectors(preprocessed_cpp_files)
    print("\nTF-IDF matrix shape:", tfidf_matrix.shape)

    # Step 4: Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(tfidf_matrix)

    # Create a DataFrame for the similarity matrix
    df_similarity = pd.DataFrame(
        similarity_matrix, index=file_names, columns=file_names
    )

    # Print formatted similarity matrix
    print("\nSimilarity Matrix:")
    print("==================")
    print(df_similarity)

    # Print pairwise similarity scores
    print("\nPairwise similarity scores:")
    print("===========================")
    for i in range(len(file_names)):
        for j in range(i + 1, len(file_names)):
            print(
                f"Similarity between {file_names[i]} and {file_names[j]}: {similarity_matrix[i, j]:.8f}"
            )
        print()


if __name__ == "__main__":
    main()
