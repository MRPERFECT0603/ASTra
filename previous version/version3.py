import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from clang import cindex  
cindex.Config.set_library_file("/opt/homebrew/opt/llvm/lib/libclang.dylib")


def generate_ast(file_path):
    # Initialize libclang index
    index = cindex.Index.create()
    # Parse the C++ file and create a TranslationUnit
    translation_unit = index.parse(file_path)
    # Get the root cursor (top-level node) of the AST
    root_cursor = translation_unit.cursor
    return root_cursor


# Function to read .cpp files and generate ASTs
def read_cpp_files_with_ast(directory):
    cpp_files = []
    file_names = []
    asts = []  # Store ASTs here
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".cpp"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    content = f.read()
                    cpp_files.append(content)
                    file_names.append(file)
                # Generate AST using Script 1's function
                root_cursor = generate_ast(file_path)
                asts.append(root_cursor)

                # Check if the file is empty
                if not content.strip():
                    print(f"Warning: {file_path} is empty.")
    return cpp_files, file_names, asts


# Function to preprocess .cpp files (can be extended as needed)
def preprocess_cpp_files(cpp_files):
    return cpp_files


# Function to compute TF-IDF vectors (including AST information)
def compute_tfidf_vectors(cpp_files, asts):
    # Combine cpp_files content with AST information
    combined_content = [content + str(ast) for content, ast in zip(cpp_files, asts)]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_content)
    return tfidf_matrix, vectorizer


# Function to compute similarity matrix for a list of .cpp files
def compute_similarity_matrix(tfidf_matrix):
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix


# Main function
def main():
    # Step 1: Read .cpp files and generate ASTs
    cpp_directory = "/Users/vivek/Desktop/ASTra/content/"
    cpp_files, file_names, asts = read_cpp_files_with_ast(cpp_directory)
    print("Read the following files:")
    for name in file_names:
        print(f"- {name}")

    # Check if any files were read
    if not cpp_files:
        print("Error: No .cpp files found in the directory.")
        return  # Exit if no files found

    # Step 2: Preprocess .cpp files (if needed)
    preprocessed_cpp_files = preprocess_cpp_files(cpp_files)
    print("\nPreprocessed contents of the files:")
    for i, file_content in enumerate(preprocessed_cpp_files):
        print(f"File: {file_names[i]}")
        print("------------------------")
        print(file_content)
        print()

    # Step 3: Compute TF-IDF vectors (including AST information)
    tfidf_matrix, vectorizer = compute_tfidf_vectors(preprocessed_cpp_files, asts)
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
            similarity_score = similarity_matrix[i, j]
            print(
                f"Similarity between {file_names[i]} and {file_names[j]}: {similarity_score:.8f}"
            )

    # Print AST information for each file
    print("\nAST Information:")
    print("================")
    for i, (file_name, ast) in enumerate(zip(file_names, asts)):
        print(f"AST for {file_name}:")
        print("--------------------")
        print(ast)

    # Additional processing or analysis can be performed here


if __name__ == "__main__":
    main()
