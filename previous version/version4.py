import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
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


# Function to plot dendrogram
def plot_dendrogram(similarity_matrix, file_names):
    linked = linkage(similarity_matrix, 'ward')
    plt.figure(figsize=(10, 7))
    dendrogram(linked, orientation='top', labels=file_names, distance_sort='descending', show_leaf_counts=True)
    plt.title('Dendrogram')
    plt.xlabel('Files')
    plt.ylabel('Distance')
    plt.show()


# Function to plot t-SNE
def plot_tsne(tfidf_matrix, file_names):
    # Ensure perplexity is less than the number of samples
    n_samples = tfidf_matrix.shape[0]
    perplexity = min(5, n_samples - 1)
    
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(tfidf_matrix.toarray())
    plt.figure(figsize=(10, 7))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
    for i, name in enumerate(file_names):
        plt.annotate(name, (tsne_results[i, 0], tsne_results[i, 1]))
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()


# Function to plot PCA
def plot_pca(tfidf_matrix, file_names):
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(tfidf_matrix.toarray())
    plt.figure(figsize=(10, 7))
    plt.scatter(pca_results[:, 0], pca_results[:, 1])
    for i, name in enumerate(file_names):
        plt.annotate(name, (pca_results[i, 0], pca_results[i, 1]))
    plt.title('PCA Visualization')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()


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

    # Step 5: Visualize the clusters
    # plot_dendrogram(similarity_matrix, file_names)
    # plot_tsne(tfidf_matrix, file_names)
    plot_pca(tfidf_matrix, file_names)


if __name__ == "__main__":
    main()
