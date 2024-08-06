import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from clang import cindex
import graphviz
import numpy as np

cindex.Config.set_library_file("/opt/homebrew/opt/llvm/lib/libclang.dylib")


def generate_ast(file_path):
    # Initialize libclang index
    index = cindex.Index.create()
    # Parse the C++ file and create a TranslationUnit
    translation_unit = index.parse(file_path)
    # Get the root cursor (top-level node) of the AST
    root_cursor = translation_unit.cursor
    return root_cursor


def get_token_kind_name(token_kind):
    token_kind_dict = {
        cindex.TokenKind.PUNCTUATION: "PUNCTUATION",
        cindex.TokenKind.KEYWORD: "KEYWORD",
        cindex.TokenKind.IDENTIFIER: "IDENTIFIER",
        cindex.TokenKind.LITERAL: "LITERAL",
    }
    return token_kind_dict.get(token_kind, "UNKNOWN")


def tokenize_cpp_code(file_path):
    index = cindex.Index.create()
    translation_unit = index.parse(file_path)
    tokens = list(translation_unit.get_tokens(extent=translation_unit.cursor.extent))

    filtered_tokens = []
    print(f"Tokens for {file_path}:")
    for token in tokens:
        token_kind_name = get_token_kind_name(token.kind)
        # Print each token for visualization in a structured format
        print(
            f"Token: {token.spelling}, Kind: {token_kind_name}, Line: {token.location.line}, Column: {token.location.column}"
        )
        # Filter out comments and preprocessor directives
        if token.kind == cindex.TokenKind.COMMENT or token.spelling.startswith("#"):
            continue
        filtered_tokens.append(token.spelling)

    return " ".join(filtered_tokens)


# Function to read .cpp files and generate ASTs after preprocessing
def read_and_preprocess_cpp_files_with_ast(directory):
    cpp_files = []
    file_names = []
    asts = []  # Store ASTs here
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".cpp"):
                file_path = os.path.join(root, file)
                tokenized_code = tokenize_cpp_code(file_path)
                cpp_files.append(tokenized_code)
                file_names.append(file)
                # Generate AST using Script 1's function
                root_cursor = generate_ast(file_path)
                asts.append(root_cursor)
                # Check if the file is empty after preprocessing
                if not tokenized_code.strip():
                    print(f"Warning: {file_path} is empty after preprocessing.")
    return cpp_files, file_names, asts


# Function to convert AST to Graphviz
def ast_to_graphviz(node, graph=None, parent_id=None, depth=0, max_depth=2):
    if graph is None:
        graph = graphviz.Digraph()

    if depth > max_depth:
        return graph

    node_id = str(node.hash)
    label = f"{node.kind.name}\n{node.spelling if node.spelling else ''}"
    graph.node(node_id, label=label)

    if parent_id:
        graph.edge(parent_id, node_id)

    for child in node.get_children():
        ast_to_graphviz(child, graph, node_id, depth + 1, max_depth)

    return graph


# Function to compute TF-IDF vectors (including AST information)
def compute_tfidf_vectors(cpp_files, asts):
    # Combine cpp_files content with AST information
    combined_content = [content + str(ast) for content, ast in zip(cpp_files, asts)]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_content)
    return tfidf_matrix, vectorizer


# Function to compute similarity matrix for a list of .cpp files
def compute_similarity_matrix(tfidf_matrix, asts):
    similarity_matrix = cosine_similarity(tfidf_matrix)
    ast_similarity_matrix = compute_ast_similarity(asts)
    return similarity_matrix, ast_similarity_matrix


# Function to plot PCA
def plot_pca(tfidf_matrix, file_names):
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(tfidf_matrix.toarray())
    plt.figure(figsize=(10, 7))
    plt.scatter(pca_results[:, 0], pca_results[:, 1])
    for i, name in enumerate(file_names):
        plt.annotate(name, (pca_results[i, 0], pca_results[i, 1]))
    plt.title("PCA Visualization")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()


def compute_ast_similarity(asts):
    num_asts = len(asts)
    ast_similarity_matrix = np.zeros((num_asts, num_asts))
    for i in range(num_asts):
        for j in range(num_asts):
            ast_similarity_matrix[i, j] = myers_diff(str(asts[i]), str(asts[j]))
    return ast_similarity_matrix


def myers_diff(a, b):
    m, n = len(a), len(b)
    d = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    return d[m][n]


def main():
    # Step 1: Read .cpp files and generate ASTs after preprocessing
    cpp_directory = "/Users/vivek/Desktop/ASTra/content/"
    cpp_files, file_names, asts = read_and_preprocess_cpp_files_with_ast(cpp_directory)
    print("Read the following files after preprocessing:")
    for name, content in zip(file_names, cpp_files):
        print(f"- {name}")
        print("------------------------")
        print(content)
        print()

    # Check if any files were read
    if not cpp_files:
        print("Error: No .cpp files found in the directory.")
        return  # Exit if no files found

    # Step 2: Compute TF-IDF vectors (including AST information)
    tfidf_matrix, vectorizer = compute_tfidf_vectors(cpp_files, asts)
    print("\nTF-IDF matrix shape:", tfidf_matrix.shape)

    # Step 3: Compute similarity matrices
    tfidf_similarity_matrix = cosine_similarity(tfidf_matrix)
    ast_similarity_matrix = compute_ast_similarity(asts)

    # Combine the similarity matrices
    combined_similarity_matrix = (tfidf_similarity_matrix + ast_similarity_matrix) / 2

    # Print the combined similarity matrix
    print("\nCombined Similarity Matrix:")
    print("==========================")
    print(combined_similarity_matrix)

    # Create a DataFrame for the combined similarity matrix
    df_combined_similarity = pd.DataFrame(
        combined_similarity_matrix, index=file_names, columns=file_names
    )

    # Print formatted combined similarity matrix
    print("\nCombined Similarity Matrix:")
    print("==========================")
    print(df_combined_similarity)

    # Print pairwise combined similarity scores
    print("\nPairwise combined similarity scores:")
    print("================================")
    for i in range(len(file_names)):
        for j in range(i + 1, len(file_names)):
            combined_similarity_score = combined_similarity_matrix[i, j]
            print(
                f"Combined similarity between {file_names[i]} and {file_names[j]}: {combined_similarity_score:.8f}"
            )

    # Print AST information for each file
    print("\nAST Information:")
    print("================")
    for i, (file_name, ast) in enumerate(zip(file_names, asts)):
        print(f"AST for {file_name}:")
        print("--------------------")
        print(ast)

    # Visualize AST using Graphviz
    # graph = ast_to_graphviz(ast)
    # graph.render(filename=file_name, format="png", cleanup=True)
    # print(f"AST visualization saved as {file_name}.png")

    # Step 4: Visualize the clusters using PCA
    # plot_pca(tfidf_matrix, file_names)

if __name__ == "__main__":
    main()