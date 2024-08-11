import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from clang import cindex

# Configure Clang library path
cindex.Config.set_library_file("/opt/homebrew/opt/llvm/lib/libclang.dylib")

def print_simplified_ast(node, indent=0):
    """Recursive function to print simplified AST nodes with indentation."""
    if node.kind in {cindex.CursorKind.FUNCTION_DECL, cindex.CursorKind.VAR_DECL, cindex.CursorKind.IF_STMT, cindex.CursorKind.FOR_STMT, cindex.CursorKind.WHILE_STMT}:
        print(' ' * indent + f'{node.kind} ({node.spelling})')
    for child in node.get_children():
        print_simplified_ast(child, indent + 2)

def generate_simplified_ast(file_path):
    index = cindex.Index.create()
    translation_unit = index.parse(file_path)
    print("Simplified AST for file:", file_path)
    print_simplified_ast(translation_unit.cursor)
    return translation_unit.cursor

def count_simplified_nodes(node):
    relevant_kinds = {
        cindex.CursorKind.FUNCTION_DECL,
        cindex.CursorKind.VAR_DECL,
        cindex.CursorKind.IF_STMT,
        cindex.CursorKind.FOR_STMT,
        cindex.CursorKind.WHILE_STMT,
    }

    count = 0
    if node.kind in relevant_kinds:
        count += 1
    for child in node.get_children():
        count += count_simplified_nodes(child)
    return count

def average_depth(node, current_depth=0):
    if node is None:
        return 0
    depths = [current_depth]
    for child in node.get_children():
        depths.append(average_depth(child, current_depth + 1))
    return sum(depths) / len(depths) if depths else 0

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
    for token in tokens:
        token_kind_name = get_token_kind_name(token.kind)
        if token.kind == cindex.TokenKind.COMMENT or token.spelling.startswith("#"):
            continue
        filtered_tokens.append(token.spelling)

    return " ".join(filtered_tokens)

def read_and_preprocess_cpp_files_with_ast(directory):
    cpp_files = []
    file_names = []
    asts = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".cpp"):
                file_path = os.path.join(root, file)
                tokenized_code = tokenize_cpp_code(file_path)
                cpp_files.append(tokenized_code)
                file_names.append(file)
                root_cursor = generate_simplified_ast(file_path)
                asts.append(root_cursor)
    return cpp_files, file_names, asts

def compute_tfidf_vectors(cpp_files, asts):
    combined_content = [content + str(ast) for content, ast in zip(cpp_files, asts)]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_content)
    return tfidf_matrix, vectorizer

def create_distance_matrix(tfidf_matrix):
    similarity_matrix = cosine_similarity(tfidf_matrix)
    distance_matrix = 1 - similarity_matrix
    distance_matrix = np.clip(distance_matrix, 0, 1)
    return distance_matrix

def create_dataset(cpp_files, file_names, asts, tfidf_matrix):
    dataset = []
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    for i in range(len(file_names)):
        for j in range(i + 1, len(file_names)):
            file1 = file_names[i]
            file2 = file_names[j]
            similarity_score = similarity_matrix[i, j]
            are_similar = 1 if similarity_score > 0.75 else 0

            num_nodes_1 = count_simplified_nodes(asts[i])
            num_nodes_2 = count_simplified_nodes(asts[j])
            avg_depth_1 = average_depth(asts[i])
            avg_depth_2 = average_depth(asts[j])

            print(f"File 1: {file1} - Number of simplified nodes: {num_nodes_1}, Average depth: {avg_depth_1}")
            print(f"File 2: {file2} - Number of simplified nodes: {num_nodes_2}, Average depth: {avg_depth_2}")

            entry = {
                "file1": file1,
                "file2": file2,
                "similarity_score": float(similarity_score),
                "tfidf_vector_1": tfidf_matrix[i].toarray().flatten().tolist(),
                "tfidf_vector_2": tfidf_matrix[j].toarray().flatten().tolist(),
                "num_nodes_1": int(num_nodes_1),
                "num_nodes_2": int(num_nodes_2),
                "avg_depth_1": float(avg_depth_1),
                "avg_depth_2": float(avg_depth_2),
                "are_similar": are_similar,
            }
            dataset.append(entry)
    return dataset

def save_dataset_to_json(dataset, output_file):
    try:
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        existing_data.extend(dataset)

        with open(output_file, 'w') as f:
            json.dump(existing_data, f, indent=4)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file: {e}")
        existing_data = dataset
        with open(output_file, 'w') as f:
            json.dump(existing_data, f, indent=4)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    cpp_directory = "/Users/vivek/Desktop/ASTra/Content1Diff"
    cpp_files, file_names, asts = read_and_preprocess_cpp_files_with_ast(cpp_directory)
    
    if not cpp_files:
        print("Error: No .cpp files found in the directory.")
        return

    tfidf_matrix, _ = compute_tfidf_vectors(cpp_files, asts)
    # Distance matrix is no longer needed

    # Create dataset without clustering
    final_dataset = create_dataset(cpp_files, file_names, asts, tfidf_matrix)

    # Save dataset to JSON
    output_file = "cpp_similarity_dataset_new.json"
    save_dataset_to_json(final_dataset, output_file)
    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    main()
