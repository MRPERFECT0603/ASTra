import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from clang import cindex

cindex.Config.set_library_file("/opt/homebrew/opt/llvm/lib/libclang.dylib")

def generate_ast(file_path):
    index = cindex.Index.create()
    translation_unit = index.parse(file_path)
    return translation_unit.cursor

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

def count_nodes(node):
    count = 1  # Count the current node
    for child in node.get_children():
        count += count_nodes(child)
    return count

def average_depth(node, current_depth=0):
    if node is None:
        return 0
    depths = [current_depth]
    for child in node.get_children():
        depths.append(average_depth(child, current_depth + 1))
    return sum(depths) / len(depths)

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
                root_cursor = generate_ast(file_path)
                asts.append(root_cursor)
    return cpp_files, file_names, asts

def compute_tfidf_vectors(cpp_files, asts):
    combined_content = [content + str(ast) for content, ast in zip(cpp_files, asts)]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_content)
    return tfidf_matrix, vectorizer

def create_dataset(cpp_files, file_names, asts, tfidf_matrix, threshold=0.75):
    dataset = []
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    for i in range(len(file_names)):
        for j in range(i + 1, len(file_names)):
            file1 = file_names[i]
            file2 = file_names[j]
            similarity_score = similarity_matrix[i, j]
            are_similar = 1 if similarity_score > threshold else 0

            entry = {
                "file1": file1,
                "file2": file2,
                "similarity_score": similarity_score,
                "tfidf_vector_1": tfidf_matrix[i].toarray().flatten().tolist(),
                "tfidf_vector_2": tfidf_matrix[j].toarray().flatten().tolist(),
                "num_nodes_1": count_nodes(asts[i]),
                "num_nodes_2": count_nodes(asts[j]),
                "avg_depth_1": average_depth(asts[i]),
                "avg_depth_2": average_depth(asts[j]),
                "are_similar": are_similar
            }
            dataset.append(entry)
    return dataset

def save_dataset_to_json(dataset, output_file):
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
        existing_data.extend(dataset)
        with open(output_file, 'w') as f:
            json.dump(existing_data, f, indent=4)
    else:
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=4)

def main():
    cpp_directory = "/Users/vivek/Desktop/ASTra/Data/"
    cpp_files, file_names, asts = read_and_preprocess_cpp_files_with_ast(cpp_directory)
    
    if not cpp_files:
        print("Error: No .cpp files found in the directory.")
        return

    tfidf_matrix, _ = compute_tfidf_vectors(cpp_files, asts)

    # Create dataset
    dataset = create_dataset(cpp_files, file_names, asts, tfidf_matrix)

    # Save dataset to JSON
    output_file = "cpp_similarity_dataset.json"
    save_dataset_to_json(dataset, output_file)
    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    main()
