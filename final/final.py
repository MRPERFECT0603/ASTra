import json
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from clang import cindex

# Set up Clang library path
cindex.Config.set_library_file("/opt/homebrew/opt/llvm/lib/libclang.dylib")

def generate_ast(file_path):
    index = cindex.Index.create()
    translation_unit = index.parse(file_path)
    return translation_unit.cursor

def tokenize_cpp_code(file_path):
    index = cindex.Index.create()
    translation_unit = index.parse(file_path)
    
    tokens = list(translation_unit.get_tokens(extent=translation_unit.cursor.extent))
    
    # print(f"Processing file: {file_path}")  # Debug print
    # print(f"Tokens before filtering: {[token.spelling for token in tokens]}")  # Debug print

    filtered_tokens = []
    for token in tokens:
        if token.kind == cindex.TokenKind.COMMENT or token.spelling.startswith("#"):
            continue
        filtered_tokens.append(token.spelling)

    # print(f"Tokens after filtering: {filtered_tokens}")  # Debug print

    return " ".join(filtered_tokens)


def load_model(model_path):
    return joblib.load(model_path)

def load_vectorizer(vectorizer_path):
    return joblib.load(vectorizer_path)

def load_and_predict(new_cpp_files, model_path, vectorizer_path):
    model = load_model(model_path)
    vectorizer = load_vectorizer(vectorizer_path)

    # Tokenize the new C++ files
    tokenized_files = [tokenize_cpp_code(file) for file in new_cpp_files]

    # Check if there are any tokenized files
    if not tokenized_files:
        print("No tokenized files available for prediction.")
        return []

    # Transform the tokenized files using the vectorizer
    tfidf_matrix = vectorizer.transform(tokenized_files)
    predictions = model.predict(tfidf_matrix)

    return predictions

# Main function
def main():
    model_path = "/Users/vivek/Desktop/ASTra/cpp_similarity_model.pkl"
    vectorizer_path = "/Users/vivek/Desktop/ASTra/tfidf_vectorizer.pkl"

    # Directory containing new C++ files
    new_cpp_directory = "/Users/vivek/Desktop/ASTra/final/cppFiles/"
    new_cpp_files = [os.path.join(new_cpp_directory, file) for file in os.listdir(new_cpp_directory) if file.endswith(".cpp")]

    # Get predictions
    predictions = load_and_predict(new_cpp_files, model_path, vectorizer_path)

    # Output the predictions
    for file, prediction in zip(new_cpp_files, predictions):
        print(f"File: {file} - Similarity Prediction: {prediction}")

if __name__ == "__main__":
    main()
