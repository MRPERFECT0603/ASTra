import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
import networkx as nx

# Load the dataset
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        dataset = json.load(f)
    return dataset

# Preprocess the data
def preprocess_data(dataset):
    data = pd.json_normalize(dataset)
    
    # Expand the tfidf vectors into separate columns
    tfidf_vectors_1 = pd.DataFrame(data['tfidf_vector_1'].tolist())
    tfidf_vectors_2 = pd.DataFrame(data['tfidf_vector_2'].tolist())
    
    # Concatenate the tfidf vectors back to the original dataframe
    data = pd.concat([data, tfidf_vectors_1.add_prefix('tfidf1_'), tfidf_vectors_2.add_prefix('tfidf2_')], axis=1)
    
    # Drop the original tfidf vector columns
    data = data.drop(columns=['tfidf_vector_1', 'tfidf_vector_2'])
    
    # Separate features and labels
    X = data.drop(columns=['file1', 'file2', 'are_similar'])
    y = data['are_similar']

    return X, y, data

def build_graph(dataset, threshold):
    G = nx.Graph()
    
    for entry in dataset:
        file1 = entry['file1']
        file2 = entry['file2']
        similarity_score = entry['similarity_score']
        
        if similarity_score >= threshold:
            G.add_edge(file1, file2)
    
    return G

# Find and print connected components
def print_connected_components(G):
    components = list(nx.connected_components(G))
    
    for component in components:
        print("Group of similar files:")
        print(", ".join(component))
        print()



# Main function to load, preprocess, train the model, and save clustered files
def main():
    dataset_file = "cpp_similarity_dataset_new.json"

    # Load dataset
    dataset = load_dataset(dataset_file)
    
    # Preprocess the data
    X, y, data = preprocess_data(dataset)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, data.index, test_size=0.5, random_state=42)
    
    # Convert idx_test to a Series for iloc usage
    idx_test = pd.Series(idx_test).reset_index(drop=True)
    
    # Convert y_test to a Series for iloc usage
    y_test = pd.Series(y_test).reset_index(drop=True)
    
    # Train a model (Random Forest in this case)
    model = RandomForestClassifier(n_estimators=10000, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Convert y_pred to a Series if it's a numpy array
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred).reset_index(drop=True)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Print results
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
    
    # Print similarity scores along with predictions
    for i in range(len(y_test)):
        idx = idx_test.iloc[i]
        print(f"File 1: {data['file1'].iloc[idx]}, File 2: {data['file2'].iloc[idx]}, Similarity Score: {data['similarity_score'].iloc[idx]}, Predicted: {y_pred.iloc[i]}, Actual: {y_test.iloc[i]}")
    
    G = build_graph(dataset, 0.75)
    
    # Find and print connected components
    print_connected_components(G)

if __name__ == "__main__":
    main()
