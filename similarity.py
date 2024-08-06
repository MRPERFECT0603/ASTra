import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For saving and loading the model and vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer  # Import this

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

    return X, y

# Main function to load, preprocess, and train the model
def main():
    dataset_file = "cpp_similarity_dataset.json"
    
    # Load dataset
    dataset = load_dataset(dataset_file)
    
    # Preprocess the data
    X, y = preprocess_data(dataset)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a model (Random Forest in this case)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, "cpp_similarity_model.pkl")
    
    # Save the vectorizer
    vectorizer = TfidfVectorizer()  # Ensure to instantiate your vectorizer
    vectorizer.fit(X_train)  # Fit it on your training data
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")  # Save the vectorizer

    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

if __name__ == "__main__":
    main()
