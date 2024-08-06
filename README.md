```markdown
# ASTra

## Overview
ASTra is a project designed to analyze C++ source files (.cpp) by reading their contents, generating Abstract Syntax Trees (ASTs), computing similarity matrices using TF-IDF vectors, and visualizing the results. The project evolved through multiple versions, each adding new features and improvements.

## Version History

### Combined Version
- **Reading .cpp Files:** Reads files from a specified directory, checks for empty files, and stores full file paths.
- **Preprocessing:** Placeholder for potential preprocessing steps, enhanced in later versions to remove comments and whitespace.
- **TF-IDF Computation:** Computes TF-IDF vectors for the file contents and combined content (file content + AST).
- **AST Generation:** Uses `libclang` to generate ASTs for each .cpp file, including detailed tokenization and filtering out comments and preprocessor directives.
- **Similarity Matrix:** Computes a cosine similarity matrix, including AST information.
- **Clustering and Visualization:** Adds clustering and visualization features, including PCA, t-SNE, and dendrograms for visualizing file clusters based on similarity.
- **AST Visualization:** Converts ASTs to Graphviz format for visualization.
- **Output:** Prints similarity matrix, pairwise similarity scores, AST information, token details, and visualizations.

## Usage

Run the `main()` function to:
1. Read `.cpp` files from the specified directory.
2. Generate ASTs for each file.
3. Preprocess the file contents.
4. Compute TF-IDF vectors, including AST information.
5. Compute and print the similarity matrix.
6. Print AST information for each file.
7. Visualize the results using PCA.
8. Visualize ASTs using Graphviz.

## Code Sections

### Part 1: Reading and Preprocessing C++ Files, Generating ASTs, and Computing Similarity Matrix

In the first part of the code, the program reads C++ files from a specified directory, ensuring that empty files are skipped. It then tokenizes the C++ code while filtering out comments and preprocessor directives. Next, the code generates ASTs using `libclang` and calculates the number of nodes and the average depth of these ASTs. The tokenized code and ASTs are combined, and TF-IDF vectors are computed for this combined content. The cosine similarity matrix is then computed using these vectors, and a dataset is created that includes file names, similarity scores, TF-IDF vectors, and AST features. Finally, this dataset is saved to a JSON file.

### Part 2: Loading, Preprocessing, and Training the Model

In the second part of the code, the program loads the dataset from the JSON file and preprocesses it for machine learning. The TF-IDF vectors are expanded into separate columns, and the features and labels are separated. The data is then split into training and testing sets. A Random Forest classifier is trained on the training data, and predictions are made on the test set. The model's accuracy and a classification report are printed. Additionally, the code prints the similarity scores along with predictions and actual labels for each pair of files in the test set.
```
