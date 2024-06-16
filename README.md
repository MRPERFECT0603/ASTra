# ASTra

## Overview
ASTra is a project designed to analyze C++ source files (.cpp) by reading their contents, generating Abstract Syntax Trees (ASTs), computing similarity matrices using TF-IDF vectors, and visualizing the results. The project evolved through multiple versions, each adding new features and improvements.

## Version 1
- **Reading .cpp Files:** Reads files from a specified directory and checks for empty files.
- **Preprocessing:** Placeholder for potential preprocessing steps (no actual preprocessing).
- **TF-IDF Computation:** Computes TF-IDF vectors for the file contents.
- **Similarity Matrix:** Computes and prints a cosine similarity matrix.
- **Output:** Prints the similarity matrix and pairwise similarity scores.

## Version 2
- **AST Generation:** Introduced the use of `libclang` to generate ASTs for each .cpp file.
- **Reading .cpp Files:** Reads files and stores full file paths.
- **Output:** Prints the root cursor kind and children nodes of each AST.

## Version 3
- **Combining Content with AST:** Merged file contents with their AST representations for TF-IDF computation.
- **Reading .cpp Files and AST Generation:** Reads .cpp files and generates ASTs.
- **Similarity Matrix:** Computes similarity matrix using combined content (file content + AST).
- **Output:** Prints similarity matrix and AST information for each file.

## Version 4
- **Clustering and Visualization:** Added clustering and visualization features including PCA, t-SNE, and dendrogram.
- **Reading .cpp Files and AST Generation:** Same as Version 3.
- **Combining Content with AST:** Same as Version 3.
- **Similarity Matrix:** Same as Version 3.
- **Visualization:** Added functions to plot dendrograms, t-SNE, and PCA for visualizing file clusters based on similarity.

## Usage
Run the `main()` function to:
1. Read `.cpp` files from the specified directory.
2. Generate ASTs for each file.
3. Preprocess the file contents.
4. Compute TF-IDF vectors including AST information.
5. Compute and print the similarity matrix.
6. Print AST information for each file.
7. Visualize the results using PCA.

This project, ASTra, provides a robust framework for analyzing and visualizing the similarities between C++ source files, leveraging both their textual content and syntactic structure.
