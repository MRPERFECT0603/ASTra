import os
from clang import cindex

cindex.Config.set_library_file("/opt/homebrew/opt/llvm/lib/libclang.dylib")


def read_cpp_files(directory):
    cpp_files = []
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".cpp"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    content = f.read()
                    cpp_files.append(content)
                    file_paths.append(file_path)  # Store full file path for later use
                # Check if the file is empty
                if not content.strip():
                    print(f"Warning: {file_path} is empty.")
    return cpp_files, file_paths


def generate_ast(file_path):
    # Set libclang library path (update with your correct path)

    # Initialize libclang and create an index
    index = cindex.Index.create()

    # Parse the C++ file and create a TranslationUnit
    translation_unit = index.parse(file_path)

    # Get the root cursor (top-level node) of the AST
    root_cursor = translation_unit.cursor

    return root_cursor


def main():
    # Step 1: Read .cpp files
    cpp_directory = "/Users/vivek/Desktop/ASTra/content/"
    cpp_files, file_paths = read_cpp_files(cpp_directory)
    print("Read the following files:")
    for path in file_paths:
        print(f"- {path}")

    # Check if any files were read
    if not cpp_files:
        print("Error: No .cpp files found in the directory.")
        return  # Exit if no files found

    # Step 2: Generate AST for each .cpp file
    asts = []
    for file_path in file_paths:
        print(f"\nGenerating AST for {file_path}...")
        root_cursor = generate_ast(file_path)
        asts.append(root_cursor)

        # Example: Print root cursor kind
        print(f"Root cursor kind: {root_cursor.kind}")

        # Example: Traverse and print children nodes
        print("Children nodes:")
        for child in root_cursor.get_children():
            print(f"- {child.kind}")

    # Additional processing or analysis of ASTs can be performed here


if __name__ == "__main__":
    main()
