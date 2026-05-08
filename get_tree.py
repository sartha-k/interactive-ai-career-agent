import os

# Files/Folders to ignore
ignore = {'.venv', 'venv', '__pycache__', '.git', 'faiss_index', '.streamlit'}

with open("clean_architecture.txt", "w", encoding="utf-8") as f:
    for root, dirs, files in os.walk("."):
        # Filter out ignored directories
        dirs[:] = [d for d in dirs if d not in ignore]
        
        level = root.replace(".", "").count(os.sep)
        indent = " " * 4 * level
        f.write(f"{indent}{os.path.basename(root)}/\n")
        
        sub_indent = " " * 4 * (level + 1)
        for file in files:
            if file not in {"clean_architecture.txt", "get_tree.py"}:
                f.write(f"{sub_indent}{file}\n")

print("✅ Success! Check 'clean_architecture.txt' now.")
