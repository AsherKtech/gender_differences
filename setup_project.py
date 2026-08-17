#!/usr/bin/env python3
import os

# 1. Define folder structure
folders = [
    "data/raw",
    "data/processed",
    "data/external",
    "data/temp",
    "notebooks",
    "src",
    "tests",
    "models",
    "reports/figures",
    "images"
]

# 2. Define .gitignore content
gitignore_content = """# --- Virtual Environments ---
**/myenv/
**/venv/
**/.venv/
.env

# --- Python & Notebooks ---
__pycache__/
*.py[cod]
*$py.class
.ipynb_checkpoints/
*/.ipynb_checkpoints/*

# Ignore all files in data folders
data/raw/*
data/processed/*
data/temp/*

# But keep the folders (use a placeholder file)
!data/raw/.gitkeep
!data/external/.gitkeep
!data/processed/.gitkeep
!data/temp/.gitkeep

# --- OS & IDE ---
.DS_Store
.Thumbs.db
.vscode/
.idea/
"""

# Create folders and .gitkeep files
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    # Create .gitkeep in data folders to ensure they are tracked by git
    if folder.startswith("data"):
        gitkeep_path = os.path.join(folder, ".gitkeep")
        if not os.path.exists(gitkeep_path):
            with open(gitkeep_path, "w") as f:
                pass

# Create/Update .gitignore
with open(".gitignore", "w") as f:
    f.write(gitignore_content)

# Ensure essential files exist
files_to_touch = [
    "README.md",
    "requirements.txt",
    "src/__init__.py"
]

for file in files_to_touch:
    if not os.path.exists(file):
        open(file, "a").close()

print("✅ Project structure and .gitignore created successfully!")