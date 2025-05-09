# Create project directory structure
import os

def create_directory_structure():
    directories = [
        "data/raw",
        "data/processed",
        "data/interim",
        "models/saved",
        "notebooks",
        "src/data_collection",
        "src/data_processing",
        "src/features",
        "src/models",
        "src/visualization",
        "config",
        "results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

create_directory_structure()