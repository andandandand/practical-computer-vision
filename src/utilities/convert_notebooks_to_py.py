import os
import subprocess
from pathlib import Path

def convert_notebooks_to_py(source_folder, output_folder):
    """Convert all .ipynb files in a folder to .py files in specified output folder"""
    source_path = Path(source_folder)
    output_path = Path(output_folder)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all notebook files
    notebook_files = list(source_path.glob("*.ipynb"))
    
    if not notebook_files:
        print("No notebook files found in the specified folder.")
        return
    
    for notebook in notebook_files:
        try:
            # Use nbconvert to convert each file with output directory
            subprocess.run([
                "jupyter", "nbconvert", 
                "--to", "python", 
                "--output-dir", str(output_path),
                str(notebook)
            ], check=True)
            print(f"Converted: {notebook.name} -> {output_path / (notebook.stem + '.py')}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting {notebook.name}: {e}")

current_dir = Path(os.getcwd())
# Usage
source_folder =  current_dir / 'notebooks'
output_folder = current_dir / 'src/notebooks_as_python_scripts'
convert_notebooks_to_py(source_folder, output_folder)
