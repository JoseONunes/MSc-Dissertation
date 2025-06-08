import os
import nbformat
import re
from nbconvert import PythonExporter

def remove_emojis_and_non_ascii(text):
    # Removes emojis and non-ASCII characters
    return re.sub(r'[^\x00-\x7F]+', '', text)

def convert_notebooks_to_scripts(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".ipynb"):
            notebook_path = os.path.join(input_folder, filename)
            with open(notebook_path, "r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)

            exporter = PythonExporter()
            script_body, _ = exporter.from_notebook_node(nb)

            # Strip emojis and non-ASCII characters
            script_body_cleaned = remove_emojis_and_non_ascii(script_body)

            script_filename = os.path.splitext(filename)[0] + ".py"
            output_path = os.path.join(output_folder, script_filename)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(script_body_cleaned)

            print(f"✅ Converted (cleaned): {filename} -> {script_filename}")

# Example usage
convert_notebooks_to_scripts("Project/Notebooks", "Report Work/Template/Notebooks")
