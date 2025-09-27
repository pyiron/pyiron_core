import requests
import json
import re
import subprocess
import tempfile
import os
from pathlib import Path
# from pwf_creator import load_extra_files, extract_code, ask_ollama, get_workflow
from IPython.display import display, Markdown, Code
import ipywidgets as widgets


def load_extra_files(file_paths):
    """Load extra files and format them for inclusion in the prompt."""
    combined_content = ""
    for file_path in file_paths:
        file_p = Path(file_path)
        if not file_p.exists():
            print(f"Warning: file {file_path} does not exist.")
            continue
        try:
            with open(file_p, "r", encoding="utf-8") as f:
                content = f.read()
            combined_content += f"\n\n--- BEGIN FILE: {file_p.name} ---\n{content}\n--- END FILE: {file_p.name} ---\n"
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return combined_content


import requests
import json

def ask_ollama(prompt, model="gpt-oss:120b", temperature: float = 0.2):
    """
    Send a prompt to Ollama and return the collected response text.

    Parameters
    ----------
    prompt : str
        The user prompt/question for the LLM.
    model : str, optional
        The Ollama model ID to use. Default is "gpt-oss:120b".
    temperature : float, optional
        Creativity/randomness in generation. 
        Lower values (~0.0) = more deterministic, 
        higher values (~1.0) = more random. Default is 0.7.

    Returns
    -------
    str
        The concatenated response text from the model.
    """
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": True
        },
        stream=True
    )

    output_text = ""
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            continue
        if "response" in data and data["response"]:
            output_text += data["response"]
    return output_text.strip()


def extract_code(text):
    """Extract Python code from triple backticks, or return full text if no blocks."""
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match2 = re.search(r"```(.*?)```", text, re.DOTALL)  # without python hint
    if match2:
        return match2.group(1).strip()
    return text.strip()


def get_workflow():
    import importlib
    import sys
    
    # Make sure the module is in sys.modules (import it if it isn’t yet)
    if "workflow" not in sys.modules:
        import workflow          # first‑time import
    else:
        import workflow          # module already present
    
    # Reload the module so any changes on disk are picked up
    importlib.reload(workflow)
    
    # Grab the refreshed `wf` attribute
    return workflow.wf