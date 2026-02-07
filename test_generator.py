import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
BASE_URL = "https://models.inference.ai.azure.com"

qa_model = ChatOpenAI(model="gpt-4o", api_key=GITHUB_TOKEN, base_url=BASE_URL, temperature=0)

def read_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f: return f.read()
    except: return ""

def write_file(filepath, content):
    with open(filepath, "w", encoding="utf-8") as f: f.write(content)

def generate_test_for_file(filepath):
    filename = os.path.basename(filepath)
    module_name = filename.replace(".py", "")
    folder = os.path.dirname(filepath)
    test_filepath = os.path.join(folder, f"test_{filename}")

    if os.path.exists(test_filepath) or filename.startswith("test_") or "agent" in filename:
        return

    print(f"🕵️  [QA Agent] Generating tests for: {filename}...")
    code_content = read_file(filepath)

    # --- UPGRADED PROMPT ---
    prompt = ChatPromptTemplate.from_template(
        """You are a Senior QA Automation Engineer. Write a Pytest suite for '{filename}'.
        
        CRITICAL RULES FOR UI/SCRIPTS:
        1. Does the code use 'streamlit', 'tkinter', or run code at global scope?
        2. IF YES: You MUST mock these modules in `sys.modules` BEFORE importing the file.
           Example:
           ```python
           import sys
           from unittest.mock import MagicMock
           sys.modules["streamlit"] = MagicMock() # Mock BEFORE import
           from {module_name} import *
           ```
        3. Logic Testing: Test functions (happy/edge cases) using standard mocks for requests/db.
        4. Return ONLY the python code.
        
        CODE: {code}"""
    )
    
    chain = prompt | qa_model | StrOutputParser()
    try:
        test_code = chain.invoke({"filename": filename, "module_name": module_name, "code": code_content})
        clean_code = test_code.replace("```python", "").replace("```", "").strip()
        write_file(test_filepath, clean_code)
        print(f"✅ Created Robust Test: {test_filepath}")
    except Exception as e:
        print(f"❌ Failed: {e}")

def scan_and_generate(root_dir):
    print(f"🚀 [QA Manager] Scanning: {root_dir}")
    for root, dirs, files in os.walk(root_dir):
        if "venv" in root or "__pycache__" in root: continue
        for file in files:
            if file.endswith(".py"):
                generate_test_for_file(os.path.join(root, file))

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    scan_and_generate(target)