import os
import sys
import shutil
import subprocess
import argparse
from typing import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
MAX_RETRIES = 3
MAX_GLOBAL_RETRIES = 5  # Stop after fixing 5 files to prevent infinite loops
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
BASE_URL = "https://models.inference.ai.azure.com"

if not GITHUB_TOKEN:
    raise ValueError("❌ MISSING GITHUB_TOKEN in .env file!")

# --- 1. DYNAMIC STATE ---
class AgentState(TypedDict):
    project_root: str   
    target_file: str    
    test_file: str     
    original_code: str
    error_log: str
    category: str 
    analysis: str
    fixed_code: str
    retry_count: int
    status: str 

# --- 2. TOOLS ---
def read_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f: return f.read()
    except Exception: return ""

def write_file(filepath, content):
    with open(filepath, "w", encoding="utf-8") as f: f.write(content)

def normalize_project(project_root):
    """Ensures requirements.txt exists at Root."""
    target_req = os.path.join(project_root, "requirements.txt")
    if not os.path.exists(target_req):
        print(f"📦 [System] Creating fresh requirements.txt at Root.")
        with open(target_req, "w") as f: f.write("")
    return target_req

def add_to_requirements(project_root, package_name):
    req_file = os.path.join(project_root, "requirements.txt")
    current_reqs = read_file(req_file)
    if package_name not in current_reqs:
        with open(req_file, "a", encoding="utf-8") as f:
            f.write(f"\n{package_name}")
        print(f"📝 [Manifest] Added '{package_name}' to requirements.")

def install_package(package_name):
    print(f"📦 [System] Installing missing package: {package_name}...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", package_name], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ [System] Successfully installed {package_name}")
        return True
    return False

def inspect_library(package_name, object_name=None):
    """
    Allows the Agent to read the docs of a library to prevent hallucinations.
    Uses Python's help() function dynamically.
    """
    try:
        if object_name:
            cmd = [sys.executable, "-c", f"import {package_name}; help({package_name}.{object_name})"]
        else:
            cmd = [sys.executable, "-c", f"import {package_name}; help({package_name})"]
        
        # Run in a subprocess to avoid crashing the agent
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Return only the first 2000 chars to save tokens (usually contains the signature and basic usage)
        if result.returncode == 0:
            return result.stdout[:2000]
        else:
            return ""
    except Exception as e:
        return f"Could not read docs: {e}"

# --- 3. MODELS ---
architect_model = ChatOpenAI(model="gpt-4o-mini", api_key=GITHUB_TOKEN, base_url=BASE_URL, temperature=0)
coder_model = ChatOpenAI(model="gpt-4o-mini", api_key=GITHUB_TOKEN, base_url=BASE_URL, temperature=0)

# --- 4. NODES ---
def analyze_node(state: AgentState):
    filename = os.path.basename(state['target_file'])
    print(f"\n🧐 [Architect] Analyzing failure in {filename} (Attempt {state['retry_count'] + 1})...")
    
    # 1. Check for HARD Dependency Errors (ModuleNotFoundError)
    if "ModuleNotFoundError" in state["error_log"] or "No module named" in state["error_log"]:
        category = "dependency"
        print(f"🔀 [Router] Decision: This is a DEPENDENCY issue (Missing Library).")
    
    # 2. Check for HARD Logic Errors (NameError, TypeError, etc.)
    elif "NameError" in state["error_log"]:
        category = "logic"
        print(f"🔀 [Router] Decision: This is a LOGIC issue (Missing Import or Typo).")
        
    else:
        # 3. Ask the AI for subtle cases
        router_prompt = ChatPromptTemplate.from_template(
            "Analyze this error: {error}\nClassify as 'dependency' or 'logic'. Return ONE WORD."
        )
        chain = router_prompt | architect_model | StrOutputParser()
        raw_category = chain.invoke({"error": state["error_log"]}).strip().lower()
        category = "dependency" if "dependency" in raw_category else "logic"
        print(f"🔀 [Router] Decision: AI classified as {category.upper()}.")
    
    analysis_text = ""
    if category == "logic":
        code_to_analyze = state["fixed_code"] if state["retry_count"] > 0 else state["original_code"]
        logic_prompt = ChatPromptTemplate.from_template(
            "Explain the bug in this code given the error.\nCODE:\n{code}\nERROR:\n{error}"
        )
        logic_chain = logic_prompt | architect_model | StrOutputParser()
        analysis_text = logic_chain.invoke({"code": code_to_analyze, "error": state["error_log"]})

    return {"category": category, "analysis": analysis_text}

def dependency_node(state: AgentState):
    print(f"🔧 [SysAdmin] Identifying missing library...")
    prompt = ChatPromptTemplate.from_template("Extract missing python package name from: {error}. Return ONLY name.")
    chain = prompt | coder_model | StrOutputParser()
    pkg_name = chain.invoke({"error": state["error_log"]}).strip().replace("'", "").replace('"', "")
    
    add_to_requirements(state["project_root"], pkg_name)
    success = install_package(pkg_name)
    
    if success: return {"fixed_code": state["original_code"]}
    else: return {"status": "failed", "retry_count": state["retry_count"] + 1, "error_log": f"Failed install: {pkg_name}"}

def fix_node(state: AgentState):
    filename = os.path.basename(state['target_file'])
    print(f"🛠️  [Coder] Applying fix to {filename}...")
    
    code_to_fix = state["fixed_code"] if state["retry_count"] > 0 else state["original_code"]
    
    # --- STEP 1: RESEARCH (PREVENT HALLUCINATION) ---
    print(f"🧠 [Coder] Checking if external documentation is needed...")
    research_prompt = ChatPromptTemplate.from_template(
        """You are fixing this error: {error}
        In this code:
        {code}
        
        Do you need to check the official documentation for any imported library to ensure correct usage?
        If yes, return the library and function name separated by a space (e.g., 'bcrypt checkpw').
        If no, return 'NO'.
        """
    )
    chain = research_prompt | coder_model | StrOutputParser()
    research_query = chain.invoke({"error": state["error_log"], "code": code_to_fix}).strip()
    
    doc_context = "No external documentation used."
    if "NO" not in research_query and len(research_query.split()) <= 2:
        try:
            parts = research_query.split()
            lib = parts[0]
            func = parts[1] if len(parts) > 1 else None
            
            print(f"📖 [Research] 🔍 Accessing Knowledge Base for: {lib}.{func if func else 'module'}...")
            doc_output = inspect_library(lib, func)
            
            if doc_output:
                doc_context = doc_output
                print(f"✅ [Research] Documentation retrieved ({len(doc_output)} chars).")
            else:
                print(f"⚠️ [Research] No documentation found.")
        except Exception as e:
            print(f"⚠️ [Research] Skipped research due to query format.")

    # --- STEP 2: WRITE CODE ---
    prompt = ChatPromptTemplate.from_template(
        """Fix the code.
        
        BROKEN CODE:
        {code}
        
        ERROR:
        {analysis}
        
        OFFICIAL DOCUMENTATION (Grounding Truth):
        {docs}
        
        Return ONLY the full code.
        """
    )
    chain = prompt | coder_model | StrOutputParser()
    fixed_code = chain.invoke({
        "code": code_to_fix, 
        "analysis": state["analysis"],
        "docs": doc_context
    })
    
    clean_code = fixed_code.replace("```python", "").replace("```", "").strip()
    return {"fixed_code": clean_code}

def test_node(state: AgentState):
    if state["fixed_code"]:
        write_file(state["target_file"], state["fixed_code"]) 
    
    # SURGICAL TEST: Run ONLY the relevant test file
    target_test = state["test_file"] if state["test_file"] and os.path.exists(state["test_file"]) else state["project_root"]
    
    print(f"🧪 [Tester] Verifying fix using: {os.path.basename(target_test)}...")
    result = subprocess.run([sys.executable, "-m", "pytest", target_test], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ [Tester] PASS: The fix is valid.")
        return {"status": "success"}
    else:
        print("❌ [Tester] FAIL: The fix did not resolve the issue.")
        return {"status": "failed", "retry_count": state["retry_count"] + 1, "error_log": result.stdout}

# --- 5. GRAPH ---
def router_logic(state: AgentState): return state["category"]
def should_continue(state: AgentState):
    if state["status"] == "success": return "end"
    if state["retry_count"] >= MAX_RETRIES: return "end"
    return "analyze"

workflow = StateGraph(AgentState)
workflow.add_node("analyze", analyze_node)
workflow.add_node("dependency", dependency_node)
workflow.add_node("logic_fix", fix_node)
workflow.add_node("test", test_node)
workflow.set_entry_point("analyze")
workflow.add_conditional_edges("analyze", router_logic, {"dependency": "dependency", "logic": "logic_fix"})
workflow.add_edge("dependency", "test")
workflow.add_edge("logic_fix", "test")
workflow.add_conditional_edges("test", should_continue, {"analyze": "analyze", "end": END})
app = workflow.compile()

# --- 6. MAIN RUNNER (THE GLOBAL LOOP) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=".", help="Path to project")
    args = parser.parse_args()
    project_root = os.path.abspath(args.path)

    print(f"\n🚀 [System] Initializing Autonomous Maintainer on: {project_root}")
    normalize_project(project_root)

    global_attempts = 0
    
    while global_attempts < MAX_GLOBAL_RETRIES:
        print(f"\n🌍 [Global Scan {global_attempts + 1}] Running full system diagnostics...")
        
        # 1. Run GLOBAL Tests
        global_test = subprocess.run([sys.executable, "-m", "pytest", project_root], capture_output=True, text=True)
        
        if global_test.returncode == 0:
            print("\n🏆 [System] GRAND VICTORY: All tests passed! The repository is clean.")
            break 
        
        print("🚨 [System] Failures Detected. Identifying the highest priority target...")
        error_log = global_test.stdout
        
        # 2. INTELLIGENT TARGETING (Find First Broken File)
        target_file = None
        detected_test_file = None

        for line in error_log.splitlines():
            # Looks for both FAILED (Logic) and ERROR (Import/Syntax)
            if ("FAILED" in line or "ERROR" in line) and ".py" in line:
                parts = line.split("::")[0].split(" ")
                for part in parts:
                    if part.endswith(".py"):
                        # Clean up path (remove colons if any)
                        failed_test = part.split("::")[0]
                        
                        # Map test_X.py -> X.py
                        folder = os.path.dirname(failed_test)
                        filename = os.path.basename(failed_test)
                        if filename.startswith("test_"):
                            source_name = filename.replace("test_", "")
                            potential_source = os.path.join(folder, source_name)
                            if os.path.exists(potential_source):
                                target_file = potential_source
                                detected_test_file = failed_test
                                print(f"🎯 [Target Acquired] Source: {source_name} | Test: {filename}")
                                break
            if target_file: break
        
        if not target_file:
            print("⚠️ [System] Could not map error to a specific source file. Halting to prevent damage.")
            break

        # 3. DEPLOY AGENT (Surgical Strike)
        print(f"🦅 [System] Deploying Agent to fix: {os.path.basename(target_file)}")
        
        initial_state = {
            "project_root": project_root,
            "target_file": target_file,
            "test_file": detected_test_file,
            "original_code": read_file(target_file),
            "error_log": error_log,
            "category": "",
            "analysis": "",
            "fixed_code": "",
            "retry_count": 0,
            "status": "start"
        }

        result = app.invoke(initial_state)
        
        if result["status"] == "success":
            print(f"✅ [System] Successfully patched {os.path.basename(target_file)}. Re-scanning project...")
        else:
            print(f"❌ [System] Failed to fix {os.path.basename(target_file)}. Moving to next cycle...")
        
        global_attempts += 1

    if global_attempts >= MAX_GLOBAL_RETRIES:
        print(f"\n🛑 [System] Maximum repair cycles reached. Please review remaining errors manually.")