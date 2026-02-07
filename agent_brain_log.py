import os
import sys
import shutil
import subprocess
import argparse
import json
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
MAX_GLOBAL_RETRIES = 5
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
BASE_URL = "https://models.inference.ai.azure.com"

if not GITHUB_TOKEN:
    # We output error as JSON even for crashes
    print(json.dumps({"type": "critical_error", "message": "MISSING GITHUB_TOKEN in .env file!"}))
    sys.exit(1)

# --- LOGGING HELPER ---
def log_event(event_type, message, data=None):
    """Outputs a structured JSON log line for Docker/API to consume."""
    log_entry = {
        "type": event_type,
        "message": message,
        "data": data or {}
    }
    print(json.dumps(log_entry), flush=True)

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
        log_event("system", "Creating fresh requirements.txt", {"path": target_req})
        with open(target_req, "w") as f: f.write("")
    return target_req

def add_to_requirements(project_root, package_name):
    req_file = os.path.join(project_root, "requirements.txt")
    current_reqs = read_file(req_file)
    if package_name not in current_reqs:
        with open(req_file, "a", encoding="utf-8") as f:
            f.write(f"\n{package_name}")
        log_event("manifest_update", f"Added {package_name} to requirements", {"package": package_name})

def install_package(package_name):
    log_event("install_start", f"Installing package: {package_name}", {"package": package_name})
    result = subprocess.run([sys.executable, "-m", "pip", "install", package_name], capture_output=True, text=True)
    if result.returncode == 0:
        log_event("install_success", f"Successfully installed {package_name}", {"package": package_name})
        return True
    else:
        log_event("install_fail", f"Failed to install {package_name}", {"error": result.stderr})
        return False

# --- 3. MODELS ---
architect_model = ChatOpenAI(model="gpt-4o-mini", api_key=GITHUB_TOKEN, base_url=BASE_URL, temperature=0)
coder_model = ChatOpenAI(model="gpt-4o-mini", api_key=GITHUB_TOKEN, base_url=BASE_URL, temperature=0)

# --- 4. NODES ---

def analyze_node(state: AgentState):
    filename = os.path.basename(state['target_file'])
    log_event("agent_action", f"Architect analyzing {filename}", {"attempt": state['retry_count'] + 1})
    
    router_prompt = ChatPromptTemplate.from_template(
        "Analyze this error: {error}\nClassify as 'dependency' or 'logic'. Return ONE WORD."
    )
    chain = router_prompt | architect_model | StrOutputParser()
    raw_category = chain.invoke({"error": state["error_log"]}).strip().lower()
    
    category = "dependency" if "dependency" in raw_category or "module" in raw_category else "logic"
    log_event("agent_decision", f"Router classified issue as {category.upper()}", {"category": category})
    
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
    log_event("agent_action", "SysAdmin identifying missing library")
    prompt = ChatPromptTemplate.from_template("Extract missing python package name from: {error}. Return ONLY name.")
    chain = prompt | coder_model | StrOutputParser()
    pkg_name = chain.invoke({"error": state["error_log"]}).strip().replace("'", "").replace('"', "")
    
    add_to_requirements(state["project_root"], pkg_name)
    success = install_package(pkg_name)
    
    if success: return {"fixed_code": state["original_code"]}
    else: return {"status": "failed", "retry_count": state["retry_count"] + 1, "error_log": f"Failed install: {pkg_name}"}

def fix_node(state: AgentState):
    filename = os.path.basename(state['target_file'])
    log_event("agent_action", f"Coder applying fix to {filename}")
    
    code_to_fix = state["fixed_code"] if state["retry_count"] > 0 else state["original_code"]
    
    prompt = ChatPromptTemplate.from_template(
        "Fix the code.\nBROKEN CODE:\n{code}\nANALYSIS:\n{analysis}\nReturn ONLY the full code."
    )
    chain = prompt | coder_model | StrOutputParser()
    fixed_code = chain.invoke({"code": code_to_fix, "analysis": state["analysis"]})
    clean_code = fixed_code.replace("```python", "").replace("```", "").strip()
    return {"fixed_code": clean_code}

def test_node(state: AgentState):
    if state["fixed_code"]:
        write_file(state["target_file"], state["fixed_code"]) 
    
    # SURGICAL TEST
    target_test = state["test_file"] if state["test_file"] and os.path.exists(state["test_file"]) else state["project_root"]
    test_name = os.path.basename(target_test)
    
    log_event("test_start", f"Verifying fix with {test_name}", {"test_file": test_name})
    result = subprocess.run([sys.executable, "-m", "pytest", target_test], capture_output=True, text=True)
    
    if result.returncode == 0:
        log_event("test_result", "Test Passed", {"status": "success", "file": test_name})
        return {"status": "success"}
    else:
        log_event("test_result", "Test Failed", {"status": "failed", "file": test_name})
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

    log_event("system_start", f"Initializing Autonomous Maintainer", {"root": project_root})
    normalize_project(project_root)

    global_attempts = 0
    
    while global_attempts < MAX_GLOBAL_RETRIES:
        log_event("global_scan_start", f"Running full system diagnostics (Round {global_attempts + 1})")
        
        # 1. Run GLOBAL Tests
        global_test = subprocess.run([sys.executable, "-m", "pytest", project_root], capture_output=True, text=True)
        
        if global_test.returncode == 0:
            log_event("system_complete", "GRAND VICTORY: All tests passed", {"success": True})
            break 
        
        log_event("scan_fail", "Failures Detected. Identifying target...", {"error_preview": global_test.stdout[:100]})
        error_log = global_test.stdout
        
        # 2. INTELLIGENT TARGETING
        target_file = None
        detected_test_file = None

        for line in error_log.splitlines():
            if ("FAILED" in line or "ERROR" in line) and ".py" in line:
                parts = line.split("::")[0].split(" ")
                for part in parts:
                    if part.endswith(".py"):
                        failed_test = part
                        folder = os.path.dirname(failed_test)
                        filename = os.path.basename(failed_test)
                        if filename.startswith("test_"):
                            source_name = filename.replace("test_", "")
                            potential_source = os.path.join(folder, source_name)
                            if os.path.exists(potential_source):
                                target_file = potential_source
                                detected_test_file = failed_test
                                log_event("target_acquired", f"Targeting {source_name}", {"source": source_name, "test": filename})
                                break
            if target_file: break
        
        if not target_file:
            log_event("system_error", "Could not map error to specific source file. Halting.", {"fatal": True})
            break

        # 3. DEPLOY AGENT
        log_event("agent_deploy", f"Deploying Agent to fix {os.path.basename(target_file)}")
        
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
            log_event("fix_success", f"Successfully patched {os.path.basename(target_file)}")
        else:
            log_event("fix_fail", f"Failed to fix {os.path.basename(target_file)}")
        
        global_attempts += 1

    if global_attempts >= MAX_GLOBAL_RETRIES:
        log_event("system_halt", "Maximum repair cycles reached. Human Intervention Required.", {"limit": MAX_GLOBAL_RETRIES})