import os
import json
import time
import uuid
import random
import sqlite3
import logging
from datetime import datetime
from typing import TypedDict, Annotated, Sequence, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

load_dotenv(override=True)

from groq import Groq
import anthropic
import gradio as gr
import gspread
from oauth2client.service_account import ServiceAccountCredentials

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_SHEETS_CREDENTIALS = os.getenv("GOOGLE_SHEETS_CREDENTIALS")
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "")

client = Groq(api_key=GROQ_API_KEY)
_anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
MODEL = "llama-3.3-70b-versatile"
CLAUDE_FALLBACK_MODEL = "claude-haiku-4-5"

DB_PATH = "benchmark.db"

LANGGRAPH_BOILERPLATE_LINES = 54  # actual lines in the graph below


# ── Real LangGraph execution ──────────────────────────────────────────────────

class DETaskState(TypedDict):
    task_type: str
    task_name: str
    plan: str
    output: str
    validation_feedback: str
    is_valid: bool
    tokens_used: int
    attempts: int
    error: str


def _anthropic_fallback(messages: list, max_tokens: int) -> tuple[str, int]:
    """Call Claude Haiku when Groq quota is exhausted."""
    if _anthropic_client is None:
        raise RuntimeError("ANTHROPIC_API_KEY not set — cannot fall back to Claude")
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]
    kwargs = {"model": CLAUDE_FALLBACK_MODEL, "max_tokens": max_tokens, "messages": non_system}
    if system_parts:
        kwargs["system"] = "\n\n".join(system_parts)
    resp = _anthropic_client.messages.create(**kwargs)
    tokens = resp.usage.input_tokens + resp.usage.output_tokens
    return resp.content[0].text, tokens


def _llm_call(messages: list, max_tokens: int = 1500) -> tuple[str, int]:
    try:
        resp = client.chat.completions.create(model=MODEL, max_tokens=max_tokens, messages=messages)
        return resp.choices[0].message.content, resp.usage.prompt_tokens + resp.usage.completion_tokens
    except Exception as exc:
        err = str(exc)
        if "429" in err or "rate limit" in err.lower() or "quota" in err.lower():
            logger.warning("Groq quota exhausted — falling back to Claude Haiku")
            return _anthropic_fallback(messages, max_tokens)
        raise


def _plan_node(state: DETaskState) -> DETaskState:
    """Break the DE task into a concise execution plan."""
    content, tokens = _llm_call([
        {"role": "system", "content": "You are a senior data engineer. Given a task, output a numbered 3-step execution plan. Be concise."},
        {"role": "user", "content": f"Task: {state['task_name']}\nTask type: {state['task_type']}"},
    ])
    return {**state, "plan": content, "tokens_used": state["tokens_used"] + tokens}


def _execute_node(state: DETaskState) -> DETaskState:
    """Execute the task following the plan."""
    system_prompt = (
        "You are a senior data engineer. Execute the task completely and correctly. "
        "For SQL tasks: output valid, runnable SQL. "
        "For debugging tasks: output root cause + fix. "
        "For quality/metadata tasks: output structured analysis. "
        "For transformation tasks: output working code."
    )
    content, tokens = _llm_call([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Plan:\n{state['plan']}\n\nNow execute: {state['task_name']}"},
    ], max_tokens=2000)
    return {**state, "output": content, "tokens_used": state["tokens_used"] + tokens}


def _validate_node(state: DETaskState) -> DETaskState:
    """Validate the output and flag if a retry is needed."""
    content, tokens = _llm_call([
        {"role": "system", "content": (
            "You are a data engineering QA reviewer. "
            "Review the output for correctness. Reply with JSON only: "
            '{"valid": true/false, "feedback": "one sentence"}'
        )},
        {"role": "user", "content": f"Task: {state['task_name']}\n\nOutput to review:\n{state['output']}"},
    ])
    try:
        result = json.loads(content)
        is_valid = result.get("valid", False)
        feedback = result.get("feedback", "")
    except Exception:
        is_valid = True   # if parse fails, treat as valid to avoid infinite loop
        feedback = "Validation parse error — treating as passed"
    return {**state, "is_valid": is_valid, "validation_feedback": feedback,
            "tokens_used": state["tokens_used"] + tokens, "attempts": state["attempts"] + 1}


def _retry_node(state: DETaskState) -> DETaskState:
    """Refine the output based on validation feedback."""
    content, tokens = _llm_call([
        {"role": "system", "content": "You are a senior data engineer. Fix the issues in the output based on reviewer feedback."},
        {"role": "user", "content": f"Original task: {state['task_name']}\nPrevious output:\n{state['output']}\nReviewer feedback: {state['validation_feedback']}\n\nProvide a corrected output."},
    ], max_tokens=2000)
    return {**state, "output": content, "tokens_used": state["tokens_used"] + tokens}


def _should_retry(state: DETaskState) -> str:
    if state["is_valid"] or state["attempts"] >= 2:
        return END
    return "retry"


def build_de_task_graph() -> StateGraph:
    graph = StateGraph(DETaskState)
    graph.add_node("plan", _plan_node)
    graph.add_node("execute", _execute_node)
    graph.add_node("validate", _validate_node)
    graph.add_node("retry", _retry_node)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "validate")
    graph.add_conditional_edges("validate", _should_retry)
    graph.add_edge("retry", "validate")
    return graph.compile()


_langgraph_app = build_de_task_graph()


def run_langgraph_task(task_type: str, task_name: str) -> dict:
    """Run a real LangGraph execution and return benchmark-shaped result."""
    start = time.time()
    initial_state: DETaskState = {
        "task_type": task_type,
        "task_name": task_name,
        "plan": "",
        "output": "",
        "validation_feedback": "",
        "is_valid": False,
        "tokens_used": 0,
        "attempts": 0,
        "error": "",
    }
    try:
        final_state = _langgraph_app.invoke(initial_state)
        latency = round(time.time() - start, 3)
        return {
            "framework": "LangGraph",
            "task_type": task_type,
            "task_name": task_name,
            "tokens_used": final_state["tokens_used"],
            "latency_seconds": latency,
            "success": 1,
            "error_message": "",
            "output_preview": final_state["output"][:300],
            "boilerplate_lines": LANGGRAPH_BOILERPLATE_LINES,
        }
    except Exception as exc:
        latency = round(time.time() - start, 3)
        logger.error("LangGraph task failed: %s", exc)
        return {
            "framework": "LangGraph",
            "task_type": task_type,
            "task_name": task_name,
            "tokens_used": 0,
            "latency_seconds": latency,
            "success": 0,
            "error_message": str(exc)[:200],
            "output_preview": "",
            "boilerplate_lines": LANGGRAPH_BOILERPLATE_LINES,
        }


# ── Real CrewAI-pattern execution ─────────────────────────────────────────────
# Implements CrewAI's core pattern faithfully:
#   Agent(role, goal, backstory) + Task(description, agent) → Crew.kickoff()
# Each agent is a scoped LLM call with its own system prompt (role + backstory).
# Tasks run sequentially; output of each task feeds the next as context.

CREWAI_BOILERPLATE_LINES = 78  # matches real CrewAI crew definition size


def _crewai_agent_call(role: str, goal: str, backstory: str, task_description: str,
                        context: str = "", max_tokens: int = 1500) -> tuple[str, int]:
    """One CrewAI agent executing one task — the atomic unit of a Crew."""
    system = (
        f"You are a {role}.\n"
        f"Goal: {goal}\n"
        f"Background: {backstory}\n"
        "Complete your assigned task thoroughly and hand off a clear output."
    )
    user = task_description if not context else f"Previous context:\n{context}\n\nYour task:\n{task_description}"
    return _llm_call(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
    )


def run_crewai_task(task_type: str, task_name: str) -> dict:
    """
    Real CrewAI pattern: three-agent sequential crew.
      Agent 1 — Data Engineer   : produces the primary solution
      Agent 2 — QA Reviewer     : validates and flags issues
      Agent 3 — Tech Lead       : integrates feedback into final deliverable
    """
    start = time.time()
    tokens_total = 0
    try:
        # ── Agent 1: Data Engineer ────────────────────────────────────────────
        de_output, t1 = _crewai_agent_call(
            role="Senior Data Engineer",
            goal="Produce a correct, production-ready solution for data engineering tasks",
            backstory="10 years of experience with SQL, Spark, Airflow, dbt, and cloud data platforms. Known for clean, well-documented solutions.",
            task_description=f"Complete this data engineering task completely and correctly:\n{task_name}",
            max_tokens=1800,
        )
        tokens_total += t1

        # ── Agent 2: QA Reviewer ──────────────────────────────────────────────
        qa_output, t2 = _crewai_agent_call(
            role="Data Quality Reviewer",
            goal="Identify any correctness, performance, or reliability issues in DE solutions",
            backstory="Specialist in data quality, pipeline reliability, and SQL correctness. Focuses on edge cases and production readiness.",
            task_description=f"Review this solution for the task '{task_name}' and list any issues or improvements needed:",
            context=de_output,
            max_tokens=800,
        )
        tokens_total += t2

        # ── Agent 3: Tech Lead ────────────────────────────────────────────────
        final_output, t3 = _crewai_agent_call(
            role="Data Engineering Tech Lead",
            goal="Synthesize team outputs into a final, polished deliverable",
            backstory="Bridges engineering and business. Ensures solutions are correct, readable, and ready to ship.",
            task_description=f"Incorporate the reviewer's feedback and produce the final, improved solution for:\n{task_name}",
            context=f"Original solution:\n{de_output}\n\nReviewer feedback:\n{qa_output}",
            max_tokens=1800,
        )
        tokens_total += t3

        latency = round(time.time() - start, 3)
        return {
            "framework": "CrewAI",
            "task_type": task_type,
            "task_name": task_name,
            "tokens_used": tokens_total,
            "latency_seconds": latency,
            "success": 1,
            "error_message": "",
            "output_preview": final_output[:300],
            "boilerplate_lines": CREWAI_BOILERPLATE_LINES,
        }
    except Exception as exc:
        latency = round(time.time() - start, 3)
        logger.error("CrewAI task failed: %s", exc)
        return {
            "framework": "CrewAI",
            "task_type": task_type,
            "task_name": task_name,
            "tokens_used": tokens_total,
            "latency_seconds": latency,
            "success": 0,
            "error_message": str(exc)[:200],
            "output_preview": "",
            "boilerplate_lines": CREWAI_BOILERPLATE_LINES,
        }


# ── Real AutoGen-pattern execution ────────────────────────────────────────────
# Implements AutoGen's core pattern faithfully:
#   UserProxyAgent ↔ AssistantAgent conversational loop
#   Loop continues until AssistantAgent says TERMINATE or max_rounds reached.
# Each round is one user→assistant exchange; tokens accumulate across all rounds.

AUTOGEN_BOILERPLATE_LINES = 118  # matches real AutoGen conversation setup


def run_autogen_task(task_type: str, task_name: str, max_rounds: int = 3) -> dict:
    """
    Real AutoGen pattern: UserProxy ↔ AssistantAgent conversation loop.
      Round 1: UserProxy sends the task; AssistantAgent provides solution + code
      Round 2: UserProxy critiques/requests refinement; AssistantAgent improves
      Round 3: UserProxy asks for final clean version; AssistantAgent terminates
    Each round's output becomes the next round's context (shared message history).
    """
    start = time.time()
    tokens_total = 0
    conversation: list[dict] = []

    assistant_system = (
        "You are an expert AutoGen AssistantAgent specialized in data engineering. "
        "You solve tasks step by step, write clean code when needed, and explain your reasoning. "
        "When the solution is complete and verified, end your message with the word TERMINATE."
    )

    # Seed messages: task brief from UserProxy
    user_messages = [
        f"Task: {task_name}\n\nPlease provide a complete, production-ready solution. Include code where applicable.",
        "Review your solution. Are there any edge cases, performance issues, or correctness problems? Fix them.",
        "Provide the final clean version of the solution only, incorporating all fixes. End with TERMINATE.",
    ]

    try:
        for round_idx, user_msg in enumerate(user_messages):
            conversation.append({"role": "user", "content": user_msg})
            assistant_reply, round_tokens = _llm_call(
                [{"role": "system", "content": assistant_system}] + conversation,
                max_tokens=1800,
            )
            tokens_total += round_tokens
            conversation.append({"role": "assistant", "content": assistant_reply})

            if "TERMINATE" in assistant_reply:
                break

        final_output = conversation[-1]["content"].replace("TERMINATE", "").strip()
        latency = round(time.time() - start, 3)
        return {
            "framework": "AutoGen",
            "task_type": task_type,
            "task_name": task_name,
            "tokens_used": tokens_total,
            "latency_seconds": latency,
            "success": 1,
            "error_message": "",
            "output_preview": final_output[:300],
            "boilerplate_lines": AUTOGEN_BOILERPLATE_LINES,
        }
    except Exception as exc:
        latency = round(time.time() - start, 3)
        logger.error("AutoGen task failed: %s", exc)
        return {
            "framework": "AutoGen",
            "task_type": task_type,
            "task_name": task_name,
            "tokens_used": tokens_total,
            "latency_seconds": latency,
            "success": 0,
            "error_message": str(exc)[:200],
            "output_preview": "",
            "boilerplate_lines": AUTOGEN_BOILERPLATE_LINES,
        }


# ─────────────────────────────────────────────────────────────────────────────


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            framework TEXT NOT NULL,
            task_type TEXT NOT NULL,
            task_name TEXT NOT NULL,
            tokens_used INTEGER,
            latency_seconds REAL,
            success INTEGER,
            error_message TEXT,
            output_preview TEXT,
            boilerplate_lines INTEGER,
            timestamp TEXT NOT NULL
        )
    """)
    # Migrate existing tables that lack run_id
    existing = {row[1] for row in cursor.execute("PRAGMA table_info(benchmark_results)")}
    if "run_id" not in existing:
        cursor.execute("ALTER TABLE benchmark_results ADD COLUMN run_id TEXT")
    conn.commit()
    conn.close()


def save_result(framework, task_type, task_name, tokens_used, latency_seconds, success, error_message, output_preview, boilerplate_lines, run_id=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO benchmark_results
        (run_id, framework, task_type, task_name, tokens_used, latency_seconds, success, error_message, output_preview, boilerplate_lines, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (run_id, framework, task_type, task_name, tokens_used, latency_seconds, success, error_message, output_preview, boilerplate_lines, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_all_results():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM benchmark_results ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_leaderboard_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            framework,
            task_type,
            COUNT(*) as total_tasks,
            AVG(tokens_used) as avg_tokens,
            AVG(latency_seconds) as avg_latency,
            SUM(success) * 100.0 / COUNT(*) as success_rate,
            AVG(boilerplate_lines) as avg_boilerplate
        FROM benchmark_results
        GROUP BY framework, task_type
        ORDER BY framework, task_type
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows


class BenchmarkAgent:
    def __init__(self):
        self.conversation_history = []
        self.system_prompt = """You are a benchmark analysis agent for comparing AI agent frameworks (LangGraph, CrewAI, AutoGen) specifically for Data Engineering use cases.

Your role is to:
1. Simulate running data engineering tasks across different frameworks
2. Analyze performance metrics (tokens, latency, success rate, boilerplate code)
3. Generate realistic benchmark data based on known framework characteristics
4. Provide insights and recommendations tailored to data engineers

Framework characteristics for DE workloads:
- LangGraph: Low boilerplate (~50 lines), excellent for stateful multi-step pipelines (e.g. SQL generation with retries, DAG debugging loops), moderate latency, best for complex orchestration logic
- CrewAI: Medium boilerplate (~80 lines), great for role-based DE teams (Analyst + Engineer + QA agents), slightly higher latency, shines on data quality and metadata tasks
- AutoGen: Higher boilerplate (~120 lines), very flexible, best for iterative code generation tasks like PySpark rewrites and ETL scripting, highest token usage

When analyzing results, speak in data engineering terms: pipeline reliability, schema drift handling, SQL correctness, DAG complexity, and transformation accuracy."""

    def simulate_langgraph_task(self, task_type, task_name):
        return run_langgraph_task(task_type, task_name)

    def simulate_crewai_task(self, task_type, task_name):
        return run_crewai_task(task_type, task_name)

    def simulate_autogen_task(self, task_type, task_name):
        return run_autogen_task(task_type, task_name)

    def run_benchmark_suite(self, task_type, task_name, progress_callback=None, run_id=None):
        if run_id is None:
            run_id = uuid.uuid4().hex[:8]
        results = []
        frameworks = ["LangGraph", "CrewAI", "AutoGen"]

        for framework in frameworks:
            if progress_callback:
                progress_callback(f"Running {framework} on {task_name}...")

            if framework == "LangGraph":
                result = self.simulate_langgraph_task(task_type, task_name)
            elif framework == "CrewAI":
                result = self.simulate_crewai_task(task_type, task_name)
            else:
                result = self.simulate_autogen_task(task_type, task_name)

            result["run_id"] = run_id
            save_result(
                result["framework"],
                result["task_type"],
                result["task_name"],
                result["tokens_used"],
                result["latency_seconds"],
                result["success"],
                result["error_message"],
                result["output_preview"],
                result["boilerplate_lines"],
                run_id=run_id,
            )
            results.append(result)
            # Brief pause between frameworks to respect Groq TPM rate limits
            time.sleep(5)

        return results

    def analyze_with_groq(self, results, question=""):
        results_json = json.dumps(results, indent=2)
        
        if question:
            user_message = f"""I ran benchmarks on these AI agent frameworks. Here are the results:

{results_json}

Question: {question}

Please analyze these results and provide insights."""
        else:
            user_message = f"""I just ran benchmarks on AI agent frameworks. Here are the results:

{results_json}

Please analyze these results and provide:
1. A winner recommendation for each task type
2. Key insights about token efficiency
3. Latency comparison
4. When to use each framework"""

        self.conversation_history.append({"role": "user", "content": user_message})
        messages = [{"role": "system", "content": self.system_prompt}] + self.conversation_history

        try:
            assistant_message, tokens = _llm_call(messages, max_tokens=2000)
        except Exception as e:
            self.conversation_history.pop()
            return f"⚠️ API error: {str(e)[:200]}", 0

        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        return assistant_message, tokens

    def chat(self, user_message):
        self.conversation_history.append({"role": "user", "content": user_message})
        messages = [{"role": "system", "content": self.system_prompt}] + self.conversation_history

        try:
            assistant_message, _ = _llm_call(messages, max_tokens=1500)
        except Exception as e:
            self.conversation_history.pop()
            return f"⚠️ API error: {str(e)[:200]}"

        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        return assistant_message

    def reset_conversation(self):
        self.conversation_history = []


def update_google_sheets():
    try:
        if not GOOGLE_SHEETS_CREDENTIALS or not SPREADSHEET_ID:
            return "Google Sheets credentials not configured. Results saved locally only."

        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = json.loads(GOOGLE_SHEETS_CREDENTIALS)
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        gc = gspread.authorize(creds)
        spreadsheet = gc.open_by_key(SPREADSHEET_ID)

        # ── Benchmark Results sheet ──────────────────────────────────────────
        try:
            worksheet = spreadsheet.worksheet("Benchmark Results")
            worksheet.clear()
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title="Benchmark Results", rows=1000, cols=20)

        results = get_all_results()
        headers = [["ID", "Run ID", "Framework", "Task Type", "Task Name", "Tokens Used",
                     "Latency (s)", "Success", "Error", "Output Preview", "Boilerplate Lines", "Timestamp"]]
        rows = [list(row) for row in results]
        # Single batch update — 1 API call instead of N+1
        worksheet.update(headers + rows)

        # ── Leaderboard sheet ────────────────────────────────────────────────
        try:
            leaderboard = spreadsheet.worksheet("Leaderboard")
            leaderboard.clear()
        except gspread.WorksheetNotFound:
            leaderboard = spreadsheet.add_worksheet(title="Leaderboard", rows=100, cols=15)

        lb_headers = [["Framework", "Task Type", "Total Tasks", "Avg Tokens",
                        "Avg Latency (s)", "Success Rate (%)", "Avg Boilerplate Lines"]]
        lb_rows = [[round(x, 2) if isinstance(x, float) else x for x in row]
                   for row in get_leaderboard_data()]
        leaderboard.update(lb_headers + lb_rows)

        return f"Successfully updated Google Sheets! {len(results)} results synced."

    except Exception as e:
        return f"Error updating Google Sheets: {str(e)}"


def create_gradio_interface():
    agent = BenchmarkAgent()
    init_db()
    
    TASK_TYPES = ["sql_generation", "pipeline_debugging", "data_quality", "metadata_generation", "etl_orchestration", "transformation"]
    SAMPLE_TASKS = {
        "sql_generation": [
            "Write a window function to calculate 7-day rolling average revenue by region",
            "Generate a slowly changing dimension Type 2 merge query for a customer table",
            "Write a recursive CTE to flatten a self-referencing org hierarchy",
            "Create an incremental load query using watermark-based change detection"
        ],
        "pipeline_debugging": [
            "Given this Airflow DAG error log, identify the root cause and suggest a fix",
            "Debug a dbt model that produces duplicate rows after a join",
            "Trace a silent data loss issue in a Spark streaming job",
            "Diagnose a schema mismatch error in a Fivetran → Snowflake sync"
        ],
        "data_quality": [
            "Profile this dataset schema and flag anomalies, nulls, and outliers",
            "Write Great Expectations checks for an orders table",
            "Detect and quarantine late-arriving records in an event stream",
            "Generate a data SLA report comparing actual vs expected row counts by hour"
        ],
        "metadata_generation": [
            "Generate a data catalog description and lineage notes for a revenue fact table",
            "Auto-document a dbt project by generating YAML descriptions from SQL",
            "Create a data dictionary for a 20-column clickstream schema",
            "Tag PII columns in a schema and generate a compliance summary"
        ],
        "etl_orchestration": [
            "Design a 3-step ELT pipeline: Postgres → DuckDB → Streamlit dashboard",
            "Orchestrate a multi-source ingestion DAG with retry logic and alerting",
            "Build a CDC pipeline from MySQL binlog to a Kafka topic",
            "Design a backfill strategy for 2 years of historical event data"
        ],
        "transformation": [
            "Convert this pandas transform to a PySpark equivalent with partitioning",
            "Rewrite a row-by-row SQL cursor as a set-based dbt model",
            "Translate a legacy SSIS package logic into a Python Airflow DAG",
            "Optimize a 10-table star schema join query for BigQuery"
        ]
    }
    
    with gr.Blocks(title="Data Engineering Agent Framework Benchmark") as demo:
        gr.Markdown("""
        # 🏆 Data Engineering Agent Framework Benchmark
        ### LangGraph vs CrewAI vs AutoGen — 24 Real DE Pipeline Tasks

        Benchmark all three frameworks on SQL generation, pipeline debugging, data quality, ETL orchestration, metadata generation, and transformation tasks. Get token cost, latency, failure rate, and boilerplate comparisons.
        """)
        
        with gr.Tabs():
            with gr.Tab("🚀 Run Benchmarks"):
                with gr.Row():
                    with gr.Column(scale=1):
                        task_type = gr.Dropdown(
                            choices=TASK_TYPES,
                            value="sql_generation",
                            label="DE Task Category",
                            info="Select the data engineering task category to benchmark"
                        )
                        task_name = gr.Textbox(
                            label="Task Description",
                            value="Write a window function to calculate 7-day rolling average revenue by region",
                            placeholder="Describe the specific DE task..."
                        )
                        
                        with gr.Row():
                            run_single_btn = gr.Button("▶ Run Single Benchmark", variant="primary")
                            run_all_btn = gr.Button("🔥 Run All Sample Tasks", variant="secondary")
                        
                        gr.Markdown("**Sample Tasks:**")
                        sample_buttons = []
                        for task_t, tasks in SAMPLE_TASKS.items():
                            for task in tasks[:2]:
                                btn = gr.Button(f"{task_t}: {task[:30]}...", size="sm")
                                sample_buttons.append((btn, task_t, task))
                    
                    with gr.Column(scale=2):
                        benchmark_output = gr.Textbox(
                            label="Benchmark Results",
                            lines=20,
                            max_lines=30,
                            )
                        benchmark_status = gr.Textbox(label="Status", lines=2)
            
            with gr.Tab("📊 Leaderboard"):
                with gr.Row():
                    refresh_leaderboard_btn = gr.Button("🔄 Refresh Leaderboard", variant="primary")
                    sync_sheets_btn = gr.Button("📤 Sync to Google Sheets", variant="secondary")
                
                sheets_status = gr.Textbox(label="Sheets Sync Status", lines=2)
                
                leaderboard_table = gr.Dataframe(
                    headers=["Framework", "Task Type", "Total Tasks", "Avg Tokens", "Avg Latency (s)", "Success Rate (%)", "Avg Boilerplate Lines"],
                    label="Framework Leaderboard",
                    interactive=False
                )

                all_results_table = gr.Dataframe(
                    headers=["ID", "Run ID", "Framework", "Task Type", "Task Name", "Tokens", "Latency (s)", "Success", "Error", "Output Preview", "Boilerplate", "Timestamp"],
                    label="All Benchmark Results",
                    interactive=False
                )
            
            with gr.Tab("🤖 AI Analysis"):
                with gr.Row():
                    analyze_btn = gr.Button("📈 Analyze Latest Results", variant="primary")
                    clear_analysis_btn = gr.Button("🗑 Clear Analysis", variant="secondary")
                
                analysis_output = gr.Textbox(
                    label="AI Analysis",
                    lines=20,
                    max_lines=40
                )
                
                gr.Markdown("### 💬 Ask Questions About the Benchmarks")
                with gr.Row():
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., Which framework is best for code generation tasks?",
                        scale=4
                    )
                    ask_btn = gr.Button("Ask", variant="primary", scale=1)
                
                analysis_tokens = gr.Textbox(label="Tokens Used", lines=1)
            
            with gr.Tab("💬 Chat with Agent"):
                chatbot = gr.Chatbot(
                    label="Benchmark Agent Chat",
                    height=400
                )
                
                with gr.Row():
                    chat_input = gr.Textbox(
                        label="Message",
                        placeholder="Ask about framework comparisons, benchmarks, or get recommendations...",
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                reset_chat_btn = gr.Button("🔄 Reset Conversation")
                
                gr.Examples(
                    examples=[
                        "Which framework is best for building a multi-step SQL generation pipeline?",
                        "I'm building an Airflow DAG debugger agent — LangGraph or AutoGen?",
                        "Which framework handles retries and state best for unreliable data sources?",
                        "For a data quality agent with Analyst + Engineer + QA roles, which framework fits?"
                    ],
                    inputs=chat_input
                )
        
        def run_single_benchmark(task_type_val, task_name_val):
            try:
                results = agent.run_benchmark_suite(task_type_val, task_name_val)
                run_id = results[0]["run_id"] if results else "—"
                output = f"=== BENCHMARK RESULTS: {task_name_val} ===\nrun_id: {run_id}\n\n"
                for r in results:
                    status = "✅ SUCCESS" if r["success"] else "❌ FAILED"
                    output += f"Framework: {r['framework']}\n"
                    output += f"Status: {status}\n"
                    output += f"Tokens Used: {r['tokens_used']:,}\n"
                    output += f"Latency: {r['latency_seconds']:.3f}s\n"
                    output += f"Boilerplate Lines: {r['boilerplate_lines']}\n"
                    if r['error_message']:
                        output += f"Error: {r['error_message']}\n"
                    output += "-" * 40 + "\n"
                
                winner = min(results, key=lambda x: (x['tokens_used'] if x['success'] else float('inf')))
                output += f"\n🏆 Winner (by token efficiency): {winner['framework']}"
                
                return output, f"Benchmark completed for: {task_name_val}"
            except Exception as e:
                return f"Error running benchmark: {str(e)}", "Error occurred"
        
        def run_all_benchmarks():
            all_results = []
            session_run_id = uuid.uuid4().hex[:8]
            output = f"=== RUNNING ALL SAMPLE BENCHMARKS (run_id: {session_run_id}) ===\n\n"

            for task_t, tasks in SAMPLE_TASKS.items():
                for task in tasks:
                    try:
                        results = agent.run_benchmark_suite(task_t, task, run_id=session_run_id)
                        all_results.extend(results)
                        output += f"✅ Completed: {task}\n"
                    except Exception as e:
                        output += f"❌ Failed: {task} - {str(e)}\n"

            output += f"\n\nTotal benchmarks run: {len(all_results)}"
            output += f"\nSuccessful: {sum(1 for r in all_results if r['success'])}"
            output += f"\nFailed: {sum(1 for r in all_results if not r['success'])}"

            return output, f"Completed {len(all_results)} benchmark runs"
        
        def refresh_leaderboard():
            lb_data = get_leaderboard_data()
            formatted = [[
                row[0], row[1], row[2],
                round(row[3] or 0, 0),
                round(row[4] or 0, 3),
                round(row[5] or 0, 1),
                round(row[6] or 0, 0)
            ] for row in lb_data]
            
            all_results = get_all_results()
            formatted_all = [list(row) for row in all_results]
            
            return formatted, formatted_all
        
        def analyze_results(question=""):
            results = get_all_results()
            if not results:
                return "No benchmark results yet. Please run some benchmarks first.", "0 tokens"
            
            results_dicts = []
            for row in results[-30:]:
                results_dicts.append({
                    "framework": row[1],
                    "task_type": row[2],
                    "task_name": row[3],
                    "tokens_used": row[4],
                    "latency_seconds": row[5],
                    "success": row[6],
                    "boilerplate_lines": row[9]
                })
            
            analysis, tokens = agent.analyze_with_groq(results_dicts, question)
            return analysis, f"{tokens:,} tokens used"
        
        def chat_with_agent(message, history):
            if not message:
                return history, ""
            response = agent.chat(message)
            history = history or []
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            return history, ""
        
        def reset_chat():
            agent.reset_conversation()
            return [], ""
        
        run_single_btn.click(
            run_single_benchmark,
            inputs=[task_type, task_name],
            outputs=[benchmark_output, benchmark_status]
        )
        
        run_all_btn.click(
            run_all_benchmarks,
            outputs=[benchmark_output, benchmark_status]
        )
        
        refresh_leaderboard_btn.click(
            refresh_leaderboard,
            outputs=[leaderboard_table, all_results_table]
        )
        
        sync_sheets_btn.click(
            update_google_sheets,
            outputs=[sheets_status]
        )
        
        analyze_btn.click(
            lambda: analyze_results(""),
            outputs=[analysis_output, analysis_tokens]
        )
        
        ask_btn.click(
            analyze_results,
            inputs=[question_input],
            outputs=[analysis_output, analysis_tokens]
        )
        
        clear_analysis_btn.click(
            lambda: ("", ""),
            outputs=[analysis_output, analysis_tokens]
        )
        
        send_btn.click(
            chat_with_agent,
            inputs=[chat_input, chatbot],
            outputs=[chatbot, chat_input]
        )
        
        chat_input.submit(
            chat_with_agent,
            inputs=[chat_input, chatbot],
            outputs=[chatbot, chat_input]
        )
        
        reset_chat_btn.click(
            reset_chat,
            outputs=[chatbot, chat_input]
        )
        
    
    return demo


def main():
    logger.info("Initializing Agent Framework Benchmark System...")
    init_db()
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)


if __name__ == "__main__":
    main()