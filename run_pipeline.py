from dotenv import load_dotenv
load_dotenv()

from ingestion.orchestrator import submit_batch

BASE = "/Users/praveen/Desktop/Vsoln/enterprise-rag-agents/tests"
files = [
    f"{BASE}/Apple_2025-10-31.htm",
    f"{BASE}/Goldman_Sachs_2026-02-25.htm",
    f"{BASE}/JPMorgan_Chase_2026-02-13.htm",
]

results = submit_batch(files)
for path, task_id in results.items():
    status = "QUEUED" if task_id else "SKIPPED"
    print(f"{status}: {path.split('/')[-1]} → {task_id}")
