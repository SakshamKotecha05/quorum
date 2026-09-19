"""Kill a real CLI process mid-research, then verify durable completed work."""
import json
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quorum.core import Status, Store

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "What makes multi-agent orchestration hard to run in production?"


def main():
    with tempfile.TemporaryDirectory(prefix="quorum-resume-") as tmp:
        store = Store(Path(tmp) / "resume.db")
        command = [sys.executable, "-m", "quorum", "--db", str(Path(tmp) / "resume.db"),
                   "run", QUESTION, "--mock", "--unthrottled", "--mock-latency", "0.4",
                   "--resume", "crash-demo"]
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and process.poll() is None:
                nodes = store.load_nodes("crash-demo")
                if (any(n.role == "researcher" and n.status is Status.DONE for n in nodes.values())
                        and any(n.status is Status.RUNNING for n in nodes.values())):
                    break
                time.sleep(0.01)
            else:
                raise RuntimeError("did not observe completed and in-flight research")
            process.kill()
            process.wait(timeout=5)
            before = store.load_nodes("crash-demo")
            completed = {nid: n.output for nid, n in before.items() if n.status is Status.DONE}
            assert any(n.status is Status.RUNNING for n in before.values())
            print("At process kill:", dict(Counter(n.status.value for n in before.values())))
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
            if result.returncode:
                raise RuntimeError(result.stderr + result.stdout)
            after = store.load_nodes("crash-demo")
            assert all(n.status is Status.DONE for n in after.values())
            assert {nid: after[nid].output for nid in completed} == completed
            calls = Counter(s["node_id"] for s in store.trace("crash-demo"))
            assert all(calls[nid] == 1 for nid in completed)
            print(f"Resumed: {len(after)} completed nodes; {len(completed)} completed nodes reused.")
            print("Previously completed model calls replayed: 0")
            print("Workflow state survives; governor accounting restarts in the new process.")
            return {"before": dict(Counter(n.status.value for n in before.values())),
                    "completed": len(after), "reused": len(completed), "replayed": 0}
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            store.db.close()


if __name__ == "__main__":
    main()
