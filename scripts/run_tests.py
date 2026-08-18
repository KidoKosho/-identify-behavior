import os
import sys
import json
import time
import datetime
import subprocess
from pathlib import Path

def main():
    root_dir = Path(__file__).parent.parent
    
    # 1. Generate RUN_ID
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["TEST_RUN_ID"] = run_id
    
    # 2. Setup directories
    results_dir = root_dir / "test-results" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    ai_logs_dir = root_dir / ".ai" / "logs"
    ai_logs_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Build command
    # We use pytest with a custom plugin to generate a basic json report if pytest-json-report is installed, 
    # but since we might not have it, we can just capture stdout and parse it, or use pytest-html
    # To be safe, we'll just run pytest and capture output.
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
    
    command_txt = results_dir / "command.txt"
    command_txt.write_text(" ".join(cmd), encoding="utf-8")
    
    start_time = time.time()
    
    print(f"🚀 Starting test run: {run_id}")
    print(f"📁 Results will be saved to: {results_dir}")
    
    # 4. Run tests and capture output
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(root_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8"
        )
        
        stdout, stderr = process.communicate()
        exit_code = process.returncode
        
    except Exception as e:
        stdout = ""
        stderr = str(e)
        exit_code = 2 # Runner error
        
    end_time = time.time()
    duration = end_time - start_time
    
    # 5. Save outputs
    (results_dir / "stdout.log").write_text(stdout, encoding="utf-8")
    (results_dir / "stderr.log").write_text(stderr, encoding="utf-8")
    
    # 6. Analyze result
    if exit_code == 0:
        status = "PASS"
    elif exit_code == 1:
        status = "FAIL"
    else:
        status = "ERROR"
        
    result_data = {
        "run_id": run_id,
        "status": status,
        "exit_code": exit_code,
        "duration_seconds": round(duration, 2),
        "start_time": datetime.datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.datetime.fromtimestamp(end_time).isoformat()
    }
    
    (results_dir / "result.json").write_text(json.dumps(result_data, indent=4), encoding="utf-8")
    
    # 7. Generate report.md
    report_md = f"# Test Report: {run_id}\n\n"
    report_md += f"**Status:** {status}\n"
    report_md += f"**Duration:** {round(duration, 2)}s\n"
    report_md += f"**Exit Code:** {exit_code}\n\n"
    
    if status != "PASS":
        report_md += "## ❌ Failures Detected\n"
        report_md += "Please check `stdout.log` for details.\n\n"
        
        # Simple failure analyzer: extract tracebacks from stdout
        failures = []
        in_failure = False
        current_fail = []
        for line in stdout.splitlines():
            if line.startswith("___") and "FAILURES" in line:
                continue
            if line.startswith("_ ") and " _" in line:
                in_failure = True
                if current_fail:
                    failures.append("\n".join(current_fail))
                current_fail = [line]
            elif in_failure:
                if line.startswith("================"):
                    in_failure = False
                    if current_fail:
                        failures.append("\n".join(current_fail))
                else:
                    current_fail.append(line)
                    
        if failures:
            for f in failures:
                report_md += f"```text\n{f[:1000]}...\n```\n\n"
            
            # Write to failure analysis log
            analysis_log = ai_logs_dir / "failure-analysis.log.md"
            with open(analysis_log, "a", encoding="utf-8") as f:
                f.write(f"\n## Run {run_id}\n")
                for fail in failures:
                    f.write(f"```text\n{fail}\n```\n")
                    
    (results_dir / "report.md").write_text(report_md, encoding="utf-8")
    
    # 8. Update general test log
    test_log = ai_logs_dir / "test.log.md"
    log_entry = f"- **{run_id}**: {status} (Took {round(duration, 2)}s) - Exit Code {exit_code}. [Report](../../test-results/{run_id}/report.md)\n"
    
    if test_log.exists():
        content = test_log.read_text(encoding="utf-8")
        test_log.write_text(log_entry + content, encoding="utf-8")
    else:
        test_log.write_text("# Automated Test Logs\n\n" + log_entry, encoding="utf-8")
        
    # 9. Update TASKS.md and HANDOFF.md if PASSED
    # For now, we just append a note to HANDOFF.md indicating the latest test result
    handoff_path = root_dir / "HANDOFF.md"
    if handoff_path.exists():
        handoff_content = handoff_path.read_text(encoding="utf-8")
        
        # Simple text replacement or append
        status_line = f"**Latest Test Status (Run {run_id}):** {status}"
        
        if "**Latest Test Status**" in handoff_content:
            import re
            handoff_content = re.sub(r"\*\*Latest Test Status.*", status_line, handoff_content)
        else:
            handoff_content += f"\n\n---\n{status_line}\n"
            
        handoff_path.write_text(handoff_content, encoding="utf-8")

    print(f"✅ Test run completed with status: {status}")
    print(f"📄 Report: {results_dir / 'report.md'}")
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
