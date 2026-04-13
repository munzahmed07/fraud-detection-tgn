"""
run_pipeline.py — Full Pipeline Runner
AI-Powered Mule Account and Financial Fraud Network Detection System

Runs all 9 modules in sequence with a single command:
    python run_pipeline.py

Options:
    python run_pipeline.py --sample 200000   # use 200k sample (default)
    python run_pipeline.py --full            # use full 6.3M dataset
    python run_pipeline.py --from 3          # resume from module 3
    python run_pipeline.py --skip-dashboard  # skip launching dashboard
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime

# ── Colors for terminal output ────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def log(msg, color=RESET):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{color}[{timestamp}] {msg}{RESET}")

def success(msg): log(f"✓ {msg}", GREEN + BOLD)
def error(msg):   log(f"✗ {msg}", RED + BOLD)
def info(msg):    log(f"  {msg}", BLUE)
def warn(msg):    log(f"⚠ {msg}", YELLOW)

def separator(title=""):
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{BLUE + BOLD}{'─' * pad} {title} {'─' * pad}{RESET}\n")
    else:
        print(f"\n{BLUE}{'─' * width}{RESET}\n")


# ── Module definitions ────────────────────────────────────────────

MODULES = [
    {
        "id":          1,
        "name":        "Data Ingestion & Feature Engineering",
        "file":        None,   # Called internally by Module 2
        "description": "Loads PaySim dataset and engineers 17 temporal features",
        "output":      None,
        "skip":        True,   # Runs as part of graph_construction
    },
    {
        "id":          2,
        "name":        "Transaction Graph Construction",
        "file":        "graph_construction.py",
        "description": "Builds 374k node graph, exports PyG data",
        "output":      "output/graph_data.npz",
        "skip":        False,
    },
    {
        "id":          3,
        "name":        "Temporal Graph Neural Network (TGN)",
        "file":        "tgnn_model.py",
        "description": "Trains TGN with memory module + temporal features",
        "output":      "output/fraud_probabilities.npy",
        "skip":        False,
    },
    {
        "id":          4,
        "name":        "Velocity & Multi-Hop Detection",
        "file":        "velocity_model.py",
        "description": "Dwell time analysis, fan-in/fan-out, fraud path tracing",
        "output":      "output/velocity_features.csv",
        "skip":        False,
    },
    {
        "id":          5,
        "name":        "Fraud Ring Detection",
        "file":        "fraud_ring_detector.py",
        "description": "Louvain community detection — finds coordinated rings",
        "output":      "output/community_scores.csv",
        "skip":        False,
    },
    {
        "id":          6,
        "name":        "Explainable AI (XAI)",
        "file":        "xai_explainer.py",
        "description": "SHAP analysis + auto-generated fraud narratives",
        "output":      "output/fraud_explanations.json",
        "skip":        False,
    },
    {
        "id":          7,
        "name":        "STR Report Generator",
        "file":        "str_generator.py",
        "description": "Auto-generates PDF Suspicious Transaction Reports",
        "output":      "output/reports",
        "skip":        False,
    },
    {
        "id":          8,
        "name":        "Evaluation Metrics & Ablation",
        "file":        "eval_metrics.py",
        "description": "Confusion matrix, PR curve, ablation study, baselines",
        "output":      "output/eval/results_summary.json",
        "skip":        False,
    },
    {
        "id":          9,
        "name":        "AML Dashboard",
        "file":        "dashboard.py",
        "description": "Streamlit dashboard — runs at localhost:8501",
        "output":      None,
        "skip":        False,
        "is_dashboard": True,
    },
]


# ── Runner ────────────────────────────────────────────────────────

def run_module(module, python_exe="python"):
    """Run a single module script and return success/failure."""
    if module.get("skip"):
        info(f"Module {module['id']} runs as part of Module 2 — skipping standalone run")
        return True

    script = module["file"]
    if not os.path.exists(script):
        error(f"Script not found: {script}")
        return False

    # Dashboard runs differently
    if module.get("is_dashboard"):
        info("Launching Streamlit dashboard...")
        info("Dashboard will open at: http://localhost:8501")
        info("Press Ctrl+C to stop the dashboard when done")
        try:
            subprocess.run([python_exe, "-m", "streamlit", "run", script],
                           check=True)
        except KeyboardInterrupt:
            info("Dashboard stopped by user")
        return True

    start = time.time()
    try:
        result = subprocess.run(
            [python_exe, script],
            capture_output=False,
            text=True,
            check=True
        )
        elapsed = time.time() - start
        success(f"Module {module['id']} completed in {elapsed:.1f}s")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        error(f"Module {module['id']} FAILED after {elapsed:.1f}s")
        return False

    except Exception as e:
        error(f"Unexpected error in Module {module['id']}: {e}")
        return False


def check_outputs():
    """Check which modules have already been run."""
    completed = []
    for m in MODULES:
        if m.get("skip") or m.get("output") is None:
            continue
        out = m["output"]
        if os.path.exists(out):
            completed.append(m["id"])
    return completed


def print_status(completed_ids):
    """Print current pipeline status."""
    separator("PIPELINE STATUS")
    for m in MODULES:
        if m.get("skip"):
            status = f"{YELLOW}PART OF MODULE 2{RESET}"
        elif m.get("is_dashboard"):
            status = f"{BLUE}RUNS LAST{RESET}"
        elif m["id"] in completed_ids:
            status = f"{GREEN}DONE{RESET}"
        else:
            status = f"{YELLOW}PENDING{RESET}"
        print(f"  Module {m['id']}: {m['name']:<45} [{status}]")
    print()


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the complete fraud detection pipeline"
    )
    parser.add_argument("--from",    type=int, default=2,
                        dest="start_from",
                        help="Start from this module number (default: 2)")
    parser.add_argument("--only",    type=int, default=None,
                        help="Run only this module number")
    parser.add_argument("--full",    action="store_true",
                        help="Use full 6.3M PaySim dataset (slower)")
    parser.add_argument("--skip-dashboard", action="store_true",
                        help="Skip launching the dashboard at the end")
    parser.add_argument("--status",  action="store_true",
                        help="Show pipeline status and exit")
    args = parser.parse_args()

    # ── Header ──
    separator("AI-POWERED FRAUD DETECTION SYSTEM")
    print(f"{BOLD}  Pipeline Runner{RESET}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Working dir: {os.getcwd()}")
    print()

    # ── Status check ──
    completed_ids = check_outputs()
    print_status(completed_ids)

    if args.status:
        return

    # ── Dataset mode ──
    if args.full:
        warn("Full dataset mode: updating graph_construction.py to use all 6.3M rows...")
        # Read and update the sample size
        if os.path.exists("graph_construction.py"):
            with open("graph_construction.py", "r") as f:
                content = f.read()
            content = content.replace(
                'df_raw.sample(n=200000, random_state=42).reset_index(drop=True)',
                'df_raw  # Full dataset — no sampling'
            )
            with open("graph_construction.py", "w") as f:
                f.write(content)
            info("graph_construction.py updated to use full dataset")

    # ── Determine which modules to run ──
    if args.only:
        modules_to_run = [m for m in MODULES if m["id"] == args.only]
        separator(f"RUNNING MODULE {args.only} ONLY")
    else:
        modules_to_run = [m for m in MODULES
                          if m["id"] >= args.start_from]
        if args.skip_dashboard:
            modules_to_run = [m for m in modules_to_run
                              if not m.get("is_dashboard")]
        separator(f"RUNNING MODULES {args.start_from} TO {modules_to_run[-1]['id']}")

    # ── Run modules ──
    total     = len([m for m in modules_to_run if not m.get("skip")])
    completed = 0
    failed    = []
    pipeline_start = time.time()

    for module in modules_to_run:
        if module.get("skip"):
            continue

        separator(f"MODULE {module['id']}: {module['name'].upper()}")
        info(module["description"])
        print()

        success_flag = run_module(module)

        if success_flag:
            completed += 1
        else:
            failed.append(module["id"])
            error(f"Module {module['id']} failed. Stopping pipeline.")
            error("Fix the error above and re-run with:")
            error(f"  python run_pipeline.py --from {module['id']}")
            break

    # ── Final summary ──
    elapsed_total = time.time() - pipeline_start
    separator("PIPELINE COMPLETE")

    print(f"  Modules run:    {completed} / {total}")
    print(f"  Total time:     {elapsed_total/60:.1f} minutes")
    print(f"  Finished:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    if failed:
        error(f"Failed modules: {failed}")
        print()
        warn("To resume from the failed module:")
        warn(f"  python run_pipeline.py --from {failed[0]}")
    else:
        success("All modules completed successfully!")
        print()
        print(f"{GREEN}{BOLD}  Output files:{RESET}")
        outputs = [
            ("output/transactions_processed.csv", "Processed transactions"),
            ("output/graph_data.npz",              "PyG graph data"),
            ("output/fraud_probabilities.npy",     "TGN fraud scores"),
            ("output/tgn_fraud_scores.csv",        "Account risk scores"),
            ("output/community_scores.csv",        "Fraud ring scores"),
            ("output/fraud_narratives.json",       "AI fraud narratives"),
            ("output/reports/",                    "PDF STR reports"),
            ("output/eval/",                       "Evaluation charts"),
        ]
        for path, desc in outputs:
            exists = "✓" if os.path.exists(path) else "✗"
            color  = GREEN if os.path.exists(path) else RED
            print(f"  {color}{exists}{RESET}  {desc:<40} {path}")

        print()
        if not args.skip_dashboard:
            info("To launch the dashboard:")
            info("  streamlit run dashboard.py")
        info("To view results:")
        info("  Check output/ folder for all generated files")


if __name__ == "__main__":
    main()
