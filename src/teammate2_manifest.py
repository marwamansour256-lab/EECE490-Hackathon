"""
Run manifest for teammate 2's scripts (AGENTS.md: "Record package versions and input hashes in a
run manifest for submission").

Rebuilds every teammate-2 output in order, then records the Python and package versions, the
SHA-256 of every input and output, and the commands used.

    python src/teammate2_manifest.py            # rerun everything, then write the manifest
    python src/teammate2_manifest.py --no-run   # only hash what is already on disk

Output: reports/teammate2_manifest.json
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next(p for p in [_here, *_here.parents] if (p / "data" / "stories_transactions.csv").exists())

STEPS = [
    ("src/external_factor_test.py", ["reports/external_factor_test.csv", "reports/external_factor_test.json",
                                     "reports/figures/external_factor_holdout.png", "app/evidence_data.js"]),
    ("src/why_drivers.py", ["reports/why_drivers.json", "app/why_data.js", "reports/figures/why_drivers.png"]),
    ("src/correlation_scan.py", ["reports/correlation_scan_pairs.csv", "reports/correlation_scan.json",
                                 "reports/figures/correlation_scan.png"]),
    ("src/metric_reconciliation.py", ["reports/metric_reconciliation.md", "reports/metric_reconciliation.json"]),
    ("src/day_ledger.py", ["app/day_data.js", "reports/day_ledger.csv"]),
]
INPUTS = ["data/stories_transactions.csv", "data/external/beirut_weather_2026H1.json",
          "data/external/batroun_weather_2026H1.json", "data/external/broummana_weather_2026H1.json",
          "data/external/external_events.json"]
STATIC = ["app/planner.html", "src/pilot_report.py", "data/pilot/pilot_log_template.csv"]


def sha(path):
    p = ROOT / path
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def version(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def main():
    runs = []
    if "--no-run" not in sys.argv:
        env = dict(os.environ, MPLBACKEND="Agg")
        for script, _ in STEPS:
            r = subprocess.run([sys.executable, "-I", "-X", "utf8", str(ROOT / script)], cwd=ROOT, env=env,
                               capture_output=True, text=True)
            runs.append({"script": script, "returncode": r.returncode, "last_line": (r.stdout.strip().splitlines() or [""])[-1]})
            print(f"{script}: exit {r.returncode}")
            if r.returncode != 0:
                print(r.stderr[-2000:])
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "owner": "Claude Code teammate 2 (Marwa)",
        "python": sys.version.split()[0], "platform": platform.platform(),
        "packages": {m: version(m) for m in ["numpy", "pandas", "matplotlib"]},
        "inputs": {p: sha(p) for p in INPUTS},
        "steps": [{"command": f"python {s}", "outputs": {o: sha(o) for o in outs}} for s, outs in STEPS],
        "static_files": {p: sha(p) for p in STATIC},
        "runs": runs,
        "note": "Raw transaction lines retained. External data are retrospective context, never strict-model inputs.",
    }
    (ROOT / "reports" / "teammate2_manifest.json").write_text(json.dumps(manifest, indent=2))
    missing = [o for _, outs in STEPS for o in outs if sha(o) is None]
    print("saved reports/teammate2_manifest.json" + (f" (missing: {missing})" if missing else ""))
    sys.exit(1 if any(r["returncode"] for r in runs) or missing else 0)


if __name__ == "__main__":
    main()
