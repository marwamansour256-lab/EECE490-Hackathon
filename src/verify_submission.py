"""One command to check the submitted models, mathematics and offline artifacts."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    steps=[ ["-m","unittest","discover","-s","tests","-q"],
        ["src/verify_strict_artifacts.py"], ["src/verify_decision_artifacts.py"],
        ["src/verify_context_artifacts.py"], ["src/verify_recommendation_artifacts.py"],
        ["src/basket_pilot_report.py","data/pilot/basket_pilot_template.csv"] ]
    for args in steps:
        print("Checking: "+" ".join(args),flush=True)
        subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
    print("Submission checks passed. Browser visual review and a prospective profit pilot are separate.",flush=True)


if __name__=="__main__":
    main()
