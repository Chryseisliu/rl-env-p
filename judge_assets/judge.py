import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_public_tests(candidate_repo: Path) -> dict:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_public.py",
    ]
    result = subprocess.run(
        command,
        cwd=candidate_repo,
        capture_output=True,
        text=True,
    )
    return {
        "passed": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-repo", required=True)
    args = parser.parse_args()

    candidate_repo = Path(args.candidate_repo).resolve()
    public_tests = run_public_tests(candidate_repo)
    hidden = {
        "score": 0.0,
        "earned_weight": 0,
        "total_weight": 0,
        "cases": [],
    }

    if public_tests["passed"]:
        try:
            from hidden_tests import run_hidden_suite
        except ModuleNotFoundError as exc:
            payload = {
                "passed": False,
                "public_tests": public_tests,
                "hidden_tests": hidden,
                "error": (
                    "The judge environment is missing a required dependency. "
                    "This prototype expects the evaluation image to include the packages listed in requirements.txt."
                ),
                "missing_module": exc.name,
            }
            print(json.dumps(payload, indent=2))
            return 1

        hidden = run_hidden_suite(candidate_repo)

    passed = public_tests["passed"] and hidden["score"] == 1.0
    payload = {
        "passed": passed,
        "public_tests": public_tests,
        "hidden_tests": hidden,
    }
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
