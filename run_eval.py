import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_patch_paths(patch_text: str) -> list[str]:
    paths = set()
    for line in patch_text.splitlines():
        if not line.startswith(("--- ", "+++ ")):
            continue
        raw_path = line[4:].strip()
        if raw_path == "/dev/null":
            continue
        if raw_path.startswith(("a/", "b/")):
            raw_path = raw_path[2:]
        paths.add(raw_path)
    return sorted(paths)


def load_protected_paths() -> list[str]:
    protected = (ROOT / "judge_assets" / "protected_paths.txt").read_text().splitlines()
    return [line.strip() for line in protected if line.strip()]


def is_protected(path: str, protected_paths: list[str]) -> bool:
    for protected_path in protected_paths:
        if path == protected_path or path.startswith(f"{protected_path}/"):
            return True
    return False


def validate_patch_paths(changed_paths: list[str], protected_paths: list[str]) -> None:
    if not changed_paths:
        raise ValueError("patch does not modify any files")

    for changed_path in changed_paths:
        if not changed_path.startswith("task_repo_baseline/"):
            raise ValueError(f"patch touches an out-of-scope path: {changed_path}")
        if is_protected(changed_path, protected_paths):
            raise ValueError(f"patch touches a protected path: {changed_path}")


def build_temp_workspace(temp_root: Path) -> None:
    shutil.copytree(ROOT / "task_repo_baseline", temp_root / "task_repo_baseline")
    shutil.copytree(ROOT / "judge_assets", temp_root / "judge_assets")


def apply_patch(temp_root: Path, patch_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=temp_root, check=True)
    subprocess.run(
        ["git", "apply", "--whitespace=nowarn", str(patch_path.resolve())],
        cwd=temp_root,
        check=True,
    )


def run_judge(temp_root: Path) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(temp_root / "judge_assets" / "judge.py"),
        "--candidate-repo",
        str(temp_root / "task_repo_baseline"),
    ]
    return subprocess.run(
        command,
        cwd=temp_root,
        capture_output=True,
        text=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("patch", help="Path to a unified diff patch")
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary evaluation directory for debugging",
    )
    args = parser.parse_args()

    patch_path = Path(args.patch).resolve()
    patch_text = patch_path.read_text()
    changed_paths = load_patch_paths(patch_text)
    protected_paths = load_protected_paths()

    try:
        validate_patch_paths(changed_paths, protected_paths)
    except ValueError as exc:
        payload = {
            "passed": False,
            "error": str(exc),
            "changed_paths": changed_paths,
        }
        print(json.dumps(payload, indent=2))
        return 1

    with tempfile.TemporaryDirectory(prefix="nano-vllm-eval-") as temp_dir:
        temp_root = Path(temp_dir)
        build_temp_workspace(temp_root)
        try:
            apply_patch(temp_root, patch_path)
        except subprocess.CalledProcessError as exc:
            payload = {
                "passed": False,
                "error": "failed to apply patch to clean baseline",
                "returncode": exc.returncode,
                "changed_paths": changed_paths,
            }
            print(json.dumps(payload, indent=2))
            return 1
        result = run_judge(temp_root)
        if args.keep_temp:
            debug_dir = ROOT / ".last_eval_workspace"
            if debug_dir.exists():
                shutil.rmtree(debug_dir)
            shutil.copytree(temp_root, debug_dir)

        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(result.stderr.strip(), file=sys.stderr)
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
