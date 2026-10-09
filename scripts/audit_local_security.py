"""Redacted local secret inventory and installed-package OSV advisory query.

Prints paths/counts/advisory IDs, never credential values. Skips local
backups, dependencies, and generated artifacts. Does not modify packages.
"""

import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


def secrets():
    paths = subprocess.check_output(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT
    ).decode("utf-8").split("\0")
    patterns = {
        "google_key": re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
        "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "github_token": re.compile(r"(?:ghp_|github_pat_)[A-Za-z0-9_]{30,}"),
        "openai_key": re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{40,}"),
        "bearer_literal": re.compile(r"Bearer\s+[A-Za-z0-9_-]{24,}(?:\.[A-Za-z0-9_-]+)*"),
    }
    scanned, findings = 0, []
    for name in sorted(set(paths)):
        if not name or name.startswith("docs/wireframes_local_backup/"):
            continue
        path = ROOT / name
        if path.suffix.lower() not in {".py", ".js", ".jsx", ".json", ".md", ".txt", ".ps1", ".yml", ".yaml", ".env"}:
            continue
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        scanned += 1
        for kind, pattern in patterns.items():
            if pattern.search(text):
                findings.append({"path": name, "kind": kind})
    print(json.dumps({"secret_files_scanned": scanned, "findings_paths_only": findings}))


def dependencies():
    packages = sorted({(d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
                       if d.metadata.get("Name")})
    queries = [{"package": {"name": name, "ecosystem": "PyPI"}, "version": version}
               for name, version in packages]
    request = urllib.request.Request("https://api.osv.dev/v1/querybatch",
                                     json.dumps({"queries": queries}).encode(),
                                     {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            results = json.load(response)["results"]
    except Exception as error:
        print(json.dumps({"python_packages_checked": len(packages),
                          "audit_blocked": type(error).__name__}))
        return
    findings = []
    for (name, version), result in zip(packages, results):
        for vuln in result.get("vulns", []):
            findings.append({"package": name, "version": version, "id": vuln["id"]})
    print(json.dumps({"python_packages_checked": len(packages), "advisories": findings}, indent=2))


if __name__ == "__main__":
    secrets()
    dependencies()
