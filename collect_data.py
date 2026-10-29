"""Optional step 0 - download release histories from the PyPI and npm registries.

Only the *release timeline* of each package is kept (version + publish date). The result is saved to
data/package_releases.jsonl, which is already included in the project, so you only need to run this
script if you want to refresh the data (requires internet access).

    python collect_data.py            # ~7000 PyPI + ~2500 npm packages
"""
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent / "data" / "package_releases.jsonl"
SEED = 42
N_PYPI, N_NPM = 7000, 2500
NPM_KEYWORDS = ["api", "util", "react", "node", "cli", "plugin", "parser", "server", "client", "test", "build",
                "component", "webpack", "babel", "express", "database", "auth", "logger", "image", "date",
                "string", "array", "config", "stream", "router", "form", "validator", "aws", "docker", "markdown",
                "css", "http", "json", "file", "mongo", "sql", "mock", "template", "email", "chart"]
session = requests.Session()
session.headers["User-Agent"] = "package-risk-student-project/1.0"


def get_json(url, **kw):
    for attempt in range(3):
        try:
            r = session.get(url, timeout=120, **kw)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (404, 410):
                return None
        except (requests.RequestException, ValueError):
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def pypi_record(name):
    data = get_json(f"https://pypi.org/pypi/{name}/json")
    if not data:
        return None
    rel = []
    for version, files in data.get("releases", {}).items():
        times = [f.get("upload_time_iso_8601") for f in files if f.get("upload_time_iso_8601")]
        if times:
            rel.append([version, min(times)])
    return {"ecosystem": "pypi", "name": name, "releases": rel}


def npm_record(name):
    data = get_json(f"https://registry.npmjs.org/{name.replace('/', '%2F')}")
    if not data or "time" not in data:
        return None
    rel = [[v, t] for v, t in data["time"].items() if v not in ("created", "modified", "unpublished")]
    return {"ecosystem": "npm", "name": name, "releases": rel}


def npm_names(n):
    rnd, names = random.Random(SEED), []
    seen = set()
    while len(seen) < n:
        kw = rnd.choice(NPM_KEYWORDS)
        data = get_json("https://registry.npmjs.org/-/v1/search",
                        params={"text": f"keywords:{kw}", "size": 250, "from": rnd.randrange(0, 4000, 250)})
        if not data:
            continue
        for o in data.get("objects", []):
            nm = o["package"]["name"]
            if nm not in seen:
                seen.add(nm); names.append(nm)
        print(f"  npm names collected: {len(seen)}", end="\r")
    return names[:n]


def main():
    rnd = random.Random(SEED)
    idx = session.get("https://pypi.org/simple/", headers={"Accept": "application/vnd.pypi.simple.v1+json"}, timeout=120).json()
    pypi = rnd.sample([p["name"] for p in idx["projects"]], N_PYPI)
    print(f"PyPI index: {len(idx['projects'])} projects, sampling {N_PYPI}")
    npm = npm_names(N_NPM)
    done = set()                                   # resume support: skip packages already saved
    if OUT.exists():
        for line in open(OUT):
            r = json.loads(line); done.add((r["ecosystem"], r["name"]))
    jobs = [(pypi_record, n) for n in pypi if ("pypi", n) not in done] + \
           [(npm_record, n) for n in npm if ("npm", n) not in done]
    stamp = datetime.now(timezone.utc).isoformat()
    ok = 0
    with open(OUT, "a") as fh, ThreadPoolExecutor(16) as ex:
        futs = [ex.submit(f, n) for f, n in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            if rec and rec["releases"]:
                rec["collected_at"] = stamp
                fh.write(json.dumps(rec) + "\n"); fh.flush(); ok += 1
            if i % 500 == 0:
                print(f"  fetched {i}/{len(jobs)} (kept {ok})", flush=True)
    print(f"done: {ok} packages written to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
