"""Langfuse drill-down: error breakdown + last observation for the top traces."""
from __future__ import annotations

import base64
import json
import os
from collections import Counter

from dotenv import load_dotenv

load_dotenv(".env", override=False)
PUBLIC = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip().strip('"').strip("'")
SECRET = os.getenv("LANGFUSE_SECRET_KEY", "").strip().strip('"').strip("'")
BASE = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com").strip().strip('"').strip("'").rstrip("/")
token = base64.b64encode(f"{PUBLIC}:{SECRET}".encode()).decode()

import urllib.request
HEADERS = {"Authorization": f"Basic {token}", "Content-Type": "application/json"}


def api(path):
    req = urllib.request.Request(f"{BASE}{path}", headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def msg(o):
    m = o.get("statusMessage") or o.get("errorMessage")
    if m:
        return str(m)
    meta = o.get("metadata") or {}
    if isinstance(meta, dict) and meta.get("error"):
        return str(meta["error"])
    return ""


TARGETS = [
    "4ba9a1ffce56ae96",  # 19:14 CST - likely the killed run
    "f5a970dc6e63152e",  # 18:42 CST
    "330b4ef1c09886e9",  # 17:51 CST - completed d6519ed3
]

for tid in TARGETS:
    # trace id may be truncated in the list; find the full id.
    traces = api("/api/public/traces?limit=15")
    items = traces.get("data", traces) if isinstance(traces, dict) else traces
    full = next((t for t in items if str(t.get("id", "")).startswith(tid)), None)
    if not full:
        print(f"\n=== {tid}: not found ===")
        continue
    fid = full["id"]
    print(f"\n{'='*72}\n=== TRACE {fid[:16]}  ts={full.get('timestamp')}\n{'='*72}")
    trace = api(f"/api/public/traces/{fid}")
    obs = [o for o in trace.get("observations", []) if isinstance(o, dict)]
    print(f"observations: {len(obs)}  | by type: {dict(Counter(o.get('type') for o in obs))}")

    # tool observations
    tools = [o for o in obs if o.get("type") == "TOOL"]
    print(f"TOOL obs: {len(tools)}")
    tool_names = Counter(o.get("name") for o in tools)
    print(f"  tool call counts: {dict(tool_names)}")

    # errors by message + tool
    errs = [o for o in obs if o.get("status") == "ERROR" or msg(o)]
    print(f"\nERROR observations: {len(errs)}")
    if errs:
        emap = Counter()
        for o in errs:
            emap[(o.get("name"), msg(o) or "(status=ERROR,no msg)")] += 1
        for (name, m), n in emap.most_common():
            print(f"  [{n}x] {name}: {m[:350]}")

    # timing: start/end of first and last observation
    def tskey(o):
        return o.get("startTime") or o.get("startTime") or ""
    if obs:
        times = sorted(o.get("startTime") for o in obs if o.get("startTime"))
        print(f"\nfirst obs start: {times[0] if times else '?'}")
        print(f"last  obs start: {times[-1] if times else '?'}")

    # last 6 observations (to see where it died)
    print("\nlast 6 observations (name | type | status | msg):")
    for o in sorted(obs, key=lambda o: o.get("startTime") or "")[-6:]:
        print(f"  - {o.get('name')!r:30} type={o.get('type')} status={o.get('status')} msg={msg(o)[:160]!r}")
