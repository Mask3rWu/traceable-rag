"""One-off Langfuse diagnostic probe. Not part of the app; safe to delete."""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from collections import Counter

from dotenv import load_dotenv

load_dotenv(".env", override=False)

PUBLIC = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip().strip('"').strip("'")
SECRET = os.getenv("LANGFUSE_SECRET_KEY", "").strip().strip('"').strip("'")
BASE = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com").strip().strip('"').strip("'").rstrip("/")

if not (PUBLIC and SECRET):
    print("ERROR: Langfuse credentials missing"); sys.exit(1)

import base64
token = base64.b64encode(f"{PUBLIC}:{SECRET}".encode()).decode()
HEADERS = {"Authorization": f"Basic {token}", "Content-Type": "application/json"}


def api(path: str, params: dict | None = None) -> dict:
    url = f"{BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def err_msg(obs: dict) -> str:
    m = obs.get("statusMessage") or obs.get("errorMessage")
    if m:
        return str(m)
    meta = obs.get("metadata") or {}
    if isinstance(meta, dict) and meta.get("error"):
        return str(meta["error"])
    return ""


print("=" * 70)
print("LANGFUSE PROBE  base:", BASE)
print("=" * 70)

# 1) List recent traces to locate the killed run (~19:29) and confirm 330b...
print("\n## Recent traces (latest 10)")
traces = api("/api/public/traces", {"limit": 10})
items = traces.get("data", traces) if isinstance(traces, dict) else traces
for t in items:
    tid = t.get("id")
    name = t.get("name", "")
    ts = t.get("timestamp", t.get("createdAt", ""))
    raw_obs = t.get("observations", [])
    n_obs = len(raw_obs)
    # In the list endpoint, observations are ID strings (not dicts), so error
    # counts aren't available here; we drill in per-trace below.
    sess = t.get("sessionId") or ""
    print(f"  {tid[:16]}  name={name!r}  ts={ts}  obs={n_obs}  sess={sess[:12]}")

# 2) Drill into the completed run's trace first.
target = "330b4ef1c09886e92aaa70d73afe1919"
print(f"\n## Fetching trace {target}")
trace = api(f"/api/public/traces/{target}")
raw_obs = trace.get("observations", [])
# Observations can be full objects or ID strings depending on endpoint; keep dicts.
obs = [o for o in raw_obs if isinstance(o, dict)]
print(f"  total observations: {len(obs)} (of {len(raw_obs)} entries)")
by_type = Counter(o.get("type") for o in obs)
print(f"  by type: {dict(by_type)}")
errors = [o for o in obs if o.get("status") == "ERROR" or err_msg(o)]
print(f"  ERROR/error-msg observations: {len(errors)}")

print("\n## Distinct error messages on this trace")
seen: dict[str, int] = {}
for o in errors:
    msg = err_msg(o) or "(status=ERROR, no message)"
    seen[msg] = seen.get(msg, 0) + 1
for msg, n in sorted(seen.items(), key=lambda kv: -kv[1]):
    print(f"  [{n}x] {msg[:400]}")

# Also surface any observation whose name looks like a tool (search/read/submit)
print("\n## Observations named like tools (name + status + message)")
for o in obs:
    name = str(o.get("name", ""))
    if any(k in name.lower() for k in ("search", "read", "submit", "tool", "chapter-worker", "evidence")):
        st = o.get("status")
        msg = err_msg(o)
        if st == "ERROR" or msg:
            print(f"  - {name!r} status={st} msg={msg[:300]!r}")

# 3) Try to find the killed run via the observations API filtered to ERROR spans.
print("\n## Latest ERROR observations across project (via /observations)")
try:
    obsresp = api("/api/public/observations", {"limit": 30, "type": "SPAN"})
    olist = obsresp.get("data", obsresp) if isinstance(obsresp, dict) else obsresp
    for o in olist:
        st = o.get("status")
        if st != "ERROR":
            continue
        print(f"  trace={o.get('traceId','')[:16]} name={o.get('name')!r} ts={o.get('startTime', o.get('createdAt'))} msg={err_msg(o)[:300]!r}")
except Exception as e:
    print(f"  (observations endpoint unavailable: {e})")
