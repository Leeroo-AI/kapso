#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recentchanges_pull.py — latest-only incremental indexer for MediaWiki (Action API)

- Resolves API by login-token (private-wiki friendly), logs in (cookie).
- Reads RC incrementally (rccontinue + last_ts backoff 1s).
- Collapses multiple edits per page to ONE "upsert_latest".
- Emits {"op":"delete"} when a page ends deleted.
- Emits {"op":"rename"} only if there is NO delete for the same pageid (coalesced).
- Keeps title→pageid map so deletes with pid=0 in RC can still delete correctly.
- Concurrently fetches HTML + links once per page.
- Appends NDJSON to OUTBOX_FILE; creates folders as needed.
"""

from __future__ import annotations
import os, json, pathlib, requests
from urllib.parse import urlparse, urlunparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

# ---------- env config ----------
WIKI        = os.getenv("WIKI_URL", "https://localhost").rstrip("/")
WIKI_SUBDOMAIN = os.getenv("WIKI_SUBDOMAIN", "").strip()
USER        = os.getenv("MW_USER", "Agent")
PASS        = os.getenv("MW_PASS", "")
STATE_PATH  = os.getenv("RC_STATE",  os.path.join(os.getcwd(), "state", "index_state.json"))
OUTBOX_FILE = os.getenv("OUTBOX_FILE", os.path.join(os.getcwd(), "outbox", "latest.ndjson"))
NS_FILTER   = os.getenv("NS_FILTER", "").strip()
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))
TIMEOUT     = int(os.getenv("TIMEOUT", "30"))
RC_INCLUDE_BOT = os.getenv("RC_INCLUDE_BOT", "true").lower() in ("1","true","yes","y")

STATE  = pathlib.Path(STATE_PATH)
OUTBOX = pathlib.Path(OUTBOX_FILE)

S = requests.Session()
S.headers["User-Agent"] = "KC-RecentChanges-Latest/1.5 (+leeroo)"
S.verify = True
API_URL: Optional[str] = None  # resolved on login


# ---------- optional subdomain routing (manager mode) ----------
def _apply_subdomain_to_base(base: str, subdomain: str) -> tuple[str, Optional[str]]:
    """Return (new_base, host_header) after applying subdomain tag for manager-mode routing.

    Behavior:
    - If host has 3+ labels (e.g. wiki.example.com) → replace leftmost with subdomain → ml.example.com
    - If host has exactly 2 labels (e.g. example.com) → prepend subdomain → ml.example.com
    - Preserve the original scheme and port.
    - Returns (original_base, None) if parsing fails or host form is not applicable.
    """
    if not subdomain:
        return base, None
    try:
        parsed = urlparse(base if "://" in base else f"https://{base}")
        host_port = parsed.netloc
        if not host_port:
            return base, None
        if ":" in host_port and host_port.count(":") == 1:
            host_only, port = host_port.split(":", 1)
            port_suffix = f":{port}"
        else:
            host_only = host_port
            port_suffix = ""
        labels = host_only.split(".") if host_only else []
        if len(labels) >= 3:
            new_host = ".".join([subdomain] + labels[1:])
        elif len(labels) == 2:
            new_host = ".".join([subdomain] + labels)
        else:
            return base, None
        final_host = f"{new_host}{port_suffix}"
        new_base = urlunparse((parsed.scheme or "https", final_host, parsed.path.rstrip("/"), "", "", "")).rstrip("/")
        return new_base, new_host
    except Exception:
        return base, None

# ---------- resolve Action API by real login-token ----------
def _login_token_on(path: str) -> Tuple[str, Optional[str]]:
    url = f"{WIKI}{path}"
    try:
        r = S.get(url, params={"action":"query","meta":"tokens","type":"login","format":"json"},
                  timeout=TIMEOUT, allow_redirects=True)
        r.raise_for_status()
        j = r.json()
        tok = j.get("query", {}).get("tokens", {}).get("logintoken")
        return url, tok
    except Exception:
        return url, None

def resolve_api_and_login() -> None:
    global API_URL
    for path in ("/api.php", "/w/api.php", "/mediawiki/api.php"):
        url, tok = _login_token_on(path)
        if tok:
            resp = S.post(url, data={
                "action":"login","format":"json",
                "lgname":USER,"lgpassword":PASS,"lgtoken":tok
            }, timeout=TIMEOUT)
            resp.raise_for_status()
            j = resp.json()
            if j.get("login",{}).get("result") == "Success":
                API_URL = url
                return
    raise RuntimeError("Cannot log in; check WIKI_URL/MW_USER/MW_PASS and that /api.php is reachable")

# ---------- Action API helper ----------
def api(params: Dict[str, Any], post: bool=False) -> Dict[str, Any]:
    if API_URL is None:
        raise RuntimeError("API_URL not resolved; call resolve_api_and_login() first")
    p = {**params, "format":"json"}
    r = S.post(API_URL, data=p, timeout=TIMEOUT) if post else S.get(API_URL, params=p, timeout=TIMEOUT)
    r.raise_for_status()
    try:
        j = r.json()
    except Exception as e:
        raise RuntimeError({"error":"non-json-response","status":r.status_code,"text":r.text[:200]}) from e
    if "error" in j: raise RuntimeError(j["error"])
    return j

# ---------- state ----------
def backoff(ts_iso: str, seconds=1) -> str:
    dt = datetime.fromisoformat(ts_iso.replace("Z","+00:00"))
    return (dt - timedelta(seconds=seconds)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def load_state() -> Dict[str, Any]:
    if STATE.exists():
        try: return json.loads(STATE.read_text())
        except Exception: pass
    return {"rccontinue":None,"last_ts":None,"last_rcid":None,"pages":{}, "title_to_pid":{}}

def save_state(st: Dict[str, Any]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, ensure_ascii=False))

# ---------- RC + fetch ----------
def fetch_rc(rccont: Optional[str], since: Optional[str]) -> Tuple[List[Dict[str,Any]], Optional[str]]:
    params = {
        "action":"query","list":"recentchanges",
        "rcprop":"rcid|title|ids|timestamp|user|comment|sha1|loginfo",
        "rclimit":"500","rctype":"edit|new|move|log|delete"
    }
    if not RC_INCLUDE_BOT:
        params["rcshow"] = "!bot"
    if NS_FILTER:
        params["rcnamespace"] = NS_FILTER
    if rccont:
        params["rccontinue"] = rccont
    elif since:
        params["rcstart"] = since
        params["rcdir"]   = "newer"
    j = api(params)
    return j.get("query",{}).get("recentchanges",[]), j.get("continue",{}).get("rccontinue")

def parse_html(title: str) -> str:
    j = api({"action":"parse","page":title,"prop":"text","redirects":"1","disablelimitreport":"1"})
    return j["parse"]["text"]["*"]

def get_links(title: str) -> List[str]:
    out, cont = [], None
    while True:
        p = {"action":"query","prop":"links","titles":title,"pllimit":"500"}
        if cont: p["plcontinue"] = cont
        j = api(p)
        page = next(iter(j["query"]["pages"].values()))
        out += [l["title"] for l in page.get("links",[])]
        cont = j.get("continue",{}).get("plcontinue")
        if not cont: break
    return out

def outbox_write(obj: Dict[str, Any]) -> None:
    OUTBOX.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTBOX, "ab") as f:
        f.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))

# ---------- main ----------
def main() -> None:
    # Manager-mode routing: if WIKI_SUBDOMAIN is set (e.g. ml|de|quant),
    # rewrite WIKI to that subdomain host and set explicit Host header so Traefik routes correctly.
    global WIKI
    if WIKI_SUBDOMAIN:
        new_base, host_header = _apply_subdomain_to_base(WIKI, WIKI_SUBDOMAIN)
        if new_base != WIKI:
            WIKI = new_base
        if host_header:
            S.headers["Host"] = host_header

    resolve_api_and_login()

    st = load_state()
    pages_state: Dict[str,int]  = st.get("pages", {})
    title_to_pid: Dict[str,int] = st.get("title_to_pid", {})

    # choose start
    if st.get("rccontinue"): since = None
    elif st.get("last_ts"):  since = backoff(st["last_ts"], 1)
    else:                    since = None

    last_rcid = st.get("last_rcid")
    latest_ts = st.get("last_ts")

    changes: Dict[int, Dict[str,Any]] = {}   # pageid -> {title,revid,timestamp}
    deletes: Dict[int, str] = {}             # pageid -> last known title
    renames: Dict[int, Dict[str,str]] = {}   # pid -> {old_title,new_title}

    while True:
        batch, cont = fetch_rc(st.get("rccontinue"), since)
        print(f"Collected {len(batch)} RC rows; rccontinue={cont} since={since}")
        if not batch and not cont: break

        for ch in batch:
            rcid = ch.get("rcid")
            if isinstance(rcid, int) and last_rcid is not None and rcid <= last_rcid:
                continue

            pageid = ch.get("pageid")
            title  = ch.get("title")
            revid  = ch.get("revid") or (ch.get("revision") or {}).get("new")
            ts     = ch.get("timestamp")
            rctype = ch.get("type") or ch.get("logtype")

            # advance watermarks
            if isinstance(rcid, int):
                last_rcid = rcid if last_rcid is None else max(last_rcid, rcid)
            if ts and (latest_ts is None or ts > latest_ts):
                latest_ts = ts

            # normalize pid
            pid = pageid if isinstance(pageid, int) and pageid > 0 else None
            if pid is None and title in title_to_pid:
                pid = title_to_pid[title]

            # delete
            if rctype in ("delete",) or (rctype == "log" and ch.get("logtype") == "delete"):
                if pid is not None:
                    deletes[pid] = title or title_to_pid.get(pid, "")
                    changes.pop(pid, None)
                    pages_state.pop(str(pid), None)
                    # title map cleanup
                    if title: title_to_pid.pop(title, None)
                else:
                    # no pid — keep title so downstream can clean title-keyed stores
                    outbox_write({"op": "delete", "title": title})
                continue

            # move
            if rctype == "move" or (rctype == "log" and ch.get("logtype") == "move"):
                logparams = ch.get("logparams") or {}
                new_title = (
                    logparams.get("target_title")
                    or logparams.get("target")
                    or logparams.get("new_title")
                    or title
                )
                if pid is not None:
                    renames[pid] = {"old_title": title, "new_title": new_title}
                    title_to_pid.pop(title, None)
                    title_to_pid[new_title] = pid
                    entry = changes.get(pid, {"revid": revid or 0})
                    entry.update({"title": new_title, "timestamp": ts})
                    changes[pid] = entry
                continue

            # edit/new
            if pid is None:
                # still no pid — skip (rare), but keep title mapping fresh next time via parse
                continue
            title_to_pid[title] = pid  # refresh mapping
            cur = changes.get(pid)
            if (cur is None) or (revid and cur.get("revid", 0) < revid):
                changes[pid] = {"title": title, "revid": revid or 0, "timestamp": ts}

        if cont:
            st["rccontinue"] = cont
            save_state(st)
            since = None
            continue
        else:
            break

    # ---- Emit events (coalesced) ----
    # 1) Deletes (these supersede renames and upserts)
    for pid, last_title in deletes.items():
        outbox_write({"op": "delete", "pageid": pid})

    # 2) Renames only for pids NOT deleted in the same window
    for pid, rn in renames.items():
        if pid not in deletes:
            outbox_write({"op": "rename", "pageid": pid, **rn})

    # 3) Upserts for pages whose latest revid increased
    to_index: List[Tuple[int, Dict[str,Any]]] = []
    for pid, meta in changes.items():
        if pid in deletes:
            continue  # final state is deleted
        last_indexed = pages_state.get(str(pid), 0)
        if meta.get("revid") and meta["revid"] > last_indexed:
            to_index.append((pid, meta))

    print(f"{len(changes)} pages changed; {len(to_index)} need re-index "
          f"(deletes={len(deletes)}, renames={len(renames)})")

    total = 0
    if to_index:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futs = {pool.submit(parse_html, m["title"]): (pid, m["title"], m["revid"], m["timestamp"]) for pid, m in to_index}
            for fut in as_completed(futs):
                pid, title, revid, ts = futs[fut]
                try:
                    html = fut.result()
                    links = get_links(title)
                    outbox_write({
                        "op": "upsert_latest",
                        "pageid": pid,
                        "revid": revid,
                        "title": title,
                        "timestamp": ts,
                        "html": html,
                        "links": links
                    })
                    pages_state[str(pid)] = revid
                    title_to_pid[title]  = pid
                    total += 1
                except Exception as e:
                    outbox_write({"op":"error","pageid":pid,"title":title,"error":str(e)})

    # finalize state
    st["rccontinue"] = None
    st["last_ts"]    = latest_ts
    st["last_rcid"]  = last_rcid
    st["pages"]      = pages_state
    st["title_to_pid"] = title_to_pid
    save_state(st)
    print(f"Done. Upserted {total} latest pages; state saved to {STATE}")
    print(f"Outbox: {OUTBOX}")


if __name__ == "__main__":
    main()
