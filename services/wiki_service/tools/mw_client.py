#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mw_client.py — robust MediaWiki Action API client + smoke test.

Key idea: resolve the Action API by attempting the REAL login-token call on
each common endpoint (/api.php, /w/api.php, /mediawiki/api.php). This works
even on private wikis and avoids HTML help-page probes.

Usage (smoke test)
  python3 mw_client.py \
    --base https://google.leeroo.com \
    --user Agent \
    --password 'SetAgentStrongPass!' \
    --prefix Sandbox:API_Test
"""

import argparse
import json
import sys
import time
import requests
from urllib.parse import urlparse, urlunparse


class MWClient:
    def __init__(self, base: str, user: str, password: str, verify: bool = True, timeout: int = 30, subdomain: str | None = None):
        self.base = base.rstrip("/")
        self.user = user
        self.password = password
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "KC-Agent/1.0 (+leeroo)"
        self.s.verify = verify
        self.timeout = timeout
        self.csrf = None
        self._api_url = None

        # Optional subdomain routing (manager mode):
        # If a subdomain tag is provided (e.g. "ml", "de", "quant"),
        # rewrite the base host to that subdomain and set explicit Host header.
        # Traefik in manager mode routes by Host(`${WIKI_HOST}`).
        if subdomain:
            self._apply_subdomain(subdomain)

    def _apply_subdomain(self, subdomain: str) -> None:
        """Rewrite self.base to target a specific subdomain and set Host header.

        Rules (simple and predictable):
        - If base host has exactly two labels (e.g. yourdomain.com): prepend sub → ml.yourdomain.com
        - If base host has 3+ labels (e.g. wiki.yourdomain.com): replace leftmost with sub → ml.yourdomain.com
        - Preserve port if present.
        - Also set the explicit Host header to the computed host for clarity.
        """
        try:
            p = urlparse(self.base if "://" in self.base else f"https://{self.base}")
            host_port = p.netloc
            if not host_port:
                return
            # Split host and port if specified
            if ":" in host_port and host_port.count(":" ) == 1:
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
                # Unrecognized host form (IP or single label). Keep base as-is.
                return
            final_host = f"{new_host}{port_suffix}"
            # Rebuild URL with updated host
            self.base = urlunparse((p.scheme or "https", final_host, p.path.rstrip("/"), "", "", "")).rstrip("/")
            # Ensure Host header matches for reverse proxy routing
            self.s.headers["Host"] = new_host
        except Exception:
            # On any parsing failure, leave base unchanged
            return

    # ---------- endpoint resolution by login-token ----------
    def _login_token_on(self, path: str):
        """Try to get a login token from base+path. Return token string or None."""
        url = f"{self.base}{path}"
        try:
            r = self.s.get(
                url,
                params={"action": "query", "meta": "tokens", "type": "login", "format": "json"},
                timeout=self.timeout,
                allow_redirects=True,
            )
            r.raise_for_status()
            j = r.json()
            tok = j.get("query", {}).get("tokens", {}).get("logintoken")
            return (url, tok) if tok else (url, None)
        except Exception:
            return (url, None)

    def _resolve_and_get_login_token(self):
        # Try common action API locations in order
        for path in ("/api.php", "/w/api.php", "/mediawiki/api.php"):
            url, tok = self._login_token_on(path)
            if tok:
                self._api_url = url
                return tok
        # As a last-ditch, do NOT guess further; tell the operator what we tried
        raise RuntimeError(
            f"Could not obtain login token from {self.base} "
            f"(tried /api.php, /w/api.php, /mediawiki/api.php)."
        )

    @property
    def api_url(self):
        if not self._api_url:
            # Will be set during login()
            raise RuntimeError("API URL not resolved yet; call login() first")
        return self._api_url

    # ---------- core helpers ----------

    def _api(self, params=None, data=None, post: bool = False):
        url = self.api_url
        if post:
            r = self.s.post(url, data=data, timeout=self.timeout)
        else:
            r = self.s.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        try:
            j = r.json()
        except Exception as e:
            raise RuntimeError({
                "error": "non-json-response",
                "status": r.status_code,
                "text_sample": r.text[:200]
            }) from e
        if "error" in j:
            raise RuntimeError(j["error"])
        return j

    def login(self):
        """Login (works with normal password or BotPassword)."""
        # Resolve API by performing the real login-token call
        tok = self._resolve_and_get_login_token()

        # Perform login (stores the cookie in the session)
        j = self.s.post(
            self._api_url,
            data={
                "action": "login",
                "format": "json",
                "lgname": self.user,
                "lgpassword": self.password,
                "lgtoken": tok,
            },
            timeout=self.timeout,
        )
        j.raise_for_status()
        j = j.json()
        if j.get("login", {}).get("result") != "Success":
            raise RuntimeError(f"login failed: {j}")

        # Get CSRF token bound to same session
        j = self.s.get(
            self._api_url,
            params={"action": "query", "meta": "tokens", "type": "csrf", "format": "json"},
            timeout=self.timeout,
        )
        j.raise_for_status()
        j = j.json()
        self.csrf = j["query"]["tokens"]["csrftoken"]

    # ---------- content ops ----------

    def edit(self, title: str, *, text: str | None = None, summary: str = "",
             createonly: bool = False, append: str | None = None,
             prepend: str | None = None, bot: bool = True, minor: bool = False):
        if not self.csrf:
            self.login()
        payload = {
            "action": "edit",
            "format": "json",
            "assert": "user",   # fail if session not logged-in
            "title": title,
            "summary": summary,
            "token": self.csrf,
        }
        if createonly:
            payload["createonly"] = 1
        if append is not None:
            payload["appendtext"] = append
        elif prepend is not None:
            payload["prependtext"] = prepend
        else:
            payload["text"] = text or ""
        if bot:
            payload["bot"] = 1
        if minor:
            payload["minor"] = 1
        r = self.s.post(self.api_url, data=payload, timeout=self.timeout)
        r.raise_for_status()
        j = r.json()
        if "error" in j:
            raise RuntimeError(j["error"])
        return j

    def move(self, old: str, new: str, *, reason: str = "", noredirect: bool = False, movetalk: bool = True):
        # noredirect=True also needs 'suppressredirect'; default False for portability
        if not self.csrf:
            self.login()
        payload = {
            "action": "move",
            "format": "json",
            "from": old,
            "to": new,
            "reason": reason,
            "token": self.csrf,
        }
        if noredirect:
            payload["noredirect"] = 1
        if movetalk:
            payload["movetalk"] = 1
        r = self.s.post(self.api_url, data=payload, timeout=self.timeout)
        r.raise_for_status()
        j = r.json()
        if "error" in j:
            raise RuntimeError(j["error"])
        return j

    def delete(self, title: str, *, reason: str = ""):
        if not self.csrf:
            self.login()
        payload = {
            "action": "delete",
            "format": "json",
            "title": title,
            "reason": reason,
            "token": self.csrf,
        }
        r = self.s.post(self.api_url, data=payload, timeout=self.timeout)
        r.raise_for_status()
        j = r.json()
        if "error" in j:
            raise RuntimeError(j["error"])
        return j

    # Optional helpers
    def get_wikitext(self, title: str) -> str:
        j = self._api(params={"action": "parse", "page": title, "prop": "wikitext", "format": "json"})
        return j["parse"]["wikitext"]["*"]

    def get_html(self, title: str) -> str:
        j = self._api(params={"action": "parse", "page": title, "prop": "text", "format": "json"})
        return j["parse"]["text"]["*"]


# ---------- CLI smoke test ----------

def _expect(cond: bool, step: str, payload):
    if cond:
        print(f"✔ {step}")
        return
    print(f"✖ {step} FAILED")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="MediaWiki API smoke test (create→append→move→delete)")
    ap.add_argument("--base", required=True, help="Base URL, e.g. https://wiki.example.com")
    ap.add_argument("--user", required=True, help="Username (or BotPassword name, e.g. Agent@APP)")
    ap.add_argument("--password", required=True, help="Password (or BotPassword app password)")
    ap.add_argument("--prefix", default="Sandbox:API_Test", help="Base title to use (default Sandbox:API_Test)")
    ap.add_argument("--insecure", action="store_true", help="Disable TLS verification")
    ap.add_argument("--noredirect", action="store_true", help="Try move without redirect (requires suppressredirect)")
    ap.add_argument("--subdomain", help="Optional subdomain tag (e.g. ml|de|quant) to select wiki in manager mode")
    args = ap.parse_args()

    mw = MWClient(args.base, args.user, args.password, verify=(not args.insecure), subdomain=args.subdomain)

    stamp = int(time.time())
    src = f"{args.prefix}_{stamp}"
    dst = f"{src}_Moved"

    # Show the effective base after any subdomain rewrite so operators can verify routing
    print(f"Testing against {mw.base} as {args.user}")
    print(f"Titles: {src} → {dst}")

    try:
        mw.login()
        print(f"✔ login ok (API: {mw.api_url})")
    except Exception as e:
        print("✖ login failed:", e)
        sys.exit(1)

    r = mw.edit(src, text="hello from agent", summary="create via test", createonly=True)
    _expect(r.get("edit", {}).get("result") == "Success", f"create {src}", r)

    r = mw.edit(src, append="\n\nappended line", summary="append via test")
    _expect(r.get("edit", {}).get("result") == "Success", f"append to {src}", r)

    r = mw.move(src, dst, reason="tidy via test", noredirect=args.noredirect, movetalk=True)
    _expect("move" in r, f"move {src} → {dst}", r)

    r = mw.delete(dst, reason="cleanup via test")
    ok_delete = ("delete" in r) or (r.get("success") == 1)
    _expect(ok_delete, f"delete {dst}", r)

    print("🎉 All API operations succeeded.")


if __name__ == "__main__":
    main()
