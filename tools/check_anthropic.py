"""One tiny Anthropic API call to confirm the key works and has credit. Prints a plain verdict, never the key."""
import os
import sys
import tomllib

import requests

key = os.environ.get("ANTHROPIC_API_KEY", "")
model = tomllib.load(open("config/system.toml", "rb")).get("llm_model", "claude-sonnet-5-5")
if not key:
    msg, ok = "ANTHROPIC key: MISSING (add the CLAUDE repo secret)", False
else:
    r = requests.post("https://api.anthropic.com/v1/messages", timeout=60, headers={
        "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={"model": model, "max_tokens": 5, "messages": [{"role": "user", "content": "Reply OK"}]})
    ok = r.status_code == 200
    if ok:
        msg = f"ANTHROPIC key: WORKS (model {model}, credit available)"
    else:
        text = r.text[:220]
        hint = ("no API credit: add credit at console.anthropic.com > Billing" if "credit balance" in text else
                "key rejected: create a new key and update the CLAUDE secret" if r.status_code == 401 else
                f"model {model} not available to this key" if r.status_code == 404 else "see error")
        msg = f"ANTHROPIC key: FAILS ({hint}). HTTP {r.status_code}: {text}"
print(msg)
print("::notice title=anthropic key check::" + msg.replace("%", "%25"))
sys.exit(0 if ok else 1)
