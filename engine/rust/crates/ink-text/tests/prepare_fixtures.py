#!/usr/bin/env python3 -I
"""Extracts the config B recordings the ink-text parity tests read (package F1) into engine/rust/testdata-local/parity/.

    python3 -I engine/rust/crates/ink-text/tests/prepare_fixtures.py [OUT_DIR]

Reads bench/raw/2026-11-23-S5-server-parity (cap1 tarball needs `zstd` and `tar`) and
bench/raw/2026-11-22-S0-goldens + engine/tools/goldens/prompts.json. Writes JSON lines, nothing else.

Package F2 adds render_requests.jsonl (the 445 chat requests of requests.json.zst with config B's recorded prompt ids
from cap1: render / stream / nonstream captures and the S0 golden case) and render_s0.jsonl (the 29 requests of
s0-golden-chat-requests.json with the S0 capture's prompt ids), the oracle of tests/render.rs.
"""
import glob
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
S5 = os.path.join(ROOT, "bench/raw/2026-11-23-S5-server-parity")
S0 = os.path.join(ROOT, "bench/raw/2026-11-22-S0-goldens")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "engine/rust/testdata-local/parity")


def write(name, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)} rows")


def texts_of(content):
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str)]
    return []


def captures(tmp):
    subprocess.run(f"zstd -dc {S5}/cap1.tar.zst | tar -x -C {tmp}", shell=True, check=True)
    for d in (os.path.join(tmp, "cap1"), os.path.join(S5, "cap2"), os.path.join(S5, "cap3")):
        for f in sorted(glob.glob(d + "/*.json")):
            if not f.endswith("manifest.json"):
                yield os.path.basename(d) + "/" + os.path.basename(f), json.load(open(f, encoding="utf-8"))


with tempfile.TemporaryDirectory() as tmp:
    chat_prompts, completions, chat_outputs, seen = [], [], [], set()
    for src, x in captures(tmp):
        r, body = x.get("response") or {}, x["body"]
        skip = body.get("skip_special_tokens", True)
        if x["endpoint"] == "chat" and r.get("prompt_token_ids"):
            ids = r["prompt_token_ids"]
            if tuple(ids) not in seen:
                seen.add(tuple(ids))
                # message texts that contain special-token spellings: the renderer encodes them as ordinary text
                lit = [t for m in body.get("messages", []) for t in texts_of(m.get("content")) if "<|" in t]
                chat_prompts.append({"src": src, "ids": ids, "literal_texts": lit})
        if not r.get("choices"):
            continue
        ch = r["choices"][0]
        out = ch.get("token_ids")
        if x["endpoint"] == "completions" and out is not None:
            p = body["prompt"]
            completions.append({"src": src, "text": p if isinstance(p, str) else None,
                                "prompt_ids": ch.get("prompt_token_ids") or (p if isinstance(p, list) else None),
                                "out_ids": out, "out_text": ch["text"], "skip_special": skip,
                                "finish": ch["finish_reason"]})
        elif x["endpoint"] == "chat" and out is not None:
            m = ch["message"]
            chat_outputs.append({"src": src, "out_ids": out, "reasoning": m.get("reasoning"),
                                 "content": m.get("content"), "tool_calls": bool(m.get("tool_calls")),
                                 "skip_special": skip, "finish": ch["finish_reason"]})
    for boot in ("2026-11-22-g5-boot1", "2026-11-22-g5-boot2"):
        for f in sorted(glob.glob(f"{S0}/{boot}/cases/*.json")):
            c = json.load(open(f, encoding="utf-8"))
            if c["kind"] == "chat" and tuple(c["prompt_ids"]) not in seen:
                seen.add(tuple(c["prompt_ids"]))
                chat_prompts.append({"src": f"S0/{boot}/{c['id']}", "ids": c["prompt_ids"], "literal_texts": []})
    write("chat_prompts.jsonl", chat_prompts)
    write("completions.jsonl", completions)
    write("chat_outputs.jsonl", chat_outputs)

    def capture(path, field):
        """ids or error status config B recorded in one capture file, None when the file does not exist"""
        if not os.path.exists(path):
            return None
        d = json.load(open(path, encoding="utf-8"))
        if d["status"] != 200:
            return {"error": d["status"]}
        if field == "sse":
            return {"ids": json.loads(d["sse"][0]["data"]).get("prompt_token_ids")} if d["sse"] else {"error": 0}
        return {"ids": d["response"].get("prompt_token_ids")}

    def s0_ids(case):
        p = os.path.join(S0, "2026-11-22-g5-boot1", "cases", case + ".json")
        return {"ids": json.load(open(p, encoding="utf-8"))["prompt_ids"]} if os.path.exists(p) else None

    requests = json.loads(subprocess.run(f"zstd -dc {S5}/requests.json.zst", shell=True, check=True,
                                         capture_output=True).stdout)["requests"]
    rows = []
    for r in requests:
        if r["endpoint"] != "chat":
            continue
        base = os.path.join(tmp, "cap1", r["id"])
        refs = {"render": capture(base + ".render.json", "response"),
                "stream": capture(base + ".stream.jsonl", "sse"),
                "nonstream": capture(base + ".nonstream.json", "response")}
        if r["id"].startswith("s0-"):
            refs["s0-golden"] = s0_ids(r["id"][3:])
        rows.append({"id": r["id"], "group": r["group"], "body": r["body"],
                     "refs": {k: v for k, v in refs.items() if v is not None}})
    write("render_requests.jsonl", rows)
    s0 = json.load(open(f"{S5}/s0-golden-chat-requests.json", encoding="utf-8"))["requests"]
    write("render_s0.jsonl", [{"id": r["id"], "group": r["group"], "body": r["body"],
                               "refs": {"s0-golden": s0_ids(r["id"][3:])}} for r in s0])

# raw text cases: recorded ids are config B's /tokenize of the text, cut to raw_tokens
texts = {c["id"]: c for c in json.load(open(os.path.join(ROOT, "engine/tools/goldens/prompts.json")))["cases"]}
raw, seen_raw = [], set()
for boot in ("2026-11-22-g5-boot1", "2026-11-22-g5-boot2"):
    for f in sorted(glob.glob(f"{S0}/{boot}/cases/*.json")):
        c = json.load(open(f, encoding="utf-8"))
        if c["kind"] == "raw" and c["id"] in texts and (c["id"], len(c["prompt_ids"])) not in seen_raw:
            seen_raw.add((c["id"], len(c["prompt_ids"])))
            raw.append({"src": f"S0/{boot}/{c['id']}", "text": texts[c["id"]]["text"], "ids": c["prompt_ids"],
                        "full_count": c["prompt_info"].get("tokenize_full_count")})
write("raw_text.jsonl", raw)
