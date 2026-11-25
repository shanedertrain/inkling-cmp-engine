#!/usr/bin/env python3
"""MiaAI-Lab protocol decode benchmark (1/2/4/8 users x structured/code/prose, T=0, 400-tok cap)."""
import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request

PROMPTS = {
    "structured": {
        "1": ["Count from 1 to 200. Output only the numbers, separated by spaces. No other text."],
        "2": [f"Request ID: decode-{i}\nCount from 1 to 200. Output only the numbers, separated by spaces. No other text." for i in range(1, 3)],
        "4": [f"Request ID: decode-{i}\nCount from 1 to 200. Output only the numbers, separated by spaces. No other text." for i in range(1, 5)],
        "8": [f"Request ID: decode-{i}\nCount from 1 to 200. Output only the numbers, separated by spaces. No other text." for i in range(1, 9)],
    },
    "code": {
        "1": ["binary_search\ndef binary_search(nums, target) -> int: index of target in a sorted list, or -1.\nOutput only Python source. No comments, no docstrings, no markdown fences. Then add tests and the helpers this needs. Keep writing code."],
        "2": [
            "binary_search\ndef binary_search(nums, target) -> int: index of target in a sorted list, or -1.\nOutput only Python source. No comments, no docstrings, no markdown fences. Then add tests and the helpers this needs. Keep writing code.",
            "merge_sort\ndef merge_sort(nums) -> list: stable sort of a list of ints, returning a new list.\nOutput only Python source. No comments, no docstrings, no markdown fences. Then add tests and the helpers this needs. Keep writing code.",
        ],
        "4": [
            "binary_search\ndef binary_search(nums, target) -> int: index of target in a sorted list, or -1.\nOutput only Python source. No comments, no docstrings, no markdown fences. Then add tests and the helpers this needs. Keep writing code.",
            "merge_sort\ndef merge_sort(nums) -> list: stable sort of a list of ints, returning a new list.\nOutput only Python source. No comments, no docstrings, no markdown fences. Then add tests and the helpers this needs. Keep writing code.",
            "lru_cache\nclass LRUCache: get(key) and put(key, value) with a fixed capacity, evicting the least recently used.\nOutput only Python source. No comments, no docstrings, no markdown fences. Then add tests and the helpers this needs. Keep writing code.",
            "token_bucket\nclass TokenBucket: allow(n) consumes n tokens refilled at a fixed rate, else returns False.\nOutput only Python source. No comments, no docstrings, no markdown fences. Then add tests and the helpers this needs. Keep writing code.",
        ],
        "8": [
            "binary_search\ndef binary_search(nums, target) -> int: index of target in a sorted list, or -1.\nOutput only Python source. No comments, no docstrings, no markdown fences. Then add tests and the helpers this needs. Keep writing code.",
            "merge_sort\ndef merge_sort(nums) -> list: stable sort of a list of ints, returning a new list.\nOutput only Python source. No comments, no docstrings, no markdown fences. Then add tests and the helpers this needs. Keep writing code.",
            "lru_cache\nclass LRUCache: get(key) and put(key, value) with a fixed capacity, evicting the least recently used.\nOutput only Python source. No comments, no docstrings, no markdown fences. Then add tests and the helpers this needs. Keep writing code.",
            "token_bucket\nclass TokenBucket: allow(n) consumes n tokens refilled at a fixed rate, else returns False.\nOutput only Python source. No comments, no docstrings, no markdown fences. Then add tests and the helpers this needs. Keep writing code.",
            "ring_buffer\nclass RingBuffer: push and pop over a fixed-capacity array, raising on overflow and underflow.\nOutput only Python source. No comments, no docstrings, no markdown fences. Then add tests and the helpers this needs. Keep writing code.",
            "dijkstra\ndef dijkstra(graph, src) -> dict: shortest path weights from src on a non-negative weighted graph.\nOutput only Python source. No comments, no docstrings, no markdown fences. Then add tests and the helpers this needs. Keep writing code.",
            "edit_distance\ndef edit_distance(a, b) -> int: Levenshtein distance between two strings.\nOutput only Python source. No comments, no docstrings, no markdown fences. Then add tests and the helpers this needs. Keep writing code.",
            "semver_cmp\ndef semver_cmp(a, b) -> int: compare dotted numeric versions, negative if a < b.\nOutput only Python source. No comments, no docstrings, no markdown fences. Then add tests and the helpers this needs. Keep writing code.",
        ],
    },
    "prose": {
        "1": ["Write a detailed step-by-step explanation of how a hash map works, including collision handling, resizing, and time complexity. Be thorough."],
        "2": [f"Request ID: decode-{i}\nWrite a detailed step-by-step explanation of how a hash map works, including collision handling, resizing, and time complexity. Be thorough." for i in range(1, 3)],
        "4": [f"Request ID: decode-{i}\nWrite a detailed step-by-step explanation of how a hash map works, including collision handling, resizing, and time complexity. Be thorough." for i in range(1, 5)],
        "8": [f"Request ID: decode-{i}\nWrite a detailed step-by-step explanation of how a hash map works, including collision handling, resizing, and time complexity. Be thorough." for i in range(1, 9)],
    },
}


def call_one(url: str, model: str, prompt: str, tag: str, max_tokens: int) -> tuple[int, float, float]:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": f"[{tag}] {prompt}"}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = None
    out_tok = 0
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            if t_first is None and ev.get("choices"):
                delta = ev["choices"][0].get("delta", {})
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    t_first = time.perf_counter()
            if ev.get("usage"):
                out_tok = ev["usage"].get("completion_tokens", out_tok)
    t1 = time.perf_counter()
    ttft = (t_first - t0) if t_first else (t1 - t0)
    dec_s = (t1 - t_first) if t_first else (t1 - t0)
    per_s = (out_tok - 1) / dec_s if dec_s > 0 else 0.0
    return out_tok, ttft, per_s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8410/v1/chat/completions")
    ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--max-tokens", type=int, default=400)
    args = ap.parse_args()

    for n in ("1", "2", "4", "8"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(n)) as ex:
            list(ex.map(lambda p: call_one(args.url, args.model, p, f"warm-{n}", args.max_tokens), PROMPTS["structured"][n]))

    out = {}
    for n in ("1", "2", "4", "8"):
        reps = 5 if n == "1" else 3
        for cat in ("structured", "code", "prose"):
            aggs, pers, ttfts = [], [], []
            for r in range(reps):
                t0 = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=int(n)) as ex:
                    res = list(ex.map(lambda p: call_one(args.url, args.model, p, f"n{n}-{cat}-r{r}", args.max_tokens), PROMPTS[cat][n]))
                wall = time.perf_counter() - t0
                tot = sum(x[0] for x in res)
                aggs.append(res[0][2] if n == "1" else tot / wall)
                pers.append(statistics.median(x[2] for x in res))
                ttfts.append(statistics.median(x[1] for x in res) * 1000.0)
            row = {
                "agg": round(statistics.median(aggs), 1),
                "per": round(statistics.median(pers), 1),
                "ttft_ms": round(statistics.median(ttfts), 1),
            }
            out[f"n{n}_{cat}"] = row
            print(f"n={n} {cat:10s}: agg={row['agg']:6.1f}  per={row['per']:5.1f}  ttft={row['ttft_ms']:5.1f}ms", flush=True)
    print("JSON_RESULT=" + json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
