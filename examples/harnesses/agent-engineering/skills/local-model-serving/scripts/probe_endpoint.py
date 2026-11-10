"""Smoke-probe an OpenAI-compatible chat endpoint for the facts a scaffold
builder needs: the models and the context window the server advertises, time
to first token and tokens per second at one stream, whether replies carry
reasoning content, whether a schema-checked synthetic tool call comes back and
whether its round trip (call, tool result, final reply) completes, what
happens when the completion budget is smaller than the model's reasoning, and
throughput at a given concurrency. It does not establish usable context
capacity or the runner's own tool round trip.

Usage:
  probe_endpoint.py --base-url http://localhost:8000/v1 --model NAME
                    [--api-key-env NAME] [--sampling '{"temperature": 0.6}']
                    [--concurrency 6] [--max-tokens 256] [--timeout 180]

Prints one JSON object on stdout. The sampling settings are the task's and are
passed through as given; the probe imposes none. The API key, when one is
named, is read from the environment variable and never printed. Standard
library only.
"""
import argparse
import http.client
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

PROBE_ERRORS = (OSError, http.client.HTTPException, json.JSONDecodeError)
SAMPLING_KEYS = {"temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "seed",
                 "reasoning_effort", "chat_template_kwargs", "thinking_token_budget", "extra_body"}
PROSE_PROMPT = "Explain in about 150 words how a queue differs from a stack."
TOOL = {
    "type": "function",
    "function": {
        "name": "record_note",
        "description": "Record a short note with a priority and a count.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "the note"},
                "priority": {"type": "string", "enum": ["low", "high"]},
                "count": {"type": "integer"},
            },
            "required": ["text", "priority", "count"],
        },
    },
}
TOOL_PROMPT = ('Call the record_note tool exactly once with text set to the two-line string "first line" '
               'then a newline then "second \\"quoted\\" line", priority high, and count 17. Do not answer in prose.')
REASONING_PROMPT = "Find the smallest positive integer divisible by 8, 12 and 15. Answer with the integer only."


def headers(api_key_env):
    out = {"Content-Type": "application/json"}
    if api_key_env:
        key = os.environ.get(api_key_env)
        if not key:
            raise SystemExit(f"environment variable {api_key_env} is not set")
        out["Authorization"] = f"Bearer {key}"
    return out


def get_json(url, hdrs, timeout):
    request = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url, payload, hdrs, timeout):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def stream_completion(url, payload, hdrs, timeout):
    """One streamed completion: ttft seconds, total seconds, usage, whether
    reasoning content appeared, and the first characters of content."""
    payload = dict(payload, stream=True, stream_options={"include_usage": True})
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    started = time.time()
    ttft = None
    usage = None
    reasoning = False
    content = []
    chunks = 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            event = json.loads(body)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("reasoning_content") or delta.get("reasoning"):
                    reasoning = True
                    if ttft is None:
                        ttft = time.time() - started
                if delta.get("content"):
                    chunks += 1
                    if ttft is None:
                        ttft = time.time() - started
                    content.append(delta["content"])
    total = time.time() - started
    return {"ttft_s": ttft, "seconds": total, "usage": usage, "reasoning_seen": reasoning,
            "content_chunks": chunks, "content_preview": "".join(content)[:120]}


def validate_call(calls):
    """Whether exactly one call matches the probe tool's schema, with the argument check's verdicts."""
    if len(calls) != 1:
        return False, {"calls": len(calls)}
    call = calls[0]
    function = call.get("function") or {}
    raw = function.get("arguments")
    verdicts = {
        "type_function": call.get("type") == "function",
        "id_present": isinstance(call.get("id"), str) and bool(call["id"]),
        "name_matches": function.get("name") == TOOL["function"]["name"],
        "arguments_parse": False, "text_string": False, "priority_enum": False, "count_integer": False,
    }
    parsed = raw if isinstance(raw, dict) else None
    if isinstance(raw, str):
        # A malformed argument string is a finding of the probe, not a crash.
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
    if isinstance(parsed, dict):
        verdicts["arguments_parse"] = True
        verdicts["text_string"] = isinstance(parsed.get("text"), str) and "\n" in parsed["text"] and '"' in parsed["text"]
        verdicts["priority_enum"] = parsed.get("priority") in ("low", "high")
        verdicts["count_integer"] = isinstance(parsed.get("count"), int) and not isinstance(parsed.get("count"), bool)
    return all(verdicts.values()), verdicts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--sampling", default="{}", help="JSON object of request settings passed through as given")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.concurrency < 1 or args.max_tokens < 1 or args.timeout <= 0:
        parser.error("concurrency, max-tokens and timeout must be positive")
    sampling = json.loads(args.sampling)
    if not isinstance(sampling, dict) or set(sampling) - SAMPLING_KEYS:
        parser.error(f"--sampling must be a JSON object with keys among {sorted(SAMPLING_KEYS)}")
    base = args.base_url.rstrip("/")
    hdrs = headers(args.api_key_env)
    report = {"base_url": base, "model": args.model, "sampling": sampling, "probed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "scope": "short-request smoke probe; usable context capacity and the runner's tool round trip are tested through the adapter"}

    try:
        models = get_json(f"{base}/models", hdrs, args.timeout)
        rows = models.get("data") or []
        report["models"] = [row.get("id") for row in rows]
        report["advertised_context_window"] = next((row.get("max_model_len") for row in rows if row.get("id") == args.model and row.get("max_model_len")), None)
    except PROBE_ERRORS as error:
        report["models_error"] = str(error)[:200]

    chat = f"{base}/chat/completions"
    base_payload = dict(sampling, model=args.model, max_tokens=args.max_tokens)

    try:
        one = stream_completion(chat, dict(base_payload, messages=[{"role": "user", "content": PROSE_PROMPT}]), hdrs, args.timeout)
        completion_tokens = (one["usage"] or {}).get("completion_tokens")
        one["tokens_per_second"] = round(completion_tokens / one["seconds"], 1) if completion_tokens and one["seconds"] else None
        report["one_stream"] = one
    except PROBE_ERRORS as error:
        report["one_stream_error"] = str(error)[:200]

    try:
        messages = [{"role": "user", "content": TOOL_PROMPT}]
        tool_reply = post_json(chat, dict(base_payload, messages=messages, tools=[TOOL], tool_choice="auto"), hdrs, args.timeout)
        choice = (tool_reply.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls = message.get("tool_calls") or []
        valid, verdicts = validate_call(calls)
        report["tool_call"] = {
            "returned": bool(calls),
            "valid_for_probe_schema": valid,
            "verdicts": verdicts,
            "content_preview": (message.get("content") or "")[:120],
            "finish_reason": choice.get("finish_reason"),
        }
        if valid:
            call = calls[0]
            followup = messages + [
                {"role": "assistant", "content": message.get("content") or None, "tool_calls": calls},
                {"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"recorded": True, "note_id": 17})},
            ]
            final = post_json(chat, dict(base_payload, messages=followup, tools=[TOOL], tool_choice="auto"), hdrs, args.timeout)
            final_choice = (final.get("choices") or [{}])[0]
            final_message = final_choice.get("message") or {}
            report["tool_call"]["round_trip"] = {
                "completed": bool((final_message.get("content") or "").strip()) and not final_message.get("tool_calls"),
                "finish_reason": final_choice.get("finish_reason"),
                "content_preview": (final_message.get("content") or "")[:120],
            }
    except PROBE_ERRORS as error:
        report["tool_call_error"] = str(error)[:200]

    try:
        small = post_json(chat, dict(base_payload, max_tokens=24, messages=[{"role": "user", "content": REASONING_PROMPT}]), hdrs, args.timeout)
        choice = (small.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        report["small_budget_reply"] = {
            "finish_reason": choice.get("finish_reason"),
            "content_empty": not (message.get("content") or "").strip(),
            "reasoning_present": bool(message.get("reasoning_content") or message.get("reasoning")),
            "usage": small.get("usage"),
        }
    except PROBE_ERRORS as error:
        report["small_budget_error"] = str(error)[:200]

    if args.concurrency > 1:
        results = []
        lock = threading.Lock()

        def worker():
            try:
                row = stream_completion(chat, dict(base_payload, messages=[{"role": "user", "content": PROSE_PROMPT}]), hdrs, args.timeout)
            except PROBE_ERRORS as error:
                row = {"error": str(error)[:200]}
            with lock:
                results.append(row)

        started = time.time()
        threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        elapsed = time.time() - started
        usages = [(row.get("usage") or {}).get("completion_tokens") for row in results if not row.get("error")]
        tokens = sum(usages) if usages and all(u is not None for u in usages) else None
        report["concurrent"] = {
            "streams": args.concurrency,
            "completed_streams": sum(1 for row in results if not row.get("error")),
            "errors": sum(1 for row in results if row.get("error")),
            "seconds": round(elapsed, 1),
            "total_tokens_per_second": round(tokens / elapsed, 1) if tokens is not None and elapsed else None,
            "ttft_s_max": max((row.get("ttft_s") or 0) for row in results) if results else None,
        }

    json.dump(report, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
