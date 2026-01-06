# Principle: Result Serialization

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
* [[source::Doc|Python JSON|https://docs.python.org/3/library/json.html]]
|-
! Domains
| [[domain::IPC]], [[domain::Serialization]], [[domain::Code_Execution]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Principle for serializing execution results and errors for transmission across process boundaries using length-prefixed JSON encoding.

=== Description ===

Result Serialization handles the output pathway from subprocess to parent:

1. **Result Packaging**: Wraps execution output with captured print statements
2. **Error Packaging**: Formats exceptions with message, description, stack trace, and stderr
3. **JSON Encoding**: Serializes to UTF-8 JSON with `default=str` for non-serializable types
4. **Length Prefixing**: Prepends 4-byte big-endian length prefix for reliable parsing
5. **Pipe Writing**: Writes length + data atomically to file descriptor

This pattern ensures:
- **Reliable Parsing**: Length prefix allows exact read sizes
- **Error Preservation**: Full context captured for debugging
- **Type Safety**: Non-serializable objects converted to strings
- **Print Capture**: Console output forwarded to caller

=== Usage ===

Apply this principle when:
- Implementing IPC for code execution results
- Building systems that need to capture both output and errors
- Creating reliable message framing over byte streams
- Designing subprocess communication protocols

== Theoretical Basis ==

Result serialization follows a **Length-Prefixed Message** pattern:

<syntaxhighlight lang="python">
# Pseudo-code for result serialization

def put_result(write_fd, result, print_args):
    # 1. Create message structure
    message = {
        "result": result,
        "print_args": truncate_print_args(print_args)
    }

    # 2. Serialize to JSON bytes
    data = json.dumps(message, default=str, ensure_ascii=False).encode("utf-8")

    # 3. Create length prefix (4 bytes, big-endian)
    length_bytes = len(data).to_bytes(4, "big")

    # 4. Write atomically: length + data
    write_all(write_fd, length_bytes)
    write_all(write_fd, data)

    # 5. Close file descriptor
    os.close(write_fd)

def put_error(write_fd, exception, stderr, print_args):
    error_info = {
        "message": str(exception),
        "description": getattr(exception, "description", ""),
        "stack": traceback.format_exc(),
        "stderr": stderr,
    }
    message = {
        "error": error_info,
        "print_args": truncate_print_args(print_args)
    }
    # ... same serialization pattern
</syntaxhighlight>

Message format:
```
[4 bytes: length][N bytes: JSON data]
```

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_TaskExecutor_put_result]]
