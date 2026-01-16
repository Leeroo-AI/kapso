# Heuristic: Streaming_Download_Large_Files

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|picoGPT|https://github.com/jaymody/picoGPT]]
* [[source::Doc|Requests Documentation|https://requests.readthedocs.io/en/latest/user/advanced/#streaming-requests]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==
Download large files using streaming with chunked iteration to avoid loading entire files into memory.

=== Description ===
When downloading large model checkpoint files (several GB), loading the entire response into memory before writing to disk can cause out-of-memory errors. Using `stream=True` with requests and iterating over chunks allows processing the download incrementally, keeping memory usage constant regardless of file size.

=== Usage ===
Use this heuristic when implementing **File Downloads** for large model weights, checkpoints, or datasets. Essential for downloading files larger than available RAM, and recommended for any file over ~100MB.

== The Insight (Rule of Thumb) ==
* **Action:** Use `requests.get(url, stream=True)` and iterate with `.iter_content(chunk_size)`
* **Value:** `chunk_size=1000` (approximately Ethernet packet size of ~1500 bytes)
* **Trade-off:** Slightly more code complexity; negligible performance difference for small files
* **Compatibility:** Works with any HTTP server; requires `requests` library

== Reasoning ==
Without streaming:
1. `requests.get(url)` downloads the entire response body into memory
2. For a 6GB checkpoint file, this requires 6GB+ of RAM just for the download
3. Only after the full download completes can you write to disk

With streaming:
1. `requests.get(url, stream=True)` only downloads headers initially
2. `.iter_content(chunk_size=1000)` yields small chunks
3. Each chunk is written to disk immediately, then garbage collected
4. Memory usage stays at ~1KB regardless of file size

The chunk size of 1000 bytes is chosen because:
- Ethernet MTU (Maximum Transmission Unit) is typically ~1500 bytes
- Smaller chunks = more Python loop iterations (overhead)
- Larger chunks = more memory usage
- 1000 bytes balances these tradeoffs well

== Code Evidence ==

From `utils.py:L23-41` (download function):
<syntaxhighlight lang="python">
url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
r = requests.get(f"{url}/{model_size}/{filename}", stream=True)
r.raise_for_status()

with open(os.path.join(model_dir, filename), "wb") as f:
    file_size = int(r.headers["content-length"])
    chunk_size = 1000
    with tqdm(
        ncols=100,
        desc="Fetching " + filename,
        total=file_size,
        unit_scale=True,
        unit="b",
    ) as pbar:
        # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(chunk_size)
</syntaxhighlight>

Key elements:
1. Line 25: `stream=True` enables streaming mode
2. Line 30: `chunk_size = 1000` sets the iteration size
3. Line 38-40: Comment explains the chunk size rationale
4. Line 39-41: Iterate, write, and update progress bar for each chunk

The `tqdm` progress bar provides user feedback during long downloads.

== Related Pages ==
* [[used_by::Implementation:Jaymody_PicoGPT_Download_Gpt2_Files]]
* [[used_by::Principle:Jaymody_PicoGPT_Model_Download]]
