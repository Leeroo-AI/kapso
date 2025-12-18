# ModularModelDetector Implementation

## Metadata

| Attribute | Value |
|-----------|-------|
| **Source File** | `/tmp/praxium_repo_d5p6fp4d/utils/modular_model_detector.py` |
| **Repository** | huggingface_transformers |
| **Commit** | f9f6619c2cf7 |
| **Lines of Code** | 913 |
| **Primary Domain** | Code Similarity Analysis |
| **Secondary Domains** | Embedding-based Search, Abstract Syntax Tree (AST) Analysis |
| **Last Updated** | 2025-12-18 |

## Overview

The `modular_model_detector.py` module provides tools for detecting code similarities between model implementations in the HuggingFace Transformers library. It uses both embedding-based and token-based (Jaccard) similarity metrics to identify similar code patterns across different model definitions, helping developers identify which models can be modularized by finding existing similar implementations.

The detector analyzes Python source code at the class and function level, computing semantic similarity scores using transformer-based embeddings and lexical similarity using token overlap metrics.

## Description

### Core Functionality

The modular model detector provides several key capabilities:

1. **Code Similarity Detection**: Analyzes Python files to find similar class and function definitions across the codebase using both semantic (embedding) and lexical (token-based) similarity

2. **Embedding-based Search**: Uses large language model embeddings (Qwen3-Embedding-4B by default) to capture semantic code similarity, enabling detection of functionally similar code even with different syntax

3. **Jaccard Token Similarity**: Computes token-set overlap using Jaccard index as a complementary lexical similarity metric

4. **Code Sanitization**: Normalizes code by removing model-specific identifiers, docstrings, and comments to focus on structural similarity

5. **Index Management**: Builds and maintains searchable indexes of all model definitions, with Hub integration for sharing precomputed embeddings

6. **Release Date Integration**: Cross-references model definitions with release dates from documentation to identify the oldest similar implementations

7. **Ranked Results**: Provides top-k similar definitions with scores, highlighting best matches, oldest implementations, and overall closest candidates

### Architecture Components

**Core Classes**:
- `CodeSimilarityAnalyzer`: Main class orchestrating similarity detection, index management, and result ranking
- Uses HuggingFace AutoModel/AutoTokenizer for embedding generation
- Integrates with HuggingFace Hub for index storage and retrieval

**Key Data Structures**:
- **Embeddings Index**: Safetensors file containing normalized embeddings for all definitions
- **Index Map**: JSON mapping integer indices to file paths and definition names
- **Tokens Map**: JSON mapping definition identifiers to sorted token lists
- **Release Date Map**: Dictionary mapping model names to release dates

**Similarity Metrics**:
1. **Embedding Similarity**: Cosine similarity on normalized embeddings (dot product of unit vectors)
2. **Jaccard Similarity**: `|intersection| / |union|` of token sets
3. **Intersection**: Definitions appearing in both top-k results

### Key Algorithms

1. **Code Sanitization**: Strips docstrings, comments, and imports, then replaces model-specific identifiers with generic "Model" placeholder
2. **Embedding Generation**: Batch processing with mean pooling over attention-masked tokens
3. **Top-K Search**: Efficient top-k retrieval using numpy's `argpartition` followed by sorting
4. **Multi-Metric Ranking**: Combines embedding scores, Jaccard scores, and release dates to highlight best candidates

## Usage

### Build Index

Build the code similarity index from all modeling files:

```bash
python utils/modular_model_detector.py --build
```

Build and push to Hub:

```bash
python utils/modular_model_detector.py --build --push-new-index
```

### Analyze a File

Analyze a modeling file to find similar implementations:

```bash
# Using model name (if in standard location)
python utils/modular_model_detector.py --modeling-file llama

# Using full path
python utils/modular_model_detector.py --modeling-file src/transformers/models/bert/modeling_bert.py
```

Enable Jaccard similarity metric:

```bash
python utils/modular_model_detector.py --modeling-file vit --use_jaccard True
```

Use custom Hub dataset:

```bash
python utils/modular_model_detector.py \
  --modeling-file gpt2 \
  --hub-dataset my-org/my-code-embeddings
```

## Code Reference

### Main Class

```python
class CodeSimilarityAnalyzer:
    """
    Analyzer for detecting code similarities between model implementations.

    This class uses embedding-based and token-based similarity metrics to identify
    similar code patterns across different model definitions in the transformers library.

    Args:
        hub_dataset: The Hub dataset repository ID containing the code embeddings index.
    """

    def __init__(self, hub_dataset: str):
        """
        Initialize the analyzer with embedding model and Hub dataset.

        Args:
            hub_dataset: Hub dataset repo ID (e.g., "hf-internal-testing/transformers_code_embeddings")
        """

    def ensure_local_index(self) -> None:
        """Ensure index files are available locally, preferring Hub cache snapshots."""

    def build_index(self) -> None:
        """Build the code similarity index from all modeling files and save to disk."""

    def analyze_file(
        self,
        modeling_file: Path,
        top_k_per_item: int = 5,
        allow_hub_fallback: bool = True,
        use_jaccard: bool = False
    ) -> dict[str, dict[str, list]]:
        """
        Analyze a modeling file and find similar code definitions in the index.

        Args:
            modeling_file: Path to the modeling file to analyze
            top_k_per_item: Number of top matches to return per definition
            allow_hub_fallback: Whether to download index from Hub if not found locally
            use_jaccard: Whether to compute Jaccard similarity in addition to embeddings

        Returns:
            Dictionary mapping definition names to their similarity results.
            Each result contains 'embedding', 'jaccard', and 'intersection' keys.
        """

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings, processing in batches.

        Args:
            texts: List of text strings to encode

        Returns:
            Stacked embeddings for all texts
        """
```

### Utility Functions

```python
def _normalize(string: str | None) -> str:
    """
    Normalize a string by removing all non-alphanumeric characters and converting to lowercase.

    Args:
        string: The string to normalize

    Returns:
        The normalized string, or empty string if input is None
    """

def _strip_source_for_tokens(code: str) -> str:
    """
    Strip docstrings, comments, and import statements from source code.

    Args:
        code: The source code to strip

    Returns:
        The stripped source code
    """

def _tokenize(code: str) -> set[str]:
    """
    Extract all Python identifiers from source code.

    Args:
        code: The source code to tokenize

    Returns:
        A set of all identifiers found in the code
    """

def _sanitize_for_embedding(code: str, model_hint: str | None, symbol_hint: str | None) -> str:
    """
    Sanitize code for embedding by replacing model-specific identifiers with generic placeholder.

    Args:
        code: The source code to sanitize
        model_hint: Hint about the model name (e.g., 'llama')
        symbol_hint: Hint about the symbol name (e.g., 'LlamaAttention')

    Returns:
        The sanitized code with model-specific identifiers replaced by 'Model'
    """

def build_date_data() -> dict[str, str]:
    """
    Scan Markdown files in docs/source/en/model_doc and build mapping of model_id to release date.

    Returns:
        Dictionary mapping model_id to ISO date string (YYYY-MM-DD)
    """
```

### Private Methods

```python
def _extract_definitions(
    self,
    file_path: Path,
    relative_to: Path | None = None,
    model_hint: str | None = None
) -> tuple[dict[str, str], dict[str, str], dict[str, list[str]], dict[str, str]]:
    """
    Extract class and function definitions from a Python file.

    Args:
        file_path: Path to the Python file to parse
        relative_to: Base path for computing relative identifiers
        model_hint: Model name hint for sanitization

    Returns:
        Tuple containing:
        - definitions_raw: Mapping of identifiers to raw source code
        - definitions_sanitized: Mapping of identifiers to sanitized source code
        - definitions_tokens: Mapping of identifiers to sorted token lists
        - definitions_kind: Mapping of identifiers to "class" or "function"
    """

def _encode_batch(self, texts: list[str]) -> np.ndarray:
    """
    Encode a batch of texts into normalized embeddings.

    Args:
        texts: List of text strings to encode

    Returns:
        Normalized embeddings as a float32 numpy array
    """

def _topk_embedding(
    self,
    query_embedding_row: np.ndarray,
    base_embeddings: np.ndarray,
    identifier_map: dict[int, str],
    self_model_normalized: str,
    self_name: str,
    k: int,
) -> list[tuple[str, float]]:
    """
    Find top-k most similar definitions using embedding cosine similarity.

    Excludes results from the same model and with the same name.
    """

def _topk_jaccard(
    self,
    query_tokens: set[str],
    identifiers: list[str],
    tokens_map: dict[str, list[str]],
    self_model_normalized: str,
    self_name: str,
    k: int,
) -> list[tuple[str, float]]:
    """
    Find top-k most similar definitions using Jaccard similarity on token sets.

    Args:
        query_tokens: Set of tokens from the query definition
        identifiers: List of all definition identifiers in the index
        tokens_map: Mapping of identifiers to their token lists
        self_model_normalized: Normalized name of the query model to exclude
        self_name: Name of the query definition to exclude
        k: Number of top results to return

    Returns:
        List of (identifier, score) tuples
    """
```

### Constants

```python
MODELS_ROOT = Path("src/transformers/models")
EMBEDDINGS_PATH = "embeddings.safetensors"
INDEX_MAP_PATH = "code_index_map.json"
TOKENS_PATH = "code_index_tokens.json"
HUB_DATASET_DEFAULT = "hf-internal-testing/transformers_code_embeddings"

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
BATCH_SIZE = 16
MAX_LENGTH = 4096

# ANSI color codes for CLI output styling
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_HEADER = "\033[1;36m"
ANSI_HIGHLIGHT_TOP = "\033[1;32m"
ANSI_HIGHLIGHT_OLD = "\033[1;33m"
ANSI_HIGHLIGHT_CANDIDATE = "\033[1;34m"
```

## I/O Contract

### Input Specifications

| Input | Type | Description | Required | Default |
|-------|------|-------------|----------|---------|
| modeling_file | `str` | Path or name of modeling file to analyze | No* | N/A |
| build | `bool` | Build index from scratch | No* | `False` |
| push_new_index | `bool` | Push index to Hub after building | No | `False` |
| hub_dataset | `str` | Hub dataset repo ID for index | No | `"hf-internal-testing/transformers_code_embeddings"` |
| use_jaccard | `bool` | Enable Jaccard similarity metric | No | `False` |

*Either `modeling_file` or `build` must be specified

**Modeling File Format**:
- Must be a Python file containing class and function definitions
- Should follow naming pattern `modeling_<model>.py`
- Can be specified as model name (e.g., "llama") or full path

### Output Specifications

| Output | Type | Description |
|--------|------|-------------|
| Similarity Results | `dict[str, dict]` | Mapping of definitions to similarity results |
| Console Report | Formatted text | Colored table showing ranked similar definitions |
| Index Files | `.safetensors`, `.json` | Persisted index for reuse |

**Result Structure**:
```python
{
    "ClassName": {
        "kind": "class",  # or "function"
        "embedding": [
            ("path/to/file.py:SimilarClass", 0.9876),
            ("other/file.py:AnotherClass", 0.9543),
            ...
        ],
        "jaccard": [  # if use_jaccard=True
            ("path/to/file.py:SimilarClass", 0.7654),
            ...
        ],
        "intersection": {  # if use_jaccard=True
            "path/to/file.py:SimilarClass",
            ...
        }
    },
    ...
}
```

**Console Output Features**:
- Color-coded results (green=highest match, yellow=oldest within 0.1, blue=closest overall)
- Grouped by classes and functions
- Shows release dates for context
- Links to GitHub Action job pages with line numbers
- Summary of closest overall candidate with aggregated scores

### Side Effects

1. **File System**:
   - Reads all `modeling_*.py` files in `src/transformers/models/`
   - Creates index files in current directory if building
   - Downloads index from Hub to cache if needed

2. **Network**:
   - Downloads embedding model from HuggingFace Hub (if not cached)
   - Downloads precomputed index from Hub (if not found locally)
   - Uploads index to Hub (if `--push-new-index` specified)

3. **Compute**:
   - GPU/CPU usage for embedding generation
   - Memory usage proportional to index size and batch size

4. **Console**:
   - Prints progress bars during encoding
   - Prints formatted similarity tables with color coding

## Usage Examples

### Example 1: Analyze a New Model

Analyze a newly created modeling file to find similar implementations:

```bash
python utils/modular_model_detector.py --modeling-file my_new_model.py
```

Example output:
```
Loading checkpoint shards: 100%|████████████████| 2/2 [00:00<00:00, 33.62it/s]
encoding 21 query definitions with Qwen/Qwen3-Embedding-4B (device=cuda, batch=16, max_length=4096)

Closest overall candidate: .../llama/modeling_llama.py (release: 2023-02-24, total score: 18.9432)

Classes
Symbol                | Path                                    | Score  | Release
---------------------|---------------------------------------|--------|------------
MyNewAttention       |                                       |        |
                     | llama/modeling_llama.py:151 (LlamaAttention) | 0.9876 | 2023-02-24
                     | mistral/modeling_mistral.py:203 (MistralAttention) | 0.9654 | 2023-09-27
                     | phi/modeling_phi.py:112 (PhiAttention) | 0.9432 | 2023-09-11
```

### Example 2: Build Custom Index

Build index for a subset of models:

```python
from pathlib import Path
from utils.modular_model_detector import CodeSimilarityAnalyzer

# Initialize analyzer
analyzer = CodeSimilarityAnalyzer(hub_dataset="my-org/my-index")

# Manually collect specific files
analyzer.models_root = Path("src/transformers/models")
files = [
    analyzer.models_root / "bert" / "modeling_bert.py",
    analyzer.models_root / "gpt2" / "modeling_gpt2.py",
    # ... more files
]

# Build index
analyzer.build_index()
```

### Example 3: Programmatic Analysis

```python
from pathlib import Path
from utils.modular_model_detector import CodeSimilarityAnalyzer

# Initialize
analyzer = CodeSimilarityAnalyzer(
    hub_dataset="hf-internal-testing/transformers_code_embeddings"
)

# Analyze file
modeling_file = Path("src/transformers/models/my_model/modeling_my_model.py")
results = analyzer.analyze_file(
    modeling_file,
    top_k_per_item=10,
    use_jaccard=True
)

# Process results
for definition_name, data in results.items():
    print(f"\n{definition_name} ({data['kind']}):")

    # Get top embedding match
    if data['embedding']:
        top_match, top_score = data['embedding'][0]
        print(f"  Best match: {top_match} (score: {top_score:.4f})")

    # Get intersection with Jaccard
    if 'intersection' in data and data['intersection']:
        print(f"  Confirmed by Jaccard: {len(data['intersection'])} matches")
        for identifier in data['intersection']:
            print(f"    - {identifier}")
```

### Example 4: Find Modularization Candidates

Identify which models can inherit from an existing implementation:

```python
from utils.modular_model_detector import CodeSimilarityAnalyzer

analyzer = CodeSimilarityAnalyzer(hub_dataset="hf-internal-testing/transformers_code_embeddings")

# Analyze new model
results = analyzer.analyze_file(
    Path("src/transformers/models/new_model/modeling_new_model.py"),
    top_k_per_item=5
)

# Find classes with high similarity to existing implementations
candidates = {}
for class_name, data in results.items():
    if data['kind'] == 'class' and data['embedding']:
        top_match, top_score = data['embedding'][0]
        if top_score > 0.95:  # Very similar
            candidates[class_name] = {
                'similar_to': top_match,
                'score': top_score
            }

print("Modularization candidates:")
for class_name, info in candidates.items():
    print(f"  {class_name} -> {info['similar_to']} ({info['score']:.4f})")
    print(f"    Consider inheriting from this class in modular definition")
```

### Example 5: Compare Models Across Releases

```python
from utils.modular_model_detector import CodeSimilarityAnalyzer, build_date_data

analyzer = CodeSimilarityAnalyzer(hub_dataset="hf-internal-testing/transformers_code_embeddings")
dates = build_date_data()

# Analyze model
results = analyzer.analyze_file(Path("src/transformers/models/model_a/modeling_model_a.py"))

# Find oldest similar implementations
for class_name, data in results.items():
    if data['kind'] == 'class':
        print(f"\n{class_name}:")

        # Sort by release date
        matches_with_dates = []
        for identifier, score in data['embedding'][:5]:
            path, match_name = identifier.split(":", 1)
            model = Path(path).parts[0]
            release = dates.get(model, "unknown")
            matches_with_dates.append((identifier, score, release))

        matches_with_dates.sort(key=lambda x: x[2])  # Sort by date

        print(f"  Oldest similar implementation: {matches_with_dates[0][0]}")
        print(f"    Released: {matches_with_dates[0][2]}")
        print(f"    Similarity: {matches_with_dates[0][1]:.4f}")
```

## Related Pages

(To be populated as wiki structure develops)

---

**Note**: This is an internal tool for HuggingFace Transformers maintainers to identify modularization opportunities by detecting code similarities across model implementations. It requires significant computational resources (GPU/VRAM recommended) for embedding generation.
