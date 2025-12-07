# File Learner
#
# Extracts knowledge from generic files (text, markdown, code).
# A versatile learner for any file-based input.

from typing import Any, Dict, List
import os

from src.knowledge.learners.base import Learner, KnowledgeChunk
from src.knowledge.learners.factory import register_learner


@register_learner("file")
class FileLearner(Learner):
    """
    Learn from generic files.
    
    Handles various file types:
    - Text files (.txt)
    - Markdown files (.md)
    - Code files (.py, .js, .ts, etc.)
    - Configuration files (.yaml, .json)
    
    Input format:
        {"path": "./notes.md"} or {"paths": ["./file1.py", "./file2.py"]}
    """
    
    @property
    def name(self) -> str:
        return "file"
    
    def learn(self, source_data: Dict[str, Any]) -> List[KnowledgeChunk]:
        """
        Extract knowledge from files.
        
        Args:
            source_data: Dict with "path" (single file) or "paths" (multiple files)
            
        Returns:
            List of KnowledgeChunk from the files
        """
        # Handle single path or multiple paths
        paths = source_data.get("paths", [])
        if not paths and "path" in source_data:
            paths = [source_data["path"]]
        
        chunks = []
        
        for path in paths:
            chunk = self._process_file(path)
            if chunk:
                chunks.append(chunk)
        
        print(f"[FileLearner] Learned from {len(chunks)} file(s)")
        return chunks
    
    def _process_file(self, path: str) -> KnowledgeChunk:
        """Process a single file and return a KnowledgeChunk."""
        # Determine chunk type based on file extension
        ext = os.path.splitext(path)[1].lower()
        code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs'}
        
        chunk_type = "code" if ext in code_extensions else "text"
        
        # TODO: Actually read and process file content
        # For now, create placeholder
        content = f"File content from {path}"
        
        # In real implementation:
        # try:
        #     with open(path, 'r', encoding='utf-8') as f:
        #         content = f.read()
        # except Exception as e:
        #     content = f"Error reading file: {e}"
        
        return KnowledgeChunk(
            content=content,
            chunk_type=chunk_type,
            source=path,
            metadata={"extension": ext, "learner": "file"}
        )

