# Commit Message Generator
#
# Generates meaningful git commit messages using a hybrid approach:
# 1. Use agent's suggestion if provided (code-aware)
# 2. Generate from diff + context using LLM (diff-based)
#
# Output follows Conventional Commits format:
# <type>(<scope>): <description>

from typing import Optional
from kapso.core.llm import LLMBackend

# Lightweight model for commit messages (cheap & fast)
COMMIT_MESSAGE_MODEL = "gpt-4.1-mini"

# Maximum diff length to send to LLM (to avoid token limits)
MAX_DIFF_LENGTH = 4000


class CommitMessageGenerator:
    """
    Generates conventional commit messages from git diffs.
    
    Used by ExperimentSession to create meaningful commits
    for agents that don't generate their own messages.
    
    Hybrid approach:
    - If agent provides a commit_message in CodingResult, use it
    - Otherwise, generate from diff + solution context
    """
    
    def __init__(self, llm: Optional[LLMBackend] = None):
        """
        Initialize the commit message generator.
        
        Args:
            llm: LLM backend for generating messages. If None, creates one.
        """
        self.llm = llm or LLMBackend()
    
    def generate(
        self,
        diff: str,
        solution_summary: Optional[str] = None,
        agent_suggestion: Optional[str] = None,
    ) -> str:
        """
        Generate commit message with fallback chain.
        
        Priority:
        1. Use agent's suggestion if provided (code-aware)
        2. Generate from diff + solution context (diff-based)
        
        Args:
            diff: Git diff of changes
            solution_summary: Optional context about what was being implemented
            agent_suggestion: Optional message from coding agent
            
        Returns:
            Formatted commit message following Conventional Commits
        """
        # Priority 1: Agent provided a message
        if agent_suggestion:
            return self._format_message(agent_suggestion)
        
        # Priority 2: Generate from diff
        if not diff or not diff.strip():
            return "chore: update code"
        
        return self._generate_from_diff(diff, solution_summary)
    
    def _generate_from_diff(
        self, 
        diff: str, 
        solution_summary: Optional[str] = None
    ) -> str:
        """
        Generate commit message using LLM analysis of diff.
        
        Args:
            diff: Git diff of changes
            solution_summary: Optional context about what was being implemented
            
        Returns:
            Generated commit message
        """
        # Truncate large diffs to avoid token limits
        truncated_diff = diff[:MAX_DIFF_LENGTH] if len(diff) > MAX_DIFF_LENGTH else diff
        
        # Build the prompt
        context_section = ""
        if solution_summary:
            # Truncate solution summary too
            summary = solution_summary[:500] if len(solution_summary) > 500 else solution_summary
            context_section = f"\nSolution being implemented:\n{summary}\n"
        
        prompt = f"""Generate a Git commit message following Conventional Commits format.

Format: <type>(<optional-scope>): <description>

Types:
- feat: New feature or capability
- fix: Bug fix
- refactor: Code restructuring without behavior change
- perf: Performance improvement
- docs: Documentation only
- style: Formatting, whitespace
- test: Adding/updating tests
- chore: Build, config, dependencies
{context_section}
Git diff:
```diff
{truncated_diff}
```

Rules:
- Subject line: max 72 characters
- Use imperative mood ("Add" not "Added")
- Be specific about what changed
- Reference key functions/classes changed
- No period at end of subject line

Output ONLY the commit message, nothing else:"""

        try:
            message = self.llm.llm_completion(
                model=COMMIT_MESSAGE_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            return self._format_message(message.strip())
        except Exception as e:
            # Fallback if LLM fails
            print(f"[CommitMessageGenerator] LLM failed: {e}")
            return "chore: update code"
    
    def _format_message(self, message: str) -> str:
        """
        Ensure message follows format constraints.
        
        Args:
            message: Raw commit message
            
        Returns:
            Formatted message
        """
        # Remove markdown code blocks if present
        message = message.strip()
        if message.startswith("```"):
            lines = message.split("\n")
            # Remove first and last lines (``` markers)
            lines = [l for l in lines if not l.strip().startswith("```")]
            message = "\n".join(lines).strip()
        
        # Remove quotes if wrapped
        if message.startswith('"') and message.endswith('"'):
            message = message[1:-1]
        if message.startswith("'") and message.endswith("'"):
            message = message[1:-1]
        
        # Ensure first line isn't too long
        lines = message.split('\n')
        if len(lines[0]) > 72:
            lines[0] = lines[0][:69] + '...'
        
        return '\n'.join(lines)

