"""
Evaluation script for AIME prompt optimization.

This script:
1. Loads a subset of AIME 2024 problems
2. Calls the optimize.solve() function for each problem
3. Extracts and grades answers against ground truth
4. Reports accuracy metric for Kapso optimization

Usage:
    python evaluate.py

The script outputs an accuracy score that Kapso uses for optimization guidance.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import concurrent.futures

from datasets import load_dataset
import optimize  # the file Kapso mutates

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
TOTAL_SAMPLES = 30  # Number of problems to evaluate
NUM_WORKERS = 30  # Concurrent LLM calls
LOG_EVERY = 5  # Print progress after this many completions
MODEL_TO_USE = "gpt-4.1-mini"  # Model for solving problems
TASK_TIMEOUT = 300  # Seconds per LLM call timeout
# ---------------------------------------------------------------------

print(f"[setup] loading {TOTAL_SAMPLES} problems from AIME 2024 …", flush=True)
DATA = load_dataset("Maxwell-Jia/AIME_2024", split=f"train[:{TOTAL_SAMPLES}]", cache_dir=".cache")


def extract_final_answer(text: str) -> str:
    """
    Extract the final AIME answer (000-999) from the LLM response.
    
    Prioritizes answers within \\boxed{}, then looks for patterns,
    and falls back to finding the last 3-digit number.
    
    Args:
        text: The LLM response text.
    
    Returns:
        A 3-digit string answer (zero-padded), or empty string if not found.
    """
    # 1. Check for \boxed{...}
    boxed_match = re.search(r"\\boxed\{(\d{1,3})\}", text)
    if boxed_match:
        return boxed_match.group(1).zfill(3)

    # 2. Check for "final answer is ..." patterns (case-insensitive)
    answer_pattern = r"(?:final|answer is|result is)[:\s]*(\d{1,3})\b"
    answer_match = re.search(answer_pattern, text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).zfill(3)

    # 3. Fallback: Find the last occurrence of a 1-3 digit number
    fallback_matches = re.findall(r"\b(\d{1,3})\b", text)
    if fallback_matches:
        return fallback_matches[-1].zfill(3)

    return ""


def grade_answer(llm_output: str, ground_truth_answer: str) -> bool:
    """
    Compare the extracted LLM answer to the ground truth.
    
    Args:
        llm_output: The raw LLM response text.
        ground_truth_answer: The correct answer from the dataset.
    
    Returns:
        True if the extracted answer matches ground truth, False otherwise.
    """
    extracted_guess = extract_final_answer(llm_output)
    try:
        return int(extracted_guess) == int(ground_truth_answer)
    except ValueError:
        return False


def run_evaluation() -> float:
    """
    Run the evaluation on the dataset and return the accuracy.
    
    Returns:
        Accuracy as a float between 0.0 and 1.0.
    """
    correct = 0
    start = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        # Submit all tasks
        futures = {
            pool.submit(optimize.solve, row["Problem"], MODEL_TO_USE): row["Answer"] 
            for row in DATA
        }

        try:
            for idx, future in enumerate(as_completed(futures), 1):
                problem_answer = futures[future]
                try:
                    llm_raw_output = future.result(timeout=TASK_TIMEOUT)
                    is_correct = grade_answer(llm_raw_output, str(problem_answer))
                    if is_correct:
                        correct += 1
                    results.append({
                        "raw_output": llm_raw_output, 
                        "correct_answer": problem_answer, 
                        "is_correct": is_correct
                    })

                except Exception as exc:
                    print(f"[error] Generated an exception: {exc}")
                    results.append({
                        "raw_output": f"Error: {exc}", 
                        "correct_answer": problem_answer, 
                        "is_correct": False
                    })

                if idx % LOG_EVERY == 0 or idx == TOTAL_SAMPLES:
                    elapsed = time.time() - start
                    current_accuracy = correct / idx if idx > 0 else 0
                    print(
                        f"[progress] {idx}/{TOTAL_SAMPLES} completed, "
                        f"accuracy: {current_accuracy:.4f}, elapsed {elapsed:.1f} s",
                        flush=True,
                    )
                    
        except concurrent.futures.TimeoutError:
            print(f"[error] LLM call timed out after {TASK_TIMEOUT}s", flush=True)
            for f in futures:
                f.cancel()
            print("Exiting due to timeout", file=sys.stderr)
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user", file=sys.stderr)
            sys.exit(1)

    # Final accuracy calculation
    total_evaluated = len(results)
    final_accuracy = correct / total_evaluated if total_evaluated > 0 else 0
    return final_accuracy


if __name__ == "__main__":
    acc = run_evaluation()
    # Kapso parses this exact line format
    print(f"accuracy: {acc:.4f}")
    print(f"\n__SCORE__: {acc}")
