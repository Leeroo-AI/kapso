"""Feedback flow for the generic strategy.

Owns the post-implementation judgment feature: the FeedbackGenerator
session over a node's evidence, the manifest-of-record score cross-check,
the handler's finalize_run_selection call at score-of-record time, and the
structured agent-result extraction (XML tags). Stateless functions only —
GenericSearch assembles arguments from its state and delegates here.
"""

import re
from typing import Any, Callable, Dict, Optional

from kapso.execution.search_strategies.base import SearchNode


def generate_feedback(
    node: SearchNode,
    *,
    feedback_generator,
    goal: str,
    design_axes: str,
    session_end_facts: str,
    clamped_timeout: Callable[[float], float],
    manifest_of_record: Callable[[SearchNode], Optional[Dict[str, Any]]],
    finalize_run_selection: Callable[[Dict[str, Any], bool], None],
) -> SearchNode:
    """
    Generate feedback for a node using the FeedbackGenerator.
    
    Updates the node in-place with feedback, score, and should_stop.
    
    Args:
        node: SearchNode with solution, evaluation_output, code_changes_summary populated
        
    Returns:
        The same node with feedback, score, should_stop populated
    """
    if feedback_generator is None:
        print("[GenericSearch] No feedback generator configured, skipping feedback")
        return node
    
    if not goal:
        print("[GenericSearch] Warning: No goal set, skipping feedback generation")
        return node
    
    print(f"[GenericSearch] Generating feedback for node {node.node_id}...")
    
    try:
        feedback_result = feedback_generator.generate(
            goal=goal,
            idea=node.solution,
            code_changes_summary=node.code_changes_summary,
            base_branch=node.parent_branch_name,
            head_branch=node.branch_name,
            evaluation_script_path=node.evaluation_script_path,
            evaluation_result=node.evaluation_output,
            workspace_dir=node.workspace_dir,
            design_axes=design_axes,
            session_end_facts=session_end_facts,
            timeout_seconds=clamped_timeout(
                feedback_generator.configured_timeout_seconds
            ),
        )

        # Update node with feedback results
        node.feedback = feedback_result.feedback
        node.evaluation_valid = feedback_result.evaluation_valid
        node.score = (
            feedback_result.score
            if feedback_result.evaluation_valid
            else None
        )
        # In registered mode the manifest line is the score of record;
        # the judge's extraction is a cross-check, and the judge keeps
        # its validity power (an invalid evaluation stays scoreless).
        manifest_record = manifest_of_record(node)
        manifest_score = (
            float(manifest_record["score"])
            if manifest_record is not None
            else None
        )
        if manifest_score is not None and node.evaluation_valid:
            if (
                node.score is not None
                and abs(node.score - manifest_score) > 1e-6
            ):
                print(
                    "[GenericSearch] Score cross-check: feedback "
                    f"extracted {node.score}, the manifest says "
                    f"{manifest_score}; the manifest is the score "
                    "of record"
                )
            node.score = manifest_score
        if manifest_record is not None:
            # Label the archive: the of-record run becomes this
            # session's registered final (or is marked invalid on a
            # judge veto / integrity flag); its intermediate siblings
            # are superseded. Handlers without run archives no-op.
            finalize_run_selection(
                manifest_record,
                bool(
                    node.evaluation_valid
                    and not node.evaluation_integrity_error
                ),
            )
        node.should_stop = (
            feedback_result.stop and feedback_result.evaluation_valid
        )
        if feedback_result.duration_seconds is not None:
            node.phase_telemetry["feedback"] = {
                "cost_usd": feedback_result.cost_usd,
                "duration_seconds": feedback_result.duration_seconds,
            }
        
        print(f"[GenericSearch] Feedback generated: stop={node.should_stop}, score={node.score}")
        
    except Exception as e:
        print(f"[GenericSearch] Error generating feedback: {e}")
        node.feedback = f"Error generating feedback: {e}"
        node.should_stop = False
    
    return node


def extract_agent_result(agent_output: str) -> dict:
    """
    Extract structured result from agent output using XML tags.
    
    The agent is instructed to return results in XML tags:
    <code_changes_summary>...</code_changes_summary>
    <evaluation_script_path>...</evaluation_script_path>
    <evaluation_output>...</evaluation_output>
    <score>...</score>
    <technical_difficulties>...</technical_difficulties>
    
    Args:
        agent_output: Raw output from the developer agent
        
    Returns:
        dict with keys: code_changes_summary, evaluation_script_path, evaluation_output, score
        Returns empty dict if extraction fails
    """
    result = {}
    
    # Extract each tag
    tags = ["code_changes_summary", "evaluation_script_path", "evaluation_output", "score", "technical_difficulties"]
    
    for tag in tags:
        pattern = rf'<{tag}>\s*(.*?)\s*</{tag}>'
        match = re.search(pattern, agent_output, re.DOTALL)
        if match:
            value = match.group(1).strip()
            # Handle score specially - convert to float
            if tag == "score":
                try:
                    if value.lower() == "null" or value == "":
                        result[tag] = None
                    else:
                        result[tag] = float(value)
                except ValueError:
                    result[tag] = None
            else:
                result[tag] = value
    
    if result:
        print(f"[GenericSearch] Extracted agent result from XML tags: {list(result.keys())}")
        return result

    # The shipped prompts mandate XML tags; there is no other format to
    # fall back to (the JSON extractor served a prompt generation that no
    # longer ships — stale-code audit 2026-08-26, B7).
    print(f"[GenericSearch] Warning: Could not extract result from agent output")
    return {}
