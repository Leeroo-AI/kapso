"""Feedback flow for the generic strategy.

Owns the post-implementation judgment feature: the FeedbackGenerator
session over a node's evidence, the manifest-of-record score cross-check,
the engine's archive labelling at score-of-record time (the strategy passes
`label_run_of_record` over its archive glob), and the structured
agent-result extraction (XML tags). Stateless functions only —
GenericSearch assembles arguments from its state and delegates here.
"""

import json
import re
from typing import Any, Callable, Dict, Optional, Tuple

from kapso.execution.search_strategies.base import SearchNode
from kapso.harness.verdict import is_void, render_verification
from kapso.execution.cost_estimator import validate_cost_value
from kapso.execution.evaluation_maintainer.maintainer import MANIFEST_MARKER


def measured_cost_from_output(
    output: str, expected_unit: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Read optional ``cost: '<number> <unit>'`` from the final manifest."""
    manifest = None
    for line in output.splitlines():
        if line.strip().startswith(MANIFEST_MARKER):
            manifest = json.loads(line.strip()[len(MANIFEST_MARKER):].strip())
            if not isinstance(manifest, dict):
                raise ValueError("Evaluation cost manifest must be an object")
    if manifest is None or "cost" not in manifest:
        return None
    cost = manifest["cost"]
    if not isinstance(cost, str) or len(cost.split(maxsplit=1)) != 2:
        raise ValueError("Evaluation cost must be '<number> <unit>'")
    amount, unit = cost.split(maxsplit=1)
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", amount):
        raise ValueError("Evaluation cost amount must be a finite number")
    return validate_cost_value(
        {"amount": float(amount), "unit": unit}, expected_unit,
    )


def generate_feedback(
    node: SearchNode,
    *,
    feedback_generator,
    goal: str,
    axes: str,
    session_end_facts: str,
    deadline_seconds: Optional[float],
    retry_deadline: Callable[[], Tuple[Optional[float], Optional[str]]],
    manifest_of_record: Callable[[SearchNode], Optional[Dict[str, Any]]],
    finalize_run_selection: Callable[[Dict[str, Any], bool], None],
    execution_valid: bool = True,
    stage_harness_for: Callable[..., Any],
    verification: Optional[Dict[str, Any]] = None,
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
    
    # The harness's feedback module: its pages in the judge's prompt, its
    # delegates and skills beside the judge (attached once per campaign).
    staged = stage_harness_for("feedback", node.workspace_dir, f"feedback/{node.branch_name}", feedback_generator.agent_type)
    feedback_generator.attach_harness(staged)
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
            axes=axes,
            session_end_facts=session_end_facts,
            domain_guidance=staged.guidance,
            verification=render_verification(verification),
            timeout_seconds=deadline_seconds,
            retry_deadline=retry_deadline,
        )
        if feedback_result.unjudged:
            # No verdict: the attempt is unjudged, by reason, keeping its
            # score; the judge's spend and time are still the attempt's.
            node.unjudged = feedback_result.unjudged
            node.phase_telemetry["feedback"] = {
                "cost_usd": feedback_result.cost_usd,
                "duration_seconds": feedback_result.duration_seconds,
            }
            print(f"[GenericSearch] node {node.node_id} unjudged: {node.unjudged}")
            return node

        # First: the session is paid for, and what follows can raise into
        # the except below.
        if feedback_result.duration_seconds is not None:
            node.phase_telemetry["feedback"] = {
                "cost_usd": feedback_result.cost_usd,
                "duration_seconds": feedback_result.duration_seconds,
            }

        # Update node with feedback results. A harness verdict of void is
        # ANDed in after the judge, so the judge's default cannot
        # resurrect it.
        node.feedback = feedback_result.feedback
        # The judge's attribution stays on the node: the trajectory carries
        # which served cards were load-bearing, so learning can settle a
        # card's claim against what the campaign actually used.
        node.cards_load_bearing = list(feedback_result.cards_load_bearing)
        node.evaluation_valid = (
            feedback_result.evaluation_valid and execution_valid and not is_void(verification)
        )
        node.score = (
            feedback_result.score
            if node.evaluation_valid
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
            feedback_result.stop and node.evaluation_valid
        )

        print(f"[GenericSearch] Feedback generated: stop={node.should_stop}, score={node.score}")
        
    except Exception as e:
        print(f"[GenericSearch] Error generating feedback: {e}")
        node.feedback = f"Error generating feedback: {e}"
        node.should_stop = False
        # No judgment reached the record: the attempt keeps its score and
        # is never a parent or delivered, as every unjudged attempt.
        node.unjudged = f"the judge failed: {e}"
        if is_void(verification):
            # The harness's void holds whether or not the judge spoke.
            node.evaluation_valid = False
            node.score = None

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
