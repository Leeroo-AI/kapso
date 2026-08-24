# Facade result types for learning (design: learn-api-design.md §3, §8.2).
#
# LessonResult mirrors SolutionResult deliberately — evolve() and learn()
# are peers, and their artifacts answer "what just happened" the same way:
# direct fields plus a human-readable explain(). MemoryStatus is the one
# place to ask "what does this agent know" across both stores.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LessonResult:
    """The artifact produced by Kapso.learn().

    Not just cards — the entire learning attempt, auditable end to end:
    which trajectory was learned from, the examined (pre-lesson) bank
    head, the resulting head, what changed, and the paper trail.
    """

    trajectory_id: str
    bank_head_before: str
    bank_head_after: str
    cards_created: List[str] = field(default_factory=list)
    cards_updated: List[str] = field(default_factory=list)
    exam_report_path: Optional[str] = None
    lesson_report_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def admitted(self) -> bool:
        """True when the lesson moved the bank (the peer of
        SolutionResult.succeeded)."""
        return self.bank_head_after != self.bank_head_before

    def explain(self) -> str:
        """Return a summary of the lesson, same idiom as
        SolutionResult.explain()."""
        lines = [
            f"Lesson from: {self.trajectory_id}",
            f"Bank: {self.bank_head_before[:9]} -> "
            f"{self.bank_head_after[:9]} (admitted: {self.admitted})",
        ]
        if self.cards_created:
            lines.append(
                f"Cards created ({len(self.cards_created)}): "
                + ", ".join(self.cards_created)
            )
        if self.cards_updated:
            lines.append(
                f"Cards updated ({len(self.cards_updated)}): "
                + ", ".join(self.cards_updated)
            )
        if not self.cards_created and not self.cards_updated:
            lines.append("Cards: no changes")
        if self.exam_report_path:
            lines.append(f"Exam report: {self.exam_report_path}")
        if self.lesson_report_path:
            lines.append(f"Lesson report: {self.lesson_report_path}")
        if self.metadata:
            lines.append("Metadata:")
            for key, value in self.metadata.items():
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)


@dataclass
class MemoryStatus:
    """What the agent knows right now, across both memory stores
    (design §8.2): knowledge (imported — KG) and experience (earned —
    the bank)."""

    knowledge_index: Optional[str]
    knowledge_backend: Optional[str]
    knowledge_enabled: bool
    bank_path: Optional[str]
    bank_head: Optional[str]
    bank_active_cards: Optional[int]
    store_trajectories: Optional[int]
    serving_enabled: bool

    def explain(self) -> str:
        lines = ["Memory:"]
        if self.knowledge_enabled:
            lines.append(
                f"  knowledge:  {self.knowledge_index or '(backends)'} "
                f"({self.knowledge_backend}) — enabled"
            )
        else:
            lines.append("  knowledge:  disabled")
        if self.bank_path and self.bank_head:
            lines.append(
                f"  experience: {self.bank_path} @ {self.bank_head[:9]} — "
                f"{self.bank_active_cards} active cards, "
                f"{self.store_trajectories} trajectories in store, "
                f"serving: {'enabled' if self.serving_enabled else 'disabled'}"
            )
        elif self.bank_path:
            lines.append(
                f"  experience: {self.bank_path} (bank not initialized), "
                f"serving: {'enabled' if self.serving_enabled else 'disabled'}"
            )
        else:
            lines.append("  experience: no bank configured")
        return "\n".join(lines)
