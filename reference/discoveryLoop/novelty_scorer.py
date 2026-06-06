"""Novelty Scorer — composite novelty scoring for conjectures.

Combines survival testing, recall detection, triviality assessment, and
structural soundness into a single composite score that ranks conjectures
by their likelihood of representing genuine inferential novelty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class NoveltyScore:
    """Composite novelty score for a single conjecture."""

    conjecture_id: str
    statement: str
    predicate_source: str
    survival_score: float
    recall_penalty: float
    triviality_penalty: float
    soundness_bonus: float
    composite_score: float
    rank: int = 0
    reason: str = ""


class NoveltyScorer:
    """Score conjectures by composite novelty.

    The composite formula is:
        survival * (1 - recall_penalty) * (1 - triviality_penalty) * (1 + soundness_bonus)

    Each component:
      - survival_score:    tested_count / max_n (capped at 1.0)
      - recall_penalty:    1.0 - confidence if likely_recall else 0.0
      - triviality_penalty: signals.get("predicate_triviality", 0.0)
      - soundness_bonus:   clamped to [0.0, 1.0]
    """

    def __init__(self, max_n: int = 10_000) -> None:
        self.max_n = max_n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_score(
        self,
        conjecture: dict[str, Any],
        recall_verdict: dict[str, Any] | None,
        soundness: float,
    ) -> NoveltyScore:
        """Compute a composite novelty score for a single conjecture.

        Parameters
        ----------
        conjecture:
            Dict with keys: id, statement, predicate_source, tested_count.
        recall_verdict:
            Dict from assess_recall() with keys: likely_recall, confidence,
            signals.  If None, recall_penalty defaults to 0.0.
        soundness:
            Structural soundness value (e.g. from refinement classifications).
            Clamped to [0.0, 1.0].

        Returns
        -------
        NoveltyScore with all component scores and a human-readable reason.
        """
        conjecture_id = conjecture.get("id", "")
        statement = conjecture.get("statement", "")
        predicate_source = conjecture.get("predicate_source", "")
        tested_count = conjecture.get("tested_count", 0)

        survival_score = self._compute_survival(tested_count)
        recall_penalty = self._compute_recall_penalty(recall_verdict)
        triviality_penalty = self._compute_triviality_penalty(recall_verdict)
        soundness_bonus = self._clamp_soundness(soundness)

        composite = (
            survival_score
            * (1.0 - recall_penalty)
            * (1.0 - triviality_penalty)
            * (1.0 + soundness_bonus)
        )

        reason = self._build_reason(
            survival_score,
            recall_penalty,
            triviality_penalty,
            soundness_bonus,
            recall_verdict,
        )

        return NoveltyScore(
            conjecture_id=conjecture_id,
            statement=statement,
            predicate_source=predicate_source,
            survival_score=round(survival_score, 4),
            recall_penalty=round(recall_penalty, 4),
            triviality_penalty=round(triviality_penalty, 4),
            soundness_bonus=round(soundness_bonus, 4),
            composite_score=round(composite, 4),
            reason=reason,
        )

    def rank_conjectures(self, scores: list[NoveltyScore]) -> list[NoveltyScore]:
        """Rank scores by composite_score descending, assigning rank field."""
        sorted_scores = sorted(scores, key=lambda s: s.composite_score, reverse=True)
        for i, s in enumerate(sorted_scores, start=1):
            s.rank = i
        return sorted_scores

    def top_n(self, scores: list[NoveltyScore], n: int = 5) -> list[NoveltyScore]:
        """Return the top N conjectures by composite score."""
        ranked = self.rank_conjectures(scores)
        return ranked[:n]

    def score_report(self, report: Any) -> list[NoveltyScore]:
        """Score all conjectures from a DiscoveryReport.

        Matches each conjecture to its recall_verdict by statement, then
        computes a NoveltyScore.  Returns ranked list.
        """
        # Build lookup: statement -> recall_verdict dict
        recall_map: dict[str, dict[str, Any]] = {}
        for rv in getattr(report, "recall_verdicts", []) or []:
            stmt = rv.get("statement", "")
            if stmt:
                recall_map[stmt] = rv

        all_conjectures = list(report.survivors) + list(report.falsified)
        if not all_conjectures:
            return []

        scores: list[NoveltyScore] = []
        for conjecture in all_conjectures:
            stmt = conjecture.get("statement", "")
            recall_verdict = recall_map.get(stmt)
            score = self.compute_score(
                conjecture=conjecture,
                recall_verdict=recall_verdict,
                soundness=getattr(report, "soundness", 0.0),
            )
            scores.append(score)

        return self.rank_conjectures(scores)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_survival(self, tested_count: int) -> float:
        """Survival score = min(1.0, tested_count / max_n)."""
        return min(1.0, tested_count / max(self.max_n, 1))

    def _compute_recall_penalty(self, recall_verdict: dict[str, Any] | None) -> float:
        """Recall penalty = 1.0 - confidence if likely_recall, else 0.0."""
        if recall_verdict is None:
            return 0.0
        if recall_verdict.get("likely_recall", False):
            confidence = recall_verdict.get("confidence", 0.0)
            return 1.0 - max(0.0, min(1.0, confidence))
        return 0.0

    def _compute_triviality_penalty(
        self, recall_verdict: dict[str, Any] | None
    ) -> float:
        """Triviality penalty from recall signals, default 0.0."""
        if recall_verdict is None:
            return 0.0
        signals = recall_verdict.get("signals", {}) or {}
        return max(0.0, min(1.0, signals.get("predicate_triviality", 0.0)))

    @staticmethod
    def _clamp_soundness(soundness: float) -> float:
        """Clamp soundness bonus to [0.0, 1.0]."""
        return max(0.0, min(1.0, float(soundness)))

    def _build_reason(
        self,
        survival: float,
        recall: float,
        triviality: float,
        soundness: float,
        recall_verdict: dict[str, Any] | None,
    ) -> str:
        """Build a human-readable rationale string."""
        parts: list[str] = []

        if survival >= 0.99:
            parts.append("full-range survival")
        elif survival >= 0.5:
            parts.append(f"partial survival ({survival:.0%})")
        else:
            parts.append(f"limited survival ({survival:.0%})")

        if recall > 0.0:
            conf = 1.0 - recall
            parts.append(f"recall suspected (confidence={conf:.2f})")
        else:
            parts.append("no recall signal")

        if triviality >= 0.7:
            parts.append("high triviality")
        elif triviality >= 0.3:
            parts.append("moderate triviality")
        else:
            parts.append("non-trivial")

        if soundness >= 0.5:
            parts.append(f"soundness bonus ({soundness:.2f})")

        # Add the recall detector's own reason if available
        if recall_verdict:
            rv_reason = recall_verdict.get("reason", "")
            if rv_reason:
                parts.append(f"recall: {rv_reason}")

        return "; ".join(parts)
