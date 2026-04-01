"""Selection policies for choosing points from the unlabeled pool."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseSelector(ABC):
    """Abstract selector interface."""

    @abstractmethod
    def select(
        self,
        candidates: pd.DataFrame,
        score_column: str,
        k: int,
    ) -> pd.DataFrame:
        """Return the chosen subset of candidates."""


class TopKSelector(BaseSelector):
    """Pick the globally highest-scoring candidates."""

    def select(self, candidates: pd.DataFrame, score_column: str, k: int) -> pd.DataFrame:
        if k <= 0:
            raise ValueError("k must be positive.")
        return candidates.nlargest(k, columns=score_column).copy()


class BalancedGroupSelector(BaseSelector):
    """Select points in a round-robin fashion across metadata groups.

    This is useful when you want donor-balanced batches without embedding a
    donor heuristic directly into the scorer.
    """

    def __init__(self, group_column: str) -> None:
        self.group_column = group_column

    def select(self, candidates: pd.DataFrame, score_column: str, k: int) -> pd.DataFrame:
        if k <= 0:
            raise ValueError("k must be positive.")
        if self.group_column not in candidates.columns:
            raise ValueError(f"Missing group column: {self.group_column}")

        ordered = candidates.sort_values(score_column, ascending=False).copy()
        grouped = {
            group: frame.reset_index(drop=True)
            for group, frame in ordered.groupby(self.group_column, sort=False)
        }

        chosen_frames = []
        cursor = {group: 0 for group in grouped}

        # Round-robin over groups until we gather k rows or exhaust the pool.
        while len(chosen_frames) < k:
            made_progress = False
            for group, frame in grouped.items():
                idx = cursor[group]
                if idx < len(frame):
                    chosen_frames.append(frame.iloc[[idx]])
                    cursor[group] += 1
                    made_progress = True
                    if len(chosen_frames) == k:
                        break
            if not made_progress:
                break

        if not chosen_frames:
            return ordered.iloc[0:0].copy()
        return pd.concat(chosen_frames, ignore_index=True)
