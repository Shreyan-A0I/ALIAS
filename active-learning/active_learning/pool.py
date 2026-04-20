"""Pool management utilities for active learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class PoolSnapshot:
    """Small immutable view of the current pool state."""

    labeled: pd.DataFrame
    unlabeled: pd.DataFrame


class PoolManager:
    """Tracks which candidates are currently labeled vs unlabeled.

    The pool manager only handles bookkeeping. It does not know anything about
    the model, scorer, or training loop.
    """

    def __init__(
        self,
        candidates: pd.DataFrame,
        id_column: str,
        initial_labeled_ids: Sequence[str] | None = None,
    ) -> None:
        if id_column not in candidates.columns:
            raise ValueError(f"Missing required id column: {id_column}")

        if candidates[id_column].duplicated().any():
            raise ValueError(f"Candidate ids in '{id_column}' must be unique.")

        self._id_column = id_column
        self._candidates = candidates.copy().reset_index(drop=True)
        self._candidates["_is_labeled"] = False

        if initial_labeled_ids:
            self.mark_labeled(initial_labeled_ids)

    @property
    def id_column(self) -> str:
        return self._id_column

    @property
    def candidates(self) -> pd.DataFrame:
        """Return a copy of the full candidate table."""

        return self._candidates.copy()

    def snapshot(self) -> PoolSnapshot:
        """Return the current labeled / unlabeled partition."""

        return PoolSnapshot(
            labeled=self.get_labeled(),
            unlabeled=self.get_unlabeled(),
        )

    def get_labeled(self) -> pd.DataFrame:
        """Return the currently labeled candidates."""

        return self._candidates[self._candidates["_is_labeled"]].copy()

    def get_unlabeled(self) -> pd.DataFrame:
        """Return the currently unlabeled candidates."""

        return self._candidates[~self._candidates["_is_labeled"]].copy()

    def mark_labeled(self, candidate_ids: Iterable[str]) -> None:
        """Mark one or more candidate ids as labeled."""

        candidate_ids = list(candidate_ids)
        if not candidate_ids:
            return

        mask = self._candidates[self._id_column].isin(candidate_ids)
        if not mask.any():
            raise ValueError("None of the provided candidate ids were found.")
        self._candidates.loc[mask, "_is_labeled"] = True

    def mark_unlabeled(self, candidate_ids: Iterable[str]) -> None:
        """Move one or more candidate ids back into the unlabeled pool."""

        candidate_ids = list(candidate_ids)
        if not candidate_ids:
            return

        mask = self._candidates[self._id_column].isin(candidate_ids)
        if not mask.any():
            raise ValueError("None of the provided candidate ids were found.")
        self._candidates.loc[mask, "_is_labeled"] = False

    def reveal(self, candidate_ids: Iterable[str]) -> pd.DataFrame:
        """Mark ids as labeled and return the revealed rows.

        This mirrors the common active-learning step of querying labels and
        then moving those examples into the labeled set.
        """

        candidate_ids = list(candidate_ids)
        self.mark_labeled(candidate_ids)
        return self._candidates[self._candidates[self._id_column].isin(candidate_ids)].copy()
