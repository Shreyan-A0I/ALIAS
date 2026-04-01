"""High-level active learning query loop helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .pool import PoolManager
from .scorers import BaseScorer
from .selectors import BaseSelector


@dataclass
class QueryResult:
    """Result of a single active-learning query."""

    scored_pool: pd.DataFrame
    selected: pd.DataFrame


class ActiveLearningEngine:
    """Coordinates scoring, selection, and pool updates.

    The engine is intentionally agnostic to model training. It assumes some
    external code is responsible for fitting or refreshing the underlying
    models used by the scorer between query rounds.
    """

    def __init__(
        self,
        pool_manager: PoolManager,
        scorer: BaseScorer,
        selector: BaseSelector,
    ) -> None:
        self.pool_manager = pool_manager
        self.scorer = scorer
        self.selector = selector

    def query(self, k: int = 1) -> QueryResult:
        """Score the unlabeled pool and select the next candidates."""

        unlabeled = self.pool_manager.get_unlabeled()
        if unlabeled.empty:
            raise ValueError("No unlabeled candidates remain in the pool.")

        bundle = self.scorer.score(unlabeled)
        scored_pool = bundle.as_frame(unlabeled, self.pool_manager.id_column)

        # Carry through the original metadata so downstream users can inspect
        # donor ids, patch paths, and any other candidate attributes.
        metadata_columns = [
            column
            for column in unlabeled.columns
            if column not in scored_pool.columns
        ]
        for column in metadata_columns:
            scored_pool[column] = unlabeled[column].to_numpy()

        selected = self.selector.select(scored_pool, score_column="score", k=k)
        return QueryResult(scored_pool=scored_pool, selected=selected)

    def commit(self, selected: pd.DataFrame) -> pd.DataFrame:
        """Move selected candidates from the unlabeled pool to the labeled set."""

        ids = selected[self.pool_manager.id_column].tolist()
        return self.pool_manager.reveal(ids)

    def step(self, k: int = 1) -> QueryResult:
        """Query and immediately commit the chosen candidates."""

        result = self.query(k=k)
        self.commit(result.selected)
        return result
