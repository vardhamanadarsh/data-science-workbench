"""Undo / redo history of dataset snapshots (UI-free).

Polars shares memory for unchanged columns, so snapshots are cheaper than full
copies, but row-filtering steps do copy data. Only the last `max_snapshots`
states are kept (the original is always kept).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


@dataclass(frozen=True, eq=False)
class Snapshot:
    label: str
    df: pl.DataFrame
    created: str


class DataHistory:
    def __init__(self, original: pl.DataFrame, max_snapshots: int = 10) -> None:
        if max_snapshots < 2:
            raise ValueError("max_snapshots must be at least 2.")
        self.max_snapshots = max_snapshots
        self._snaps: list[Snapshot] = [Snapshot("Original data", original, _now())]
        self._i = 0

    @property
    def snapshots(self) -> list[Snapshot]:
        return self._snaps

    @property
    def index(self) -> int:
        return self._i

    @property
    def current(self) -> Snapshot:
        return self._snaps[self._i]

    @property
    def can_undo(self) -> bool:
        return self._i > 0

    @property
    def can_redo(self) -> bool:
        return self._i < len(self._snaps) - 1

    def push(self, label: str, df: pl.DataFrame) -> None:
        """Record a new state. Anything that could have been redone is discarded."""
        del self._snaps[self._i + 1:]
        self._snaps.append(Snapshot(label, df, _now()))
        self._i = len(self._snaps) - 1
        while len(self._snaps) > self.max_snapshots:
            del self._snaps[1]          # drop the oldest step, keep the original
            self._i -= 1

    def undo(self) -> Snapshot:
        if self.can_undo:
            self._i -= 1
        return self.current

    def redo(self) -> Snapshot:
        if self.can_redo:
            self._i += 1
        return self.current

    def rollback_to(self, index: int) -> Snapshot:
        if not 0 <= index < len(self._snaps):
            raise IndexError("No such step.")
        self._i = index
        return self.current

    def timeline(self) -> list[dict[str, str | int]]:
        return [
            {
                "step": i,
                "action": s.label,
                "rows": s.df.height,
                "columns": s.df.width,
                "time": s.created,
                "current": "◀" if i == self._i else "",
            }
            for i, s in enumerate(self._snaps)
        ]