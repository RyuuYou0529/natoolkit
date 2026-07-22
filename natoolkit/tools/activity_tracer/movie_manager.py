from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .models import MovieState


@dataclass
class ManagedMovie:
    layer: Any
    state: MovieState
    source_path: Path | None = None
    imported: bool = False
    motion_source_data: Any | None = None
    motion_corrected: bool = False

    @property
    def key(self) -> int:
        return id(self.layer)

    @property
    def motion_input(self) -> Any:
        return self.motion_source_data if self.motion_source_data is not None else self.state.source_data


class MovieManager:
    """Track movie layers by identity and preserve their full source arrays."""

    def __init__(self) -> None:
        self.records: dict[int, ManagedMovie] = {}
        self.states: dict[int, MovieState] = {}

    def state_for(self, layer: Any) -> MovieState:
        key = id(layer)
        if key not in self.records:
            state = MovieState(
                stop=int(layer.data.shape[0]),
                layer_name=layer.name,
                source_data=layer.data,
            )
            self.states[key] = state
            self.records[key] = ManagedMovie(layer=layer, state=state)
        state = self.records[key].state
        state.layer_name = layer.name
        return state

    def mark_imported(self, layer: Any, source_path: Path) -> None:
        self.state_for(layer)
        record = self.records[id(layer)]
        record.source_path = source_path
        record.imported = True

    def imported_movies(self) -> list[ManagedMovie]:
        return [record for record in self.records.values() if record.imported]

    def remove(self, layer: Any) -> None:
        key = id(layer)
        self.records.pop(key, None)
        self.states.pop(key, None)

    def apply_registered_data(self, layer: Any, data: np.ndarray) -> MovieState:
        record = self.records[id(layer)]
        if record.motion_source_data is None:
            record.motion_source_data = record.state.source_data
        record.state.source_data = data
        record.motion_corrected = True
        return record.state

    def restore_motion_input(self, layer: Any) -> MovieState:
        record = self.records[id(layer)]
        state = record.state
        if not record.motion_corrected:
            return state
        registered = state.source_data
        if record.motion_source_data is not None:
            state.source_data = record.motion_source_data
            state.stop = min(state.stop, int(state.source_data.shape[0]))
            layer.data = state.source_data[state.start : state.stop]
        record.motion_corrected = False
        if isinstance(registered, np.memmap):
            registered._mmap.close()
        return state
