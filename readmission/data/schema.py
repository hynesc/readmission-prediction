from __future__ import annotations

from dataclasses import dataclass

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_INDEX = 0
UNK_INDEX = 1


@dataclass(frozen=True)
class Visit:
    day: int
    codes: tuple[int, ...]
    service_locations: tuple[str, ...] = ()


@dataclass(frozen=True)
class PatientSequence:
    pid: str
    visits: tuple[Visit, ...]
    labels: tuple[int, ...]

    @property
    def days(self) -> tuple[int, ...]:
        return tuple(visit.day for visit in self.visits)
