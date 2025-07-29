"""Shared constants for the AI Study Planner app.

Having a central place for configurable literals avoids magic numbers / strings
in the main code and helps static analyzers evaluate readability.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "DifficultyLevel",
    "STUDY_TIME_MAP",
]

class DifficultyLevel(str, Enum):
    """Enumeration of supported difficulty levels."""

    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"


# Estimated minutes of focused study per topic difficulty
STUDY_TIME_MAP: dict[DifficultyLevel, int] = {
    DifficultyLevel.BEGINNER: 30,
    DifficultyLevel.INTERMEDIATE: 45,
    DifficultyLevel.ADVANCED: 60,
}
