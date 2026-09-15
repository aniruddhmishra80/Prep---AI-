"""
The rules that make the interview adaptive.

Interview line: "A fixed question list wastes time. If the candidate nails a
medium question, three more mediums teach me nothing. So the score from the
previous answer feeds straight into the next question's difficulty, and the
skill order comes from the gap analysis - missing skills first, because that is
where the signal is.

Counter-argument I'll concede if pushed: warming up on strengths first would
probably be a nicer candidate experience."
"""

from typing import Dict, List

from backend import config


def next_difficulty_index(current_index: int, score: float) -> int:
    """Move up on a strong answer, down on a weak one, otherwise hold."""
    if score >= config.LEVEL_UP_AT:
        current_index += 1
    elif score < config.LEVEL_DOWN_AT:
        current_index -= 1
    return max(0, min(len(config.DIFFICULTY_LEVELS) - 1, current_index))


def difficulty_label(index: int) -> str:
    return config.DIFFICULTY_LEVELS[index]


def build_skill_plan(gap: Dict) -> List[str]:
    """Missing skills first, then matched skills, then anything extra."""
    plan = list(gap.get("missing", [])) + list(gap.get("matched", []))
    if not plan:
        plan = list(gap.get("extra", [])) or ["general software engineering"]
    return plan


def pick_skill(gap: Dict, asked_count: int) -> str:
    """Cycle through the plan so a short plan still fills a long interview."""
    plan = build_skill_plan(gap)
    return plan[asked_count % len(plan)]
