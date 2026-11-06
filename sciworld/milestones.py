"""Milestone definitions and ordering."""

# Ordered from lowest to highest achievement.
MILESTONE_ORDER = [
    "none",
    "method_runs",
    "core_claim",
    "core_claim_plus",
    "secondary_claims",
    "majority",
    "near_complete",
    "full",
]

# Special value: not ranked.
NOT_POSSIBLE = "not_possible"

MILESTONE_RANK = {name: i for i, name in enumerate(MILESTONE_ORDER)}


def milestone_rank(milestone: str) -> int:
    """Return numeric rank for a milestone string (-1 for unknown)."""
    return MILESTONE_RANK.get(milestone, -1)


def meets_minimum(milestone: str, minimum: str) -> bool:
    """Check if *milestone* is at or above *minimum*."""
    if milestone == NOT_POSSIBLE:
        return False
    m_rank = milestone_rank(milestone)
    min_rank = milestone_rank(minimum)
    if m_rank < 0 or min_rank < 0:
        return False
    return m_rank >= min_rank
