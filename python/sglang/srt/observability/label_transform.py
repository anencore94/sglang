from typing import Optional

PRIORITY_TIERS = ("P0_Critical", "P1_High", "P2_Normal", "P3_Batch")

UNKNOWN_PRIORITY_VALUE = PRIORITY_TIERS[3]


def transform_priority(priority: Optional[int]) -> str:
    """Classify raw values without changing scheduling order or direction."""
    if priority is None:
        return UNKNOWN_PRIORITY_VALUE
    if priority < 10:
        return PRIORITY_TIERS[0]
    if priority < 30:
        return PRIORITY_TIERS[1]
    if priority < 50:
        return PRIORITY_TIERS[2]
    return PRIORITY_TIERS[3]
