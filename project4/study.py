"""Running a session: what each participant is shown, and in what order.

The brief specifies uniform-random slates, which is also the right choice for the comparison
being made. An adaptive selector would help each interface by a different amount, so any
difference in the measured outcome could no longer be attributed to the interface itself.

Slates are drawn from a generator seeded on the participant's own token, so a session is
reproducible from the database alone, and the two blocks draw from disjoint pools so nothing
seen in one condition is seen again in the other.
"""

import numpy as np

from . import data
from .models import PAIRWISE, RANKING

TASKS_PER_BLOCK = {PAIRWISE: 12, RANKING: 3}
SLATE_SIZE = {PAIRWISE: 2, RANKING: 10}

# Enough films to fill both blocks without overlap, drawn once per participant.
POOL_SIZE = sum(TASKS_PER_BLOCK[c] * SLATE_SIZE[c] for c in (PAIRWISE, RANKING))


def _generator(participant):
    return np.random.default_rng(participant.token.int % (2**32))


def build_slates(participant):
    """Every slate for this participant, as {(block, position): [film indices]}.

    Both blocks are drawn up front from one disjoint pool, so a film can never appear twice in
    a session and the two conditions cannot differ by having been given easier films.
    """
    rng = _generator(participant)
    pool = rng.choice(len(data.catalogue()), size=POOL_SIZE, replace=False).tolist()

    slates, cursor = {}, 0
    for block, condition in enumerate(participant.conditions_in_order(), start=1):
        size = SLATE_SIZE[condition]
        for position in range(1, TASKS_PER_BLOCK[condition] + 1):
            slates[(block, position)] = pool[cursor : cursor + size]
            cursor += size
    return slates


def condition_for(participant, block):
    return participant.conditions_in_order()[block - 1]


def tasks_in_block(condition):
    return TASKS_PER_BLOCK[condition]


def next_step(participant):
    """Where the participant should go next: a task, or the end of the study."""
    answered = {(task.block, task.position) for task in participant.tasks.filter(response__isnull=False)}
    for block, condition in enumerate(participant.conditions_in_order(), start=1):
        for position in range(1, TASKS_PER_BLOCK[condition] + 1):
            if (block, position) not in answered:
                return block, position
    return None


def rankings_by_condition(participant):
    """The orderings this participant gave, split by which interface produced them."""
    collected = {PAIRWISE: [], RANKING: []}
    for task in participant.tasks.filter(response__isnull=False).select_related("response"):
        collected[task.condition].append(task.response.ordering)
    return collected


def time_by_condition(participant):
    """Seconds spent in each condition, which the primary measure is normalised by."""
    spent = {PAIRWISE: 0.0, RANKING: 0.0}
    for task in participant.tasks.filter(response__isnull=False).select_related("response"):
        spent[task.condition] += task.response.duration_ms / 1000.0
    return spent
