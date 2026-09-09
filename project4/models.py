"""What a run of the study records.

Data minimisation is the governing constraint, from lecture 12. A participant is an opaque
identifier and nothing else: no name, no email, no IP. Everything stored is either a preference
they deliberately expressed or a timing needed to answer the research question. If the study
were actually run, this is the table a participant could point at when asking what is held
about them, and the honest answer is "an anonymous id, some film orderings, and how long they
took".
"""

import uuid

from django.db import models

PAIRWISE = "pairwise"
RANKING = "ranking"
CONDITIONS = [(PAIRWISE, "Pairwise choice"), (RANKING, "Rank ten films")]


class Participant(models.Model):
    """One person's session, identified only by a random token."""

    token = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    started_at = models.DateTimeField(auto_now_add=True)
    consented_at = models.DateTimeField(null=True, blank=True)
    # Which interface they see first. Alternating this across participants is the
    # counterbalancing that stops order effects loading onto one condition.
    first_condition = models.CharField(max_length=20, choices=CONDITIONS)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        # Sessions started in the same clock tick share a timestamp, so the primary key is
        # the only stable tiebreak. Ordering on started_at alone is not deterministic.
        ordering = ["started_at", "id"]

    def __str__(self):
        return f"participant {self.token}"

    @property
    def second_condition(self):
        return RANKING if self.first_condition == PAIRWISE else PAIRWISE

    def conditions_in_order(self):
        return [self.first_condition, self.second_condition]


class ElicitationTask(models.Model):
    """One slate put in front of a participant, in one of the two conditions."""

    participant = models.ForeignKey(Participant, on_delete=models.CASCADE, related_name="tasks")
    condition = models.CharField(max_length=20, choices=CONDITIONS)
    block = models.PositiveSmallIntegerField()
    position = models.PositiveSmallIntegerField()
    # Catalogue row indices, in the order they were shown.
    film_indices = models.JSONField()
    presented_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["block", "position"]
        unique_together = ("participant", "block", "position")

    def __str__(self):
        return f"{self.condition} task {self.block}.{self.position}"


class Response(models.Model):
    """What the participant said, as a ranking from most to least preferred.

    A pairwise choice is stored the same way as a ten-item ranking, just with two entries. That
    is not a storage convenience: under Plackett-Luce they are the same kind of evidence, so
    keeping one shape means the analysis never has to special-case a condition.
    """

    task = models.OneToOneField(ElicitationTask, on_delete=models.CASCADE, related_name="response")
    ordering = models.JSONField()
    duration_ms = models.PositiveIntegerField()
    submitted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"response to {self.task}"
