from io import BytesIO

import numpy as np
from django.http import FileResponse
from django.shortcuts import redirect, render
from django.utils import timezone

from . import data, preferences, report, study
from .forms import ConsentForm, PairwiseForm, RankingForm
from .models import PAIRWISE, RANKING, ElicitationTask, Participant, Response

SESSION_KEY = "project4_participant"


def index(request):
    """The landing page: read the design, or take part.

    The brief asks for exactly these two doors, so they are the page.
    """
    return render(request, "project4/index.html", {"summary": data.summary()})


def report_pdf(request):
    return FileResponse(
        BytesIO(report.build()),
        as_attachment=True,
        filename="project4-study-design.pdf",
        content_type="application/pdf",
    )


def consent(request):
    """Informed consent, and the point at which a participant record is created."""
    if request.method == "POST":
        form = ConsentForm(request.POST)
        if form.is_valid():
            # Alternate who sees which interface first: this is the counterbalancing. The
            # parity comes from the participant's own primary key, which the database
            # allocates atomically, rather than from a count() that two simultaneous
            # consents could both read before either row exists.
            participant = Participant.objects.create(
                first_condition=PAIRWISE, consented_at=timezone.now()
            )
            if participant.pk % 2 == 0:
                participant.first_condition = RANKING
                participant.save(update_fields=["first_condition"])
            request.session[SESSION_KEY] = str(participant.token)
            return redirect("project4:task")
    else:
        form = ConsentForm()
    return render(request, "project4/consent.html", {"form": form})


def _current(request):
    token = request.session.get(SESSION_KEY)
    return Participant.objects.filter(token=token).first() if token else None


def _bind(condition, films, payload=None):
    form_class = PairwiseForm if condition == PAIRWISE else RankingForm
    return form_class(films, payload) if payload is not None else form_class(films)


def _ordering_from(condition, form, films):
    if condition == RANKING:
        return form.cleaned_data["ordering"]
    chosen = int(form.cleaned_data["choice"])
    return [chosen] + [index for index in films if index != chosen]


def task(request):
    """Present the next slate, in whichever condition the participant is currently in."""
    participant = _current(request)
    if participant is None:
        return redirect("project4:consent")

    step = study.next_step(participant)
    if step is None:
        return redirect("project4:results")

    block, position = step
    condition = study.condition_for(participant, block)

    record, _ = ElicitationTask.objects.get_or_create(
        participant=participant,
        block=block,
        position=position,
        defaults={
            "condition": condition,
            "film_indices": study.build_slates(participant)[(block, position)],
        },
    )
    # Render what was recorded, not a second derivation of it. The two agree today because the
    # generator is seeded on the token, but only one of them is what the participant saw.
    films = record.film_indices

    if request.method == "POST":
        form = _bind(condition, films, request.POST)
        if form.is_valid():
            elapsed = (timezone.now() - record.presented_at).total_seconds() * 1000
            Response.objects.create(
                task=record,
                ordering=_ordering_from(condition, form, films),
                duration_ms=max(int(elapsed), 0),
            )
            if study.next_step(participant) is None:
                participant.finished_at = timezone.now()
                participant.save(update_fields=["finished_at"])
            return redirect("project4:task")
    else:
        form = _bind(condition, films)

    described = [data.describe(index) for index in films]
    # Pair each film with its own rank field here rather than trying to match them up by name
    # inside the template, where `"rank_"|add:index` silently yields an empty string.
    rows = [
        {"film": film, "field": form[f"rank_{film['index']}"]}
        for film in described
    ] if condition == RANKING else []

    return render(
        request,
        "project4/task.html",
        {
            "form": form,
            "condition": condition,
            "films": described,
            "rows": rows,
            "block_number": block,
            "position": position,
            "total": study.tasks_in_block(condition),
            "is_first_of_block": position == 1,
        },
    )


def results(request):
    """What the participant's answers imply about their taste.

    Not demanded by the brief, but a twenty-minute study that gives nothing back is one people
    abandon. It also demonstrates the task 2 model working end to end rather than only in tests.
    """
    participant = _current(request)
    if participant is None:
        return redirect("project4:consent")

    by_condition = study.rankings_by_condition(participant)
    everything = by_condition[PAIRWISE] + by_condition[RANKING]
    if not everything:
        return redirect("project4:task")

    X = data.matrix()
    weights = preferences.fit(everything, X)
    seen = {index for ranking in everything for index in ranking}

    names = data.feature_names()
    order = np.argsort(weights)[::-1]
    strongest = [
        {"feature": names[i], "weight": round(float(weights[i]), 3)}
        for i in list(order[:5]) + list(order[-5:])
    ]

    spent = study.time_by_condition(participant)
    per_condition = [
        {
            "condition": condition,
            "comparisons": sum(len(r) - 1 for r in by_condition[condition]),
            "seconds": round(spent[condition], 1),
        }
        for condition in (PAIRWISE, RANKING)
    ]

    return render(
        request,
        "project4/results.html",
        {
            "recommendations": [
                data.describe(i) for i in preferences.recommend(weights, X, k=8, exclude=seen)
            ],
            "strongest": strongest,
            "per_condition": per_condition,
        },
    )
