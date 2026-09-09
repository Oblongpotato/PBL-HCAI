from django import forms


class ConsentForm(forms.Form):
    """Consent has to be an explicit act, not a pre-ticked box."""

    consent = forms.BooleanField(
        required=True,
        label="I have read the above and agree to take part.",
        error_messages={"required": "You have to agree before the study can start."},
    )


class PairwiseForm(forms.Form):
    """Which of the two films the participant would rather watch."""

    choice = forms.ChoiceField(widget=forms.HiddenInput)

    def __init__(self, films, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["choice"].choices = [(str(index), str(index)) for index in films]


class RankingForm(forms.Form):
    """One rank per film, and the ranks together have to be a permutation.

    Numbered dropdowns rather than drag-and-drop: it needs no JavaScript, it works with a
    keyboard and a screen reader, and a participant can revise one position without disturbing
    the others.
    """

    def __init__(self, films, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.films = list(films)
        choices = [("", "-")] + [(str(n), str(n)) for n in range(1, len(self.films) + 1)]
        for index in self.films:
            self.fields[f"rank_{index}"] = forms.ChoiceField(
                choices=choices, required=True, label=str(index)
            )

    def clean(self):
        cleaned = super().clean()
        ranks = [cleaned.get(f"rank_{index}") for index in self.films]
        if any(rank in (None, "") for rank in ranks):
            raise forms.ValidationError("Give every film a position.")
        if len(set(ranks)) != len(ranks):
            raise forms.ValidationError("Each position can only be used once.")
        cleaned["ordering"] = [
            index for _, index in sorted(zip((int(r) for r in ranks), self.films))
        ]
        return cleaned
