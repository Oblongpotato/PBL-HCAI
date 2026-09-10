"""Tests for project 4.

The model tests check properties rather than remembered numbers: that the gradient matches
finite differences, that Plackett-Luce at n = 2 really is Bradley-Terry, and that the fitter
recovers a preference vector it was not told. The interface tests walk a whole session the way
a participant would.
"""

from unittest import mock

import numpy as np
from django.test import TestCase, override_settings
from django.utils.html import escape
from django.urls import reverse

from . import data, preferences, study, views
from .forms import RankingForm
from .models import PAIRWISE, RANKING, ElicitationTask, Participant, Response


class CatalogueTests(TestCase):
    def test_cleaning_leaves_a_usable_catalogue(self):
        summary = data.summary()
        self.assertGreater(summary["n_films"], 4000)
        self.assertEqual(summary["n_films"] + summary["n_dropped"], summary["n_raw"])

    def test_titles_lose_their_trailing_non_breaking_space(self):
        titles = data.catalogue()["movie_title"]
        self.assertFalse(titles.str.contains("\xa0").any())
        self.assertEqual(titles.iloc[0], titles.iloc[0].strip())

    def test_no_film_appears_twice(self):
        frame = data.catalogue()
        self.assertFalse(frame.duplicated(subset=["movie_title", "title_year"]).any())

    def test_standardised_columns_have_zero_mean_and_unit_spread(self):
        X, names = data.features()
        for name in ("length", "acclaim", "director fame", "cast fame", "reach"):
            column = X[:, names.index(name)]
            self.assertAlmostEqual(column.mean(), 0.0, places=8, msg=name)
            self.assertAlmostEqual(column.std(), 1.0, places=8, msg=name)

    def test_genre_multi_hot_matches_the_source_string(self):
        X, names = data.features()
        genre_columns = {n[6:]: i for i, n in enumerate(names) if n.startswith("genre:")}
        for row in (0, 100, 2500):
            expected = set(data.catalogue().iloc[row]["genre_list"])
            actual = {g for g, i in genre_columns.items() if X[row, i] == 1.0}
            self.assertEqual(actual, expected)

    def test_one_hot_blocks_select_exactly_one_value(self):
        X, names = data.features()
        for prefix in ("era:", "rating:"):
            block = [i for i, n in enumerate(names) if n.startswith(prefix)]
            self.assertTrue(np.allclose(X[:, block].sum(axis=1), 1.0), prefix)

    def test_money_columns_are_excluded_on_purpose(self):
        # Missing and denominated in unstated currencies, so they encode data quality, not taste.
        self.assertNotIn("budget", data.feature_names())
        self.assertNotIn("gross", data.feature_names())


class PreferenceModelTests(TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.X = self.rng.normal(size=(40, 6))
        self.w = self.rng.normal(size=6)

    def _simulate(self, slate, w):
        remaining, order = list(slate), []
        while remaining:
            utilities = self.X[remaining] @ w
            probabilities = np.exp(utilities - utilities.max())
            probabilities /= probabilities.sum()
            order.append(remaining.pop(self.rng.choice(len(remaining), p=probabilities)))
        return order

    def test_gradient_matches_finite_differences(self):
        rankings = [self._simulate(self.rng.choice(40, 5, replace=False).tolist(), self.w)
                    for _ in range(10)]
        point = self.rng.normal(size=6) * 0.3
        analytic = preferences.gradient(point, rankings, self.X)

        numeric = np.zeros_like(point)
        step = 1e-6
        for j in range(len(point)):
            up, down = point.copy(), point.copy()
            up[j] += step
            down[j] -= step
            numeric[j] = (
                preferences.log_likelihood(up, rankings, self.X)
                - preferences.log_likelihood(down, rankings, self.X)
            ) / (2 * step)
        self.assertLess(np.abs(analytic - numeric).max(), 1e-6)

    def test_pairwise_case_is_exactly_bradley_terry(self):
        # The identity the whole study rests on: both interfaces feed one likelihood.
        for winner, loser in ((3, 17), (0, 39), (12, 5)):
            plackett_luce = np.exp(preferences.log_likelihood(self.w, [[winner, loser]], self.X))
            bradley_terry = preferences.pairwise_probability(self.w, winner, loser, self.X)
            self.assertAlmostEqual(plackett_luce, bradley_terry, places=12)

    def test_a_ranking_of_n_contributes_n_minus_one_comparisons(self):
        ranking = [[1, 2, 3, 4]]
        self.assertAlmostEqual(
            preferences.held_out_log_likelihood(self.w, ranking, self.X),
            preferences.log_likelihood(self.w, ranking, self.X) / 3,
        )

    def test_fitting_recovers_a_known_preference_vector(self):
        rankings = [self._simulate(self.rng.choice(40, 8, replace=False).tolist(), self.w)
                    for _ in range(300)]
        estimate = preferences.fit(rankings, self.X, prior_variance=10.0)
        correlation = np.corrcoef(self.w, estimate)[0, 1]
        self.assertGreater(correlation, 0.9)

    def test_more_evidence_predicts_held_out_answers_better(self):
        rankings = [self._simulate(self.rng.choice(40, 8, replace=False).tolist(), self.w)
                    for _ in range(260)]
        held_out = rankings[:60]
        few = preferences.fit(rankings[60:80], self.X, prior_variance=10.0)
        many = preferences.fit(rankings[60:], self.X, prior_variance=10.0)
        self.assertGreater(
            preferences.held_out_log_likelihood(many, held_out, self.X),
            preferences.held_out_log_likelihood(few, held_out, self.X),
        )

    def test_no_evidence_leaves_the_prior_mean(self):
        self.assertTrue(np.array_equal(preferences.fit([], self.X), np.zeros(6)))

    def test_recommendations_can_exclude_what_was_already_seen(self):
        seen = [0, 1, 2]
        picks = preferences.recommend(self.w, self.X, k=5, exclude=seen)
        self.assertEqual(len(picks), 5)
        self.assertFalse(set(picks) & set(seen))


class RankingFormTests(TestCase):
    films = [10, 11, 12]

    def test_a_permutation_becomes_an_ordering(self):
        form = RankingForm(self.films, {"rank_10": "2", "rank_11": "1", "rank_12": "3"})
        self.assertTrue(form.is_valid())
        self.assertEqual(form.cleaned_data["ordering"], [11, 10, 12])

    def test_a_repeated_position_is_rejected(self):
        form = RankingForm(self.films, {"rank_10": "1", "rank_11": "1", "rank_12": "3"})
        self.assertFalse(form.is_valid())
        self.assertIn("only be used once", str(form.errors))

    def test_a_missing_position_is_rejected(self):
        form = RankingForm(self.films, {"rank_10": "1", "rank_12": "3"})
        self.assertFalse(form.is_valid())


class SlateTests(TestCase):
    def _participant(self, first=PAIRWISE):
        return Participant.objects.create(first_condition=first)

    def test_a_film_is_never_shown_twice_in_a_session(self):
        slates = study.build_slates(self._participant())
        shown = [index for slate in slates.values() for index in slate]
        self.assertEqual(len(shown), len(set(shown)))

    def test_slates_are_reproducible_from_the_token_alone(self):
        participant = self._participant()
        self.assertEqual(study.build_slates(participant), study.build_slates(participant))

    def test_each_condition_gets_its_own_slate_size(self):
        participant = self._participant()
        slates = study.build_slates(participant)
        for (block, _), films in slates.items():
            condition = study.condition_for(participant, block)
            self.assertEqual(len(films), study.SLATE_SIZE[condition])


@override_settings(ALLOWED_HOSTS=["testserver"])
class InterfaceTests(TestCase):
    def test_the_landing_page_offers_the_two_doors_the_brief_asks_for(self):
        response = self.client.get(reverse("project4:index"))
        self.assertContains(response, reverse("project4:report"))
        self.assertContains(response, reverse("project4:consent"))
        self.assertContains(response, "designed, not run")

    def test_the_study_cannot_be_entered_without_consenting(self):
        self.assertEqual(self.client.get(reverse("project4:task")).status_code, 302)
        self.client.post(reverse("project4:consent"), {})
        self.assertFalse(Participant.objects.exists())

    def test_condition_order_alternates_between_participants(self):
        seen = []
        for _ in range(4):
            self.client.post(reverse("project4:consent"), {"consent": "on"})
            seen.append(Participant.objects.order_by("id").last().first_condition)
            self.client.session.flush()
        self.assertTrue(all(seen[i] != seen[i + 1] for i in range(len(seen) - 1)), seen)
        self.assertEqual(seen.count(PAIRWISE), seen.count(RANKING))

    def _walk(self):
        """Answer every task the way a participant would, and return the participant."""
        self.client.post(reverse("project4:consent"), {"consent": "on"})
        participant = Participant.objects.order_by("id").last()
        while self.client.get(reverse("project4:task")).status_code == 200:
            task = ElicitationTask.objects.filter(
                participant=participant, response__isnull=True
            ).first()
            films = task.film_indices
            if task.condition == PAIRWISE:
                payload = {"choice": str(films[0])}
            else:
                payload = {f"rank_{f}": str(i + 1) for i, f in enumerate(films)}
            self.client.post(reverse("project4:task"), payload)
        return participant

    def test_a_session_token_that_is_not_a_uuid_is_discarded(self):
        # A cookie can outlive a secret-key change or a database reset. The token used to go
        # straight into the query, where the UUID field raised instead of simply not matching.
        session = self.client.session
        session[views.SESSION_KEY] = "not-a-uuid"
        session.save()
        self.assertEqual(self.client.get(reverse("project4:task")).status_code, 302)

    def test_consenting_twice_keeps_the_same_session(self):
        # A second consent used to mint a second participant, strand the first one unfinished
        # and consume another counterbalancing slot.
        self.client.post(reverse("project4:consent"), {"consent": "on"})
        token = self.client.session[views.SESSION_KEY]
        self.client.post(reverse("project4:consent"), {"consent": "on"})
        self.assertEqual(self.client.session[views.SESSION_KEY], token)
        self.assertEqual(Participant.objects.count(), 1)

    def test_answering_the_same_task_twice_is_not_an_error(self):
        # Response.task is unique, so two submissions racing past the next-step check used to
        # end in an IntegrityError rather than in the first answer simply winning. Pinning
        # next_step is what a double-click on a slow connection does: both requests read the
        # same step before either of them has written.
        self.client.post(reverse("project4:consent"), {"consent": "on"})
        self.client.get(reverse("project4:task"))
        task = ElicitationTask.objects.get()
        films = task.film_indices
        payload = (
            {"choice": str(films[0])}
            if task.condition == PAIRWISE
            else {f"rank_{f}": str(i + 1) for i, f in enumerate(films)}
        )
        with mock.patch.object(views.study, "next_step", return_value=(task.block, task.position)):
            first = self.client.post(reverse("project4:task"), payload)
            second = self.client.post(reverse("project4:task"), payload)
        self.assertEqual(first.status_code, 302)
        self.assertEqual(second.status_code, 302)
        self.assertEqual(Response.objects.filter(task=task).count(), 1)

    def test_a_pairwise_submission_with_no_choice_says_so(self):
        # Only non-field errors were rendered, and PairwiseForm fails on the field, so pressing
        # Enter rather than clicking a card redisplayed the page with nothing explaining why.
        self.client.post(reverse("project4:consent"), {"consent": "on"})
        while True:
            response = self.client.get(reverse("project4:task"))
            pending = ElicitationTask.objects.filter(response__isnull=True).first()
            if pending.condition == PAIRWISE:
                break
            self.client.post(
                reverse("project4:task"),
                {f"rank_{f}": str(i + 1) for i, f in enumerate(pending.film_indices)},
            )
        body = self.client.post(reverse("project4:task"), {}).content.decode()
        self.assertEqual(Response.objects.filter(task=pending).count(), 0)
        self.assertIn("callout--danger", body)

    def test_the_ranking_page_offers_one_dropdown_per_film(self):
        # The films rendered but the rank fields did not, because the template tried to match
        # them by name with a filter chain that silently produced an empty string.
        self.client.post(reverse("project4:consent"), {"consent": "on"})
        participant = Participant.objects.order_by("id").last()
        while True:
            response = self.client.get(reverse("project4:task"))
            self.assertEqual(response.status_code, 200)
            pending = ElicitationTask.objects.filter(
                participant=participant, response__isnull=True
            ).first()
            if pending.condition == RANKING:
                body = response.content.decode()
                self.assertEqual(body.count("<select"), study.SLATE_SIZE[RANKING])
                for rank in range(1, study.SLATE_SIZE[RANKING] + 1):
                    self.assertIn(f'value="{rank}"', body)
                return
            self.client.post(
                reverse("project4:task"), {"choice": str(pending.film_indices[0])}
            )

    def test_the_page_shows_exactly_the_slate_that_was_recorded(self):
        # The view used to re-derive the slate rather than read the stored one, so the record
        # and what the participant saw were two independent derivations of the same thing.
        self.client.post(reverse("project4:consent"), {"consent": "on"})
        participant = Participant.objects.order_by("id").last()
        body = self.client.get(reverse("project4:task")).content.decode()
        record = ElicitationTask.objects.filter(participant=participant).first()
        for index in record.film_indices:
            # Escaped, because roughly one film title in twenty carries an apostrophe or an
            # ampersand. Comparing raw titles made this test fail whenever a participant's
            # randomly drawn slate happened to contain one.
            self.assertIn(escape(data.describe(index)["title"]), body)

    def test_no_template_internals_leak_into_the_page(self):
        # `block` is a Django tag name, so a context variable called `block` renders the
        # BlockNode's repr into the page. Status-code assertions never see this.
        self.client.post(reverse("project4:consent"), {"consent": "on"})
        for url in (reverse("project4:index"), reverse("project4:task")):
            body = self.client.get(url).content.decode()
            for leak in ("Block Node", "TextNode", "IfNode", "object at 0x"):
                self.assertNotIn(leak, body, f"{leak} leaked into {url}")

    def test_the_progress_line_reads_sensibly(self):
        self.client.post(reverse("project4:consent"), {"consent": "on"})
        self.assertRegex(
            self.client.get(reverse("project4:task")).content.decode(),
            "Block [0-9]+ &middot; question [0-9]+ of [0-9]+",
        )

    def test_a_whole_session_runs_through_both_conditions(self):
        participant = self._walk()
        counts = {
            condition: participant.tasks.filter(condition=condition).count()
            for condition in (PAIRWISE, RANKING)
        }
        self.assertEqual(counts[PAIRWISE], study.TASKS_PER_BLOCK[PAIRWISE])
        self.assertEqual(counts[RANKING], study.TASKS_PER_BLOCK[RANKING])
        self.assertEqual(Response.objects.count(), sum(counts.values()))
        participant.refresh_from_db()
        self.assertIsNotNone(participant.finished_at)

    def test_the_results_page_recommends_films_that_were_not_shown(self):
        self._walk()
        response = self.client.get(reverse("project4:results"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Films you might like")

    def test_a_pairwise_answer_is_stored_as_a_two_item_ranking(self):
        self._walk()
        pairwise = Response.objects.filter(task__condition=PAIRWISE).first()
        self.assertEqual(len(pairwise.ordering), 2)
        self.assertEqual(pairwise.ordering[0], pairwise.task.film_indices[0])

    def test_nothing_personal_is_stored(self):
        self._walk()
        fields = {f.name for f in Participant._meta.get_fields()}
        for forbidden in ("name", "email", "ip", "ip_address", "address"):
            self.assertNotIn(forbidden, fields)

    def test_the_report_downloads_as_a_pdf(self):
        response = self.client.get(reverse("project4:report"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Content-Type"], "application/pdf")
        self.assertIn("attachment", response.headers["Content-Disposition"])
        body = b"".join(response.streaming_content)
        self.assertTrue(body.startswith(b"%PDF"))
        self.assertGreater(len(body), 10_000)
