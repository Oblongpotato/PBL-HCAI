"""Tests for project 2.

The interesting assertions are numerical: that the hand-written PDP and ALE are what they
claim to be, and that the counterfactuals really are classified as the target by the model
that was selected.
"""

import numpy as np
from django.test import TestCase, override_settings
from pandas.api import types as ptypes
from django.urls import reverse

from . import counterfactuals, data, effects, training

FEATURE = "bill_length_mm"


class DataTests(TestCase):
    def test_complete_case_dataset(self):
        self.assertEqual(len(data.load()), 333)
        self.assertEqual(data.species(), ["Adelie", "Chinstrap", "Gentoo"])

    def test_year_is_treated_as_a_category(self):
        self.assertIn("year", data.CATEGORICAL_FEATURES)
        self.assertFalse(ptypes.is_numeric_dtype(data.load()["year"]))


class SelectionTests(TestCase):
    def test_lambda_never_increases_complexity(self):
        for family in training.FAMILIES:
            top = training.lambda_max(family)
            complexities = [
                training.select(family, top * step / 20)["complexity"] for step in range(21)
            ]
            self.assertEqual(complexities, sorted(complexities, reverse=True))

    def test_each_family_offers_several_models_across_its_own_range(self):
        # A single shared range left one family with a dead slider.
        for family in training.FAMILIES:
            top = training.lambda_max(family)
            chosen = {training.select(family, top * step / 100)["value"] for step in range(101)}
            self.assertGreaterEqual(len(chosen), 3, family)

    def test_zero_lambda_maximises_accuracy(self):
        for family in training.FAMILIES:
            best = max(entry["accuracy"] for entry in training.pool(family))
            self.assertEqual(training.select(family, 0.0)["accuracy"], best)

    def test_tree_complexity_is_its_leaf_count(self):
        entry = training.select("tree", 0.0)
        self.assertEqual(entry["complexity"], entry["pipeline"].named_steps["model"].get_n_leaves())

    def test_fully_penalised_logistic_model_keeps_no_features(self):
        entry = training.select("logistic", training.lambda_max("logistic"))
        self.assertEqual(entry["complexity"], 0)
        self.assertEqual(training.used_features(entry["pipeline"]), [])

    def test_logistic_complexity_counts_surviving_features(self):
        entry = training.select("logistic", 0.05)
        self.assertEqual(entry["complexity"], len(training.used_features(entry["pipeline"])))

    def test_tree_thresholds_stay_in_original_units(self):
        # Scaling a tree would make its rendered thresholds unreadable.
        pipeline = training.select("tree", 0.0)["pipeline"]
        self.assertGreater(pipeline.named_steps["model"].tree_.threshold.max(), 10)


class CounterfactualTests(TestCase):
    def test_counterfactuals_are_classified_as_the_target(self):
        for family in training.FAMILIES:
            pipeline = training.select(family, 0.0)["pipeline"]
            _, x = data.example(7)
            found = counterfactuals.generate(pipeline, x, "Gentoo")
            self.assertFalse(found.empty)
            predicted = pipeline.predict(found[data.FEATURES])
            self.assertTrue(all(name == "Gentoo" for name in predicted))

    def test_results_are_ordered_by_distance(self):
        pipeline = training.select("tree", 0.0)["pipeline"]
        _, x = data.example(7)
        distances = counterfactuals.generate(pipeline, x, "Chinstrap")["distance"].tolist()
        self.assertEqual(distances, sorted(distances))

    def test_mad_is_positive_for_every_numeric_feature(self):
        self.assertTrue(all(value > 0 for value in counterfactuals.mad().values()))


class EffectTests(TestCase):
    def test_pdp_probabilities_sum_to_one(self):
        pipeline = training.select("logistic", 0.0)["pipeline"]
        grid = effects.grid_for(FEATURE)
        curves = effects.pdp(pipeline, FEATURE, grid)
        self.assertTrue(np.allclose(sum(curves.values()), 1.0))

    def test_exact_and_discretised_ale_agree_for_logistic_regression(self):
        pipeline = training.select("logistic", 0.0)["pipeline"]
        _, finite = effects.ale_finite(pipeline, FEATURE)
        _, exact = effects.ale_exact(pipeline, FEATURE)
        for name in data.species():
            self.assertLess(np.abs(finite[name] - exact[name]).max(), 0.01)

    def test_a_tree_has_no_exact_derivative(self):
        pipeline = training.select("tree", 0.0)["pipeline"]
        edges, curves = effects.ale_exact(pipeline, FEATURE)
        self.assertIsNone(edges)
        self.assertIsNone(curves)

    def test_ale_is_centred(self):
        pipeline = training.select("tree", 0.0)["pipeline"]
        _, curves = effects.ale_finite(pipeline, FEATURE)
        for values in curves.values():
            self.assertLess(abs(float(values.mean())), 0.5)


@override_settings(ALLOWED_HOSTS=["testserver"])
class PageTests(TestCase):
    def test_page_renders_for_both_families(self):
        for family in training.FAMILIES:
            response = self.client.get(reverse("project2:index"), {"family": family})
            self.assertEqual(response.status_code, 200)
            self.assertContains(response, "333 complete rows")

    def test_every_section_is_present(self):
        response = self.client.get(reverse("project2:index"))
        for heading in ("1. The data", "3. The model", "4. Counterfactuals",
                        "5. Feature effect plots"):
            self.assertContains(response, heading)

    def test_bad_query_parameters_do_not_break_the_page(self):
        for params in (
            {"family": "nonsense"},
            {"lam": "abc"},
            {"lam": "-5"},
            {"lam": "999"},
            {"row": "99999"},
            {"row": "abc"},
            {"target": "Penguin"},
            {"feature": "nonsense"},
        ):
            self.assertEqual(
                self.client.get(reverse("project2:index"), params).status_code, 200, params
            )

    def test_the_derivative_note_matches_the_model(self):
        tree = self.client.get(reverse("project2:index"), {"family": "tree"})
        self.assertContains(tree, "piecewise constant")
        logistic = self.client.get(reverse("project2:index"), {"family": "logistic"})
        self.assertNotContains(logistic, "piecewise constant")
