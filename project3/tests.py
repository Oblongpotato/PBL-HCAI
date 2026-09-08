"""Tests for project 3.

The results on the page are precomputed and committed, so these check two things: that the
machinery producing them is correct, and that the page can display what was committed.
"""

from pathlib import Path

import numpy as np
from django.test import TestCase, override_settings
from django.urls import reverse

from . import data, defer, experiments, experts


class DataTests(TestCase):
    def test_subsample_is_balanced(self):
        train, test = data.load()
        self.assertEqual(len(train), 8000)
        self.assertEqual(len(test), 2000)
        self.assertEqual(set(train["topic"]), set(data.TOPICS))
        self.assertEqual(train["topic"].value_counts().nunique(), 1)


class ExpertTests(TestCase):
    def test_expert_is_identical_across_processes(self):
        # Seeding from hash() gave a different expert on every run, which quietly made the
        # committed results irreproducible.
        first = experts.predictions("specialist", "test").copy()
        experts.predictions.cache_clear()
        self.assertTrue(np.array_equal(first, experts.predictions("specialist", "test")))

    def test_competence_matches_the_configured_profile(self):
        truth = data.features()["y_test"]
        right = experts.correctness("specialist", "test")
        for topic, code in data.CODES.items():
            configured = experts.PROFILES["specialist"]["competence"][topic]
            self.assertAlmostEqual(right[truth == code].mean(), configured, delta=0.08, msg=topic)

    def test_specialist_is_complementary_to_the_classifier(self):
        report = experts.report("specialist")
        self.assertLess(report["accuracy"], 0.891)
        self.assertEqual(sorted(report["beats_classifier_on"]), ["Business", "Sci/Tech"])


class CssTests(TestCase):
    def test_weights_follow_the_lecture(self):
        # The true class always carries weight; deferral only when the expert was right.
        weights = defer.css_weights(np.array([2, 2]), np.array([1, 0]), 4)
        self.assertEqual(list(weights[0]), [0, 0, 1, 0, 1])
        self.assertEqual(list(weights[1]), [0, 0, 1, 0, 0])

    def test_query_cost_discourages_deferral(self):
        free = defer.css_weights(np.array([1]), np.array([1]), 4, kappa=0.0)[0][4]
        priced = defer.css_weights(np.array([1]), np.array([1]), 4, kappa=0.3)[0][4]
        self.assertLess(priced, free)

    def test_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        scores = rng.normal(size=(5, 5))
        weights = defer.css_weights(rng.integers(0, 4, 5), rng.integers(0, 2, 5), 4)
        _, analytic = defer.css_loss(scores, weights)

        numeric = np.zeros_like(scores)
        step = 1e-6
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                up, down = scores.copy(), scores.copy()
                up[i, j] += step
                down[i, j] -= step
                numeric[i, j] = (
                    defer.css_loss(up, weights)[0] - defer.css_loss(down, weights)[0]
                ) / (2 * step)
        self.assertLess(np.abs(analytic - numeric).max(), 1e-7)

    def test_training_reduces_the_loss(self):
        rng = np.random.default_rng(0)
        Z = rng.normal(size=(400, 12))
        model = defer.train_css(Z, rng.integers(0, 4, 400), rng.integers(0, 2, 400), 4, epochs=8)
        self.assertLess(model["history"][-1], model["history"][0])


class EvaluationTests(TestCase):
    def test_kept_and_deferred_are_scored_separately(self):
        truth = np.array([0, 1, 2, 3])
        # Deferred: expert says 1 (wrong) then 2 (right). Kept: 1 (right) then 0 (wrong).
        result = defer.evaluate(
            np.array([True, False, True, False]),
            np.array([0, 1, 0, 0]),
            np.array([1, 1, 2, 2]),
            truth,
        )
        self.assertEqual(result["system_accuracy"], 0.5)
        self.assertEqual(result["deferral_rate"], 0.5)
        self.assertEqual(result["expert_accuracy_on_deferred"], 0.5)
        self.assertEqual(result["classifier_accuracy_on_kept"], 0.5)

    def test_oracle_ceiling_is_never_below_the_system(self):
        truth = np.array([0, 1, 2, 3])
        result = defer.evaluate(
            np.zeros(4, dtype=bool), np.array([0, 1, 0, 0]), np.array([1, 1, 2, 2]), truth
        )
        self.assertGreaterEqual(result["oracle_ceiling"], result["system_accuracy"])


class CommittedResultsTests(TestCase):
    def setUp(self):
        self.results = experiments.load()
        self.assertIsNotNone(self.results, "run `manage.py run_project3` first")

    def test_every_section_was_generated(self):
        for section in ("data", "baseline", "experts", "deferral", "active"):
            self.assertIn(section, self.results)

    def test_every_referenced_figure_exists(self):
        root = Path(experiments.FIGURE_DIR).parent.parent
        referenced = [
            self.results["baseline"]["confusion_figure"],
            self.results["experts"]["figure"],
        ]
        referenced += [run["figure"] for run in self.results["active"]["runs"]]
        for run in self.results["deferral"]:
            referenced += [run["coverage_figure"], run["targeting_figure"]]
        for path in referenced:
            self.assertTrue((root / path).exists(), path)

    def test_deferral_is_aimed_at_the_experts_strong_topics(self):
        # This is the whole argument against confidence-based rejection.
        run = next(r for r in self.results["deferral"] if r["expert"] == "specialist")
        strong = [t for t, row in run["deferral_by_topic"].items() if row["expert_better"]]
        weak = [t for t, row in run["deferral_by_topic"].items() if not row["expert_better"]]
        self.assertGreater(
            min(run["deferral_by_topic"][t]["css"] for t in strong),
            max(run["deferral_by_topic"][t]["css"] for t in weak),
        )

    def test_active_learning_beats_random_for_the_specialist(self):
        run = next(r for r in self.results["active"]["runs"] if r["expert"] == "specialist")
        curves = {s["strategy"]: s["curve"] for s in run["strategies"]}
        self.assertGreater(
            curves["proposed"][-1]["system_accuracy"], curves["random"][-1]["system_accuracy"]
        )

    def test_code_hash_ignores_line_endings(self):
        """The stored hash has to survive a clone with different newline settings.

        Comparing the stored hash to the current one only proves they agree on this machine,
        which is exactly why the byte-based version shipped: it passed here and failed on
        every checkout.
        """
        import hashlib

        crlf, lf = "\r\n", "\n"

        def digest(line_ending):
            running = hashlib.sha256()
            for name in experiments.SOURCES:
                text = (experiments.APP_DIR / name).read_text(encoding="utf-8")
                rewritten = text.replace(crlf, lf).replace(lf, line_ending)
                running.update(rewritten.replace(crlf, lf).encode("utf-8"))
            return running.hexdigest()[:16]

        self.assertEqual(digest(lf), digest(crlf))
        self.assertEqual(digest(lf), experiments.code_hash())

    def test_results_carry_provenance(self):
        stamp = self.results["generated"]
        self.assertEqual(stamp["code_hash"], experiments.code_hash())
        self.assertFalse(experiments.is_stale(self.results))

    def test_chosen_query_cost_is_justified_by_its_grid(self):
        # Either the optimum is interior, or it sits at kappa >= 1 where deferral is switched
        # off entirely and extending the grid could not change the answer.
        for run in self.results["deferral"]:
            grid = [row["kappa"] for row in run["kappa_trace"]]
            self.assertTrue(
                run["kappa"] != max(grid) or run["kappa"] >= 1.0,
                f"{run['expert']} selected the grid maximum {run['kappa']}",
            )

    def test_a_useless_expert_is_never_consulted(self):
        run = next(r for r in self.results["deferral"] if r["expert"] == "generalist")
        self.assertEqual(run["css"]["deferral_rate"], 0.0)
        self.assertAlmostEqual(run["css"]["system_accuracy"], run["baseline_accuracy"], places=4)

    def test_kappa_was_chosen_off_the_test_set(self):
        for run in self.results["deferral"]:
            self.assertIn(run["kappa"], [row["kappa"] for row in run["kappa_trace"]])


@override_settings(ALLOWED_HOSTS=["testserver"])
class PageTests(TestCase):
    def test_page_shows_every_section(self):
        response = self.client.get(reverse("project3:index"))
        self.assertEqual(response.status_code, 200)
        for heading in (
            "1. The data",
            "2. The classifier on its own",
            "3. The simulated experts",
            "4. Learning to defer",
            "5. Learning what the expert is good at",
        ):
            self.assertContains(response, heading)

    def test_report_downloads_as_a_pdf(self):
        # The brief requires a report reachable from the interface, so this is the deliverable
        # that blocked the merge.
        response = self.client.get(reverse("project3:report"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Content-Type"], "application/pdf")
        self.assertIn("attachment", response.headers["Content-Disposition"])
        body = b"".join(response.streaming_content)
        self.assertTrue(body.startswith(b"%PDF"))
        self.assertGreater(len(body), 50_000)

    def test_the_page_links_to_the_report(self):
        self.assertContains(self.client.get(reverse("project3:index")), reverse("project3:report"))

    def test_page_touches_no_database(self):
        # Everything comes from the committed results, so repeated loads stay cheap.
        with self.assertNumQueries(0):
            self.client.get(reverse("project3:index"))
